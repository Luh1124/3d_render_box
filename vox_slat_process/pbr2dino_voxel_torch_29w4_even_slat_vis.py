import os  
import copy
import importlib
import argparse
import json
from typing import List, Dict, Tuple, Optional
from matplotlib.pyplot import thetagrids
import numpy as np
import pandas as pd  
import pyexr  
import cv2  
import utils3d  
from tqdm import tqdm  
from concurrent.futures import ThreadPoolExecutor  
from queue import Queue
import argparse  
import numpy as np
from PIL import Image
from torchvision import transforms
import open3d as o3d
import torch
import torch.nn.functional as F
from easydict import EasyDict as edict
import logging
import gc
import time
import traceback
from queue import Empty, Full

def clear_gpu_memory():
    """清理GPU内存"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

logger = logging.getLogger(__name__)

os.environ['ATTN_BACKEND'] = 'xformers'   # Can be 'flash-attn' or 'xformers', default is 'flash-attn'

import trellis.models as models
import trellis.modules.sparse as sp

'''
x,y,z 3 channels, roughness 1 channel、metallic 1channel、base color 3 channels、相机空间法线 (0-1 范围) 3 channels
'''

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

torch.set_grad_enabled(False)
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HOME'] = '/baai-cwm-vepfs/cwm/hong.li/.cache/huggingface'
os.environ['TORCH_HOME'] = '/baai-cwm-vepfs/cwm/hong.li/.cache/torch'
os.environ['GRADIO_TEMP_DIR'] = '/baai-cwm-vepfs/cwm/hong.li/.cache/gradio'


def adaptive_depth_threshold(depth_values: torch.Tensor) -> float:
    """
    根据深度值范围自适应计算深度阈值
    
    Args:
        depth_values: 深度值张量
        
    Returns:
        适应性阈值
    """
    depth_range = depth_values.max() - depth_values.min()
    # float16 精度约为 0.001 在 [1,2] 范围内，根据深度范围调整
    base_threshold = 0.005  # float16 基础阈值
    adaptive_threshold = base_threshold + depth_range * 0.01
    return float(adaptive_threshold.clamp(0.005, 0.1))  # 限制在合理范围内


def robust_depth_visibility_check(positions: torch.Tensor, 
                                 depth_maps: torch.Tensor, 
                                 extrinsics: torch.Tensor, 
                                 intrinsics: torch.Tensor,
                                 base_threshold: float = 0.01,
                                 use_adaptive: bool = True,
                                 use_percentile: bool = True,
                                 neighborhood_size: int = 3) -> torch.Tensor:
    """
    针对 float16 深度图优化的可见性检查
    
    Args:
        positions: [N, 3] 3D点坐标
        depth_maps: [B, H, W] 各视角的深度图 (float16)
        extrinsics: [B, 4, 4] 相机外参矩阵
        intrinsics: [B, 3, 3] 相机内参矩阵
        base_threshold: 基础深度阈值
        use_adaptive: 是否使用自适应阈值
        use_percentile: 是否使用百分位数滤波
        neighborhood_size: 邻域大小用于深度滤波
        
    Returns:
        visibility_mask: [B, N] 布尔掩码，True表示可见
    """
    B, H, W = depth_maps.shape
    N = positions.shape[0]
    
    # 转换为float32进行精确计算
    depth_maps_f32 = depth_maps.float()
    
    # 将3D点投影到各个视角
    uv, depths_pred = utils3d.torch.project_cv(positions, extrinsics, intrinsics)
    depths_pred = depths_pred.float()  # 确保使用float32
    
    # 标准化UV坐标到[-1,1]
    uv_normalized = uv * 2 - 1  # [B, N, 2]
    
    # 多种采样策略组合
    visibility_masks = []
    
    # 策略1: 双线性插值采样
    depths_sampled_bilinear = F.grid_sample(
        depth_maps_f32.unsqueeze(1),  # [B, 1, H, W]
        uv_normalized.unsqueeze(2),   # [B, N, 1, 2]
        mode='bilinear',
        align_corners=False,
        padding_mode='border'
    ).squeeze(1).squeeze(2)  # [B, N]
    
    # 策略2: 最近邻采样（更鲁棒对float16误差）
    depths_sampled_nearest = F.grid_sample(
        depth_maps_f32.unsqueeze(1),
        uv_normalized.unsqueeze(2),
        mode='nearest',
        align_corners=False,
        padding_mode='border'
    ).squeeze(1).squeeze(2)  # [B, N]
    
    # 策略3: 邻域采样（获取局部深度统计）
    if neighborhood_size > 1:
        # 创建邻域偏移
        offset_range = neighborhood_size // 2
        offsets = torch.meshgrid(
            torch.arange(-offset_range, offset_range + 1, device=uv.device),
            torch.arange(-offset_range, offset_range + 1, device=uv.device),
            indexing='ij'
        )
        offsets = torch.stack([offsets[1], offsets[0]], dim=-1).float()  # [nh, nw, 2]
        offsets = offsets.reshape(-1, 2) / torch.tensor([W, H], device=uv.device) * 2  # 归一化
        
        # 为每个点采样邻域
        neighborhood_depths = []
        for offset in offsets:
            offset_uv = uv_normalized + offset.unsqueeze(0).unsqueeze(0)  # [B, N, 2]
            neighbor_depth = F.grid_sample(
                depth_maps_f32.unsqueeze(1),
                offset_uv.unsqueeze(2),
                mode='bilinear',
                align_corners=False,
                padding_mode='border'
            ).squeeze(1).squeeze(2)
            neighborhood_depths.append(neighbor_depth)
        
        neighborhood_depths = torch.stack(neighborhood_depths, dim=-1)  # [B, N, num_neighbors]
        
        # 使用中位数或百分位数作为鲁棒估计
        if use_percentile:
            depths_sampled_robust = torch.median(neighborhood_depths, dim=-1)[0]
        else:
            depths_sampled_robust = neighborhood_depths.mean(dim=-1)
    else:
        depths_sampled_robust = depths_sampled_bilinear
    
    # 计算自适应阈值
    thresholds = []
    for b in range(B):
        if use_adaptive:
            valid_depths = depth_maps_f32[b][depth_maps_f32[b] > 0]
            if len(valid_depths) > 0:
                adaptive_thresh = adaptive_depth_threshold(valid_depths)
                thresholds.append(adaptive_thresh)
            else:
                thresholds.append(base_threshold)
        else:
            thresholds.append(base_threshold)
    
    thresholds = torch.tensor(thresholds, device=positions.device)  # [B]
    
    # 多策略可见性判断
    strategies = [
        ('bilinear', depths_sampled_bilinear),
        ('nearest', depths_sampled_nearest), 
        ('robust', depths_sampled_robust)
    ]
    
    strategy_masks = []
    for strategy_name, depths_sampled in strategies:
        # 绝对深度差异
        depth_diff = torch.abs(depths_pred - depths_sampled)
        
        # 相对深度差异（对远距离点更宽松）
        relative_threshold = thresholds.unsqueeze(1) * (1 + depths_pred * 0.1)
        
        # 组合判断
        abs_vis = depth_diff < thresholds.unsqueeze(1)
        rel_vis = depth_diff < relative_threshold
        
        strategy_mask = abs_vis | rel_vis  # 任一条件满足即可见
        strategy_masks.append(strategy_mask)
    
    # 投票机制：多数策略认为可见才算可见
    visibility_vote = torch.stack(strategy_masks, dim=-1).sum(dim=-1)  # [B, N]
    visibility_mask = visibility_vote >= (len(strategies) // 2 + 1)
    
    # 基本约束检查
    in_bounds = (uv[..., 0] >= 0) & (uv[..., 0] <= 1) & \
                (uv[..., 1] >= 0) & (uv[..., 1] <= 1)
    valid_depth = depths_pred > 0
    sampled_valid = depths_sampled_robust > 0
    
    # 深度合理性检查：避免过远或过近的点
    depth_reasonable = (depths_pred > 0.1) & (depths_pred < 10.0)  # 根据场景调整
    
    # 综合可见性判断
    final_mask = visibility_mask & in_bounds & valid_depth & sampled_valid & depth_reasonable
    
    return final_mask


def statistical_outlier_filtering(visibility_mask: torch.Tensor, 
                                positions: torch.Tensor,
                                k: int = 10, 
                                std_ratio: float = 2.0) -> torch.Tensor:
    """
    基于统计的异常值过滤，去除孤立的可见点
    
    Args:
        visibility_mask: [B, N] 初始可见性掩码
        positions: [N, 3] 3D点坐标
        k: 邻居数量
        std_ratio: 标准差比率阈值
        
    Returns:
        filtered_mask: [B, N] 过滤后的可见性掩码
    """
    from sklearn.neighbors import NearestNeighbors
    
    B, N = visibility_mask.shape
    filtered_mask = visibility_mask.clone()
    
    # 对每个视角独立处理
    for b in range(B):
        visible_indices = visibility_mask[b].nonzero().squeeze(1)
        if len(visible_indices) < k:
            continue
            
        visible_positions = positions[visible_indices].cpu().numpy()
        
        # 构建KD树
        nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='kd_tree').fit(visible_positions)
        distances, _ = nbrs.kneighbors(visible_positions)
        
        # 计算到邻居的平均距离（排除自己）
        mean_distances = distances[:, 1:].mean(axis=1)
        
        # 统计滤波
        global_mean = mean_distances.mean()
        global_std = mean_distances.std()
        
        # 标记异常值
        outliers = mean_distances > (global_mean + std_ratio * global_std)
        
        # 更新掩码
        outlier_indices = visible_indices[outliers]
        filtered_mask[b, outlier_indices] = False
    
    return filtered_mask


def enhanced_visibility_check_float16(positions: torch.Tensor,
                                     data: List[Dict],
                                     use_statistical_filter: bool = True,
                                     **kwargs) -> torch.Tensor:
    """
    专门针对float16深度图的增强可见性检查
    
    Args:
        positions: [N, 3] 3D点坐标
        data: 包含各视角数据的列表
        use_statistical_filter: 是否使用统计滤波
        **kwargs: 其他参数
        
    Returns:
        visibility_mask: [B, N] 最终可见性掩码
    """
    # 准备数据
    depth_maps = torch.stack([d['depth'] for d in data])  # [B, H, W]
    extrinsics = torch.stack([d['extrinsics'] for d in data])  # [B, 4, 4]
    intrinsics = torch.stack([d['intrinsics'] for d in data])  # [B, 3, 3]
    
    # 确保深度图为float16但计算用float32
    if depth_maps.dtype != torch.float16:
        print(f"Warning: depth_maps dtype is {depth_maps.dtype}, expected float16")
    
    # 鲁棒深度可见性检查
    visibility_mask = robust_depth_visibility_check(
        positions=positions,
        depth_maps=depth_maps,
        extrinsics=extrinsics,
        intrinsics=intrinsics,
        base_threshold=kwargs.get('depth_threshold', 0.015),  # float16 适配
        use_adaptive=kwargs.get('use_adaptive', True),
        use_percentile=kwargs.get('use_percentile', True),
        neighborhood_size=kwargs.get('neighborhood_size', 3)
    )
    
    # 统计异常值过滤
    if use_statistical_filter:
        visibility_mask = statistical_outlier_filtering(
            visibility_mask, positions,
            k=kwargs.get('filter_k', 10),
            std_ratio=kwargs.get('filter_std_ratio', 2.0)
        )
    
    return visibility_mask


# 修改后的特征采样函数
def float16_optimized_feature_sampling(positions: torch.Tensor,
                                      data: List[Dict],
                                      patchtokens: torch.Tensor,
                                      opt,
                                      **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    针对float16深度图优化的特征采样
    """
    n_views = len(data)
    N, C = positions.shape[0], patchtokens.shape[1]
    
    # 获取可见性掩码
    visibility_mask = enhanced_visibility_check_float16(
        positions=positions,
        data=data,
        depth_threshold=0.015,  # 针对float16调整
        use_adaptive=True,
        use_percentile=True,
        neighborhood_size=3,
        use_statistical_filter=True,
        filter_k=8,
        filter_std_ratio=1.5
    )
    
    # 投影坐标计算
    extrinsics = torch.stack([d['extrinsics'] for d in data])
    intrinsics = torch.stack([d['intrinsics'] for d in data])
    uv, _ = utils3d.torch.project_cv(positions, extrinsics, intrinsics)
    uv_normalized = uv * 2 - 1  # [B, N, 2]
    
    # 加权特征聚合
    aggregated_features = torch.zeros(N, C, device=positions.device, dtype=torch.float32)
    view_weights = torch.zeros(N, device=positions.device, dtype=torch.float32)
    
    for i in range(0, n_views, opt.batch_size):
        end_idx = min(i + opt.batch_size, n_views)
        batch_size = end_idx - i
        
        batch_tokens = patchtokens[i:end_idx]  # [batch_size, C, H, W]
        batch_uv = uv_normalized[i:end_idx]    # [batch_size, N, 2]
        batch_visibility = visibility_mask[i:end_idx]  # [batch_size, N]
        
        # 采样特征
        sampled_features = F.grid_sample(
            batch_tokens,
            batch_uv.unsqueeze(2),  # [batch_size, N, 1, 2]
            mode='bilinear',
            align_corners=False,
            padding_mode='zeros'
        ).squeeze(3).permute(0, 2, 1)  # [batch_size, N, C]
        
        # 应用可见性权重
        for b in range(batch_size):
            valid_mask = batch_visibility[b]  # [N]
            if valid_mask.any():
                # 简单权重策略：可见即权重为1
                weight = valid_mask.float()
                aggregated_features += sampled_features[b] * weight.unsqueeze(1)
                view_weights += weight
    
    # 归一化
    valid_points = view_weights > 0
    if valid_points.any():
        aggregated_features[valid_points] /= view_weights[valid_points].unsqueeze(1)
    
    # 过滤有效点
    min_views = kwargs.get('min_visible_views', 2)  # float16情况下降低要求
    sufficient_mask = view_weights >= min_views
    
    if sufficient_mask.sum() == 0:
        print("Warning: No points have sufficient visibility, using all points")
        sufficient_mask = torch.ones(N, dtype=torch.bool, device=positions.device)
    
    valid_positions = positions[sufficient_mask]
    valid_features = aggregated_features[sufficient_mask]
    valid_indices = ((valid_positions + 0.5) * 64).long().to(torch.int32)
    
    print(f"Visibility filtering: {N} -> {sufficient_mask.sum()} points "
          f"({sufficient_mask.sum()/N*100:.1f}% retained)")
    
    return valid_features, valid_indices, sufficient_mask


def sampled_mean_grid_sample_chunked(tensor, uv, num_chunks=8):  
    B = uv.shape[0]  
    chunk_size = (B + num_chunks - 1) // num_chunks  # 向上取整分块大小  
    
    total = 0  
    count = 0  
    for i in range(num_chunks):  
        start = i * chunk_size  
        end = min(start + chunk_size, B)  
        if start >= end:  # 防止块超出范围  
            break  
        
        uv_chunk = uv[start:end]  # [chunk, N, 2]  
        tensor_chunk = tensor[start:end]  # [chunk, C, H, W]  
        
        sampled = F.grid_sample(  
            tensor_chunk,  
            uv_chunk.unsqueeze(1),  # [chunk, 1, N, 2]  
            mode='bilinear',  
            align_corners=False  
        )  # [chunk, C, 1, N]  
        
        sampled = sampled.squeeze(2).permute(0, 2, 1)  # [chunk, N, C]  
        total += sampled.sum(dim=0)  # 累积 [N, C]  
        count += (end - start)  
        
    mean = total / count  # [N, C]  
    return mean  

def image_uv(height: int, width: int, left: int = None, top: int = None, right: int = None, bottom: int = None, device: torch.device = None, dtype: torch.dtype = None) -> torch.Tensor:
    """
    Get image space UV grid, ranging in [0, 1]. 

    >>> image_uv(10, 10):
    [[[0.05, 0.05], [0.15, 0.05], ..., [0.95, 0.05]],
     [[0.05, 0.15], [0.15, 0.15], ..., [0.95, 0.15]],
      ...             ...                  ...
     [[0.05, 0.95], [0.15, 0.95], ..., [0.95, 0.95]]]

    Args:
        width (int): image width
        height (int): image height

    Returns:
        torch.Tensor: shape (height, width, 2)
    """
    if left is None: left = 0
    if top is None: top = 0
    if right is None: right = width
    if bottom is None: bottom = height
    u = torch.linspace((left + 0.5) / width, (right - 0.5) / width, right - left, device=device, dtype=dtype)
    v = torch.linspace((top + 0.5) / height, (bottom - 0.5) / height, bottom - top, device=device, dtype=dtype)
    u, v = torch.meshgrid(u, v, indexing='xy')
    uv = torch.stack([u, v], dim=-1)
    return uv

def depth_to_points(depth: torch.Tensor, intrinsics: torch.Tensor, extrinsics: torch.Tensor = None):
    height, width = depth.shape[-2:]
    uv = image_uv(width=width, height=height, dtype=depth.dtype, device=depth.device)
    pts = utils3d.torch.transforms.unproject_cv(uv, depth, intrinsics=intrinsics[..., None, :, :], extrinsics=extrinsics[..., None, :, :] if extrinsics is not None else None)
    return pts

def filter_metadata(metadata, opt):
    """过滤已处理的元数据条目"""
    logger.info(f"开始过滤元数据，共{len(metadata)}条记录")
    
    # 步骤1: 并行检查文件存在性
    with ThreadPoolExecutor(max_workers=8) as executor:
        # 生成检查任务 - 直接使用 SHA256 构建文件路径
        futures = [  
            executor.submit(  
                lambda sha256=sha256: 
                    os.path.exists(  
                        os.path.join(  
                            opt.output_dir,  
                            opt.feature_dir,  
                            f"{sha256}.npz"  
                        )   
                    )  
            )  
            for sha256 in metadata['sha256'].values  
        ] 
        
        # 获取存在文件的sha256列表
        exists_flags = []
        for future in tqdm(futures, desc='检查已处理文件'):
            exists_flags.append(future.result())
    
    # 步骤2: 批量过滤数据
    filtered_metadata = metadata[~np.array(exists_flags)].copy()
    
    logger.info(f"过滤完成，共找到{sum(exists_flags)}个已处理文件，剩余{len(filtered_metadata)}条待处理")
    return filtered_metadata.reset_index(drop=True)


def get_data(input_dir, frames, sha256):  
    """Load depth, PBR maps, and camera parameters for each view."""  
    with ThreadPoolExecutor(max_workers=16) as executor:  
    # with ThreadPoolExecutor(max_workers=1) as executor:  
        def worker(view):  
            depth_path = os.path.join(input_dir, sha256, view['file_path'].replace("image", "depth").replace(".png", "_depth.exr"))  
            normal_path = os.path.join(input_dir, sha256, view['file_path'].replace("image", "normal").replace(".png", "_normal.exr"))  
            # color_path = os.path.join(input_dir, sha256, view['file_path'].replace("image", "Base Color").replace(".png", "_Base Color.png"))  
            # roughness_path = os.path.join(input_dir, sha256, view['file_path'].replace("image", "Roughness").replace(".png", "_Roughness.exr"))  
            # metallic_path = os.path.join(input_dir, sha256, view['file_path'].replace("image", "Metallic").replace(".png", "_Metallic.exr"))  
            image_path = os.path.join(input_dir, sha256, view['file_path'].replace("image/", ""))

            try:  
                # import ipdb; ipdb.set_trace()
                # if depth_path == "/baai-cwm-1/baai_cwm_ml/cwm/hong.li/code/3dgen/data/zeroverse/dorabenchmark/even/Level2/DTC_1_0_Basketball_B09QMRRHKG_BlackBlue_3d-asset/depth/137_depth.exr":
                    # print(depth_path)
                depth = pyexr.read(depth_path, precision=pyexr.HALF).astype(np.float32)  
                normal = pyexr.read(normal_path, precision=pyexr.HALF).astype(np.float32)  
                # color = cv2.cvtColor(cv2.imread(color_path), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0   
                # roughness = pyexr.read(roughness_path, precision=pyexr.HALF).astype(np.float32)
                # metallic = pyexr.read(metallic_path, precision=pyexr.HALF).astype(np.float32)

                image = Image.open(image_path)
                # color = Image.open(color_path)

            except pyexr.exr.ExrError as e:  
                print(f"EXR读取错误: {e}") 
                return None

            except Exception as e:  
                # print(f"Error loading image {sha256}: {e}")  
                os.makedirs("error_images", exist_ok=True)
                with open(f"error_images/{sha256}.txt", 'w') as f:
                    f.write(f"Error loading image {sha256}: {e}")
                return None  

            image_size = (518, 518)
            depth = cv2.resize(depth, image_size, interpolation=cv2.INTER_NEAREST_EXACT)  # shape 
            normal = cv2.resize(normal, image_size, interpolation=cv2.INTER_NEAREST_EXACT)
            # roughness = cv2.resize(roughness, image_size, interpolation=cv2.INTER_NEAREST_EXACT)  
            # metallic = cv2.resize(metallic, image_size, interpolation=cv2.INTER_NEAREST_EXACT)  

            image = image.resize(image_size, Image.Resampling.LANCZOS)
            # color = color.resize(image_size, Image.Resampling.LANCZOS)

            image = np.array(image).astype(np.float32) / 255
            # color = np.array(color).astype(np.float32) / 255

            # color = color[:, :, :3] * image[:, :, 3:] 
            image = image[:, :, :3] * image[:, :, 3:]

            image = torch.from_numpy(image).permute(2, 0, 1).float()
            # color = torch.from_numpy(color).permute(2, 0, 1).float()

            mask = ((depth[..., 1] < 2.5) & (depth[..., 1] > 1.5)).astype(np.float32)             
            # mask = np.where(depth[..., 1] > 1., 1.0, 0.0)

            mask = torch.from_numpy(mask).float()

            depth = torch.from_numpy(depth).float()
            normal = torch.from_numpy(normal).float()
            # roughness = torch.from_numpy(roughness).float()
            # metallic = torch.from_numpy(metallic).float()

            # depth, normal, color, roughness, metallic = cp.array(depth), cp.array(normal), cp.array(color), cp.array(roughness), cp.array(metallic)

            # Compute mask from depth  
            # mask = cp.where(depth[..., 1] < 5., 1.0, 0.0)  

            # Compute camera extrinsics and intrinsics  
            # c2w = np.array(view['transform_matrix'], dtype=np.float32)  
            c2w = torch.tensor(view['transform_matrix'])

            normal_reshape = normal[...,:3].reshape(-1, 3)  
            normal_camera = normal_reshape @ c2w[:3, :3]  
            normal_camera = normal_camera.reshape(518, 518, 3)
            normal_camera_norm = normal_camera * 0.5 + 0.5  

            c2w[:3, 1:3] *= -1  
            # Compute camera intrinsics  
            extrinsics = torch.inverse(c2w)

            fov = view['camera_angle_x']  
            intrinsics = utils3d.torch.intrinsics_from_fov_xy(torch.tensor(fov), torch.tensor(fov))

            return {  
                'depth': depth[..., 1].cuda(),
                'normal': normal_camera_norm[...,:3].cuda(),
                # 'color': color.cuda(),
                # 'roughness': roughness[..., None].cuda(),  
                # 'metallic': metallic[..., None].cuda(),
                'image': image.cuda(),
                'mask': mask.cuda(),
                'extrinsics': extrinsics.cuda(),  
                'intrinsics': intrinsics.cuda()
            }  
        
        # Process all frames in parallel  
        datas = executor.map(worker, frames)  
        for data in datas:  
            if data is not None:  
                yield data 
        # return datas 


@torch.no_grad()
def gen_pbr_slat_1_step():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, required=False, default='/baai-cwm-backup/cwm/hong.li/data/3diclight_rendering/3diclight_even_not_in_pbr33w_29w4',
                        help='Directory to save the metadata')
    parser.add_argument('--input_dir', type=str, required=False, default='/baai-cwm-backup/cwm/hong.li/data/3diclight_rendering/3diclight_even_not_in_pbr33w_29w4/eevee_all',
                        help='Directory where the renders are stored')
    parser.add_argument('--feature_dir', type=str, required=False, default='pbr2dino_voxel_latent_visbility',
                        help='Directory to save the features')
    parser.add_argument('--model', type=str, default='dinov2_vitl14_reg',
                        help='Feature extraction model')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=6)

    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--queue_size', type=int, default=4)

    parser.add_argument('--debug', action='store_true', default=False)

    opt = parser.parse_args()
    # opt = edict(vars(opt))

    # feature_name = opt.model
    out_dir = os.path.join(opt.output_dir, opt.feature_dir)
    os.makedirs(out_dir, exist_ok=True)

        # 设置日志  
    logging.basicConfig(  
        level=logging.INFO,  
        format='%(asctime)s - %(levelname)s - %(message)s',  
        handlers=[  
            logging.FileHandler(os.path.join(opt.output_dir, "processing.log")),  
            logging.StreamHandler()  
        ]  
    )  

    dinov2_model = torch.hub.load('/baai-cwm-vepfs/cwm/hong.li/.cache/torch/hub/facebookresearch_dinov2_main', "dinov2_vitl14_reg", pretrained=True, source='local')    
    dinov2_model.eval().cuda()
    transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    n_patch = 518 // 14

    encoder = models.from_pretrained('/baai-cwm-vepfs/cwm/hong.li/code/3dgen/TRELLIS/TRELLIS-image-large/ckpts/slat_enc_swin8_B_64l8_fp16').eval().cuda()


    # 加载元数据
    if os.path.exists("metadatas/not_pbr_in_33w_29w4.csv"):
        metadata = pd.read_csv("metadatas/not_pbr_in_33w_29w4.csv")
    else:
        raise ValueError('not_pbr_in_33w_29w4.csv not found')

    start = len(metadata) * opt.rank // opt.world_size
    end = len(metadata) * (opt.rank + 1) // opt.world_size
    metadata = metadata[start:end]
    
        # 调试模式
    if opt.debug:
        metadata = metadata.head(10)
        opt.num_workers = 1
        opt.queue_size = 1

    metadata = filter_metadata(metadata, opt)

    print(f"Processing {len(metadata)} objects")

    sha256s = list(metadata['sha256'].values)

    # 添加停止事件
    import threading
    stop_event = threading.Event()
    
    # extract features
    load_queue = Queue(maxsize=opt.queue_size * 2)
    completed_count = 0  # 添加完成计数器

    try:
        with ThreadPoolExecutor(max_workers=opt.num_workers) as loader_executor, \
            ThreadPoolExecutor(max_workers=opt.num_workers) as saver_executor:
            
            def loader(sha256):
                try:
                    if stop_event.is_set():
                        return
                        
                    with open(os.path.join(opt.input_dir, sha256, 'transforms.json'), 'r') as f:
                        metadata = json.load(f)
                    frames = metadata['frames']
                    data = []
                    for datum in get_data(opt.input_dir, frames, sha256):
                        datum['image'] = transform(datum['image'])
                        datum['depth'] = datum['depth']
                        datum['normal'] = datum['normal']
                        datum['mask'] = datum['mask']
                        datum['extrinsics'] = datum['extrinsics']
                        datum['intrinsics'] = datum['intrinsics']
                        data.append(datum)

                    # 检查数据有效性
                    if len(data) < 5:  # 假设至少需要5个有效视图
                        print(f"Not enough valid views for {sha256}: {len(data)}")
                        return

                    voxel_path = os.path.join("/baai-cwm-backup/cwm/hong.li/data/3diclight_rendering/3diclight_even_not_in_pbr33w_29w4", 'voxel', f'{sha256}.ply')

                    if not os.path.exists(voxel_path):
                        print(f"Voxel file not found: {voxel_path}")
                        return

                    positions = utils3d.io.read_ply(voxel_path)[0]

                    # 添加超时和停止检查
                    timeout = 300  # 5分钟超时
                    start_time = time.time()
                    
                    while not stop_event.is_set():
                        try:
                            load_queue.put((sha256, data, positions), timeout=10)  # 短超时以便定期检查停止信号
                            break
                        except Full:
                            if time.time() - start_time > timeout:
                                print(f"Timeout putting {sha256} into queue after {timeout}s")
                                return
                            # 简短等待后重试
                            time.sleep(1)

                except Exception as e:
                    print(f"Error loading data for {sha256}: {e}")
                    traceback.print_exc()

            # 限制同时执行的加载任务数量，而不是一次提交所有
            active_loaders = set()
            pending_sha256s = list(sha256s)
            
            def submit_next_loaders():
                while len(active_loaders) < opt.num_workers * 2 and pending_sha256s and not stop_event.is_set():
                    sha256 = pending_sha256s.pop(0)
                    future = loader_executor.submit(loader, sha256)
                    active_loaders.add(future)
                    future.add_done_callback(lambda f: active_loaders.remove(f))
            
            # 提交初始批次的加载任务
            submit_next_loaders()

            def saver(sha256, pack):
                try:
                    save_path = os.path.join(opt.output_dir, opt.feature_dir, f'{sha256}.npz')
                    np.savez_compressed(save_path, **pack)
                    print(f"Successfully saved {sha256}")
                except Exception as e:
                    print(f"Error saving {sha256}: {e}")
                    traceback.print_exc()

            # 处理队列中的项目
            with tqdm(total=len(sha256s), desc='Processing') as pbar:
                while completed_count < len(sha256s) and not stop_event.is_set():
                    try:
                        # 尝试从队列获取项目，添加超时以检查停止事件
                        sha256, data, positions = load_queue.get(timeout=30)
                        
                        try:
                            # 将位置转换为张量
                            positions = torch.from_numpy(positions).float().cuda()
                            indices = ((positions + 0.5) * 64).long().to(torch.int32)
                            
                            # 使用try/except替代assert
                            if not (torch.all(indices >= 0) and torch.all(indices < 64)):
                                print(f"Vertices out of bounds for {sha256}, skipping")
                                completed_count += 1
                                pbar.update(1)
                                # 提交下一批加载任务
                                submit_next_loaders()
                                continue

                            n_views = len(data)
                            N = positions.shape[0]
                                        
                            patchtokens_lst = []
                            uv_lst = []

                            for i in range(0, n_views, opt.batch_size):
                                batch_data = data[i:i+opt.batch_size]
                                bs = len(batch_data)

                                # 准备批量数据
                                batch_image = torch.stack([d['image'] for d in batch_data])
                                batch_extrinsics = torch.stack([d['extrinsics'] for d in batch_data])
                                batch_intrinsics = torch.stack([d['intrinsics'] for d in batch_data])

                                # 提取特征
                                try:
                                    combined_features = dinov2_model(batch_image, is_training=True)  
                                    features_combined = combined_features['x_prenorm']  
                                except RuntimeError as e:
                                    if "CUDA out of memory" in str(e):
                                        print(f"CUDA OOM processing {sha256}, clearing memory and reducing batch size")
                                        clear_gpu_memory()
                                        
                                        # 尝试较小的批处理大小
                                        half_bs = max(1, bs // 2)
                                        features_list = []
                                        
                                        for j in range(0, bs, half_bs):
                                            mini_batch = batch_image[j:j+half_bs]
                                            mini_features = dinov2_model(mini_batch, is_training=True)
                                            features_list.append(mini_features['x_prenorm'])
                                            
                                        features_combined = torch.cat(features_list, dim=0)
                                    else:
                                        raise

                                uv = utils3d.torch.project_cv(positions, batch_extrinsics, batch_intrinsics)[0] * 2 - 1  
                                patchtokens = features_combined[:, dinov2_model.num_register_tokens + 1:].permute(0, 2, 1).reshape(bs, 1024, n_patch, n_patch)
                                
                                patchtokens_lst.append(patchtokens)
                                uv_lst.append(uv) 
                                
                                # 清理中间变量
                                del batch_image, batch_extrinsics, batch_intrinsics, combined_features, features_combined
                                clear_gpu_memory()

                            patchtokens = torch.cat(patchtokens_lst, dim=0)
                            uv = torch.cat(uv_lst, dim=0)

                            # 采样特征
                            # patchtokens_sampled = sampled_mean_grid_sample_chunked(patchtokens, uv, num_chunks=4)
                            patchtokens_sampled, valid_indices, visibility_mask = float16_optimized_feature_sampling(
                                positions=positions,
                                data=data,
                                patchtokens=patchtokens,
                                opt=opt,
                                depth_threshold=0.015,      # 针对float16调整
                                min_visible_views=2,        # 降低要求
                                use_adaptive=True,
                                neighborhood_size=3
                            )            

                            # 清理大型临时变量
                            del patchtokens, patchtokens_lst, uv, uv_lst
                            clear_gpu_memory()

                            batch_col = torch.zeros(valid_indices.shape[0], 1, dtype=torch.int32, device='cuda')
                            feats = sp.SparseTensor(
                                feats=patchtokens_sampled.float(),
                                coords=torch.cat([batch_col, valid_indices], dim=1),
                            )

                            latent = encoder(feats, sample_posterior=False)
                            # 准备结果包
                            pack = {
                                'feats': latent.feats.cpu().numpy().astype(np.float32),
                                'coords': latent.coords[:, 1:].cpu().numpy().astype(np.uint8),
                            }

                            # 清理最终变量
                            del feats, latent, patchtokens_sampled, batch_col, indices
                            clear_gpu_memory()

                            # 提交保存任务
                            saver_executor.submit(saver, sha256, pack)
                            
                        except Exception as e:
                            print(f"Error processing {sha256}: {e}")
                            traceback.print_exc()
                        
                        # 无论成功还是失败，都更新计数和进度条
                        completed_count += 1
                        pbar.update(1)
                        
                        # 提交下一批加载任务
                        submit_next_loaders()
                        
                    except Empty:
                        # 队列超时 - 检查是否所有加载器都完成了
                        if len(active_loaders) == 0 and len(pending_sha256s) == 0:
                            # 所有加载器都完成了，但队列为空，说明已经处理完了所有对象
                            if completed_count >= len(sha256s):
                                break
                            else:
                                # 可能有些对象被跳过了
                                print(f"Queue empty but only processed {completed_count}/{len(sha256s)}")
                                break
                    
                    except Exception as e:
                        print(f"Error in main loop: {e}")
                        traceback.print_exc()
                        stop_event.set()  # 发生异常时停止所有线程

            # 等待保存完成
            saver_executor.shutdown(wait=True)
            
    except KeyboardInterrupt:
        print("Keyboard interrupt detected, stopping gracefully")
        stop_event.set()  # 通知线程停止
        
    except Exception as e:
        print(f"Fatal error: {e}")
        traceback.print_exc()
        stop_event.set()  # 通知线程停止
        
    finally:
        print(f"Completed {completed_count}/{len(sha256s)} objects")

if __name__ == '__main__':
    gen_pbr_slat_1_step()
