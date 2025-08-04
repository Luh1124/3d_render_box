import os  
import copy
import importlib
import argparse
import json  
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
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--output_dir', type=str, required=False, default='/baai-cwm-1/baai_cwm_ml/cwm/hong.li/code/3dgen/data/zeroverse/3diclight_matsynth/even_slat_6k_debug',
    #                     help='Directory to save the metadata')
    # parser.add_argument('--input_dir', type=str, required=False, default='/baai-cwm-1/baai_cwm_ml/cwm/hong.li/code/3dgen/data/zeroverse/3diclight_matsynth/even_render_6k',
    #                     help='Directory where the renders are stored')
    # parser.add_argument('--feature_dir', type=str, required=False, default='pbr2dino_voxel_latent',
    #                     help='Directory to save the features')
    # parser.add_argument('--model', type=str, default='dinov2_vitl14_reg',
    #                     help='Feature extraction model')
    # parser.add_argument('--enc_model', type=str, default=None,
    #                     help='Encoder model. if specified, use this model instead of pretrained model')
    # parser.add_argument('--batch_size', type=int, default=16)
    # parser.add_argument('--rank', type=int, default=0)
    # parser.add_argument('--world_size', type=int, default=1)
    # opt = parser.parse_args()

    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, required=False, default='/baai-cwm-backup/cwm/hong.li/data/3diclight_rendering/3diclight_even_not_in_pbr33w_29w4',
                        help='Directory to save the metadata')
    parser.add_argument('--input_dir', type=str, required=False, default='/baai-cwm-backup/cwm/hong.li/data/3diclight_rendering/3diclight_even_not_in_pbr33w_29w4/eevee_all',
                        help='Directory where the renders are stored')
    parser.add_argument('--feature_dir', type=str, required=False, default='pbr2dino_voxel_latent',
                        help='Directory to save the features')
    parser.add_argument('--model', type=str, default='dinov2_vitl14_reg',
                        help='Feature extraction model')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=12)

    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--queue_size', type=int, default=4)

    parser.add_argument('--debug', action='store_true', default=False)

    opt = parser.parse_args()
    # opt = edict(vars(opt))

    # feature_name = opt.model
    out_dir = os.path.join(opt.output_dir, opt.feature_dir)
    os.makedirs(out_dir, exist_ok=True)

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
                        # 向队列发送失败标记而不是直接返回
                        timeout = 300
                        start_time = time.time()
                        while not stop_event.is_set():
                            try:
                                load_queue.put((sha256, None, None), timeout=10)  # 发送失败标记
                                break
                            except Full:
                                if time.time() - start_time > timeout:
                                    print(f"Timeout putting failed {sha256} into queue after {timeout}s")
                                    break
                                time.sleep(1)
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
                    try:
                        load_queue.put((sha256, None, None), timeout=10)  # 发送失败标记
                    except:
                        pass

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
                        
                        # 检查是否为失败标记
                        if data is None or positions is None:
                            print(f"Skipping {sha256} due to loading failure")
                            completed_count += 1
                            pbar.update(1)
                            submit_next_loaders()
                            continue
                        
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
                            patchtokens_sampled = sampled_mean_grid_sample_chunked(patchtokens, uv, num_chunks=16)
                            
                            # 清理大型临时变量
                            del patchtokens, patchtokens_lst, uv, uv_lst
                            clear_gpu_memory()

                            # 稀疏特征编码
                            batch_col = torch.zeros(indices.shape[0], 1, dtype=torch.int32, device='cuda')
                            feats = sp.SparseTensor(
                                feats = patchtokens_sampled.float(),
                                coords = torch.cat([batch_col, indices], dim=1),
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
    # python 2_dino_surface_torch.py --output_dir /baai-cwm-1/baai_cwm_ml/algorithm/hong.li/code/3dgen/data/data_factory_blender/datasets/3diclight_even --subdir renders_3diclightv3_eevee_even_150views --feature_dir surface_features

    gen_pbr_slat_1_step()
