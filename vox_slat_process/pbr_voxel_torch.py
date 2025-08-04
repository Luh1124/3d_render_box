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

'''
x,y,z 3 channels, roughness 1 channel、metallic 1channel、base color 3 channels、相机空间法线 (0-1 范围) 3 channels
'''
torch.set_grad_enabled(False)
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HOME'] = '/baai-cwm-vepfs/cwm/hong.li/.cache/huggingface'
os.environ['TORCH_HOME'] = '/baai-cwm-vepfs/cwm/hong.li/.cache/torch'
os.environ['GRADIO_TEMP_DIR'] = '/baai-cwm-vepfs/cwm/hong.li/.cache/gradio'

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

    # 步骤1: 并行检查文件存在性
    with ThreadPoolExecutor(max_workers=8) as executor:
        # 生成检查任务
        futures = [  
            executor.submit(  
                lambda npz_path=npz_path: 
                    os.path.exists(  
                        os.path.join(  
                            opt.output_dir,  
                            opt.feature_dir,  
                            npz_path  
                    ) 
                )  
            )  
            for npz_path in metadata['npz_paths'].values  
        ] 
        
        # 获取存在文件的sha256列表
        exists_flags = []
        for future in tqdm(futures, desc='Checking files'):
            exists_flags.append(future.result())
    
    # 步骤2: 批量过滤数据
    to_remove = metadata['npz_paths'][exists_flags].tolist()
    filtered_metadata = metadata[~metadata['npz_paths'].isin(to_remove)].copy()
    return filtered_metadata.reset_index(drop=True)


def get_data(input_dir, frames, render_path):  
    """Load depth, PBR maps, and camera parameters for each view."""  
    with ThreadPoolExecutor(max_workers=16) as executor:  
    # with ThreadPoolExecutor(max_workers=1) as executor:  
        def worker(view):  
            depth_path = os.path.join(input_dir, render_path, view['file_path'].replace("image", "depth").replace(".png", "_depth.exr"))  
            normal_path = os.path.join(input_dir, render_path, view['file_path'].replace("image", "normal").replace(".png", "_normal.exr"))  
            color_path = os.path.join(input_dir, render_path, view['file_path'].replace("image", "Base Color").replace(".png", "_Base Color.png"))  
            roughness_path = os.path.join(input_dir, render_path, view['file_path'].replace("image", "Roughness").replace(".png", "_Roughness.exr"))  
            metallic_path = os.path.join(input_dir, render_path, view['file_path'].replace("image", "Metallic").replace(".png", "_Metallic.exr"))  
            image_path = os.path.join(input_dir, render_path, view['file_path'].replace("image/", ""))

            try:  
                # import ipdb; ipdb.set_trace()
                depth = pyexr.read(depth_path, precision=pyexr.HALF).astype(np.float32)  
                normal = pyexr.read(normal_path, precision=pyexr.HALF).astype(np.float32)  
                color = cv2.cvtColor(cv2.imread(color_path), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0   
                # roughness = cv2.imread(roughness_path).astype(np.float32) / 255.0  
                # metallic = cv2.imread(metallic_path).astype(np.float32) / 255.0  
                roughness = pyexr.read(roughness_path, precision=pyexr.HALF).astype(np.float32)
                metallic = pyexr.read(metallic_path, precision=pyexr.HALF).astype(np.float32)

                image = Image.open(image_path)
                color = Image.open(color_path)

            except Exception as e:  
                # print(f"Error loading image {sha256}: {e}")  
                os.makedirs("error_images", exist_ok=True)
                with open(f"error_images/{render_path}.txt", 'w') as f:
                    f.write(f"Error loading image {render_path}: {e}")
                return None  

            image_size = (518, 518)
            depth = cv2.resize(depth, image_size, interpolation=cv2.INTER_NEAREST_EXACT)  # shape 
            normal = cv2.resize(normal, image_size, interpolation=cv2.INTER_NEAREST_EXACT)
            roughness = cv2.resize(roughness, image_size, interpolation=cv2.INTER_NEAREST_EXACT)  
            metallic = cv2.resize(metallic, image_size, interpolation=cv2.INTER_NEAREST_EXACT)  

            image = image.resize(image_size, Image.Resampling.LANCZOS)
            color = color.resize(image_size, Image.Resampling.LANCZOS)

            image = np.array(image).astype(np.float32) / 255
            color = np.array(color).astype(np.float32) / 255

            color = color[:, :, :3] * image[:, :, 3:] 
            image = image[:, :, :3] * image[:, :, 3:]

            image = torch.from_numpy(image).permute(2, 0, 1).float()
            color = torch.from_numpy(color).permute(2, 0, 1).float()

            mask = ((depth[..., 1] < 2.5) & (depth[..., 1] > 1.5)).astype(np.float32)             
            # mask = np.where(depth[..., 1] > 1., 1.0, 0.0)

            mask = torch.from_numpy(mask).float()

            depth = torch.from_numpy(depth).float()
            normal = torch.from_numpy(normal).float()
            roughness = torch.from_numpy(roughness).float()
            metallic = torch.from_numpy(metallic).float()

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
                'color': color.cuda(),
                'roughness': roughness[..., None].cuda(),  
                'metallic': metallic[..., None].cuda(),
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


def gen_pbr_slat_1_step():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, required=False, default='/baai-cwm-1/baai_cwm_ml/cwm/hong.li/code/3dgen/data/zeroverse/3diclight_matsynth/even_slat_6k_debug',
                        help='Directory to save the metadata')
    parser.add_argument('--input_dir', type=str, required=False, default='/baai-cwm-1/baai_cwm_ml/cwm/hong.li/code/3dgen/data/zeroverse/3diclight_matsynth/even_render_6k',
                        help='Directory where the renders are stored')
    parser.add_argument('--feature_dir', type=str, required=False, default='pbr_voxel_slats',
                        help='Directory to save the features')
    parser.add_argument('--model', type=str, default='dinov2_vitl14_reg',
                        help='Feature extraction model')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    opt = parser.parse_args()

    # parser = argparse.ArgumentParser()
    # parser.add_argument('--output_dir', type=str, required=False, default='/baai-cwm-1/baai_cwm_ml/cwm/hong.li/code/3dgen/data/zeroverse/dorabenchmark',
    #                     help='Directory to save the metadata')
    # parser.add_argument('--input_dir', type=str, required=False, default='/baai-cwm-1/baai_cwm_ml/cwm/hong.li/code/3dgen/data/zeroverse/dorabenchmark/even',
    #                     help='Directory where the renders are stored')
    # parser.add_argument('--feature_dir', type=str, required=False, default='pbr_voxel_slats',
    #                     help='Directory to save the features')
    # parser.add_argument('--model', type=str, default='dinov2_vitl14_reg',
    #                     help='Feature extraction model')
    # parser.add_argument('--batch_size', type=int, default=16)
    # parser.add_argument('--rank', type=int, default=0)
    # parser.add_argument('--world_size', type=int, default=1)
    # opt = parser.parse_args()
    # # opt = edict(vars(opt))

    # feature_name = opt.model
    os.makedirs(os.path.join(opt.output_dir, opt.feature_dir), exist_ok=True)
    dinov2_model = torch.hub.load('/baai-cwm-1/baai_cwm_ml/cwm/hong.li/.cache/torch/hub/facebookresearch_dinov2_main', "dinov2_vitl14_reg", pretrained=True, source='local')    
    dinov2_model.eval().cuda()
    transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    n_patch = 518 // 14

    # get file list
    if os.path.exists("/baai-cwm-1/baai_cwm_ml/cwm/hong.li/code/3dgen/data/zeroverse/metadatas/metadata_3didlight_6k_new_paths.csv"):
        metadata = pd.read_csv("/baai-cwm-1/baai_cwm_ml/cwm/hong.li/code/3dgen/data/zeroverse/metadatas/metadata_3didlight_6k_new_paths.csv")
    else:
        raise ValueError('dora_bencmetadata_3didlight_6k_new_pathsh_meshes_paths.csv not found')

    start = len(metadata) * opt.rank // opt.world_size
    end = len(metadata) * (opt.rank + 1) // opt.world_size
    metadata = metadata[start:end]
    
    metadata = filter_metadata(metadata, opt)

    # import ipdb; ipdb.set_trace()

    # start = len(metadata) * opt.rank // opt.world_size
    # end = len(metadata) * (opt.rank + 1) // opt.world_size
    # metadata = metadata[start:end]


    print(f"Processing {len(metadata)} objects")

    render_paths = list(metadata['render_paths'].values)

    # extract features
    load_queue = Queue(maxsize=2)
    try:
        with ThreadPoolExecutor(max_workers=2) as loader_executor, \
            ThreadPoolExecutor(max_workers=2) as saver_executor:
            def loader(render_path):
                try:
                    with open(os.path.join(opt.input_dir, render_path, 'transforms.json'), 'r') as f:
                        metadata = json.load(f)
                    frames = metadata['frames']
                    data = []
                    for datum in get_data(opt.input_dir , frames, render_path):
                        datum['image'] = transform(datum['image'])
                        datum['depth'] = datum['depth']
                        datum['normal'] = datum['normal']
                        datum['tcolor'] = transform(datum['color'])
                        datum['color'] = datum['color']
                        
                        datum['roughness'] = datum['roughness']
                        datum['metallic'] = datum['metallic']
                        datum['mask'] = datum['mask']
                        datum['extrinsics'] = datum['extrinsics']
                        datum['intrinsics'] = datum['intrinsics']
                        datum['brm'] = transform(torch.cat([torch.zeros_like(datum['roughness']), datum['roughness'], datum['metallic']], dim=-1).permute(2, 0, 1))
                        data.append(datum)

                    positions = utils3d.io.read_ply(os.path.join("/baai-cwm-1/baai_cwm_ml/cwm/hong.li/code/3dgen/data/zeroverse/3diclight_matsynth/even_render_6k_voxel", f'{render_path}.ply'))[0]

                    load_queue.put((render_path, data, positions))

                except Exception as e:
                    print(f"Error loading data for {render_path}: {e}")
            
            loader_executor.map(loader, render_paths)

            def saver(sha256, pack, patchtokens, patchtokens_color, patchtokens_brm, uv):
                pack['patchtokens'] = F.grid_sample(
                    patchtokens,
                    uv.unsqueeze(1),
                    mode='bilinear',
                    align_corners=False,
                ).squeeze(2).permute(0, 2, 1).cpu().numpy()
                pack['patchtokens'] = np.mean(pack['patchtokens'], axis=0).astype(np.float16)
                pack['patchtokens_color'] = F.grid_sample(
                    patchtokens_color,
                    uv.unsqueeze(1),
                    mode='bilinear',
                    align_corners=False,
                ).squeeze(2).permute(0, 2, 1).cpu().numpy()
                pack['patchtokens_color'] = np.mean(pack['patchtokens_color'], axis=0).astype(np.float16)
                pack['patchtokens_brm'] = F.grid_sample(
                    patchtokens_brm,
                    uv.unsqueeze(1),
                    mode='bilinear',
                    align_corners=False,
                ).squeeze(2).permute(0, 2, 1).cpu().numpy()
                pack['patchtokens_brm'] = np.mean(pack['patchtokens_brm'], axis=0).astype(np.float16)
                save_path = os.path.join(opt.output_dir, opt.feature_dir, f'{sha256}.npz')
                np.savez_compressed(save_path, **pack)
                # records.append({'sha256': sha256, f'feature_{feature_name}' : True})

            for _ in tqdm(range(len(render_paths)), desc='Processing pbr slat'):
                render_path, data, positions = load_queue.get()
                positions = torch.from_numpy(positions).float().cuda()
                indices = ((positions + 0.5) * 64).long()
                assert torch.all(indices >= 0) and torch.all(indices < 64), "Some vertices are out of bounds"

                n_views = len(data)
                N = positions.shape[0]
                pack = {
                    'indices': indices.cpu().numpy().astype(np.uint8),
                }

                patchtokens_lst = []
                patchtokens_color_lst = []
                patchtokens_brm_lst = []

                uv_lst = []

                for i in range(0, n_views, opt.batch_size):
                    batch_data = data[i:i+opt.batch_size]
                    bs = len(batch_data)

                    # 准备批量数据
                    batch_image = torch.stack([d['image'] for d in batch_data])
                    batch_depth = torch.stack([d['depth'] for d in batch_data])
                    batch_normal = torch.stack([d['normal'] for d in batch_data])
                    batch_tcolor = torch.stack([d['tcolor'] for d in batch_data])
                    batch_color = torch.stack([d['color'] for d in batch_data])
                    batch_roughness = torch.stack([d['roughness'] for d in batch_data])
                    batch_metallic = torch.stack([d['metallic'] for d in batch_data])
                    batch_mask = torch.stack([d['mask'] for d in batch_data])
                    batch_extrinsics = torch.stack([d['extrinsics'] for d in batch_data])
                    batch_intrinsics = torch.stack([d['intrinsics'] for d in batch_data])
                    batch_brm = torch.stack([d['brm'] for d in batch_data])


                    # 提取特征
                    with torch.no_grad():
                        features = dinov2_model(batch_image, is_training=True)
                        features_color = dinov2_model(batch_tcolor, is_training=True)
                        features_brm = dinov2_model(batch_brm, is_training=True)

                    uv = utils3d.torch.project_cv(positions, batch_extrinsics, batch_intrinsics)[0] * 2 - 1
                    patchtokens = features['x_prenorm'][:, dinov2_model.num_register_tokens + 1:].permute(0, 2, 1).reshape(bs, 1024, n_patch, n_patch)
                    patchtokens_lst.append(patchtokens)
                    uv_lst.append(uv)
                    patchtokens_color = features_color['x_prenorm'][:, dinov2_model.num_register_tokens + 1:].permute(0, 2, 1).reshape(bs, 1024, n_patch, n_patch)
                    patchtokens_color_lst.append(patchtokens_color)

                    patchtokens_brm = features_brm['x_prenorm'][:, dinov2_model.num_register_tokens + 1:].permute(0, 2, 1).reshape(bs, 1024, n_patch, n_patch)
                    patchtokens_brm_lst.append(patchtokens_brm)

                patchtokens = torch.cat(patchtokens_lst, dim=0)
                patchtokens_color = torch.cat(patchtokens_color_lst, dim=0)
                patchtokens_brm = torch.cat(patchtokens_brm_lst, dim=0)
                uv = torch.cat(uv_lst, dim=0)

                saver_executor.submit(saver, render_path, pack, patchtokens, patchtokens_color, patchtokens_brm, uv)

            saver_executor.shutdown(wait=True)
    except Exception as e:
        print(f"Error happened during processing: {e}")

def test_single_sha256():
    output_dir = "/baai-cwm-1/baai_cwm_ml/algorithm/hong.li/code/3dgen/data/data_factory_blender/datasets/3diclight_even"

    subdir = "renders_3diclightv3_eevee_even_150views"

    # sha256 = "e0a4c1ee8b745f608b85d27a10c6fa86c6b92f30831b8338af8f7aa990496433"
    # sha256 = "71833f49de1b5592b324d54f897378683fc456be3035b578737bffc790758933"
    sha256 = "2d8fb27dff453cc91239957512e9564e241cf67921865d3ac7a3c8d1998f36f6"

    feature_dir = "surface_features_debug"

    mode_name = 'dinov2_vitl14_reg'

    os.makedirs(os.path.join(output_dir, feature_dir, mode_name), exist_ok=True)


    batch_size = 16

    # import ipdb; ipdb.set_trace()

    # dinov2_model = torch.hub.load('facebookresearch/dinov2', mode_name)
    # dinov2_model.eval().cuda()

    transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    n_patch = 518 // 14


    def loader(sha256):
        with open(os.path.join(output_dir, subdir, sha256, 'transforms.json'), 'r') as f:
            metadata = json.load(f)

        frames = metadata['frames']

        data = []

        for datum in get_data(output_dir + '/' + subdir, frames, sha256):
            datum['image'] = transform(datum['image'])
            datum['depth'] = datum['depth']
            datum['normal'] = datum['normal']
            datum['color'] = datum['color']
            datum['roughness'] = datum['roughness']
            datum['metallic'] = datum['metallic']
            datum['mask'] = datum['mask']
            datum['extrinsics'] = datum['extrinsics']
            datum['intrinsics'] = datum['intrinsics']

            data.append(datum)
        
        return data

    data = loader(sha256)

    n_views = len(data)

    # patchtokens_lst = []
    # patchtokens_color_lst = []

    pts_valid_lst = []
    features_valid_lst = []

    # uv_lst = []

    for i in range(0, n_views, batch_size):
        batch_data = data[i:i+batch_size]
        bs = len(batch_data)
        
        # 准备批量数据
        batch_image = torch.stack([d['image'] for d in batch_data]).cuda()
        batch_depth = torch.stack([d['depth'] for d in batch_data]).cuda()
        batch_normal = torch.stack([d['normal'] for d in batch_data]).cuda()
        batch_color = torch.stack([d['color'] for d in batch_data]).cuda()
        batch_roughness = torch.stack([d['roughness'] for d in batch_data]).cuda()
        batch_metallic = torch.stack([d['metallic'] for d in batch_data]).cuda()
        batch_mask = torch.stack([d['mask'] for d in batch_data]).cuda()
        batch_extrinsics = torch.stack([d['extrinsics'] for d in batch_data]).cuda()
        batch_intrinsics = torch.stack([d['intrinsics'] for d in batch_data]).cuda()
        
        # 提取特征
        # with torch.no_grad():
            # features = dinov2_model(batch_image, is_training=True) # 提取特征
            # features_color = dinov2_model(batch_color, is_training=True) # 提取特征
        
        # 深度图转换为点云
        pts = utils3d.torch.depth_to_points(batch_depth, batch_intrinsics, batch_extrinsics)
        pts_valid = pts[batch_mask > 0]

        # clip points to ensure they are within the bounding box
        min_bound = torch.tensor([-0.5, -0.5, -0.5]).to(pts_valid.device)
        max_bound = torch.tensor([0.5, 0.5, 0.5]).to(pts_valid.device)
        pts_valid = torch.clip(pts_valid, min_bound + 1e-6, max_bound - 1e-6)

        normal_valid = batch_normal[batch_mask > 0]
        roughness_valid = batch_roughness[batch_mask > 0]
        metallic_valid = batch_metallic[batch_mask > 0]

        pts_valid_lst.append(pts_valid)
        features_valid_lst.append(torch.cat([roughness_valid, metallic_valid, normal_valid], dim=1))

        # uv = utils3d.torch.project_cv(pts_valid, batch_extrinsics, batch_intrinsics)[0] * 2 - 1
        
        # 保存点云以便可视化
        # utils3d.io.write_ply(f'{sha256}_batch_{i}.ply', pts_valid.cpu().numpy().reshape(-1, 3))
        
        # 提取 patch tokens
        # patchtokens = features['x_prenorm'][:, dinov2_model.num_register_tokens + 1:].permute(0, 2, 1).reshape(bs, 1024, n_patch, n_patch)
        # patchtokens_color = features_color['x_prenorm'][:, dinov2_model.num_register_tokens + 1:].permute(0, 2, 1).reshape(bs, 1024, n_patch, n_patch)

        # patchtokens_lst.append(patchtokens)
        # patchtokens_color_lst.append(patchtokens_color)

        # uv_lst.append(uv)
    
    all_pts = torch.cat(pts_valid_lst, dim=0)  # (N_total, 3)
    all_features = torch.cat(features_valid_lst, dim=0)  # (N_total, 5)

    # 保存点云以便可视化
    utils3d.io.write_ply(f'{sha256}_all.ply', all_pts.cpu().numpy().reshape(-1, 3))

    # 将坐标映射到 [0, 64) 整数范围
    scaled_pts = (all_pts + 0.5) * 64 # (N_total, 3)
    indices = torch.floor(scaled_pts).long()  # (N_total, 3)

    # 计算每个体素的唯一索引
    unique_indices, inverse, counts = torch.unique(indices, return_inverse=True, return_counts=True, dim=0)  # (N_unique, 3), (N_total), (N_unique)

    # 计算每个体素的中心坐标 （输出范围[-0.5, 0.5]）
    voxel_centers = (unique_indices.float() + 0.5) / 64.0 - 0.5  # (N_unique, 3) 转换为体素中心

    # 为每个体素计算平均特征
    sum_features = torch.zeros(unique_indices.size(0), 5).to(all_features.device)  # (N_unique, 5)
    sum_features.scatter_add_(0, inverse.unsqueeze(1).expand(-1, 5), all_features)  # (N_total, 5) -> (N_unique, 5)
    mean_features = sum_features / counts.unsqueeze(1)  # (N_unique, 5)

    # 去除无效体素
    valid_mask = counts > 0

    mean_features = mean_features[valid_mask]
    voxel_centers = voxel_centers[valid_mask]

    # 保存点云以便可视化
    utils3d.io.write_ply(f'{sha256}_voxel_centers.ply', voxel_centers.cpu().numpy().reshape(-1, 3))

    # 保存特征和点云数据
    np.savez_compressed(f'{sha256}_features.npz', voxel_centers=voxel_centers.cpu().numpy(), features=mean_features.cpu().numpy())

if __name__ == '__main__':
    # python 2_dino_surface_torch.py --output_dir /baai-cwm-1/baai_cwm_ml/algorithm/hong.li/code/3dgen/data/data_factory_blender/datasets/3diclight_even --subdir renders_3diclightv3_eevee_even_150views --feature_dir surface_features

    # test_single_sha256()

    gen_pbr_slat_1_step()
