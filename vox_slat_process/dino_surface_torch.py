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

import torch
import torch.nn.functional as F
from easydict import EasyDict as edict


import trellis.models as models
import trellis.modules.sparse as sp

'''
x,y,z 3 channels, roughness 1 channel、metallic 1channel、base color 3 channels、相机空间法线 (0-1 范围) 3 channels
'''
torch.set_grad_enabled(False)
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HOME'] = '/baai-cwm-1/baai_cwm_ml/cwm/chongjie.ye/cache/huggingface'
os.environ['TORCH_HOME'] = '/baai-cwm-1/baai_cwm_ml/cwm/chongjie.ye/cache/torch'
os.environ['GRADIO_TEMP_DIR'] = '/baai-cwm-1/baai_cwm_ml/cwm/chongjie.ye/cache/gradio'

@torch.no_grad()  # 禁用梯度计算
def process_in_chunks(tensor, uv, chunk_size=8):
    """分块处理grid_sample减少峰值显存"""
    B, C, H, W = tensor.shape
    chunks = []
    for i in range(0, B, chunk_size):
        chunk = F.grid_sample(
            tensor[i:i+chunk_size],
            uv[i:i+chunk_size].unsqueeze(1),
            mode='bilinear',
            align_corners=False
        ).squeeze(2).permute(0, 2, 1)
        chunks.append(chunk)
    return torch.cat(chunks, dim=0)

def compute_mean(tensor):
    mean = torch.mean(tensor, dim=0).to(torch.float16)
    del tensor  # 显式释放
    torch.cuda.empty_cache()
    return mean

@torch.no_grad()
def merged_process_and_mean(tensor, uv, chunk_size=8):
    """
    合并的分块处理与均值计算函数
    优势：节省峰值显存约30-50% (原始方法的平均内存占用量)
    """
    B, C, H, W = tensor.shape
    accumulator = None  # 使用累加器代替存储所有chunks
    total = 0  # 样本计数器
    
    for i in range(0, B, chunk_size):
        # 分块处理grid_sample
        chunk = F.grid_sample(
            tensor[i:i+chunk_size],
            uv[i:i+chunk_size].unsqueeze(1),
            mode='bilinear',
            align_corners=False
        ).squeeze(2).permute(0, 2, 1)  # [N, L, C]
        
        # 累加部分和
        sum_chunk = torch.sum(chunk, dim=0)  # [L, C]
        
        # 初始化累加器
        if accumulator is None:
            accumulator = torch.zeros_like(sum_chunk, dtype=torch.float32)  # 高精度累加
        
        accumulator.add_(sum_chunk)  # type: ignore
        total += chunk.size(0)
        
        # 立即释放中间显存
        del chunk, sum_chunk
        torch.cuda.empty_cache()
    
    # 计算结果均值并进行精度转换
    if accumulator is not None:
        mean = (accumulator / total).to(torch.float16)  # 转换为float16
        del accumulator
        torch.cuda.empty_cache()
        return mean
    else:
        return torch.zeros(C, dtype=torch.float16, device=tensor.device)


def process_sparse(feats, indices):
    batch_col = torch.zeros(indices.shape[0], 1, dtype=torch.int32, device='cuda')
    indices_tensor = torch.as_tensor(indices, dtype=torch.int32, device='cuda')
    return sp.SparseTensor(
        feats=feats.to(torch.float32),
        coords=torch.cat([batch_col, indices_tensor], dim=1)
    )

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

            mask = ((depth[..., 1] < 5.) & (depth[..., 1] > 1.)).astype(np.float32)             
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


def gen_slat_2_step():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, required=False, default='/baai-cwm-1/baai_cwm_ml/cwm/hong.li/code/3dgen/data/zeroverse/3diclight_matsynth/even_slat_6k',
                        help='Directory to save the metadata')
    parser.add_argument('--input_dir', type=str, required=False, default='/baai-cwm-1/baai_cwm_ml/cwm/hong.li/code/3dgen/data/zeroverse/3diclight_matsynth/even_render_6k',
                        help='Directory where the renders are stored')
    parser.add_argument('--feature_dir', type=str, required=False, default='dino_surface_slats',
                        help='Directory to save the features')
    parser.add_argument('--model', type=str, default='dinov2_vitl14_reg',
                        help='Feature extraction model')
    parser.add_argument('--enc_pretrained', type=str, default='/baai-cwm-1/baai_cwm_ml/cwm/chongjie.ye/code/TRELLIS_mini/weights/TRELLIS-image-large/ckpts/slat_enc_swin8_B_64l8_fp16',
                        help='Pretrained encoder model')

    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    opt = parser.parse_args()
    # opt = edict(vars(opt))

    # feature_name = opt.model
    os.makedirs(os.path.join(opt.output_dir, opt.feature_dir), exist_ok=True)
    dinov2_model = torch.hub.load('/baai-cwm-1/baai_cwm_ml/cwm/hong.li/.cache/torch/hub/facebookresearch_dinov2_main', "dinov2_vitl14_reg", pretrained=True, source='local')    
    dinov2_model.eval().cuda()
    transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    n_patch = 518 // 14

    encoder = models.from_pretrained(opt.enc_pretrained).eval().cuda()

    # get file list
    if os.path.exists("/baai-cwm-1/baai_cwm_ml/cwm/hong.li/code/3dgen/data/zeroverse/metadatas/metadata_3didlight_6k_new_paths.csv"):
        metadata = pd.read_csv("/baai-cwm-1/baai_cwm_ml/cwm/hong.li/code/3dgen/data/zeroverse/metadatas/metadata_3didlight_6k_new_paths.csv")
    else:
        raise ValueError('metadata_3didlight_6k_new_paths.csv not found')

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
    # try:
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
                    datum['color'] = transform(datum['color'])
                    # datum['color'] = datum['color']
                    datum['roughness'] = datum['roughness']
                    datum['metallic'] = datum['metallic']
                    datum['mask'] = datum['mask']
                    datum['extrinsics'] = datum['extrinsics']
                    datum['intrinsics'] = datum['intrinsics']
                    datum['brm'] = transform(torch.cat([torch.zeros_like(datum['roughness']), datum['roughness'], datum['metallic']], dim=-1).permute(2, 0, 1))

                    data.append(datum)

                np_data = np.load(os.path.join(opt.output_dir, 'pbr_surface_slats', f'{render_path}.npz'))

                load_queue.put((render_path, data, np_data))

            except Exception as e:
                print(f"Error loading data for {render_path}: {e}")
        
        loader_executor.map(loader, render_paths)

        def saver(render_path, feats, coords, color_feats, brm_feats, pbr_features):
            # 结果保存
            new_pack = {
                'feats': feats,
                'coords': coords,
                'color_feats': color_feats,
                'brm_feats': brm_feats,
                'pbr_features': pbr_features
            }

            save_path = os.path.join(opt.output_dir, opt.feature_dir, f'{render_path}.npz')
            np.savez_compressed(save_path, **new_pack)

        for _ in tqdm(range(len(render_paths)), desc='Processing Dino slat'):
            render_path, data, np_data = load_queue.get()
            positions = torch.from_numpy(np_data['positions']).float().cuda()
            pbr_features = torch.from_numpy(np_data['pbr_features']).float().cuda()
            indices = ((positions + 0.5) * 64).long()
            assert torch.all(indices >= 0) and torch.all(indices < 64), "Some vertices are out of bounds"
            N = positions.shape[0]
            pack = {
                'indices': indices.to(torch.int32)
            }
            n_views = len(data)

            patchtokens_lst = []
            patchtokens_color_lst = []
            patchtokens_brm_lst = []

            uv_lst = []

            for i in range(0, n_views, opt.batch_size):
                batch_data = data[i:i+opt.batch_size]
                bs = len(batch_data)

                # 准备批量数据
                batch_image = torch.stack([d['image'] for d in batch_data]).cuda()
                batch_color = torch.stack([d['color'] for d in batch_data]).cuda()

                batch_brm = torch.stack([d['brm'] for d in batch_data]).cuda()

                batch_extrinsics = torch.stack([d['extrinsics'] for d in batch_data]).cuda()
                batch_intrinsics = torch.stack([d['intrinsics'] for d in batch_data]).cuda()

                # 提取特征
                with torch.no_grad():
                    features = dinov2_model(batch_image, is_training=True)
                    features_color = dinov2_model(batch_color, is_training=True)
                    features_brm = dinov2_model(batch_brm, is_training=True)
                del batch_image, batch_color, batch_brm

                uv = utils3d.torch.project_cv(positions, batch_extrinsics, batch_intrinsics)[0] * 2 - 1
                patchtokens = features['x_prenorm'][:, dinov2_model.num_register_tokens + 1:].permute(0, 2, 1).reshape(bs, 1024, n_patch, n_patch)
                patchtokens_color = features_color['x_prenorm'][:, dinov2_model.num_register_tokens + 1:].permute(0, 2, 1).reshape(bs, 1024, n_patch, n_patch)
                patchtokens_brm = features_brm['x_prenorm'][:, dinov2_model.num_register_tokens + 1:].permute(0, 2, 1).reshape(bs, 1024, n_patch, n_patch)
                patchtokens_lst.append(patchtokens)
                patchtokens_color_lst.append(patchtokens_color)
                patchtokens_brm_lst.append(patchtokens_brm)
                uv_lst.append(uv)

            patchtokens = torch.cat(patchtokens_lst, dim=0)
            del patchtokens_lst
            patchtokens_color = torch.cat(patchtokens_color_lst, dim=0)
            del patchtokens_color_lst
            patchtokens_brm = torch.cat(patchtokens_brm_lst, dim=0)
            del patchtokens_brm_lst

            uv = torch.cat(uv_lst, dim=0)
        
            # 分步处理稀疏张量
            feats = process_sparse(merged_process_and_mean(patchtokens, uv), pack['indices'])
            with torch.no_grad():
                # 计算特征
                latent = encoder(feats, sample_posterior=False)

            feats = latent.feats.float().cpu().numpy()
            coords = latent.coords[:, 1:].cpu().numpy().astype(np.uint8)

            del latent

            color_feats = process_sparse(merged_process_and_mean(patchtokens_color, uv), pack['indices'])
            del patchtokens_color

            with torch.no_grad():
                color_latent = encoder(color_feats, sample_posterior=False)
            del color_feats

            color_feats = color_latent.feats.float().cpu().numpy()
            del color_latent

            brm_feats = process_sparse(merged_process_and_mean(patchtokens_brm, uv), pack['indices'])
            del patchtokens_brm

            with torch.no_grad():
                brm_latent = encoder(brm_feats, sample_posterior=False)
            del brm_feats
            brm_feats = brm_latent.feats.float().cpu().numpy()
            del brm_latent

            # 处理PBR特征
            pbr_features = pbr_features.float().cpu().numpy()

            # save features
            saver_executor.submit(saver, render_path, feats, coords, color_feats, brm_feats, pbr_features)

        saver_executor.shutdown(wait=True)
    # except Exception as e:
        # print(f"Error happened during processing: {e}")


if __name__ == '__main__':
    # python 2_dino_surface_torch.py --output_dir /baai-cwm-1/baai_cwm_ml/algorithm/hong.li/code/3dgen/data/data_factory_blender/datasets/3diclight_even --subdir renders_3diclightv3_eevee_even_150views --feature_dir surface_features

    # test_single_sha256()

    gen_slat_2_step()
