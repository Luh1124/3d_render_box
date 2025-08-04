import os
import json
import copy
import sys
import importlib
import argparse
import pandas as pd
from easydict import EasyDict as edict
from functools import partial
from subprocess import DEVNULL, call
import numpy as np
import tqdm
from scipy.spatial import cKDTree

import open3d as o3d
import utils3d

from concurrent.futures import ThreadPoolExecutor, as_completed  

def point_cloud_intersection(points1, points2, threshold=1e-4):
    """
    计算两组点云坐标的交集
    
    参数:
    - points1: 第一组点云坐标 (N, 3)
    - points2: 第二组点云坐标 (M, 3)
    - threshold: 点匹配的距离阈值
    
    返回:
    - intersect_points: 交集点坐标
    """
    # 构造 KDTree
    tree1 = cKDTree(points1)
    
    # 查找交集点
    intersect_indices = []
    for point in points2:
        # 查找最近点
        dist, idx = tree1.query(point, k=1)
        if dist <= threshold:
            intersect_indices.append(idx)
    
    # 获取唯一的交集点
    intersect_indices = np.unique(intersect_indices)
    intersect_points = points1[intersect_indices]
    
    return intersect_points

def _voxelize(file, sha256, level, output_dir, out_basename, voxel_dir, dino_dir):  
    voxel_path = os.path.join(output_dir, voxel_dir, f'level_{level}', f'{sha256}.ply')  
    npz_path = os.path.join(output_dir, dino_dir, f'level_{level}', f'{sha256}.npz')  

    pcd = o3d.io.read_point_cloud(voxel_path)  
    points1 = ((np.asarray(pcd.points) + 0.5) * 64).astype(np.uint8)  

    np_data = np.load(npz_path)  
    points2 = np_data['coords'].astype(np.uint8)  # (N,3)  
    
    # 保存原始特征  
    pbr_features = np_data['pbr_features']  # (N, features...)  
    color_feats = np_data['color_feats']  # (N, features...)  
    feats = np_data['feats']  # (N, features...)  
    brm_feats = np_data['brm_feats']  # (N, features...)  

    # 计算交集点  
    intersect_points = point_cloud_intersection(points1, points2, threshold=0.1)  

    # 找到交集点在 points2 中的索引  
    tree2 = cKDTree(points2)  
    intersect_indices = []  
    for point in intersect_points:  
        dist, idx = tree2.query(point, k=1)  
        if dist <= 0.1:  
            intersect_indices.append(idx)  
    
    # 转换为唯一的索引  
    intersect_indices = np.unique(intersect_indices)  

    # 提取对应的特征  
    intersect_pbr_features = pbr_features[intersect_indices]  
    intersect_color_feats = color_feats[intersect_indices]  
    intersect_feats = feats[intersect_indices]  
    intersect_brm_feats = brm_feats[intersect_indices]  

    # 准备顶点坐标（从 uint8 转换回原始坐标系）  
    vertices = (intersect_points + 0.5) / 64 - 0.5  

    # 保存 PLY 文件  
    utils3d.io.write_ply(os.path.join(output_dir, out_basename, f'level_{level}', f'{sha256}.ply'), vertices)  

    # 保存 NPZ 文件（包含特征）  
    np.savez(  
        os.path.join(output_dir, out_basename, f'level_{level}', f'{sha256}.npz'),  
        coords=intersect_points,  
        pbr_features=intersect_pbr_features,  
        color_feats=intersect_color_feats,  
        feats=intersect_feats,  
        brm_feats=intersect_brm_feats  
    )  

    return {  
        'sha256': sha256,   
        'voxelized': True,   
        'num_voxels': len(vertices),  
        'num_pbr_features': intersect_pbr_features.shape[1] if intersect_pbr_features.size > 0 else 0,  
        'num_color_feats': intersect_color_feats.shape[1] if intersect_color_feats.size > 0 else 0,  
        'num_feats': intersect_feats.shape[1] if intersect_feats.size > 0 else 0,  
        'num_brm_feats': intersect_brm_feats.shape[1] if intersect_brm_feats.size > 0 else 0  
    }  

def check_processed(sha256, level, output_dir, out_basename):  
    mesh_path = os.path.join(output_dir, out_basename, f'level_{level}', f'{sha256}.ply')
    # return sha256 if os.path.exists(path) else None 
    return sha256 if os.path.exists(mesh_path) else None

def filter_processed(metadata, opt, max_workers=16):  
    print("Filter out objects that are already processed")  

    sha256_list = copy.copy(metadata['sha256'].values)  
    level_list = copy.copy(metadata['level'].values)
    processed = []  

    with ThreadPoolExecutor(max_workers=max_workers) as executor:  
        futures = {executor.submit(check_processed, sha, lvl, opt.output_dir, opt.out_basename): sha for sha, lvl in zip(sha256_list, level_list)}
        for future in tqdm.tqdm(as_completed(futures), total=len(futures), desc="Checking processed"):  
            sha = future.result()  
            if sha is not None:  
                processed.append({'sha256': sha, 'rendered': True})  

    processed_sha256 = set(p['sha256'] for p in processed)  
    filtered_metadata = metadata[~metadata['sha256'].isin(processed_sha256)]  

    print(f'Processing {len(filtered_metadata)} objects...')  

    return filtered_metadata, processed 


if __name__ == '__main__':
    dataset_utils = importlib.import_module(f'datasets.{sys.argv[1]}')

    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default="3diclight_matsynth",
                        help='Directory to save the metadata')
    parser.add_argument('--metadata_dir', type=str, default="metadatas",
                        help='Directory to save the metadata')
    parser.add_argument('--voxel_dir', type=str, default="even_render_6k_voxel", help='The basename of the output directory')
    parser.add_argument('--dino_dir', type=str, default="even_slat_6k/dino_surface_slats", help='The basename of the output directory')
    parser.add_argument('--out_basename', type=str, default="even_render_6k_voxel_surface_interact", help='The basename of the output directory')

    parser.add_argument('--debug', action='store_true', help='Debug mode')

    dataset_utils.add_args(parser)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--max_workers', type=int, default=2)
    opt = parser.parse_args(sys.argv[2:])
    opt = edict(vars(opt))

    output_dir = os.path.join(opt.output_dir, opt.out_basename)
    for level in range(1, 5):
        os.makedirs(os.path.join(output_dir, f'level_{level}'), exist_ok=True)

    print(opt)
    
    print('Checking blender...', flush=True)

    # get file list
    if not os.path.exists(os.path.join(opt.metadata_dir, 'metadata_3didlight_6k_new.csv')):
        raise ValueError('metadata_3didlight_6k_new.csv not found')
    metadata = pd.read_csv(os.path.join(opt.metadata_dir, 'metadata_3didlight_6k_new.csv'))


    start = len(metadata) * opt.rank // opt.world_size
    end = len(metadata) * (opt.rank + 1) // opt.world_size
    metadata = metadata[start:end]

    if opt.debug:
        metadata = metadata.head(10)

    records = []

    print("filter out objects that are already processed")
    metadata, records = filter_processed(metadata, opt)
                
    print(f'Processing {len(metadata)} objects...')

    # process objects
    func = partial(_voxelize, output_dir=opt.output_dir, out_basename=opt.out_basename, voxel_dir=opt.voxel_dir, dino_dir=opt.dino_dir)
    voxelized = dataset_utils.foreach_instance(metadata, opt.output_dir, func, max_workers=opt.max_workers, desc='Voxelizing')
    voxelized = pd.concat([voxelized, pd.DataFrame.from_records(records)])
    voxelized.to_csv(os.path.join(opt.output_dir, f'voxelized_{opt.rank}.csv'), index=False)
