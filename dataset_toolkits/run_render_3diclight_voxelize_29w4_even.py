import os
import copy
import sys
import importlib
import argparse
import pandas as pd
from easydict import EasyDict as edict
from functools import partial
from subprocess import DEVNULL, call
import numpy as np
from utils import sphere_hammersley_sequence, equator_cameras_sequence
import tqdm
from pprint import pprint
import open3d as o3d
import utils3d

from concurrent.futures import ThreadPoolExecutor, as_completed  

EVEN_DIR = "/baai-cwm-backup/cwm/hong.li/data/3diclight_rendering/3diclight_even_not_in_pbr33w_29w4/eevee_all"

def _voxelize(file, sha256, output_dir, out_basename):
    mesh = o3d.io.read_triangle_mesh(os.path.join(EVEN_DIR, sha256, 'mesh.ply'))
    # clamp vertices to the range [-0.5, 0.5]
    vertices = np.clip(np.asarray(mesh.vertices), -0.5 + 1e-6, 0.5 - 1e-6)
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(mesh, voxel_size=1/64, min_bound=(-0.5, -0.5, -0.5), max_bound=(0.5, 0.5, 0.5))
    vertices = np.array([voxel.grid_index for voxel in voxel_grid.get_voxels()])
    assert np.all(vertices >= 0) and np.all(vertices < 64), "Some vertices are out of bounds"
    vertices = (vertices + 0.5) / 64 - 0.5
    utils3d.io.write_ply(os.path.join(output_dir, out_basename, f'{sha256}.ply'), vertices)
    return {'sha256': sha256, 'voxelized': True, 'num_voxels': len(vertices)}


def check_processed(sha256, output_dir, out_basename):
    mesh_path = os.path.join(output_dir, out_basename, f'{sha256}.ply')
    # return sha256 if os.path.exists(path) else None 
    return sha256 if os.path.exists(mesh_path) else None

def filter_processed(metadata, opt, max_workers=64):  
    print("Filter out objects that are already processed")  

    sha256_list = copy.copy(metadata['sha256'].values)  
    processed = []  

    with ThreadPoolExecutor(max_workers=max_workers) as executor:  
        futures = {executor.submit(check_processed, sha, opt.output_dir, opt.out_basename): sha for sha in sha256_list}
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
    parser.add_argument('--output_dir', type=str, default="/baai-cwm-backup/cwm/hong.li/data/3diclight_rendering/3diclight_even_not_in_pbr33w_29w4",
                        help='Directory to save the metadata')
    parser.add_argument('--metadata_dir', type=str, default="metadatas",
                        help='Directory to save the metadata')
    parser.add_argument('--out_basename', type=str, default="voxel", help='The basename of the output directory')
    
    parser.add_argument('--debug', action='store_true', help='Debug mode')

    dataset_utils.add_args(parser)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--max_workers', type=int, default=16)
    opt = parser.parse_args(sys.argv[2:])
    opt = edict(vars(opt))

    output_dir = os.path.join(opt.output_dir, opt.out_basename)
    os.makedirs(output_dir, exist_ok=True)    

    print("Args:")
    pprint(vars(opt), indent=2, width=80)    
    print('Checking processed...', flush=True)

    # get file list
    if not os.path.exists(os.path.join(opt.metadata_dir, 'not_pbr_in_33w_29w4.csv')):
        raise ValueError('not_pbr_in_33w_29w4.csv not found')
    metadata = pd.read_csv(os.path.join(opt.metadata_dir, 'not_pbr_in_33w_29w4.csv'))


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
    func = partial(_voxelize, output_dir=opt.output_dir, out_basename=opt.out_basename)
    voxelized = dataset_utils.foreach_instance_ori(metadata, opt.output_dir, func, max_workers=opt.max_workers, desc='Voxelizing')
    voxelized = pd.concat([voxelized, pd.DataFrame.from_records(records)])
    voxelized.to_csv(os.path.join(opt.output_dir, f'voxelized_{opt.rank}.csv'), index=False)
