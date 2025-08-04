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
from utils import sphere_hammersley_sequence, equator_cameras_sequence
import tqdm

from concurrent.futures import ThreadPoolExecutor, as_completed  


"""
512 分辨率
正交相机 0.8
渲染 depth、RGB、PBR、normal, low_normal
"""

BLENDER_LINK = 'https://download.blender.org/release/Blender4.3/blender-4.3.2-linux-x64.tar.xz'
BLENDER_INSTALLATION_PATH = '/baai-cwm-1/baai_cwm_ml/cwm/hong.li/tools'
BLENDER_PATH = f'{BLENDER_INSTALLATION_PATH}/blender-4.3.2-linux-x64/blender'

def _install_blender():
    if not os.path.exists(BLENDER_PATH):
        os.system('sudo apt-get update')
        os.system('sudo apt-get install -y libxrender1 libxi6 libxkbcommon-x11-0 libsm6')
        os.system(f'wget {BLENDER_LINK} -P {BLENDER_INSTALLATION_PATH}')
        # os.system(f'tar -xvf {BLENDER_INSTALLATION_PATH}/blender-3.0.1-linux-x64.tar.xz -C {BLENDER_INSTALLATION_PATH}')
        os.system(f'tar -xvf {BLENDER_INSTALLATION_PATH}/blender-4.3.2-linux-x64.tar.xz -C {BLENDER_INSTALLATION_PATH}')
        os.system('export EGL_DRIVER=nvidia')
        os.system('export __EGL_VENDOR_LIBRARY_DIRS=/usr/share/glvnd/egl_vendor.d')
        print('Blender installed', flush=True)


def _render(file_path, sha256, level, output_dir, num_views=1, num_lights=10, low_normal=True, out_basename="render"):

    output_folder = os.path.join(output_dir, out_basename, f'level_{level}', sha256)
    
    hdri_file_paths = os.listdir("/baai-cwm-nas/public_data/rendering_data/high_4k_hdri/4k_exr")
    hdri_file_path = os.path.join("/baai-cwm-nas/public_data/rendering_data/high_4k_hdri/4k_exr", np.random.choice(hdri_file_paths))

    args = [
        BLENDER_PATH, '-b', '-P', os.path.join(os.path.dirname(__file__), 'blender_script', 'render_ps_v3.py'),
        '--',
        '--num_views', f'{num_views}',
        '--num_lights', f'{num_lights}',
        '--object', file_path,
        '--resolution', '512',
        '--output_folder', output_folder,
        '--engine', 'CYCLES',
        # '--engine', 'BLENDER_EEVEE_NEXT',
        '--film_transparent',
        # '--save_mesh',
        '--save_image',
        '--save_normal',
        '--save_depth',
        '--save_pbr',
        # '--save_albedo',
        # '--save_glossycol',
        # '--save_env',
        # '--save_ao',
        # '--save_glossydir',
        # '--save_diffdir',
        # '--save_shadow',
        # '--save_diffind',
        # '--save_glossyind',
        '--hdri_file_path', hdri_file_path,
    ]
        
    if file_path.endswith('.blend'):
        args.insert(1, file_path)
    # import pdb; pdb.set_trace()
    # call(args, stdout=DEVNULL, stderr=DEVNULL)
    call(args)

    if low_normal:
        args = [
            BLENDER_PATH, '-b', '-P', os.path.join(os.path.dirname(__file__), 'blender_script', 'render_ps_v3.py'),
            '--',
            '--num_views', f'{num_views}',
            '--num_lights', f'{num_lights}',
            '--object', file_path,
            '--resolution', '512',
            '--output_folder', output_folder,
            '--engine', 'CYCLES',
            # '--engine', 'BLENDER_EEVEE_NEXT',
            '--film_transparent',
            '--save_low_normal',
            '--save_low_image',
            '--hdri_file_path', hdri_file_path,
        ]

        if file_path.endswith('.blend'):
            args.insert(1, file_path)

        # call(args, stdout=DEVNULL, stderr=DEVNULL)
        call(args)
    
    transforms_json_path = os.path.join(output_folder, 'transforms.json')
    transforms_low_json_path = os.path.join(output_folder, 'transforms_low.json')
    if os.path.exists(transforms_json_path) and os.path.exists(transforms_low_json_path):
        return {'sha256': sha256, 'rendered': True}

def check_processed(sha256, level, output_dir, out_basename):  
    path = os.path.join(output_dir, out_basename, f'level_{level}', sha256, 'transforms.json')
    low_path = os.path.join(output_dir, out_basename, f'level_{level}', sha256, 'transforms_low.json')
    return sha256 if os.path.exists(path) and os.path.exists(low_path) else None 

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
    parser.add_argument('--output_dir', type=str, default="render_all",
                        help='Directory to save the metadata')
    parser.add_argument('--metadata_dir', type=str, default="metadatas",
                        help='Directory to save the metadata')
    parser.add_argument('--num_views', type=int, default=1,
                        help='Number of views to render')
    parser.add_argument('--num_lights', type=int, default=20,
                        help='Number of lights')
    parser.add_argument('--out_basename', type=str, default="PS_plane_2K_render", help='The basename of the output directory')
    
    parser.add_argument('--debug', action='store_true', help='Debug mode')

    dataset_utils.add_args(parser)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--max_workers', type=int, default=1)
    opt = parser.parse_args(sys.argv[2:])
    opt = edict(vars(opt))

    os.makedirs(os.path.join(opt.output_dir, opt.out_basename), exist_ok=True)
    print(opt)
    
    print('Checking blender...', flush=True)
    _install_blender()

    # get file list
    if not os.path.exists(os.path.join(opt.metadata_dir, 'metadata_PS_plane_2k.csv')):
        raise ValueError('metadata_PS_plane_2k.csv not found')
    metadata = pd.read_csv(os.path.join(opt.metadata_dir, 'metadata_PS_plane_2k.csv'))


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
    func = partial(_render, output_dir=opt.output_dir, num_views=opt.num_views, num_lights=opt.num_lights, low_normal=True, out_basename=opt.out_basename)
    rendered = dataset_utils.foreach_instance(metadata, opt.output_dir, func, max_workers=opt.max_workers, desc='Rendering objects')
    rendered = pd.concat([rendered, pd.DataFrame.from_records(records)])
    rendered.to_csv(os.path.join(opt.output_dir, opt.out_basename, f'rendered_{opt.rank}.csv'), index=False)
