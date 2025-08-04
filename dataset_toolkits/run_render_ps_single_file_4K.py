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


def _render(file_path, sha256, output_dir, num_views=1, num_lights=20, low_normal=True, out_basename="render"):

    output_folder = os.path.join(output_dir, out_basename, sha256)
    
    hdri_file_dir = "/baai-cwm-nas/public_data/rendering_data/high_4k_hdri/4k_exr"
    hdri_txt = "/baai-cwm-1/baai_cwm_ml/cwm/hong.li/code/3dgen/data/zeroverse/metadatas/hdri_cross_list.txt"
    with open(hdri_txt, 'r') as f:
        hdri_file_paths = f.read().splitlines()
    hdri_file_path = os.path.join(hdri_file_dir, np.random.choice(hdri_file_paths))

    args = [
        BLENDER_PATH, '-b', '-P', os.path.join(os.path.dirname(__file__), 'blender_script', 'render_ps_v3.py'),
        '--',
        '--num_views', f'{num_views}',
        '--num_lights', f'{num_lights}',
        '--object', file_path,
        '--resolution', '4096',
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
            '--resolution', '1024',
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
    
if __name__ == '__main__':
    # files_dir = "/baai-cwm-1/baai_cwm_ml/cwm/hong.li/code/3dgen/data/zeroverse/unips_bench/M18"
    # import glob
    # file_paths = glob.glob(os.path.join(files_dir, '*', '*.FBX'))

    import glob
    files_dir = "/baai-cwm-1/baai_cwm_ml/cwm/hong.li/code/3dgen/data/zeroverse/unips_bench/unzips"
    file_paths = glob.glob(os.path.join(files_dir, '*', '*.stl'))

    selected_file_paths = [file_path for file_path in file_paths if 'supported' not in file_path]

    output_dir = "render_all"
    out_basename = 'unips_bench/unzips4K'

    os.makedirs(os.path.join(output_dir, out_basename), exist_ok=True)
    print(f'Output directory: {output_dir}, Base name: {out_basename}', flush=True)
    
    print('Checking blender...', flush=True)
    _install_blender()

    for file_path in tqdm.tqdm(selected_file_paths):
        print(f'Processing {file_path}...', flush=True)
        sha256 = os.path.basename(file_path).split('.')[0]
        _render(file_path, sha256, output_dir, num_views=1, num_lights=20, low_normal=False, out_basename=out_basename)





                