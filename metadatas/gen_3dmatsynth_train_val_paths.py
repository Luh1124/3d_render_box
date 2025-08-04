import os
import pandas as pd
import glob
import numpy as np

import tqdm
import pathlib as pl

import concurrent.futures  
import numpy as np  
import tqdm  

def load_and_check(path):  
    np_data = np.load(path)['feats']  
    if len(np_data) == 0:  
        return f"np_data is empty: {path}"  
    return None  

# create train csv and val csv

pbr2dino_voxel_latent_dir = "/baai-cwm-1/baai_cwm_ml/cwm/hong.li/code/3dgen/data/zeroverse/3diclight_matsynth/even_slat_6k_debug/pbr2dino_voxel_latent"


pbr2dino_voxel_latent_paths = glob.glob(os.path.join(pbr2dino_voxel_latent_dir, '*', "*.npz"))

sha256s = [os.path.basename(path).split('.')[0] for path in pbr2dino_voxel_latent_paths]

even_render_6k_dir = "/baai-cwm-1/baai_cwm_ml/cwm/hong.li/code/3dgen/data/zeroverse/3diclight_matsynth/even_render_6k"

relight_render_6k_dir = "/baai-cwm-1/baai_cwm_ml/cwm/hong.li/code/3dgen/data/zeroverse/3diclight_matsynth/relight_render_6k"


csv_path = "/baai-cwm-1/baai_cwm_ml/cwm/hong.li/code/3dgen/data/zeroverse/3diclight_matsynth/even_slat_6k_debug/train_6k.csv"
df = pd.DataFrame()

df['sha256'] = sha256s
df['slat_path'] = pbr2dino_voxel_latent_paths
df['even'] = [os.path.join(even_render_6k_dir, os.path.basename(os.path.dirname(path)), os.path.basename(path).split('.')[0]) for path in pbr2dino_voxel_latent_paths]
df['relight'] = [os.path.join(relight_render_6k_dir, os.path.basename(os.path.dirname(path)), os.path.basename(path).split('.')[0]) for path in pbr2dino_voxel_latent_paths]

# check pbr2dino_voxel_latent_paths readload

with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:  
    futures = [executor.submit(load_and_check, path) for path in pbr2dino_voxel_latent_paths]  

    for f in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(futures)):  
        msg = f.result()  
        if msg:  
            print(msg)  

# 拆分训练和测试
train_df = df.sample(frac=0.95, random_state=42)
val_df = df.drop(train_df.index)
train_df.to_csv(csv_path, index=False)
val_csv_path = "/baai-cwm-1/baai_cwm_ml/cwm/hong.li/code/3dgen/data/zeroverse/3diclight_matsynth/even_slat_6k_debug/val_6k.csv"
val_df.to_csv(val_csv_path, index=False)
print(f"train csv saved to {csv_path}")
print(f"val csv saved to {val_csv_path}")