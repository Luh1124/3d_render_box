import os
import pandas as pd
import glob
import numpy as np

import tqdm
import pathlib as pl

# create train csv and val csv

pbr2dino_voxel_latent_dir = "/baai-cwm-1/baai_cwm_ml/cwm/hong.li/code/3dgen/data/zeroverse/dorabenchmark/pbr2dino_voxel_latent"


pbr2dino_voxel_latent_paths = glob.glob(os.path.join(pbr2dino_voxel_latent_dir, '*', "*.npz"))

sha256s = [os.path.basename(path).split('.')[0] for path in pbr2dino_voxel_latent_paths]

even_render_6k_dir = "/baai-cwm-1/baai_cwm_ml/cwm/hong.li/code/3dgen/data/zeroverse/dorabenchmark/even"

relight_render_6k_dir = "/baai-cwm-1/baai_cwm_ml/cwm/hong.li/code/3dgen/data/zeroverse/dorabenchmark/relight"


csv_path = "/baai-cwm-1/baai_cwm_ml/cwm/hong.li/code/3dgen/data/zeroverse/dorabenchmark/val.csv"
df = pd.DataFrame()

df['sha256'] = sha256s
df['latent_path'] = pbr2dino_voxel_latent_paths
df['even_render_6k'] = [os.path.join(even_render_6k_dir, os.path.basename(os.path.dirname(path)), os.path.basename(path)) for path in pbr2dino_voxel_latent_paths]
df['relight_render_6k'] = [os.path.join(relight_render_6k_dir, os.path.basename(os.path.dirname(path)), os.path.basename(path)) for path in pbr2dino_voxel_latent_paths]

# check pbr2dino_voxel_latent_paths readload

for path in tqdm.tqdm(pbr2dino_voxel_latent_paths, total=len(pbr2dino_voxel_latent_paths)):
    np_data = np.load(path)['feats']

    if len(np_data) == 0:
        print(f"np_data is empty: {path}")

# 拆分训练和测试
df.to_csv(csv_path, index=False)