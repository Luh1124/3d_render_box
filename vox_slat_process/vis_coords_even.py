import os  
import numpy as np  
import pandas as pd  
import open3d as o3d  
import tqdm

csv_path = "/baai-cwm-1/baai_cwm_ml/cwm/hong.li/code/3dgen/data/zeroverse/3diclight_matsynth/metadata_3didlight_6k_new_paths.csv"  
df = pd.read_csv(csv_path)  
npz_paths = df['npz_paths'].values  

# base_path = "/baai-cwm-1/baai_cwm_ml/cwm/hong.li/code/3dgen/data/data_factory_blender/datasets/3diclight_even/dino_surface_slats/dinov2_vitl14_reg_latent_pbr_v2"  
base_path = "/baai-cwm-1/baai_cwm_ml/cwm/hong.li/code/3dgen/data/zeroverse/3diclight_matsynth/even_slat_6k/dino_surface_slats"
# sample_npz_paths = np.random.choice(npz_paths, 200, replace=False)  
sample_npz_paths = npz_paths[:10] 
sample_npz_paths = [os.path.join(base_path, path) for path in sample_npz_paths]  

output_dir = "./ply_outputs"  
os.makedirs(output_dir, exist_ok=True)  

for npz_path in tqdm.tqdm(sample_npz_paths, desc="Processing PLY files", total=len(sample_npz_paths)):
    np_data = np.load(npz_path)  
    coords = np_data['coords'] / 64. - 0.5  # (N,3)  
    pbr_features = np_data['pbr_features']  # (N, features...)  
    colors_raw = pbr_features[:, 2:5]        # (N,3), RGB or similar  

    N = coords.shape[0]  
    sample_num = 1000  

    if N >= sample_num:  
        chosen_idx = np.random.choice(N, sample_num, replace=False)  
    else:  
        chosen_idx = np.arange(N)  

    coords_sampled = coords[chosen_idx]  
    colors_sampled = colors_raw[chosen_idx]  

    # 归一化颜色到[0,1]  
    colors_min = colors_sampled.min(axis=0, keepdims=True)  
    colors_max = colors_sampled.max(axis=0, keepdims=True)  
    colors_norm = (colors_sampled - colors_min) / (colors_max - colors_min + 1e-8)  
    colors_norm = colors_norm.astype(np.float64)  # open3d expects float64 or float32 colors  

    # 构造open3d点云对象  
    pcd = o3d.geometry.PointCloud()  
    pcd.points = o3d.utility.Vector3dVector(coords_sampled)  
    pcd.colors = o3d.utility.Vector3dVector(colors_norm)  

    ply_save_path = os.path.join(output_dir, os.path.basename(npz_path).replace('.npz', '.ply'))  
    o3d.io.write_point_cloud(ply_save_path, pcd)  

print(f"Saved {len(sample_npz_paths)} PLY files to {output_dir}")  