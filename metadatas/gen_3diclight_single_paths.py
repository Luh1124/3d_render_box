import os  
import glob  
import hashlib  
import pandas as pd  
import ast  
from concurrent.futures import ProcessPoolExecutor, as_completed  
from concurrent.futures import ThreadPoolExecutor, as_completed  

from tqdm import tqdm  

metadatacsv = "/baai-cwm-1/baai_cwm_ml/cwm/hong.li/code/3dgen/data/zeroverse/metadatas/metadata_3didlight_6k_new.csv"
new_metadatacsv = "/baai-cwm-1/baai_cwm_ml/cwm/hong.li/code/3dgen/data/zeroverse/metadatas/metadata_3didlight_6k_new_paths.csv"

df = pd.read_csv(metadatacsv)

def combine_path(blend_path, sha256, level):  
    """  
    根据 blend_path、sha256 和 level 组合新的路径  
    
    Args:  
        blend_path (str): 原始 Blend 文件路径  
        sha256 (str): 文件的 SHA256 哈希值  
        level (str): 层级  
    
    Returns:  
        str: 新组合的路径  
    """  
    # 从原路径提取目录  
    
    new_path = os.path.join(f"level_{level}/{sha256}")  
    
    return new_path  

df['render_paths'] = df.apply(lambda row: combine_path(row['blend_path'], row['sha256'], row['level']), axis=1)
df['npz_paths'] = df.apply(lambda row: combine_path(row['blend_path'], row['sha256'], row['level']) + '.npz', axis=1)


#save df to csv
df.to_csv(new_metadatacsv, index=False)

