import os  
import glob  
import hashlib  
import pandas as pd  
import glob
from concurrent.futures import ProcessPoolExecutor, as_completed  
from concurrent.futures import ThreadPoolExecutor, as_completed  

from tqdm import tqdm  

dora_meshs_dir = '/baai-cwm-1/baai_cwm_ml/cwm/hong.li/code/3dgen/data/zeroverse/dora_bench_meshes'

dora_meshs_paths = glob.glob(os.path.join(dora_meshs_dir, '*/*.glb'))

dora_metadata_dict = {
    "blend_path": [],
    "sha256": [],
    "level": [],
    "render_paths": [],
    "npz_paths": []
}

def split_path(path):
    # Split the path into directory and filename
    sha256 = os.path.basename(path).split('.')[0]
    level = os.path.basename(os.path.dirname(path))
    return sha256, level

for path in dora_meshs_paths:
    sha256, level = split_path(path)
    dora_metadata_dict["blend_path"].append(path)
    dora_metadata_dict["sha256"].append(sha256)
    dora_metadata_dict["level"].append(level)
    dora_metadata_dict["render_paths"].append(os.path.join(f"{level}", sha256))
    dora_metadata_dict["npz_paths"].append(os.path.join(f"{level}", sha256 + '.npz'))

# Convert the dictionary to a DataFrame
dora_metadata_df = pd.DataFrame(dora_metadata_dict)
# Save the DataFrame to a CSV file
dora_metadata_df.to_csv('/baai-cwm-1/baai_cwm_ml/cwm/hong.li/code/3dgen/data/zeroverse/metadatas/dora_bench_meshes_paths.csv', index=False)

# Print the number of rows in the DataFrame
print(f"Number of rows in the DataFrame: {len(dora_metadata_df)}")
# Print the first few rows of the DataFrame
print(dora_metadata_df.head())
    
