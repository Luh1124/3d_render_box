import os  
import glob  
import hashlib  
import pandas as pd  
import ast  
from concurrent.futures import ProcessPoolExecutor, as_completed  
from concurrent.futures import ThreadPoolExecutor, as_completed  

from tqdm import tqdm  

# PS_v1_path = "/baai-cwm-1/baai_cwm_ml/cwm/hong.li/code/3dgen/data/zeroverse/output_all/PS_v1"  
PS_v1_path = "/baai-cwm-1/baai_cwm_ml/cwm/hong.li/code/3dgen/data/zeroverse/output_all/PS_v2"  

def compute_sha256(filepath):  
    hash_sha256 = hashlib.sha256()  
    with open(filepath, "rb") as f:  
        for chunk in iter(lambda: f.read(4096), b""):  
            hash_sha256.update(chunk)  
    return hash_sha256.hexdigest()  

def parse_txt_file(txt_path):  
    data = {"uuid": None, "level": None, "glb_files": []}  
    with open(txt_path, 'r') as f:  
        for line in f:  
            line = line.strip()  
            if line.startswith("uuid:"):  
                data["uuid"] = line.split(":", 1)[1].strip()  
            elif line.startswith("level:"):  
                data["level"] = line.split(":", 1)[1].strip()  
            elif line.startswith("glb_files:"):  
                list_str = line.split(":", 1)[1].strip()  
                try:  
                    data["glb_files"] = ast.literal_eval(list_str)  
                except:  
                    data["glb_files"] = []  
    return data  

def process_single_blend(blend_path):  
    try:  
        sha256_hash = compute_sha256(blend_path)  
        txt_path = os.path.splitext(blend_path)[0] + ".txt"  
        if not os.path.exists(txt_path):  
            # 返回 None 或特殊标记，主进程处理过滤  
            return None  

        parsed = parse_txt_file(txt_path)  
        return {  
            "blend_path": blend_path,  
            "sha256": sha256_hash,  
            "uuid": parsed["uuid"],  
            "level": parsed["level"],  
            "glb_files": ",".join(parsed["glb_files"])  # CSV中用逗号分隔  
        }  
    except Exception as e:  
        print(f"Error processing {blend_path}: {e}")  
        return None  

def find_blends_in_dir(dir_path):  
    if not os.path.exists(dir_path):  
        return []  
    return glob.glob(os.path.join(dir_path, "*.blend"))  

def main():  
    blend_file_dirs = [os.path.join(PS_v1_path, f"level_{int(i / 10000 + 1)}", f"{i:06d}") for i in range(40000)]  
    blend_files = []  
    max_workers = 32  # 可根据CPU和磁盘性能调节线程数  

    with ThreadPoolExecutor(max_workers=max_workers) as executor:  
        futures = {executor.submit(find_blends_in_dir, d): d for d in blend_file_dirs}  

        for future in tqdm(as_completed(futures), total=len(futures), desc="Finding blend files"):  
            try:  
                files = future.result()  
                blend_files.extend(files)  
            except Exception as e:  
                print(f"Error processing directory {futures[future]}: {e}")  

    print(f"Total .blend files found: {len(blend_files)}")  

    results = []  

    with ProcessPoolExecutor(max_workers=24) as executor:  
        futures = {executor.submit(process_single_blend, p): p for p in blend_files}  

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing blends"):  
            res = future.result()  
            if res is not None:  
                results.append(res)  

    if results:  
        df = pd.DataFrame(results)  
        output_csv = os.path.join(PS_v1_path, "blend_files_summary_from_txt.csv")  
        df.to_csv(output_csv, index=False, encoding="utf-8-sig")  
        print(f"Summary CSV saved to: {output_csv}")  
    else:  
        print("No results to save.")  

if __name__ == "__main__":  
    main() 