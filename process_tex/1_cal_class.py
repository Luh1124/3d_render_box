import os
import numpy as np
from PIL import Image
import csv
import glob
from tqdm import tqdm

# 配置参数
MATSYNTH_DIR = "/baai-cwm-1/baai_cwm_ml/algorithm/hong.li/code/3dgen/data/BlenderProc/resources/matsynth_processed_v2"
OUTPUT_CSV = "./matsynth_metal_stats.csv"
METAL_THRESHOLD = 0.2  # 像素值阈值
MATERIAL_THRESHOLD = 0.2  # 材质判定阈值

def analyze_metallic_images():
    stats = []
    
    # 遍历所有材质文件夹
    for material_dir in tqdm(glob.glob(os.path.join(MATSYNTH_DIR, "*"))):
        if not os.path.isdir(material_dir):
            continue
            
        # 查找金属度贴图
        metallic_files = glob.glob(os.path.join(material_dir, "*Metallness.png"))
        if not metallic_files:
            continue
            
        try:
            # 读取并处理金属度贴图
            with Image.open(metallic_files[0]) as img:
                gray_img = img.convert("L")
                img_array = np.array(gray_img, dtype=np.float32) / 255.0
                
                # 计算金属区域
                metal_mask = img_array > METAL_THRESHOLD
                metal_pixels = np.sum(metal_mask)
                total_pixels = img_array.size
                metal_ratio = metal_pixels / total_pixels
                
                # 记录统计信息
                stats.append({
                    "name": os.path.basename(material_dir),
                    "metal_pixels": metal_pixels,
                    "total_pixels": total_pixels,
                    "metal_ratio": metal_ratio,
                    "is_metal": metal_ratio > MATERIAL_THRESHOLD
                })
                
        except Exception as e:
            print(f"处理失败: {material_dir} - {str(e)}")
    
    # 导出统计结果
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "metal_pixels", "total_pixels", "metal_ratio", "is_metal"])
        writer.writeheader()
        writer.writerows(stats)
        
    return stats

if __name__ == "__main__":
    stats = analyze_metallic_images()
    metal_count = sum(1 for s in stats if s["is_metal"])
    non_metal_count = len(stats) - metal_count
    
    print(f"\n统计完成 | 总材质: {len(stats)}")
    print(f"金属材质: {metal_count} ({metal_count/len(stats):.1%})")
    print(f"非金属材质: {non_metal_count} ({non_metal_count/len(stats):.1%})")
    print(f"结果已保存至: {OUTPUT_CSV}")