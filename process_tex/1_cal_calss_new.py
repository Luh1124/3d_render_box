import os
import numpy as np
from PIL import Image
import csv
import glob
from tqdm import tqdm

# 配置参数
MATSYNTH_DIR = "/baai-cwm-1/baai_cwm_ml/cwm/hong.li/code/3dgen/data/BlenderProc/resources/matsynth_processed_v2"
OUTPUT_CSV = "./matsynth_material_stats.csv"
METAL_THRESHOLD = 0.2  # 金属度贴图阈值
MATERIAL_THRESHOLD = 0.2
ROUGH_MIRROR_THRESHOLD = 0.3  # 低于该阈值认为是镜面区域（低粗糙度）
MIRROR_RATIO_THRESHOLD = 0.2  # 判定镜面反射材质阈值
ROUGH_DIFFUSE_THRESHOLD = 0.6  # 高于该阈值认为是漫反射区域（高粗糙度）
DIFFUSE_RATIO_THRESHOLD = 0.2  # 判定漫反射材质阈值

def analyze_materials():
    stats = []

    for material_dir in tqdm(glob.glob(os.path.join(MATSYNTH_DIR, "*"))):
        if not os.path.isdir(material_dir):
            continue
        
        # 金属度分析
        metallic_files = glob.glob(os.path.join(material_dir, "*Metallness.png"))
        metal_ratio = 0.0
        metal_pixels = 0
        total_pixels = 0
        is_metal = False
        
        if metallic_files:
            try:
                with Image.open(metallic_files[0]) as img:
                    gray_img = img.convert("L")
                    img_array = np.array(gray_img, dtype=np.float32) / 255.0
                    metal_mask = img_array > METAL_THRESHOLD
                    metal_pixels = np.sum(metal_mask)
                    total_pixels = img_array.size
                    metal_ratio = metal_pixels / total_pixels
                    is_metal = (metal_ratio > MATERIAL_THRESHOLD)
            except Exception as e:
                print(f"金属度处理失败: {material_dir} - {e}")
        
        # 粗糙度分析 (用以判定镜面/漫反射)
        roughness_files = glob.glob(os.path.join(material_dir, "*Roughness.png"))
        mirror_ratio = 0.0
        mirror_pixels = 0
        diffuse_ratio = 0.0
        diffuse_pixels = 0
        is_mirror = False
        is_diffuse = False
        
        if roughness_files:
            try:
                with Image.open(roughness_files[0]) as rough_img:
                    rough_gray = rough_img.convert("L")
                    rough_array = np.array(rough_gray, dtype=np.float32) / 255.0
                    mirror_mask = rough_array < ROUGH_MIRROR_THRESHOLD
                    mirror_pixels = np.sum(mirror_mask)
                    mirror_ratio = mirror_pixels / rough_array.size
                    is_mirror = mirror_ratio > MIRROR_RATIO_THRESHOLD

                    diffuse_mask = rough_array > ROUGH_DIFFUSE_THRESHOLD
                    diffuse_pixels = np.sum(diffuse_mask)
                    diffuse_ratio = diffuse_pixels / rough_array.size
                    is_diffuse = diffuse_ratio > DIFFUSE_RATIO_THRESHOLD
            except Exception as e:
                print(f"粗糙度处理失败: {material_dir} - {e}")

        stats.append({
            "name": os.path.basename(material_dir),
            "metal_pixels": metal_pixels,
            "total_pixels": total_pixels,
            "metal_ratio": metal_ratio,
            "is_metal": is_metal,
            "mirror_pixels": mirror_pixels,
            "mirror_ratio": mirror_ratio,
            "is_mirror": is_mirror,
            "diffuse_pixels": diffuse_pixels,
            "diffuse_ratio": diffuse_ratio,
            "is_diffuse": is_diffuse
        })

    # 保存结果
    with open(OUTPUT_CSV, "w", newline="") as f:
        fieldnames = ["name", "metal_pixels", "total_pixels", "metal_ratio", "is_metal",
                      "mirror_pixels", "mirror_ratio", "is_mirror",
                      "diffuse_pixels", "diffuse_ratio", "is_diffuse"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(stats)

    return stats


if __name__ == "__main__":
    stats = analyze_materials()

    metal_count = sum(1 for s in stats if s["is_metal"])
    mirror_count = sum(1 for s in stats if s["is_mirror"])
    diffuse_count = sum(1 for s in stats if s["is_diffuse"])
    total = len(stats)
    neither_count = sum(1 for s in stats if not (s["is_metal"] or s["is_mirror"] or s["is_diffuse"]))

    print(f"\n统计完成 | 总材质数: {total}")
    print(f"金属材质: {metal_count} ({metal_count / total:.1%})")
    print(f"镜面反射材质: {mirror_count} ({mirror_count / total:.1%})")
    print(f"漫反射材质: {diffuse_count} ({diffuse_count / total:.1%})")
    print(f"未归类材质（非金属/非镜面/非漫反射）: {neither_count} ({neither_count / total:.1%})")
    print(f"详细结果已保存至: {OUTPUT_CSV}")