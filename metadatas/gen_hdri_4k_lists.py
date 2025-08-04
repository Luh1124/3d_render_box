import os

hdri_4k = "/baai-cwm-nas/public_data/rendering_data/high_4k_hdri/4k_exr"

hdri_512 = "/baai-cwm-1/baai_cwm_ml/cwm/hong.li/code/3dgen/TRELLIS_train/relight_datasets/hdri_512"

# in 512 and 4k, the file name is the same
hdri_4k_list = os.listdir(hdri_4k)
hdri_512_list = os.listdir(hdri_512)

hdri_4k_set = set(hdri_4k_list)
hdri_512_set = set(hdri_512_list)

cross_list = list(hdri_4k_set & hdri_512_set)
print(f"hdri_4k: {len(hdri_4k_list)}, hdri_512: {len(hdri_512_list)}, cross: {len(cross_list)}")


with open("hdri_cross_list.txt", "w") as f:  
    for item in cross_list:
        f.write(item + "\n")