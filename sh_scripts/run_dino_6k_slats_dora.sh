export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/baai-cwm-1/baai_cwm_ml/cwm/chongjie.ye/cache/huggingface
export TORCH_HOME=/baai-cwm-1/baai_cwm_ml/cwm/chongjie.ye/cache/torch
export GRADIO_TEMP_DIR=/baai-cwm-1/baai_cwm_ml/cwm/chongjie.ye/cache/gradio

export DEBIAN_FRONTEND=noninteractive  
# export NVIDIA_VISIBLE_DEVICES=all  
export NVIDIA_DRIVER_CAPABILITIES=all  

pip install pandas pydantic
echo "deb http://mirrors.aliyun.com/ubuntu/ jammy main" | sudo tee -a /etc/apt/sources.list  
apt-get update && apt-get -q install -y --no-install-recommends --fix-missing libc6
strings /usr/lib/x86_64-linux-gnu/libstdc++.so.6 | grep GLIBC

cd /baai-cwm-1/baai_cwm_ml/cwm/hong.li/code/3dgen/data/zeroverse/vox_slat_process

echo "当前路径: $(pwd)"

# cuda是否可以使用？
nvidia-smi

# 查看当前python路径
which python

# Default parameters  

# DEFAULT_OUTPUT_DIR="/baai-cwm-1/baai_cwm_ml/cwm/hong.li/code/3dgen/data/zeroverse/3diclight_matsynth/even_slat_6k"
# DEFAULT_SUBDIR="renders_3diclightv3_eevee_even_150views"
# DEFAULT_FEATURE="dino_surface_slats"
RANK=0
WORLD_SIZE=10

# Check command-line arguments and use default values if not provided  
RANK=${1:-$RANK}  # Use default output directory if not provided  
WORLD_SIZE=${2:-$WORLD_SIZE}            # Use default subset if not provided 
# OUTPUT_DIR=${3:-$DEFAULT_OUTPUT_DIR}  # Use default output directory if not provided
# SUBDIR=${4:-$DEFAULT_SUBDIR}  # Use default output directory if not provided
# FEATURE=${5:-$DEFAULT_FEATURE}  # Use default output directory if not provided


CUDA_VISIBLE_DEVICES=0 python pbr2dino_voxel_torch_dora.py \
 --rank $RANK \
 --world_size $WORLD_SIZE 

wait  

echo "All rendering tasks for subset '$SUBDIR' completed, output saved in '$OUTPUT_DIR'."  
