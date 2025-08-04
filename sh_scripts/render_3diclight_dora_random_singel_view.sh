#!/bin/bash  

# source /baai-cwm-1/baai_cwm_ml/algorithm/hong.li/miniconda3/bin/activate

# conda activate trellis

export DEBIAN_FRONTEND=noninteractive  
# export NVIDIA_VISIBLE_DEVICES=all  
export NVIDIA_DRIVER_CAPABILITIES=all  

apt-get update && apt-get -q install -y --no-install-recommends --fix-missing \
    automake \
    autoconf \
    build-essential \
    git \
    libbz2-dev \
    libegl1 \
    libfontconfig1 \
    libgl1 \
    libglvnd-dev \
    libgtk-3-0 \
    libsm6 \
    libtool \
    libx11-6 \
    libx11-dev \
    libxcursor1 \
    libxext6 \
    libxext-dev \
    libxi6 \
    libxinerama1 \
    libxkbcommon0 \
    libxrandr2 \
    libxrender1 \
    libxxf86vm1 \
    mesa-utils \
    pkg-config \
    wget \
    python3 \
    python3-pip

apt-get -q install -y --no-install-recommends --fix-missing \
    x11proto-dev \
    x11proto-gl-dev \
    libxrender1 \
    xvfb

apt-get clean && rm -rf /var/lib/apt/lists/*

git clone https://githubfast.com/NVIDIA/libglvnd.git /tmp/libglvnd \
    && cd /tmp/libglvnd \
    && ./autogen.sh \
    && ./configure \
    && make -j$(nproc) \
    && make install \
    && mkdir -p /usr/share/glvnd/egl_vendor.d/ \
    && printf "{\n\
    \"file_format_version\" : \"1.0.0\",\n\
    \"ICD\": {\n\
        \"library_path\": \"libEGL_nvidia.so.0\"\n\
    }\n\
    }" > /usr/share/glvnd/egl_vendor.d/10_nvidia.json \
    && cd / \
    && rm -rf /tmp/libglvnd

export EGL_DRIVER=nvidia
export __EGL_VENDOR_LIBRARY_DIRS=/usr/share/glvnd/egl_vendor.d

cd /baai-cwm-vepfs/cwm/hong.li/code/3dgen/data/zeroverse

echo "当前路径: $(pwd)"

# cuda是否可以使用？
nvidia-smi

# 查看当前python路径
which python

# Default parameters  
DEFAULT_OUTPUT_DIR="dorabenchmark"  
DEFAULT_SUBSET="3diclight"  
DEFAULT_OUTBASE="relight_random"  # Default output base name
RANK=0
WORLD_SIZE=8
MAX_WORKERS=12

# Check command-line arguments and use default values if not provided  
RANK=${1:-$RANK}  # Use default output directory if not provided  
WORLD_SIZE=${2:-$WORLD_SIZE}            # Use default subset if not provided 
MAX_WORKERS=${3:-$MAX_WORKERS}
OUTPUT_DIR=${4:-$DEFAULT_OUTPUT_DIR}  # Use default output directory if not provided  
SUBSET=${5:-$DEFAULT_SUBSET}            # Use default subset if not provided  
OUTBASE=${6:-$DEFAULT_OUTBASE}          # Use default output base name if not provided

# chmod -R 777 "${OUTPUT_DIR}/renders_pbr"

echo "python dataset_toolkits/run_render_3diclight_dora_relight_random_for_single_view.py $SUBSET --output_dir $OUTPUT_DIR --out_basename $OUTBASE --rank $RANK --world_size $WORLD_SIZE --max_workers $MAX_WORKERS"

# CUDA_VISIBLE_DEVICES=0 python dataset_toolkits/render_pbr.py $SUBSET --output_dir $OUTPUT_DIR --rank $RANK --world_size $WORLD_SIZE --max_workers $MAX_WORKERS

python dataset_toolkits/run_render_3diclight_dora_relight_random_for_single_view.py $SUBSET --output_dir $OUTPUT_DIR --out_basename $OUTBASE --rank $RANK --world_size $WORLD_SIZE --max_workers $MAX_WORKERS


# python dataset_toolkits/render_pbr.py $SUBSET --output_dir $OUTPUT_DIR --rank $RANK --world_size $WORLD_SIZE --max_workers $MAX_WORKERS

# chmod -R 777 "${OUTPUT_DIR}/renders_pbr"

# Wait for all background processes to complete  
wait  

echo "All rendering tasks for subset '$SUBSET' completed, output saved in '$OUTPUT_DIR'."  
