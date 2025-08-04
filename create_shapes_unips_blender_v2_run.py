import os  
import uuid  
from subprocess import DEVNULL, call  
from tqdm.contrib.concurrent import process_map  


BLENDER_LINK = 'https://download.blender.org/release/Blender4.0/blender-4.0.2-linux-x64.tar.xz'  
BLENDER_INSTALLATION_PATH = '/baai-cwm-1/baai_cwm_ml/cwm/hong.li/tools'  
BLENDER_PATH = f'{BLENDER_INSTALLATION_PATH}/blender-4.0.2-linux-x64/blender'  


def _install_blender():  
    if not os.path.exists(BLENDER_PATH):  
        os.system('sudo apt-get update')  
        os.system('sudo apt-get install -y libxrender1 libxi6 libxkbcommon-x11-0 libsm6')  
        os.system(f'wget {BLENDER_LINK} -P {BLENDER_INSTALLATION_PATH}')  
        os.system(f'tar -xvf {BLENDER_INSTALLATION_PATH}/blender-4.0.2-linux-x64.tar.xz -C {BLENDER_INSTALLATION_PATH}')  
        os.system('export EGL_DRIVER=nvidia')  
        os.system('export __EGL_VENDOR_LIBRARY_DIRS=/usr/share/glvnd/egl_vendor.d')  
        print('Blender installed', flush=True)  

def render_task(index):  
    """  
    单独渲染任务函数，调用Blender渲染指定index对应场景。  
    level根据index每1万个区间划分。  
    """  
    output_dir = "/baai-cwm-1/baai_cwm_ml/cwm/hong.li/code/3dgen/data/zeroverse/output_all/PS_v2_6w"  
    os.makedirs(output_dir, exist_ok=True)
    level = index // 15000 + 1  
    scene_uuid = uuid.uuid4().hex  
    script_path = os.path.join(os.path.dirname(__file__), 'create_shapes_unips_blender_v2.py')  

    args = [  
        BLENDER_PATH, '-b', '-P', script_path,  
        '--',  
        '--index', str(index),  
        '--output_dir', output_dir,  
        '--uuid', scene_uuid,  
        '--level', str(level),  
    ]  
    # call时屏蔽输出，改成需要时去掉 stdout/stderr 禁止重定向  
    call(args, stdout=DEVNULL, stderr=DEVNULL)  
    # call(args)  

    # 返回index调试追踪用可选  
    return index  

if __name__ == "__main__":  
    _install_blender()  # 如果有安装步骤  

    num = 60000  
    max_workers = 16  # 推荐根据CPU核心数调整  
    chunksize = 10    # 任务批大小，调节性能与响应速度权衡  

    # process_map内部分配任务，多进程执行render_task自动显示进度  
    results = process_map(render_task, range(num), max_workers=max_workers, chunksize=chunksize)
