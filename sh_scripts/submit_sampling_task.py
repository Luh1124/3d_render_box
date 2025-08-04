import os

template = """# self define e.g text_classfication
TaskName: "{task_name}"
# description for this task
Description: ""
# entry point command
Entrypoint: "{command}"
Tags: []
# the code path you want to upload locally
UserCodePath: ""
# remote path mount in training container
RemoteMountCodePath: ""
# user define env var
Envs:
  - Name: "DEBIAN_FRONTEND"
    Value: "noninteractive"
  - Name: "NVIDIA_DRIVER_CAPABILITIES"
    Value: "all"
# queue created under Resource Group, empty as default queue
ResourceQueueID: {queue_id}
# distributed framework, support: TensorFlowPS / PyTorchDDP / MPI / BytePS / Custom
Framework: "Custom"
TaskRoleSpecs:
    - RoleName: "worker"
      RoleReplicas: {num_gpus}
      Flavor: "{flavor}"
ActiveDeadlineSeconds: 864000
DelayExitTimeSeconds: 120
# enable tensor board or not
EnableTensorBoard: false
# storages
Storages:
    - Type: "Vepfs"
      MountPath: "/baai-cwm-vepfs"
      VepfsId: "vepfs-cnbj4b621a1f4a2c"
    - Type: "Nas"                      # 挂载 NAS 数据盘
      MountPath: "/baai-cwm-backup"       # 容器中的挂载目录
      NasId: "cnas-cnbjb0e2acbfba11e1"  # NAS 实例的挂载点地址
ImageUrl: "vemlp-cn-beijing.cr.volces.com/preset-images/cuda:11.8.0-py3.10-ubuntu20.04"
CacheType: "Cloudfs"
# user define retry options
RetryOptions:
    EnableRetry: false
    MaxRetryTimes: 5
    IntervalSeconds: 120
    PolicySets: []
# diagnosis options
DiagOptions:
    - Name: "EnvironmentalDiagnosis"
      Enable: false
    - Name: "PythonDetection"
      Enable: false
    - Name: "LogDetection"
      Enable: false"""
      

import uuid
import time

flavor = {
    'L20render': 'ml.gni3cgd.5xlarge',
    '4090': 'ml.xni3c.5xlarge',
}
# For 4090d [ml.xni3c.5xlarge, ml.xni3c.11xlarge, ml.xni3c.22xlarge, ml.xni3c.45xlarge]
# For L20 [ml.gni3cgd.5xlarge, ml.gni3cgd.11xlarge, ml.gni3cgd.22xlarge, ml.gni3cgd.45xlarge] 
# For A800 [ml.pni2l.3xlarge, ml.pni2l.7xlarge, ml.pni2l.14xlarge, ml.pni2l.28xlarge]

queue_id = {
    'L20render': 'q-20250701160101-65wrj',
    '4090': 'q-20250618153607-mvb77',
}

num_gpus = 1

# ========== For 5d aigc v3 =================================
# taskname_prefix = "unips_ps_v3_6w"

# world_size = 24
# max_workers = 12

# # for rank in range(0, world_size):
# # for rank in range(0, world_size):
# for rank in [11,]:
#     taskname = f'{taskname_prefix}_{rank}_{world_size}_{max_workers}'
#     command = f'''cd /baai-cwm-1/baai_cwm_ml/cwm/hong.li/code/3dgen/data/zeroverse &&
#     source activate /baai-cwm-1/baai_cwm_ml/cwm/hong.li/miniconda3/envs/Dora &&
#     bash sh_scripts/render_p1_v3.sh {rank} {world_size} {max_workers}'''
    
#     with open(f'sh_scripts/task_yaml/{taskname}.yaml', 'w') as f:
#         f.write(template.format(task_name=taskname, command=command, queue_id=queue_id['L20'], flavor=flavor['L20'], num_gpus=num_gpus))
#     print(f'volc ml_task submit --conf ./sh_scripts/task_yaml/{taskname}.yaml')
#     # time.sleep(3)
#     os.system(f'volc ml_task submit --conf ./sh_scripts/task_yaml/{taskname}.yaml')
#     # remove config file
#     os.system(f'rm ./sh_scripts/task_yaml/{taskname}.yaml')

# taskname_prefix = "render_3diclight_even_6k"

# world_size = 24
# max_workers = 12

# # for rank in range(0, world_size):
# # for rank in range(0, 1):
# # for rank in range(1, 4):
# for rank in range(4, world_size):
# # for rank in [11,]:
#     taskname = f'{taskname_prefix}_{rank}_{world_size}_{max_workers}'
#     command = f'''cd /baai-cwm-1/baai_cwm_ml/cwm/hong.li/code/3dgen/data/zeroverse &&
#     source activate /baai-cwm-1/baai_cwm_ml/cwm/hong.li/miniconda3/envs/Dora &&
#     bash sh_scripts/render_3diclight_even.sh {rank} {world_size} {max_workers}'''
    
#     with open(f'sh_scripts/task_yaml/{taskname}.yaml', 'w') as f:
#         f.write(template.format(task_name=taskname, command=command, queue_id=queue_id['L20render'], flavor=flavor['L20render'], num_gpus=num_gpus))
#         # f.write(template.format(task_name=taskname, command=command, queue_id=queue_id['L20'], flavor=flavor['L20'], num_gpus=num_gpus))
#     print(f'volc ml_task submit --conf ./sh_scripts/task_yaml/{taskname}.yaml')
#     # time.sleep(3)
#     os.system(f'volc ml_task submit --conf ./sh_scripts/task_yaml/{taskname}.yaml')
#     # remove config file
#     os.system(f'rm ./sh_scripts/task_yaml/{taskname}.yaml')


# taskname_prefix = "render_3diclight_relight_6k"

# world_size = 12
# max_workers = 12

# for rank in range(0, 1):
# # for rank in range(0, world_size):
# # for rank in [11,]:
#     taskname = f'{taskname_prefix}_{rank}_{world_size}_{max_workers}'
#     command = f'''cd /baai-cwm-1/baai_cwm_ml/cwm/hong.li/code/3dgen/data/zeroverse &&
#     source activate /baai-cwm-1/baai_cwm_ml/cwm/hong.li/miniconda3/envs/Dora &&
#     bash sh_scripts/render_3diclight_relight.sh {rank} {world_size} {max_workers}'''
    
#     with open(f'sh_scripts/task_yaml/{taskname}.yaml', 'w') as f:
#         f.write(template.format(task_name=taskname, command=command, queue_id=queue_id['L20'], flavor=flavor['L20'], num_gpus=num_gpus))
#     print(f'volc ml_task submit --conf ./sh_scripts/task_yaml/{taskname}.yaml')
#     # time.sleep(3)
#     os.system(f'volc ml_task submit --conf ./sh_scripts/task_yaml/{taskname}.yaml')
#     # remove config file
#     os.system(f'rm ./sh_scripts/task_yaml/{taskname}.yaml')

# taskname_prefix = "gen_dino_6k_slats"

# world_size = 12

# # for rank in range(0, 1):
# # for rank in range(1, world_size):
# for rank in range(0, world_size):
# # for rank in [11,]:
#     taskname = f'{taskname_prefix}_{rank}_{world_size}'
#     command = f'''cd /baai-cwm-1/baai_cwm_ml/cwm/hong.li/code/3dgen/data/zeroverse &&
#     source activate /baai-cwm-1/baai_cwm_ml/cwm/hong.li/miniconda3/envs/trellis_train &&
#     bash sh_scripts/run_dino_6k_slats.sh {rank} {world_size}'''
    
#     with open(f'sh_scripts/task_yaml/{taskname}.yaml', 'w') as f:
#         f.write(template.format(task_name=taskname, command=command, queue_id=queue_id['L20'], flavor=flavor['L20'], num_gpus=num_gpus))
#     print(f'volc ml_task submit --conf ./sh_scripts/task_yaml/{taskname}.yaml')
#     # time.sleep(3)
#     os.system(f'volc ml_task submit --conf ./sh_scripts/task_yaml/{taskname}.yaml')
#     # remove config file
#     os.system(f'rm ./sh_scripts/task_yaml/{taskname}.yaml')

# taskname_prefix = "render_dora_even"

# world_size = 8

# # for rank in range(0, 1):
# for rank in range(1, world_size):
# # for rank in [11,]:
#     taskname = f'{taskname_prefix}_{rank}_{world_size}'
#     command = f'''cd /baai-cwm-vepfs/cwm/hong.li/code/3dgen/data/zeroverse &&
#     source activate /baai-cwm-vepfs/cwm/hong.li/miniconda3/envs/trellis &&
#     bash sh_scripts/render_3diclight_dora_even.sh {rank} {world_size}'''
    
#     with open(f'sh_scripts/task_yaml/{taskname}.yaml', 'w') as f:
#         f.write(template.format(task_name=taskname, command=command, queue_id=queue_id['4090'], flavor=flavor['4090'], num_gpus=num_gpus))
#     print(f'volc ml_task submit --conf ./sh_scripts/task_yaml/{taskname}.yaml')
#     # time.sleep(3)
#     os.system(f'volc ml_task submit --conf ./sh_scripts/task_yaml/{taskname}.yaml')
#     # remove config file
#     os.system(f'rm ./sh_scripts/task_yaml/{taskname}.yaml')


# taskname_prefix = "render_dora_relight"

# world_size = 16

# # for rank in range(0, 1):
# # for rank in range(1, world_size):
# for rank in range(0, world_size):
# # for rank in [11,]:
#     taskname = f'{taskname_prefix}_{rank}_{world_size}'
#     command = f'''cd /baai-cwm-1/baai_cwm_ml/cwm/hong.li/code/3dgen/data/zeroverse &&
#     source activate /baai-cwm-1/baai_cwm_ml/cwm/hong.li/miniconda3/envs/trellis_train &&
#     bash sh_scripts/render_3diclight_dora_relight.sh {rank} {world_size}'''
    
#     with open(f'sh_scripts/task_yaml/{taskname}.yaml', 'w') as f:
#         f.write(template.format(task_name=taskname, command=command, queue_id=queue_id['L20'], flavor=flavor['L20'], num_gpus=num_gpus))
#     print(f'volc ml_task submit --conf ./sh_scripts/task_yaml/{taskname}.yaml')
#     # time.sleep(3)
#     os.system(f'volc ml_task submit --conf ./sh_scripts/task_yaml/{taskname}.yaml')
#     # remove config file
#     os.system(f'rm ./sh_scripts/task_yaml/{taskname}.yaml')


# taskname_prefix = "run_dino_6k_slats"

# world_size = 50

# # for rank in range(0, 1):
# for rank in range(0, world_size):
# # for rank in [11,]:
#     taskname = f'{taskname_prefix}_{rank}_{world_size}'
#     command = f'''cd /baai-cwm-1/baai_cwm_ml/cwm/hong.li/code/3dgen/data/zeroverse &&
#     source activate /baai-cwm-1/baai_cwm_ml/cwm/hong.li/miniconda3/envs/trellis_train &&
#     bash sh_scripts/run_dino_6k_slats.sh {rank} {world_size}'''
    
#     with open(f'sh_scripts/task_yaml/{taskname}.yaml', 'w') as f:
#         # f.write(template.format(task_name=taskname, command=command, queue_id=queue_id['L20render'], flavor=flavor['L20render'], num_gpus=num_gpus))
#         f.write(template.format(task_name=taskname, command=command, queue_id=queue_id['L20'], flavor=flavor['L20'], num_gpus=num_gpus))
#     print(f'volc ml_task submit --conf ./sh_scripts/task_yaml/{taskname}.yaml')
#     # time.sleep(3)
#     os.system(f'volc ml_task submit --conf ./sh_scripts/task_yaml/{taskname}.yaml')
#     # remove config file
#     os.system(f'rm ./sh_scripts/task_yaml/{taskname}.yaml')

# taskname_prefix = "gen_dora_bench_slats"

# world_size = 6

# for rank in range(0, 1):
# # for rank in range(1, world_size):
# # for rank in [11,]:
#     taskname = f'{taskname_prefix}_{rank}_{world_size}'
#     command = f'''cd /baai-cwm-1/baai_cwm_ml/cwm/hong.li/code/3dgen/data/zeroverse &&
#     source activate /baai-cwm-1/baai_cwm_ml/cwm/hong.li/miniconda3/envs/trellis_train &&
#     bash sh_scripts/run_dino_6k_slats_dora.sh {rank} {world_size}'''
    
#     with open(f'sh_scripts/task_yaml/{taskname}.yaml', 'w') as f:
#         f.write(template.format(task_name=taskname, command=command, queue_id=queue_id['L20render'], flavor=flavor['L20render'], num_gpus=num_gpus))
#     print(f'volc ml_task submit --conf ./sh_scripts/task_yaml/{taskname}.yaml')
#     # time.sleep(3)
#     os.system(f'volc ml_task submit --conf ./sh_scripts/task_yaml/{taskname}.yaml')
#     # remove config file
#     os.system(f'rm ./sh_scripts/task_yaml/{taskname}.yaml')





# ========== For 5d aigc v3 =================================
# taskname_prefix = "render_ps_render_plane_v1"

# world_size = 12
# max_workers = 12

# # for rank in range(0, 1):
# # for rank in range(0, 8):
# # for rank in range(0, 3):
# for rank in range(3, world_size):
# # for rank in [11,]:
#     taskname = f'{taskname_prefix}_{rank}_{world_size}_{max_workers}'
#     command = f'''cd /baai-cwm-1/baai_cwm_ml/cwm/hong.li/code/3dgen/data/zeroverse &&
#     source activate /baai-cwm-1/baai_cwm_ml/cwm/hong.li/miniconda3/envs/Dora &&
#     bash sh_scripts/render_ps_render_plane.sh {rank} {world_size} {max_workers}'''
    
#     with open(f'sh_scripts/task_yaml/{taskname}.yaml', 'w') as f:
#         # f.write(template.format(task_name=taskname, command=command, queue_id=queue_id['L20render'], flavor=flavor['L20render'], num_gpus=num_gpus))
#         f.write(template.format(task_name=taskname, command=command, queue_id=queue_id['L20'], flavor=flavor['L20'], num_gpus=num_gpus))
#     print(f'volc ml_task submit --conf ./sh_scripts/task_yaml/{taskname}.yaml')
#     # time.sleep(3)
#     os.system(f'volc ml_task submit --conf ./sh_scripts/task_yaml/{taskname}.yaml')
#     # remove config file
#     os.system(f'rm ./sh_scripts/task_yaml/{taskname}.yaml')


# taskname_prefix = "render_relight_slat_6k"

# world_size = 14
# max_workers = 12

# # for rank in range(0, 1):
# # for rank in range(0, 8):
# # for rank in range(0, 3):
# for rank in range(1, world_size):
# # for rank in [11,]:
#     taskname = f'{taskname_prefix}_{rank}_{world_size}_{max_workers}'
#     command = f'''cd /baai-cwm-vepfs/cwm/hong.li/code/3dgen/data/zeroverse &&
#     source activate /baai-cwm-vepfs/cwm/hong.li/miniconda3/envs/blender &&
#     bash sh_scripts/render_3diclight_relight_slat.sh {rank} {world_size} {max_workers}'''

#     with open(f'sh_scripts/task_yaml/{taskname}.yaml', 'w') as f:
#         # f.write(template.format(task_name=taskname, command=command, queue_id=queue_id['L20render'], flavor=flavor['L20render'], num_gpus=num_gpus))
#         f.write(template.format(task_name=taskname, command=command, queue_id=queue_id['4090'], flavor=flavor['4090'], num_gpus=num_gpus))
#     print(f'volc ml_task submit --conf ./sh_scripts/task_yaml/{taskname}.yaml')
#     # time.sleep(3)
#     os.system(f'volc ml_task submit --conf ./sh_scripts/task_yaml/{taskname}.yaml')
#     # remove config file
#     os.system(f'rm ./sh_scripts/task_yaml/{taskname}.yaml')


# taskname_prefix = "render_relight_slat_6k_after_vox_do_dino"

# world_size = 14
# # max_workers = 12

# # for rank in range(0, 1):
# # for rank in range(0, 8):
# # for rank in range(0, 3):
# for rank in range(1, world_size):
# # for rank in [11,]:
#     taskname = f'{taskname_prefix}_{rank}_{world_size}'
#     command = f'''cd /baai-cwm-vepfs/cwm/hong.li/code/3dgen/data/zeroverse &&
#     source activate /baai-cwm-vepfs/cwm/hong.li/miniconda3/envs/trellis &&
#     bash sh_scripts/run_dino_6k_slats_6k_relight_slat.sh {rank} {world_size}'''

#     with open(f'sh_scripts/task_yaml/{taskname}.yaml', 'w') as f:
#         # f.write(template.format(task_name=taskname, command=command, queue_id=queue_id['L20render'], flavor=flavor['L20render'], num_gpus=num_gpus))
#         f.write(template.format(task_name=taskname, command=command, queue_id=queue_id['4090'], flavor=flavor['4090'], num_gpus=num_gpus))
#     print(f'volc ml_task submit --conf ./sh_scripts/task_yaml/{taskname}.yaml')
#     # time.sleep(3)
#     os.system(f'volc ml_task submit --conf ./sh_scripts/task_yaml/{taskname}.yaml')
#     # remove config file
#     os.system(f'rm ./sh_scripts/task_yaml/{taskname}.yaml')


# taskname_prefix = "render_dora_relight_slat_6k"

# world_size = 8
# max_workers = 12

# # for rank in range(0, 1):
# # for rank in range(0, 8):
# # for rank in range(0, 3):
# for rank in range(1, world_size):
# # for rank in [11,]:
#     taskname = f'{taskname_prefix}_{rank}_{world_size}_{max_workers}'
#     command = f'''cd /baai-cwm-vepfs/cwm/hong.li/code/3dgen/data/zeroverse &&
#     source activate /baai-cwm-vepfs/cwm/hong.li/miniconda3/envs/blender &&
#     bash sh_scripts/render_3diclight_dora_relight_slat.sh {rank} {world_size} {max_workers}'''

#     with open(f'sh_scripts/task_yaml/{taskname}.yaml', 'w') as f:
#         f.write(template.format(task_name=taskname, command=command, queue_id=queue_id['L20render'], flavor=flavor['L20render'], num_gpus=num_gpus))
#         # f.write(template.format(task_name=taskname, command=command, queue_id=queue_id['4090'], flavor=flavor['4090'], num_gpus=num_gpus))
#     print(f'volc ml_task submit --conf ./sh_scripts/task_yaml/{taskname}.yaml')
#     # time.sleep(3)
#     os.system(f'volc ml_task submit --conf ./sh_scripts/task_yaml/{taskname}.yaml')
#     # remove config file
#     os.system(f'rm ./sh_scripts/task_yaml/{taskname}.yaml')


taskname_prefix = "dino_29w4_pbr_even_slat_extract_0722"

world_size = 12

# for rank in range(0, 1):
# for rank in range(0, 8):
# for rank in range(0, 3):
for rank in range(0, world_size):
# for rank in [11,]:
    taskname = f'{taskname_prefix}_{rank}_{world_size}'
    command = f'''cd /baai-cwm-vepfs/cwm/hong.li/code/3dgen/data/zeroverse &&
    source activate /baai-cwm-vepfs/cwm/hong.li/miniconda3/envs/trellis &&
    bash sh_scripts/run_dino_6k_slats_29w4_even_slat.sh {rank} {world_size}'''

    with open(f'sh_scripts/task_yaml/{taskname}.yaml', 'w') as f:
        # f.write(template.format(task_name=taskname, command=command, queue_id=queue_id['L20render'], flavor=flavor['L20render'], num_gpus=num_gpus))
        f.write(template.format(task_name=taskname, command=command, queue_id=queue_id['4090'], flavor=flavor['4090'], num_gpus=num_gpus))
    print(f'volc ml_task submit --conf ./sh_scripts/task_yaml/{taskname}.yaml')
    # time.sleep(3)
    os.system(f'volc ml_task submit --conf ./sh_scripts/task_yaml/{taskname}.yaml')
    # remove config file
    os.system(f'rm ./sh_scripts/task_yaml/{taskname}.yaml')

# taskname_prefix = "render_dora_random_for_singleview_gt"

# world_size = 8
# max_workers = 12

# # for rank in range(0, 1):
# # for rank in range(0, 8):
# # for rank in range(0, 3):
# for rank in range(1, world_size):
# # for rank in [11,]:
#     taskname = f'{taskname_prefix}_{rank}_{world_size}_{max_workers}'
#     command = f'''cd /baai-cwm-vepfs/cwm/hong.li/code/3dgen/data/zeroverse &&
#     source activate /baai-cwm-vepfs/cwm/hong.li/miniconda3/envs/blender &&
#     bash sh_scripts/render_3diclight_dora_random_singel_view.sh {rank} {world_size} {max_workers}'''

#     with open(f'sh_scripts/task_yaml/{taskname}.yaml', 'w') as f:
#         f.write(template.format(task_name=taskname, command=command, queue_id=queue_id['L20render'], flavor=flavor['L20render'], num_gpus=num_gpus))
#         # f.write(template.format(task_name=taskname, command=command, queue_id=queue_id['4090'], flavor=flavor['4090'], num_gpus=num_gpus))
#     print(f'volc ml_task submit --conf ./sh_scripts/task_yaml/{taskname}.yaml')
#     # time.sleep(3)
#     os.system(f'volc ml_task submit --conf ./sh_scripts/task_yaml/{taskname}.yaml')
#     # remove config file
#     os.system(f'rm ./sh_scripts/task_yaml/{taskname}.yaml')