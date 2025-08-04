import os
import argparse, sys, os, math, re, glob
from typing import *
import bpy
from mathutils import Vector, Matrix, Euler, Quaternion
import numpy as np
import json
import random

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import *

LIGHTS_GROUPS = {
    0: ["point"],
    1: ["area"],
    2: ["hdri"],
    3: ["point", "area"],
    4: ["point", "hdri"],
    5: ["area", "hdri"],
    6: ["point", "background"],
    7: ["area", "background"],
    8: ["point", "area", "hdri"],
    9: ["point", "area", "background"],
}

def random_light_yaw_pitch(radius_min=2.0, radius_max=2.5, max_angle_deg=50):  
    # yaw: 0 ~ 360°  
    yaw = random.uniform(0, 2*math.pi)  
    
    # pitch: 0 ~ max_angle_deg（转为弧度）  
    max_angle_rad = math.radians(max_angle_deg)  
    
    # 为达到均匀分布，采样 cos(pitch) 在 [cos(50°), 1]之间均匀  
    cos_pitch_min = math.cos(max_angle_rad)  
    
    cos_pitch = random.uniform(cos_pitch_min, 1.0)  
    pitch = math.acos(cos_pitch)  # 这样 pitch 会均匀分布在【0°, 50°】  
    
    # radius 随机  
    radius = random.uniform(radius_min, radius_max)  
    
    # 返回弧度单位的yaw和pitch，以及radius  
    return yaw, pitch, radius  

def spherical_to_cartesian_y_axis_neg(yaw, pitch, radius):  
    """  
    以 -y 轴为极轴的极坐标转笛卡尔坐标  
    """  
    # 极轴是 -y 轴，所以设:  
    # pitch 是和 -y 轴的夹角  
    # yaw 是绕 -y 轴的旋转角  
    #  
    # 先计算相对于 -y 轴的方向向量  
    #  
    # 设极轴(极点)方向 n = (0,-1,0)  
    # 将球坐标转换为相对于 y 轴:  
    # x = r sin(pitch) cos(yaw)  
    # y = -r cos(pitch)     # 负y轴方向  
    # z = r sin(pitch) sin(yaw)  
    x = radius * math.sin(pitch) * math.cos(yaw)  
    y = -radius * math.cos(pitch)  
    z = radius * math.sin(pitch) * math.sin(yaw)  
    return x, y, z  

def add_point_light(num=1):
    # 创建光源  
    points_light_info = {'energy': [], 'location': []}
    for _ in range(num):  # 添加循环以创建多个光源
        bpy.ops.object.light_add(type='POINT', location=(0, 0, 0))  
        light = bpy.context.active_object  
        light.data.energy = random.uniform(64, 512 // (num + 2))  # 设置光源强度 
        light.data.color = (1, 1, 1)  # 设置光源颜色为白色  
        light.data.use_shadow = True
        light.data.use_shadow_jitter = True

        # 随机化光源位置  
        yaw, pitch, radius = random_light_yaw_pitch(2.0, 2.5, 50)  
        x, y, z = spherical_to_cartesian_y_axis_neg(yaw, pitch, radius)  
        light.location = (x, y, z)  
        
        points_light_info['energy'].append(light.data.energy)
        points_light_info['location'].append((x, y, z))
    
    return points_light_info

def add_area_light(num=1):
    # 创建光源  
    area_light_info = {'energy': [], 'location': [], 'scale': []}
    for _ in range(num):  # 添加循环以创建多个光源
        bpy.ops.object.light_add(type='AREA', location=(0, 0, 0))  
        light = bpy.context.active_object  
        light.data.energy = random.uniform(64, 144)  # 设置光源强度
        light.data.color = (1, 1, 1)  # 设置光源颜色为白色  
        scale = random.uniform(1., 3.)  # 随机化光源大小
        light.scale = (scale, scale, scale)  # 设置光源大小
        light.data.use_shadow = True
        light.data.use_shadow_jitter = True

        # 约束朝向 0 0 0
        constraint = light.constraints.new(type='TRACK_TO')
        if 'Empty' not in bpy.data.objects:
            empty = bpy.data.objects.new("Empty", None)
            empty.location = (0, 0, 0)
            bpy.context.collection.objects.link(empty)
        else:
            empty = bpy.data.objects['Empty']
        constraint.target = empty
        constraint.track_axis = 'TRACK_NEGATIVE_Z'
        constraint.up_axis = 'UP_Y'
        
        # 随机化光源位置  
        yaw, pitch, radius = random_light_yaw_pitch(4.0, 5.0, 50)
        x, y, z = spherical_to_cartesian_y_axis_neg(yaw, pitch, radius)  
        light.location = (x, y, z)

        area_light_info['energy'].append(light.data.energy)
        area_light_info['location'].append((x, y, z))
        area_light_info['scale'].append(scale)

    return area_light_info

def init_unips_lightings():
    del_lighting()

    # 创建一个空对象，作为光源的目标
    target = bpy.data.objects.new("Target", None)
    bpy.context.collection.objects.link(target)
    target.location = (0, 0, 0)

    # Add a point light
    point_light = bpy.data.objects.new("Point_Light", bpy.data.lights.new("Point_Light", type="POINT"))
    bpy.context.collection.objects.link(point_light)
    point_light.data.energy = 1000
    point_light.location = (4, 1, 6)
    point_light.rotation_euler = (0, 0, 0)
    point_light.data.use_shadow = True
    point_light.data.use_shadow_jitter = True
    # 点光源不需要约束

    # Add an area light
    area_light = bpy.data.objects.new("Area_Light", bpy.data.lights.new("Area_Light", type="AREA"))
    bpy.context.collection.objects.link(area_light)
    area_light.data.energy = 10000
    area_light.location = (0, 0, 10)
    area_light.scale = (100, 100, 100)
    area_light.data.use_shadow = True
    area_light.data.use_shadow_jitter = True
    # 添加 Track To 约束  使得光源指向世界中心
    area_light_constraint = area_light.constraints.new(type='TRACK_TO')
    area_light_constraint.target = None
    # 设置约束目标
    area_light_constraint.target = target
    # 设置约束轴
    area_light_constraint.track_axis = 'TRACK_NEGATIVE_Z'
    area_light_constraint.up_axis = 'UP_Y'


    # Add a background light
    # set_even_background_lighting_with_strength(strength=0.2) # 均匀光照 0.6

    return {
        "point_light": point_light,
        "area_light": area_light,
    }


def set_area_lighting(area_light: Tuple[bpy.types.Object, bpy.types.Light], energy: float, location: Tuple[float, float, float], scale: Tuple[float, float, float], color: Tuple[float, float, float] = (1, 1, 1)):
    area_light.data.energy = energy
    area_light.location = location
    area_light.scale = scale
    area_light.data.color = color

def set_point_lighting(point_light: Tuple[bpy.types.Object, bpy.types.Light], energy: float, location: Tuple[float, float, float], color: Tuple[float, float, float] = (1, 1, 1)):
    point_light.data.energy = energy
    point_light.location = location
    point_light.data.color = color


def render_current_scene(file_postfix, i, j, args, save_light=True, outputs=None):
    """渲染当前场景并保存指定AOV输出
    
    Args:
        file_postfix: 文件名后缀标识
        i: view索引号
        j: 光照索引号
        args: 命令行参数对象
        save_light: 是否保存光照相关通道
        outputs: AOV输出配置字典
    """
    # bpy.context.scene.render.filepath = os.path.join(arg.output_folder, f'{i:03d}.png') if not arg.save_low_normal else os.path.join(arg.output_folder, 'low_normal_image', f'{i:03d}_low_normal.png')
    bpy.context.scene.render.filepath = os.path.join(args.output_folder, f'{i:03d}_{j:03d}_{file_postfix}.png') if not args.save_low_normal else os.path.join(args.output_folder, 'low_normal_image', f'{i:03d}_{j:03d}_low_normal_{file_postfix}.png')

    for name, output in outputs.items():
        os.makedirs(os.path.join(args.output_folder, f'{name.split(".")[0]}'), exist_ok=True)
        output.mute = False
        if name in AOV_NAMES_WITHOUT:
            if save_light and j == 0:
                output.file_slots[0].path = os.path.join(args.output_folder, f'{name.split(".")[0]}', f'{i:03d}_{j:03d}_{name}')
            else:
                # #import pdb; pdb.set_trace()
                output.mute = True
        elif name in AOV_NAMES_WITH: # 名字跟随光照变化
                output.file_slots[0].path = os.path.join(args.output_folder, f'{name.split(".")[0]}', f'{i:03d}_{j:03d}_{name}_{file_postfix}')
            
    # Render the scene
    bpy.ops.render.render(write_still=True, animation=False)
    bpy.context.view_layer.update()
    #import pdb; pdb.set_trace()

    for name, output in outputs.items():
        ext = EXT[output.format.file_format]
        path = f'{output.file_slots[0].path}0001.{ext}'
        if os.path.exists(path):
            os.rename(path, f'{output.file_slots[0].path}.{ext}') 


def main(args):
    os.makedirs(args.output_folder, exist_ok=True)
    
    if args.object.endswith(".blend"):
        delete_invisible_objects()
    else:
        init_scene()
        load_object(args.object, set_origin=True)
        bpy.context.object.rotation_euler[0] = 1.5708

        if args.split_normal:
            split_mesh_normal()
        # delete_custom_normals()
    # #import pdb; pdb.set_trace()
    
    delete_animation_data() # lihong add 20250126
    # delete_armature() # lihong add 20250126
    delete_gltf_not_imported() # lihong add 20250126

    # remove_unwanted_objects() # lihong add 20250304
    print('[INFO] Scene initialized.')
    
    # normalize scene
    scale, offset = normalize_scene()

    print('[INFO] Scene normalized.')
    
    # Initialize camera and lighting
    cam = init_orthographic_camera() # Initialize orthographic camera
    # del_lighting() # delete all existing lights
    # set_even_background_lighting_with_strength(strength=0.6) # 均匀光照 0.6

    print('[INFO] Camera and lighting initialized.')

    # Initialize context
    init_render(engine=args.engine, resolution=args.resolution, geo_mode=args.geo_mode, film_transparent=args.film_transparent)
    # import pdb; pdb.set_trace()
    outputs, spec_nodes, aovs = init_nodes(
        save_image=args.save_image,
        save_low_image=args.save_low_image,
        save_alpha=args.save_alpha,
        save_depth=args.save_depth,
        save_normal=args.save_normal,
        save_low_normal=args.save_low_normal,
        save_albedo=args.save_albedo,
        save_glossycol=args.save_glossycol,
        save_mist=args.save_mist,
        save_pbr=args.save_pbr,
        save_env=args.save_env,
        save_pos=args.save_pos,
        save_ao=args.save_ao,
        save_glossydir=args.save_glossydir,
        save_diffdir=args.save_diffdir,
        save_shadow=args.save_shadow,
        save_diffind=args.save_diffind,
        save_glossyind=args.save_glossyind
    )

    # Override material
    if args.geo_mode:
        override_material()
    
    # Create a list of views
    to_export = {
        "aabb": [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        "scale": scale,
        "offset": [offset.x, offset.y, offset.z],
        "resolution": args.resolution,
        "frames": []
    }

    # random selet hdri envmap and rotation
    # rotation_euler = (0, 0, np.random.uniform(0, 2*np.pi))

    # random select hdri envmap and 10 rotations
    # rotation_eulers = [(np.random.uniform(0, 2*np.pi), np.random.uniform(0, 2*np.pi), np.random.uniform(0, 2*np.pi)) for _ in range(args.num_lights)]
    # rotation_eulers = [(0, 0, np.random.uniform(0, 2*np.pi)) for _ in range(args.num_lights)]

    # assert args.num_lights % 2 == 0
    # 生成半球均匀分布的点光源位置
    # point_lighting_rotations =  hemisphere_uniform_sequence(args.num_lights//2, top=True) + hemisphere_uniform_sequence(args.num_lights//2, top=False)
    # 生成半球均匀分布的面光源位置
    # area_lighting_rotations =  hemisphere_uniform_sequence(args.num_lights//2, top=True) + hemisphere_uniform_sequence(args.num_lights//2, top=False)

    for i in range(args.num_views):
        for j in range(args.num_lights):

            del_lighting()

            light_group = random.choice(LIGHTS_GROUPS)
            
            points_light_info = None
            area_light_info = None       

            if 'point' in light_group:
                points_light_info = add_point_light(num=random.randint(1, 3))

            if 'area' in light_group:
                area_light_info = add_area_light(num=1)

            if 'background' in light_group:
                backgroud_energy = np.random.uniform(0.1, 0.3)
                set_even_background_lighting_with_strength(strength=backgroud_energy)
            else:
                del_world_tree()
            
            if 'hdri' in light_group:
                rotation_euler = (0, 0, np.random.uniform(0, 2*np.pi))
                set_hdri(path_to_hdr_file=args.hdri_file_path, strength=1.0, rotation_euler=rotation_euler)
            else:
                del_world_tree()

            file_postfix = '_'.join(light_group)
            # save_light = False if args.save_low_normal else True
            # if args.save_low_normal:
                # import pdb; pdb.set_trace()
            
            render_current_scene(file_postfix=file_postfix, i=i, j=j, args=args, save_light=True, outputs=outputs)

            # Save camera parameters
            metadata = {
                "file_path": f'{os.path.join("image", f"{i:03d}_{j:03d}_{file_postfix}.png")}' if not args.save_low_normal else f'{os.path.join("low_normal_image", f"{i:03d}_{j:03d}_low_normal_{file_postfix}.png")}',
                "camera_type": cam.data.type,
                "camera_ortho_scale": cam.data.ortho_scale,
                "engine": args.engine,
                "transform_matrix": get_transform_matrix(cam),
                "hdri_file_path": args.hdri_file_path,
                "rotation_euler": rotation_euler if 'hdri' in light_group else None,
                "background_energy": backgroud_energy if 'background' in light_group else None,
                "points_light_info": points_light_info,
                "area_light_info": area_light_info
            }

            to_export["frames"].append(metadata)

    # Save the camera parameters
    if args.save_low_normal:
        # import pdb; pdb.set_trace()
        with open(os.path.join(args.output_folder, 'transforms_low.json'), 'w') as f:
            json.dump(to_export, f, indent=4)
    else:
        with open(os.path.join(args.output_folder, 'transforms.json'), 'w') as f:
            json.dump(to_export, f, indent=4)
        
    if args.save_mesh:
        # triangulate meshes
        unhide_all_objects()
        convert_to_meshes()
        triangulate_meshes()
        print('[INFO] Meshes triangulated.')
        
        # export ply mesh
        bpy.ops.wm.ply_export(filepath=os.path.join(args.output_folder, 'mesh.ply'))

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Renders given obj file by rotation a camera around it.')
    parser.add_argument('--num_views', type=int, help='Number of views to be used.')
    parser.add_argument('--num_lights', type=int, default=10, help='Number of lights to be used.')
    parser.add_argument('--object', type=str, help='Path to the 3D model file to be rendered.')
    parser.add_argument('--output_folder', type=str, default='/tmp', help='The path the output will be dumped to.')
    parser.add_argument('--resolution', type=int, default=512, help='Resolution of the images.')
    parser.add_argument('--engine', type=str, default='CYCLES', help='Blender internal engine for rendering. E.g. CYCLES, BLENDER_EEVEE, ...')
    parser.add_argument('--film_transparent', action='store_true', help='Film transparent mode for rendering.')

    parser.add_argument('--geo_mode', action='store_true', help='Geometry mode for rendering.')
    parser.add_argument('--save_alpha', action='store_true', help='Save the alpha maps.')
    parser.add_argument('--save_depth', action='store_true', help='Save the depth maps.')
    parser.add_argument('--save_normal', action='store_true', help='Save the normal maps.')
    parser.add_argument('--save_low_normal', action='store_true', help='Save the low normal maps.')
    parser.add_argument('--save_low_image', action='store_true', help='Save the low image.')

    parser.add_argument('--split_normal', action='store_true', help='Split the normals of the mesh.')

    parser.add_argument('--save_image', action='store_true', help='Save the image.')
    parser.add_argument('--save_albedo', action='store_true', help='Save the albedo maps.')
    parser.add_argument('--save_glossycol', action='store_true', help='Save the glossycol maps.')
    parser.add_argument('--save_mist', action='store_true', help='Save the mist distance maps.')
    parser.add_argument('--save_pbr', action='store_true', help='Save the pbr maps.')
    parser.add_argument('--save_env', action='store_true', help='Save the env maps.')
    parser.add_argument('--save_pos', action='store_true', help='Save the pos maps.')
    parser.add_argument('--save_ao', action='store_true', help='Save the ao maps.')

    parser.add_argument('--save_diffdir', action='store_true', help='Save the diffdir maps.')
    parser.add_argument('--save_glossydir', action='store_true', help='Save the glossydir maps.')
    parser.add_argument('--save_shadow', action='store_true', help='Save the shadow maps.')

    parser.add_argument('--save_diffind', action='store_true', help='Save the diffind maps.')
    parser.add_argument('--save_glossyind', action='store_true', help='Save the glossyind maps.')

    parser.add_argument('--save_mesh', action='store_true', help='Save the mesh as a .ply file.')

    parser.add_argument('--hdri_file_path', type=str, default=None, help='The number of hdri envmaps.')

    argv = sys.argv[sys.argv.index("--") + 1:]
    args = parser.parse_args(argv)

    main(args)
    

# commond
'''
/baai-cwm-1/baai_cwm_ml/algorithm/hong.li/tools/blender-3.0.1-linux-x64/blender -b -P /baai-cwm-1/baai_cwm_ml/algorithm/hong.li/code/3dgen/TRELLIS/dataset_toolkits/blender_script/render_pbr.py --\
    --views 150 --object /baai-cwm-1/baai_cwm_ml/algorithm/hong.li/code/3dgen/TRELLIS/datasets/ObjaverseXL_sketchfab/raw/hf-objaverse-v1/glbs/000-034/1bb177d4e6f6470ba167ef5e4d8e2596.glb \
    --resolution 512 --output_folder /baai-cwm-1/baai_cwm_ml/algorithm/hong.li/code/3dgen/TRELLIS/datasets/tmp \
    --engine CYCLES --save_mesh --save_normal
'''
