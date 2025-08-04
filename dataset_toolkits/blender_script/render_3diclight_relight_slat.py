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


def init_nodes_v2(save_alpha=False, save_low_image=False, save_image=False, save_depth=False, save_normal=False, save_low_normal=False, save_albedo=False,  save_glossycol=False, save_mist=False, save_pbr=False, save_env=False, save_pos=False, save_ao=False, save_glossydir=False, save_diffdir=False, save_shadow=False, save_diffind=False, save_glossyind=False):
    if not any([save_alpha, save_low_image, save_image, save_depth, save_normal, save_low_normal, save_albedo, save_glossycol, save_mist, save_pbr, save_env, save_pos, save_ao, save_glossydir, save_diffdir, save_shadow, save_diffind, save_glossyind]):
        return {}, {}, {}
    outputs = {}
    spec_nodes = {}
    aovs = {}

    bpy.context.scene.render.use_compositing = True
    bpy.context.scene.use_nodes = True
    
    bpy.context.scene.view_settings.view_transform = 'AgX'

    nodes = bpy.context.scene.node_tree.nodes
    links = bpy.context.scene.node_tree.links
    for n in nodes:
        nodes.remove(n)
    
    render_layers = nodes.new('CompositorNodeRLayers')
    
    if save_alpha:
        alpha_file_output = nodes.new('CompositorNodeOutputFile')
        alpha_file_output.base_path = ''
        alpha_file_output.file_slots[0].use_node_format = True
        alpha_file_output.format.file_format = 'PNG'
        alpha_file_output.format.color_mode = 'BW'
        alpha_file_output.format.color_depth = '8'

        links.new(render_layers.outputs['Alpha'], alpha_file_output.inputs[0])

        outputs['alpha'] = alpha_file_output

    if save_low_image:
        image_file_output = nodes.new('CompositorNodeOutputFile')
        image_file_output.base_path = ''
        image_file_output.file_slots[0].use_node_format = True
        image_file_output.format.file_format = 'OPEN_EXR'
        image_file_output.format.color_mode = 'RGBA'
        image_file_output.format.color_depth = '16'

        alpha_image = nodes.new('CompositorNodeSetAlpha')
        links.new(render_layers.outputs['Image'], alpha_image.inputs['Image'])
        links.new(render_layers.outputs['Alpha'], alpha_image.inputs['Alpha'])
        links.new(alpha_image.outputs['Image'], image_file_output.inputs['Image'])

        outputs['low_image'] = image_file_output
    
    if save_image:
        image_file_output = nodes.new('CompositorNodeOutputFile')
        image_file_output.base_path = ''
        image_file_output.file_slots[0].use_node_format = True
        image_file_output.format.file_format = 'OPEN_EXR'
        image_file_output.format.color_mode = 'RGBA'
        image_file_output.format.color_depth = '16'

        links.new(render_layers.outputs['Image'], image_file_output.inputs[0])

        outputs['image'] = image_file_output

    if save_depth:
        bpy.context.view_layer.use_pass_z = save_depth
        depth_file_output = nodes.new('CompositorNodeOutputFile')
        depth_file_output.base_path = ''
        depth_file_output.file_slots[0].use_node_format = True
        depth_file_output.format.file_format = 'OPEN_EXR'
        depth_file_output.format.color_depth = '16'
        depth_file_output.format.color_mode = 'RGBA'

        alpha_depth = nodes.new('CompositorNodeSetAlpha')
        links.new(render_layers.outputs['Depth'], alpha_depth.inputs['Image'])
        links.new(render_layers.outputs['Alpha'], alpha_depth.inputs['Alpha'])
        links.new(alpha_depth.outputs['Image'], depth_file_output.inputs['Image'])
        # links.new(render_layers.outputs["Depth"], depth_file_output.inputs["Image"])

        outputs['depth'] = depth_file_output
    
    if save_normal:
        bpy.context.view_layer.use_pass_normal = save_normal
        normal_file_output = nodes.new('CompositorNodeOutputFile')
        normal_file_output.base_path = ''
        normal_file_output.file_slots[0].use_node_format = True
        normal_file_output.format.file_format = 'OPEN_EXR'
        normal_file_output.format.color_mode = 'RGBA'
        normal_file_output.format.color_depth = '16'

        alpha_normal = nodes.new('CompositorNodeSetAlpha')
        links.new(render_layers.outputs['Normal'], alpha_normal.inputs['Image'])
        links.new(render_layers.outputs['Alpha'], alpha_normal.inputs['Alpha'])
        links.new(alpha_normal.outputs['Image'], normal_file_output.inputs['Image'])
        # links.new(render_layers.outputs['Normal'], normal_file_output.inputs[0])

        outputs['normal'] = normal_file_output
    
    if save_low_normal:
        bpy.context.view_layer.use_pass_normal = save_low_normal
        # link remove the normal node
        unable_normals_texture_output()

        ### ----------------- Normal down ----------------- ###
        low_normal_file_output = nodes.new('CompositorNodeOutputFile')
        low_normal_file_output.base_path = ''
        low_normal_file_output.file_slots[0].use_node_format = True
        low_normal_file_output.format.file_format = 'OPEN_EXR'
        low_normal_file_output.format.color_mode = 'RGBA'
        low_normal_file_output.format.color_depth = '16'

        alpha_low_normal = nodes.new('CompositorNodeSetAlpha')
        links.new(render_layers.outputs['Normal'], alpha_low_normal.inputs['Image'])
        links.new(render_layers.outputs['Alpha'], alpha_low_normal.inputs['Alpha'])
        links.new(alpha_low_normal.outputs['Image'], low_normal_file_output.inputs['Image'])
        # links.new(render_layers.outputs['Normal'], low_normal_file_output.inputs[0])

        outputs['low_normal'] = low_normal_file_output
            
    if save_albedo:
        bpy.context.view_layer.use_pass_diffuse_color = save_albedo
        albedo_file_output = nodes.new('CompositorNodeOutputFile')
        albedo_file_output.base_path = ''
        albedo_file_output.file_slots[0].use_node_format = True
        albedo_file_output.format.file_format = 'OPEN_EXR'
        albedo_file_output.format.color_mode = 'RGB'
        albedo_file_output.format.color_depth = '16'
        
        links.new(render_layers.outputs['DiffCol'], albedo_file_output.inputs[0])
        
        outputs['albedo'] = albedo_file_output
    
    if save_glossycol:
        bpy.context.view_layer.use_pass_glossy_color = save_glossycol
        glossycolor_file_output = nodes.new('CompositorNodeOutputFile')
        glossycolor_file_output.base_path = ''
        glossycolor_file_output.file_slots[0].use_node_format = True
        glossycolor_file_output.format.file_format = 'OPEN_EXR'
        glossycolor_file_output.format.color_mode = 'RGB'
        glossycolor_file_output.format.color_depth = '16'

        links.new(render_layers.outputs['GlossCol'], glossycolor_file_output.inputs[0])

        outputs['glossycol'] = glossycolor_file_output
        
    if save_mist:
        bpy.context.view_layer.use_pass_mist = save_mist
        bpy.data.worlds['World'].mist_settings.start = 0
        bpy.data.worlds['World'].mist_settings.depth = 10
        
        mist_file_output = nodes.new('CompositorNodeOutputFile')
        mist_file_output.base_path = ''
        mist_file_output.file_slots[0].use_node_format = True
        mist_file_output.format.file_format = 'PNG'
        mist_file_output.format.color_mode = 'BW'
        mist_file_output.format.color_depth = '16'
        
        links.new(render_layers.outputs['Mist'], mist_file_output.inputs[0])

        
        outputs['mist'] = mist_file_output
    
    if save_pbr: # need AOV output
        aov_base_color = enable_pbr_output(rl=render_layers, attr_name="Base Color", color_mode="RGBA", file_format="PNG", color_depth="16", outputs=outputs)
        aov_roughness = enable_pbr_output(rl=render_layers, attr_name="Roughness", color_mode="BW", file_format="OPEN_EXR", color_depth="16",outputs=outputs)
        aov_metallic = enable_pbr_output(rl=render_layers, attr_name="Metallic", color_mode="BW", file_format="OPEN_EXR", color_depth="16",outputs=outputs)

        aovs['base_color'] = aov_base_color
        aovs['roughness'] = aov_roughness
        aovs['metallic'] = aov_metallic
    
    if save_env:
        bpy.context.view_layer.use_pass_environment = save_env
        env_file_output = nodes.new('CompositorNodeOutputFile')
        env_file_output.base_path = ''
        env_file_output.file_slots[0].use_node_format = True
        env_file_output.format.file_format = 'PNG'
        env_file_output.format.color_mode = 'RGB'
        env_file_output.format.color_depth = '8'

        links.new(render_layers.outputs['Env'], env_file_output.inputs[0])

        outputs['env'] = env_file_output

    if save_pos:
        bpy.context.view_layer.use_pass_position = save_pos
        pos_file_output = nodes.new('CompositorNodeOutputFile')
        pos_file_output.base_path = ''
        pos_file_output.file_slots[0].use_node_format = True
        pos_file_output.format.file_format = 'OPEN_EXR'
        pos_file_output.format.color_mode = 'RGBA'
        pos_file_output.format.color_depth = '16'

        links.new(render_layers.outputs['Position'], pos_file_output.inputs[0])

        outputs['position'] = pos_file_output
    
    if save_ao:
        bpy.context.view_layer.use_pass_ambient_occlusion = save_ao
        ao_file_output = nodes.new('CompositorNodeOutputFile')
        ao_file_output.base_path = ''
        ao_file_output.file_slots[0].use_node_format = True
        ao_file_output.format.file_format = 'OPEN_EXR'
        ao_file_output.format.color_mode = 'BW'
        ao_file_output.format.color_depth = '16'

        links.new(render_layers.outputs['AO'], ao_file_output.inputs[0])

        outputs['ao'] = ao_file_output

    if save_diffdir:
        bpy.context.view_layer.use_pass_diffuse_direct = save_diffdir
        diffdir_file_output = nodes.new('CompositorNodeOutputFile')
        diffdir_file_output.base_path = ''
        diffdir_file_output.file_slots[0].use_node_format = True
        diffdir_file_output.format.file_format = 'OPEN_EXR'
        diffdir_file_output.format.color_mode = 'RGB'
        diffdir_file_output.format.color_depth = '16'

        links.new(render_layers.outputs['DiffDir'], diffdir_file_output.inputs[0])

        outputs['diffdir'] = diffdir_file_output
    
    if save_glossydir:
        bpy.context.view_layer.use_pass_glossy_direct = save_glossydir
        glossydir_file_output = nodes.new('CompositorNodeOutputFile')
        glossydir_file_output.base_path = ''
        glossydir_file_output.file_slots[0].use_node_format = True
        glossydir_file_output.format.file_format = 'OPEN_EXR'
        glossydir_file_output.format.color_mode = 'RGB'
        glossydir_file_output.format.color_depth = '16'

        links.new(render_layers.outputs['GlossDir'], glossydir_file_output.inputs[0])

        outputs['glossydir'] = glossydir_file_output
    
    if save_shadow:
        bpy.context.view_layer.use_pass_shadow = save_shadow
        shadow_file_output = nodes.new('CompositorNodeOutputFile')
        shadow_file_output.base_path = ''
        shadow_file_output.file_slots[0].use_node_format = True
        shadow_file_output.format.file_format = 'OPEN_EXR'
        shadow_file_output.format.color_mode = 'RGB'
        shadow_file_output.format.color_depth = '16'

        links.new(render_layers.outputs['Shadow'], shadow_file_output.inputs[0])

        outputs['shadow'] = shadow_file_output

    if save_diffind:
        bpy.context.view_layer.use_pass_diffuse_indirect = save_diffind
        diffind_file_output = nodes.new('CompositorNodeOutputFile')
        diffind_file_output.base_path = ''
        diffind_file_output.file_slots[0].use_node_format = True
        diffind_file_output.format.file_format = 'OPEN_EXR'
        diffind_file_output.format.color_mode = 'RGB'
        diffind_file_output.format.color_depth = '16'

        links.new(render_layers.outputs['DiffInd'], diffind_file_output.inputs[0])

        outputs['diffind'] = diffind_file_output
    
    if save_glossyind:
        bpy.context.view_layer.use_pass_glossy_indirect = save_glossyind
        glossyind_file_output = nodes.new('CompositorNodeOutputFile')
        glossyind_file_output.base_path = ''
        glossyind_file_output.file_slots[0].use_node_format = True
        glossyind_file_output.format.file_format = 'OPEN_EXR'
        glossyind_file_output.format.color_mode = 'RGB'
        glossyind_file_output.format.color_depth = '16'

        links.new(render_layers.outputs['GlossInd'], glossyind_file_output.inputs[0])

        outputs['glossyind'] = glossyind_file_output
    
    return outputs, spec_nodes, aovs

def add_light_env(env=(1, 1, 1, 1), strength=1, rot_vec_rad=(0, 0, 0), scale=(1, 1, 1)):
    r"""Adds environment lighting.
    Args:
        env (tuple(float) or str, optional): Environment map. If tuple,
            it's RGB or RGBA, each element of which :math:`\in [0,1]`.
            Otherwise, it's the path to an image.
        strength (float, optional): Light intensity.
        rot_vec_rad (tuple(float), optional): Rotations in radians around x, y and z.
        scale (tuple(float), optional): If all changed simultaneously, then no effects.
    """

    engine = bpy.context.scene.render.engine
    # assert engine == "CYCLES", "Rendering engine is not Cycles"

    if isinstance(env, str):
        bpy.data.images.load(env, check_existing=True)
        env = bpy.data.images[os.path.basename(env)]
    else:
        if len(env) == 3:
            env += (1,)
        assert len(env) == 4, "If tuple, env must be of length 3 or 4"

    world = bpy.context.scene.world
    world.use_nodes = True
    node_tree = world.node_tree
    nodes = node_tree.nodes
    links = node_tree.links

    bg_node = nodes.new("ShaderNodeBackground")
    links.new(bg_node.outputs["Background"], nodes["World Output"].inputs["Surface"])

    if isinstance(env, tuple):
        # Color
        bg_node.inputs["Color"].default_value = env
        print(("Environment is pure color, " "so rotation and scale have no effect"))
    else:
        # Environment map
        texcoord_node = nodes.new("ShaderNodeTexCoord")
        env_node = nodes.new("ShaderNodeTexEnvironment")
        env_node.image = env
        mapping_node = nodes.new("ShaderNodeMapping")
        mapping_node.inputs["Rotation"].default_value = rot_vec_rad
        mapping_node.inputs["Scale"].default_value = scale
        links.new(texcoord_node.outputs["Generated"], mapping_node.inputs["Vector"])
        links.new(mapping_node.outputs["Vector"], env_node.inputs["Vector"])
        links.new(env_node.outputs["Color"], bg_node.inputs["Color"])

    bg_node.inputs["Strength"].default_value = strength

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
        base_path = "/baai-cwm-vepfs/cwm/hong.li/code/3dgen/data/zeroverse/output_all/"  
        for image in bpy.data.images:
            original_path = image.filepath  
            
            if original_path.startswith("//"):  
                # import pdb; pdb.set_trace()  # 调试用，查看路径转换  

                # 去掉开头的 '//'  
                rel_path = original_path[2:]  
                parts = rel_path.split("/")  
                
                try:  
                    idx = parts.index("output_all")  
                    # 保留 output_all 及后续路径部分  
                    relative_subpath = os.path.join(*parts[idx+1:])  # 跳过 'output_all'，仅拼后面的子目录和文件名  
                    # 构造新绝对路径  
                    new_path = os.path.join(base_path, relative_subpath)  
                    # 转为 Blender 需要的绝对路径格式  
                    image.filepath = new_path  
                    print(f"Updated: {original_path} => {new_path}")  
                except ValueError:  
                    # 路径中没有 output_all，跳过  
                    print(f"Skipped (no output_all): {original_path}")
        image.reload()  
    else:
        init_scene()
        load_object(args.object)
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
    # cam = init_orthographic_camera() # Initialize orthographic camera
    # del_lighting() # delete all existing lights
    cam = init_camera() # Initialize camera, add camera to scene
    del_lighting() # delete all existed lights

    print('[INFO] Camera and lighting initialized.')

    # Initialize context
    init_render(engine=args.engine, resolution=args.resolution, geo_mode=args.geo_mode, film_transparent=args.film_transparent, color_depth='8')
    # import pdb; pdb.set_trace()
    outputs, spec_nodes, aovs = init_nodes_v2(
        save_alpha=args.save_alpha,
        save_image=args.save_image,
        save_low_image=args.save_low_image,
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
    

    hdri_file_path = args.hdri_file_path
    # rotation_euler = (0, 0, np.random.uniform(0, 2*np.pi))
    rotation_euler = (0, 0, 0)
    hdri_node = set_hdri(path_to_hdr_file=hdri_file_path, strength=1.0, rotation_euler=rotation_euler)

        # Create a list of views
    to_export = {
        "aabb": [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        "scale": scale,
        "offset": [offset.x, offset.y, offset.z],
        "resolution": args.resolution,
        "frames": [],
        "hdri": {
            "path": hdri_file_path,
            "rotation_euler": (0,0,0)
        }
    }

    views = json.loads(args.views)

    for i, view in enumerate(views):
        cam.location = (
            view['radius'] * np.cos(view['yaw']) * np.cos(view['pitch']),
            view['radius'] * np.sin(view['yaw']) * np.cos(view['pitch']),
            view['radius'] * np.sin(view['pitch'])
        )
        cam.data.lens = 16 / np.tan(view['fov'] / 2)

        bpy.context.scene.render.filepath = os.path.join(args.output_folder, f'{i:03d}.png') if not args.save_low_normal else os.path.join(args.output_folder, 'low_normal_image', f'{i:03d}_low_normal.png')
        for name, output in outputs.items():
            os.makedirs(os.path.join(args.output_folder, f'{name.split(".")[0]}'), exist_ok=True)
            output.file_slots[0].path = os.path.join(args.output_folder, f'{name.split(".")[0]}', f'{i:03d}_{name}')

        bpy.ops.render.render(write_still=True, animation=False)
        bpy.context.view_layer.update()
        for name, output in outputs.items():
            ext = EXT[output.format.file_format]
            path = glob.glob(f'{output.file_slots[0].path}0001.{ext}')[0]
            os.rename(path, f'{output.file_slots[0].path}.{ext}')
        metadata = {
            "file_path": f'{os.path.join("image", f"{i:03d}.png")}',
            "camera_angle_x": view['fov'],
            "transform_matrix": get_transform_matrix(cam)
        }

        to_export["frames"].append(metadata)

    if args.save_low_normal:
        return
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
    parser.add_argument('--views', type=str, help='JSON string of views. Contains a list of {yaw, pitch, radius, fov} object.')
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
    parser.add_argument('--hdri_file_path', type=str, default=None, help='The path to the HDRI file .')

    argv = sys.argv[sys.argv.index("--") + 1:]
    args = parser.parse_args(argv)

    main(args)
    
