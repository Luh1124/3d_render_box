

import bpy
import os
import csv
import random
import pandas as pd
import math
from mathutils import Vector, Euler
import uuid  
from tqdm import tqdm, trange


# ----- 工具函数 -----

def init_scene() -> None:
    """清空场景材质贴图和所有对象"""
    for obj in bpy.data.objects:
        bpy.data.objects.remove(obj, do_unlink=True)
    for m in bpy.data.materials:
        bpy.data.materials.remove(m, do_unlink=True)
    for tex in bpy.data.textures:
        bpy.data.textures.remove(tex, do_unlink=True)
    for img in bpy.data.images:
        bpy.data.images.remove(img, do_unlink=True)

def delete_gltf_not_imported():
    collection = bpy.data.collections.get("glTF_not_exported")
    if collection:
        bpy.data.collections.remove(collection, do_unlink=True)

def scene_bbox(objs):
    bbox_min = (math.inf,) * 3
    bbox_max = (-math.inf,) * 3
    found = False
    meshes = [obj for obj in objs if obj.type == 'MESH']
    for obj in meshes:
        found = True
        for v in obj.bound_box:
            coord = Vector(v)
            coord = obj.matrix_world @ coord
            bbox_min = tuple(min(x, y) for x, y in zip(bbox_min, coord))
            bbox_max = tuple(max(x, y) for x, y in zip(bbox_max, coord))
    if not found:
        raise RuntimeError("no objects to compute bounding box")
    return Vector(bbox_min), Vector(bbox_max)

def normalize_scene(objs):
    roots = [o for o in objs if not o.parent]
    if len(roots) > 1:
        parent_empty = bpy.data.objects.new("ParentEmpty", None)
        bpy.context.scene.collection.objects.link(parent_empty)
        for o in roots:
            o.parent = parent_empty
        scene = parent_empty
    else:
        scene = roots[0]

    bbox_min, bbox_max = scene_bbox(objs)
    scale = 1 / max(bbox_max - bbox_min)
    scene.scale = scene.scale * scale

    bpy.context.view_layer.update()

    bbox_min, bbox_max = scene_bbox(objs)
    offset = -(bbox_min + bbox_max) / 2
    scene.matrix_world.translation += offset

    bpy.ops.object.select_all(action="DESELECT")
    return scene, scale, offset

def random_transform_objects(objs,
                             scale_range=(0.8, 1.2),
                             rotation_range_deg=180,
                             translation_range=0.05):

    bpy.context.object.rotation_mode = 'XYZ'

    for obj in objs:
        scl = random.uniform(*scale_range)
        obj.scale *= scl

        rot_x = math.radians(random.uniform(-rotation_range_deg, rotation_range_deg))
        rot_y = math.radians(random.uniform(-rotation_range_deg, rotation_range_deg))
        rot_z = math.radians(random.uniform(-rotation_range_deg, rotation_range_deg))
        random_rot = Euler((rot_x, rot_y, rot_z), 'XYZ')
        current_rot = obj.rotation_euler
        new_rot_mat = current_rot.to_matrix() @ random_rot.to_matrix()
        obj.rotation_euler = new_rot_mat.to_euler('XYZ')
        # print(obj.rotation_euler)

        trans_x = random.uniform(-translation_range, translation_range)
        trans_y = random.uniform(-translation_range, translation_range)
        trans_z = random.uniform(-translation_range, translation_range)
        obj.location += Vector((trans_x, trans_y, trans_z))

def create_material_from_folder(folder_path, name):
    if not os.path.exists(folder_path):
        print(f"材质文件夹不存在：{folder_path}")
        return None

    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    for node in nodes:
        nodes.remove(node)

    output = nodes.new('ShaderNodeOutputMaterial')
    output.location = (400, 0)

    principled = nodes.new('ShaderNodeBsdfPrincipled')
    principled.location = (0, 0)
    links.new(principled.outputs['BSDF'], output.inputs['Surface'])

    def add_tex_node(texfile, label, color_space):
        path = os.path.join(folder_path, texfile)
        if not os.path.exists(path):
            print(f"缺失贴图文件: {path}")
            return None
        node = nodes.new('ShaderNodeTexImage')
        node.location = (-400, 0)
        node.label = label
        node.image = bpy.data.images.load(path)
        node.image.colorspace_settings.name = color_space
        return node

    basecolor_node = add_tex_node(f'{name}_2K_PNG_Color.png', 'BaseColor', 'sRGB')
    if basecolor_node:
        links.new(basecolor_node.outputs['Color'], principled.inputs['Base Color'])

    metallic_node = add_tex_node(f'{name}_2K_PNG_Metallness.png', 'Metallic', 'Non-Color')
    if metallic_node:
        links.new(metallic_node.outputs['Color'], principled.inputs['Metallic'])

    roughness_node = add_tex_node(f'{name}_2K_PNG_Roughness.png', 'Roughness', 'Non-Color')
    if roughness_node:
        links.new(roughness_node.outputs['Color'], principled.inputs['Roughness'])

    normal_map_node = nodes.new('ShaderNodeNormalMap')
    normal_map_node.location = (-200, -200)

    normal_tex_node = add_tex_node(f'{name}_2K_PNG_NormalIGL.png', 'Normal', 'Non-Color')
    if normal_tex_node:
        normal_tex_node.location = (-400, -200)
        links.new(normal_tex_node.outputs['Color'], normal_map_node.inputs['Color'])
        links.new(normal_map_node.outputs['Normal'], principled.inputs['Normal'])

    return mat

def replace_material_random(obj, materials_info):
    r = random.random()
    if r < 0.2:
        return  # 保留原材质

    if r < 0.6:
        pool = [m for m in materials_info if not m['is_metal']]
    else:
        pool = [m for m in materials_info if m['is_metal']]

    if not pool:
        print("未找到匹配材质，跳过替换")
        return

    selected = random.choice(pool)
    mat_name = selected['name']
    mat = bpy.data.materials.get(mat_name)
    if mat is None:
        mat = create_material_from_folder(os.path.join(TEXTURE_LIB_BASEPATH, mat_name), mat_name)
        if mat is None:
            return

    if obj.type == 'MESH':
        if len(obj.data.materials) == 0:
            obj.data.materials.append(mat)
        else:
            for i in range(len(obj.data.materials)):
                obj.data.materials[i] = mat

def parse_materials_csv(csv_path):
    materials = []
    with open(csv_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            materials.append({
                "name": row['name'],
                "is_metal": row['is_metal'].lower() == 'true',
            })
    return materials

def import_and_process_single_glb(filepath, materials_info):
    bpy.ops.import_scene.gltf(filepath=filepath)
    imported_objs = bpy.context.selected_objects

    delete_gltf_not_imported()
    scene, scale, offset = normalize_scene(imported_objs)
    # random_transform_objects(imported_objs)

    for obj in imported_objs:
        replace_material_random(obj, materials_info)

    return imported_objs

# ======= 主程序 ========

# 设置输出目录  
start = 2172
end = 10000
OUTPUT_DIR = f"/baai-cwm-1/baai_cwm_ml/cwm/hong.li/code/3dgen/data/zeroverse/output_all/no_animation/{start:05d}-{end:05d}"  # 请替换为实际输出路径  
os.makedirs(OUTPUT_DIR, exist_ok=True)  

# 创建日志文件记录生成情况  
log_file_path = os.path.join(OUTPUT_DIR, "generation_log.txt")  
log_file = open(log_file_path, "w")  

# 设置材质和 GLB 文件路径  
TEXTURE_LIB_BASEPATH = "/baai-cwm-1/baai_cwm_ml/cwm/hong.li/code/3dgen/data/BlenderProc/resources/matsynth_processed_v2"  
CSV_PATH = "/baai-cwm-1/baai_cwm_ml/cwm/hong.li/code/3dgen/data/zeroverse/process_tex/matsynth_metal_stats.csv"  
GLB_FILES_CSV = "/baai-cwm-1/baai_cwm_ml/cwm/hong.li/code/3dgen/data/utils3dgen/23_unips/glb_animation_check_results_3w9.csv"  

glbs_info = pd.read_csv(GLB_FILES_CSV)
glbs_info = glbs_info[glbs_info['has_animation']==False]

print(len(glbs_info))

materials_info = parse_materials_csv(CSV_PATH)  

# 生成 10000 个场景  
for i in trange(start,end):  
    try:  
        # 清空场景  
        init_scene()  

        # 随机选择 4-6 个 GLB 文件  
        num_files = random.randint(4, 6)  
        GLB_FILES = glbs_info['local_path'].sample(n=num_files).tolist()  

        # 导入并处理 GLB 文件  
        all_objs = []  
        for path in GLB_FILES:  
            objs = import_and_process_single_glb(path, materials_info)  
            all_objs.extend(objs)  

        # 随机变换物体  
        random_transform_objects(all_objs)  

        # 归一化场景  
        final_scene, final_scale, final_offset = normalize_scene(all_objs)  

        # 生成唯一文件名  
        scene_uuid = uuid.uuid4().hex  
        blend_save_path = os.path.join(OUTPUT_DIR, f"{scene_uuid}.blend")  
        # glb_save_path = os.path.join(OUTPUT_DIR, f"{scene_uuid}.glb")  

        # 保存 Blender 文件  
        bpy.ops.wm.save_as_mainfile(filepath=blend_save_path, compress=True)  

        # 导出 GLB 文件  
        # bpy.ops.export_scene.gltf(filepath=glb_save_path)  

        # 记录日志  
        log_file.write(f"Scene {i+1}: {scene_uuid}\n")  
        # log_file.write(f"  GLB Files: {GLB_FILES}\n")  
        log_file.write(f"  Blend Path: {blend_save_path}\n")  
        # log_file.write(f"  GLB Path: {glb_save_path}\n")  
        log_file.write(f"  Scale: {final_scale}, Offset: {final_offset}\n\n")  
        log_file.flush()  

        print(f"Generated scene {i+1}: {scene_uuid}")  

    except Exception as e:  
        # 记录错误日志  
        log_file.write(f"Error in scene {i+1}: {str(e)}\n")  
        log_file.flush()  
        print(f"Error in scene {i+1}: {str(e)}")  

# 关闭日志文件  
log_file.close()  

print("Scene generation complete!")  




