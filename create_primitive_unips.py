import bpy  
import os  
import random  
from math import radians  
from datetime import datetime  
import glob  

# =================配置参数=================  
TEXTURE_DIR = "/baai-cwm-1/baai_cwm_ml/algorithm/hong.li/code/3dgen/data/BlenderProc/resources/matsynth_processed_v2"  # 法线贴图目录  
EXPORT_PATH = "export_unips/output.glb"  # 导出文件路径  

os.makedirs(os.path.dirname(EXPORT_PATH), exist_ok=True)  

NUM_OBJECTS = 1                          # 生成物体数量  
AREA_SIZE = 0                           # 物体分布范围  
NORMAL_STRENGTH_RANGE = (-2.0, 2.0)    # 法线强度随机范围  
# ==========================================  

# 材质预设库  
MATERIAL_PRESETS = {  
    "Cu": {  
        "base_color": (0.83, 0.48, 0.21, 1),  
        "metallic": 1.0,  
        "roughness": 0.3  
    },  
    "methylene": {  
        "base_color": (0.92, 0.92, 0.92, 1),  
        "metallic": 0.0,  
        "roughness": 0.8  
    },  
    "ACRYLIC": {  
        "base_color": (0.75, 0.85, 0.95, 1),  
        "metallic": 0.2,  
        "roughness": 0.4  
    }  
}  

def create_primitive():  
    """创建随机基元几何体"""  
    primitives = ['CUBE', 'CYLINDER', 'CONE', 'UV_SPHERE', 'ICO_SPHERE', 'TORUS', 'Monkey']  
    primitive_type = random.choice(primitives)  

    operator_name = f"primitive_{primitive_type.lower()}_add"  
    operator = getattr(bpy.ops.mesh, operator_name)  

    location = (  
        random.uniform(-AREA_SIZE, AREA_SIZE),  
        random.uniform(-AREA_SIZE, AREA_SIZE),  
        random.uniform(-AREA_SIZE, AREA_SIZE)  
    )  
    rotation = (  
        radians(random.randint(0, 360)),  
        radians(random.randint(0, 360)),  
        radians(random.randint(0, 360))  
    )  

    # Monkey primitive does not accept rotation argument  
    if primitive_type.lower() == "monkey":  
        operator(location=location)  
    else:  
        operator(location=location, rotation=rotation)  

    return bpy.context.object  

def setup_material(obj, mat_name):  
    """创建PBR材质节点"""  
    mat = bpy.data.materials.new(name=mat_name)  
    mat.use_nodes = True  
    nodes = mat.node_tree.nodes  
    links = mat.node_tree.links  

    # 清除默认节点  
    nodes.clear()  

    # 创建Principled BSDF节点和输出节点  
    bsdf = nodes.new('ShaderNodeBsdfPrincipled')  
    output = nodes.new('ShaderNodeOutputMaterial')  
    links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])  

    # 设置节点位置  
    bsdf.location = (0, 0)  
    output.location = (400, 0)  

    return mat, bsdf  

def load_normal_map(bsdf_node, mat_tree):  
    """加载随机法线贴图"""  
    normal_maps = glob.glob(os.path.join(TEXTURE_DIR, "*", "*NormalIGL.png"))  

    if not normal_maps:  
        raise FileNotFoundError("未找到法线贴图文件！")  

    selected_map = random.choice(normal_maps)  # selected_map is already full path  

    tex_image = bpy.data.images.load(selected_map)  
    tex_node = mat_tree.nodes.new('ShaderNodeTexImage')  
    tex_node.image = tex_image  
    tex_node.location = (-600, 0)  

    normal_node = mat_tree.nodes.new('ShaderNodeNormalMap')  
    normal_node.location = (-300, 0)  
    normal_node.inputs['Strength'].default_value = random.uniform(*NORMAL_STRENGTH_RANGE)  

    mat_tree.links.new(tex_node.outputs['Color'], normal_node.inputs['Color'])  
    mat_tree.links.new(normal_node.outputs['Normal'], bsdf_node.inputs['Normal'])  

    return normal_node  

def export_glb(filepath):  
    """导出GLB文件"""  
    # 选中所有物体  
    for obj in bpy.data.objects:  
        obj.select_set(True)  

    # 导出设置  
    bpy.ops.export_scene.gltf(  
        filepath=filepath,  
        check_existing=False,  
        export_format='GLB',  
        export_yup=True,  
        export_apply=True,  
        export_cameras=False,  
        export_lights=False,  
        export_materials='EXPORT',  
        export_normals=True,  
        export_tangents=False,  
        export_animations=False,  
        export_skins=False,  
        export_morph=False  
    )  

def main():  
    # 清理场景  
    bpy.ops.object.select_all(action='SELECT')  
    bpy.ops.object.delete()  

    # 创建指定数量的物体  
    for _ in range(NUM_OBJECTS):  
        try:  
            breakpoint()
            obj = create_primitive()  
            obj.select_set(True)
            bpy.ops.object.shade_smooth()
            obj.select_set(False)
            mat_type = random.choice(list(MATERIAL_PRESETS.keys()))  
            preset = MATERIAL_PRESETS[mat_type]  

            # 创建材质  
            mat_name = f"{mat_type}_Material_{random.randint(1000, 9999)}"  
            material, bsdf = setup_material(obj, mat_name)  

            # 设置材质属性  
            bsdf.inputs['Base Color'].default_value = preset['base_color']  
            bsdf.inputs['Metallic'].default_value = preset['metallic']  
            bsdf.inputs['Roughness'].default_value = preset['roughness']  

            # 添加法线贴图节点  
            load_normal_map(bsdf, material.node_tree)  

            # 应用材质到物体  
            if obj.data.materials:  
                obj.data.materials[0] = material  
            else:  
                obj.data.materials.append(material)  

        except Exception as e:  
            print(f"创建物体时出错: {str(e)}")  

    # 导出GLB文件，文件名包含时间戳  
    try:  
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  
        export_path = EXPORT_PATH.replace(".glb", f"_{timestamp}.glb")  
        export_glb(export_path)  
        print(f"成功导出到: {export_path}")  
    except Exception as e:  
        print(f"导出失败: {str(e)}")  

if __name__ == "__main__":  
    main()  