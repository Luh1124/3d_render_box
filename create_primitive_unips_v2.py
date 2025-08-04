import bpy  
import os  
import math  
import random  

# --- 材质预设（可继续添加/调整） ---  
MATERIALS = {  
    "POM":      {"base_color": (0.78, 0.76, 0.70, 1), "metallic": 0, "roughness": 0.38, "specular": 0.55},  
    "PP":       {"base_color": (0.90, 0.90, 0.82, 1), "metallic": 0, "roughness": 0.41, "specular": 0.35},  
    "Nylon":    {"base_color": (0.83, 0.85, 0.82, 1), "metallic": 0, "roughness": 0.50, "specular": 0.32},  
    "PVC":      {"base_color": (0.7, 0.74, 0.77, 1),  "metallic": 0, "roughness": 0.21, "specular": 0.18},  
    "ABS":      {"base_color": (0.22, 0.22, 0.25, 1), "metallic": 0, "roughness": 0.33, "specular": 0.41},  
    "Bakelite": {"base_color": (0.35, 0.23, 0.13, 1), "metallic": 0, "roughness": 0.62, "specular": 0.27},  
    "AL":       {"base_color": (0.83, 0.85, 0.89, 1), "metallic": 1, "roughness": 0.18, "specular": 0.8},  
    "Cu":       {"base_color": (0.96, 0.64, 0.41, 1), "metallic": 1, "roughness": 0.24, "specular": 0.92},  
    "STEEL":    {"base_color": (0.65, 0.68, 0.70, 1), "metallic": 1, "roughness": 0.12, "specular": 0.9},  
    "Acrylic":  {"base_color": (0.98, 0.99, 1.0, 1),  "metallic": 0,"roughness": 0.03, "specular": 0.9, "ior": 1.49, "transmission": 1.0},  
}  

# ------- 基础几何生成区 -------  
def create_ball():  
    bpy.ops.mesh.primitive_uv_sphere_add(segments=64, ring_count=32, radius=1)  
    return bpy.context.active_object  

def create_golf():  
    obj = create_ball()  
    tex = bpy.data.textures.new("golf_noise", type='VORONOI')  
    dis = obj.modifiers.new("GolfDisplace", 'DISPLACE')  
    dis.texture = tex  
    dis.strength = 0.14  
    bpy.context.view_layer.objects.active = obj  
    bpy.ops.object.modifier_apply(modifier=dis.name)  
    return obj  

def create_spike():  
    obj = create_ball()  
    # 在球面发射刺（低多边形实现，多可用GeometryNodes）  
    spikes = []  
    for i in range(28):  
        theta = random.uniform(0, math.pi)  
        phi = random.uniform(0, 2*math.pi)  
        r = 1.12  
        x = r*math.sin(theta)*math.cos(phi)  
        y = r*math.sin(theta)*math.sin(phi)  
        z = r*math.cos(theta)  
        bpy.ops.mesh.primitive_cone_add(vertices=12, radius1=0.05, depth=0.32, location=(x, y, z))  
        cone = bpy.context.active_object  
        cone.rotation_euler = (  
            theta,   
            phi,   
            0  
        )  
        spikes.append(cone)  
    # 并集布尔  
    for spike in spikes:  
        mod = obj.modifiers.new("UNION", 'BOOLEAN')  
        mod.object = spike  
        mod.operation = 'UNION'  
        bpy.context.view_layer.objects.active = obj  
        bpy.ops.object.modifier_apply(modifier=mod.name)  
        bpy.data.objects.remove(spike)  
    return obj  

def create_nut():  
    # 六边柱+内孔  
    bpy.ops.mesh.primitive_cylinder_add(vertices=6, radius=1, depth=0.6)  
    nut = bpy.context.active_object  
    bpy.ops.mesh.primitive_cylinder_add(vertices=32, radius=0.5, depth=0.7, location=nut.location)  
    hole = bpy.context.active_object  
    mod = nut.modifiers.new("BOOL", 'BOOLEAN')  
    mod.object = hole  
    mod.operation = 'DIFFERENCE'  
    bpy.context.view_layer.objects.active = nut  
    bpy.ops.object.modifier_apply(modifier=mod.name)  
    bpy.data.objects.remove(hole)  
    return nut  

def create_square():  
    bpy.ops.mesh.primitive_cube_add(size=1.6)  
    return bpy.context.active_object  

def create_pentagon():  
    bpy.ops.mesh.primitive_cylinder_add(vertices=5, radius=1, depth=1.0)  
    return bpy.context.active_object  

def create_hexagon():  
    bpy.ops.mesh.primitive_cylinder_add(vertices=6, radius=1, depth=1.0)  
    return bpy.context.active_object  

def create_propeller():  
    bpy.ops.mesh.primitive_cylinder_add(vertices=24, radius=1, depth=0.4)  
    core = bpy.context.active_object  
    blades = []  
    for i in range(3):  
        angle = math.radians(i*120)  
        bpy.ops.mesh.primitive_cube_add(size=0.65, location=(math.cos(angle), math.sin(angle), 0.18))  
        blade = bpy.context.active_object  
        blade.scale.y = 0.15  
        blade.rotation_euler = (0, 0, angle)  
        blades.append(blade)  
    # Boolean，将叶片合体  
    for blade in blades:  
        mod = core.modifiers.new("UNION", 'BOOLEAN')  
        mod.object = blade  
        mod.operation = 'UNION'  
        bpy.context.view_layer.objects.active = core  
        bpy.ops.object.modifier_apply(modifier=mod.name)  
        bpy.data.objects.remove(blade)  
    return core  

def create_turbine():  
    bpy.ops.mesh.primitive_cylinder_add(vertices=32, radius=1, depth=0.5)  
    core = bpy.context.active_object  
    blades = []  
    for i in range(6):  
        angle = math.radians(i*60)  
        bpy.ops.mesh.primitive_cube_add(size=0.6, location=(math.cos(angle)*0.8, math.sin(angle)*0.8, 0.12))  
        blade = bpy.context.active_object  
        blade.scale.y = 0.08  
        blade.scale.x = 0.28  
        blade.rotation_euler = (0, 0, angle+0.25)  
        blades.append(blade)  
    for blade in blades:  
        mod = core.modifiers.new("UNION", 'BOOLEAN')  
        mod.object = blade  
        mod.operation = 'UNION'  
        bpy.context.view_layer.objects.active = core  
        bpy.ops.object.modifier_apply(modifier=mod.name)  
        bpy.data.objects.remove(blade)  
    return core  

def import_bunny():  
    # 建议提前本地准备好 Stanford Bunny glb/obj，并设路径  
    bunny_path = "/your/path/bunny.obj"  
    bpy.ops.import_scene.obj(filepath=bunny_path)  
    # 取最新导入的object  
    obj = bpy.context.selected_objects[-1]  
    bpy.context.view_layer.objects.active = obj  
    return obj  

GEOMETRY_FUNCS = {  
    "Ball": create_ball,  
    "Golf": create_golf,  
    "Spike": create_spike,  
    "Nut": create_nut,  
    "Square": create_square,  
    "Pentagon": create_pentagon,  
    "Hexagon": create_hexagon,  
    "Propeller": create_propeller,  
    "Turbine": create_turbine,  
    # "Bunny": import_bunny,  # 需要准备好路径  
}  

# ------- 材质生成 -------  
def create_material(name, params):  
    mat = bpy.data.materials.new(name=name)  
    mat.use_nodes = True  
    nodes = mat.node_tree.nodes  
    bsdf = nodes.get("Principled BSDF")  
    for k,v in params.items():  
        if k == "base_color":  
            bsdf.inputs["Base Color"].default_value = v  
        elif k.capitalize() in bsdf.inputs:  
            bsdf.inputs[k.capitalize()].default_value = v  
        elif k in bsdf.inputs:  
            bsdf.inputs[k].default_value = v  
    # 增加轻微噪声Bump（法线多样化，可丰富）  
    bump = nodes.new("ShaderNodeBump")  
    noise = nodes.new("ShaderNodeTexNoise")  
    nodes["Noise Texture"].inputs["Scale"].default_value = random.uniform(7, 24)  
    nodes["Bump"].inputs["Strength"].default_value = random.uniform(0.01, 0.12)  
    mat.node_tree.links.new(noise.outputs["Color"], bump.inputs["Height"])  
    mat.node_tree.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])  
    return mat  

def assign_material(obj, mat):  
    if obj.data.materials:  
        obj.data.materials[0] = mat  
    else:  
        obj.data.materials.append(mat)  

# -------- GLB 导出 --------  
def export_glb(obj, save_path):  
    bpy.ops.object.select_all(action='DESELECT')  
    obj.select_set(True)  
    bpy.ops.export_scene.gltf(  
        filepath=save_path,  
        export_format='GLB',  
        export_apply=True,  
        use_selection=True,  
        export_materials='EXPORT'  
    )  

# --------- 主批量生成流程 ---------  
def batch_main(export_dir="export_unips_v2"):  
    # 创建目录  
    if not os.path.exists(export_dir):  
        os.makedirs(export_dir)  

    # 清空场景  
    bpy.ops.object.select_all(action='SELECT')  
    bpy.ops.object.delete(use_global=False)  
    # 组合生成  
    for geo_name, geo_func in GEOMETRY_FUNCS.items():  
        for mat_name, params in MATERIALS.items():  
            # 生成几何  
            obj = geo_func()  
            # 材质  
            mat = create_material(mat_name, params)  
            assign_material(obj, mat)  
            bpy.context.view_layer.objects.active = obj  
            # 优化网格表现  
            bpy.ops.object.shade_smooth()  
            if hasattr(obj.data, "use_auto_smooth"):  
                obj.data.use_auto_smooth = True  
                obj.data.auto_smooth_angle = 3.14159  
            # GLB名称  
            glb_path = os.path.join(export_dir, f"{geo_name}_{mat_name}.glb")  
            # 导出  
            export_glb(obj, glb_path)  
            print(f"导出: {glb_path}")  
            # 清除当前object，为下一对组合  
            bpy.data.objects.remove(obj, do_unlink=True)  

if __name__ == "__main__":  
    batch_main("export_unips_v2")  # 改成你的实际输出目录  