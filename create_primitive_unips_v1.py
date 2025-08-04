import bpy  
import random  
import os  

# 新版Principled端口映射表（支持自动兼容老版本/别名）  
PARAM_PORTS = {  
    "base_color":      ["Base Color"],  
    "metallic":        ["Metallic"],  
    "roughness":       ["Roughness"],  
    "ior":             ["IOR"],  
    "alpha":           ["Alpha"],  
    "normal":          ["Normal"],  
    "weight":          ["Weight"],  
    "diffuse_roughness": ["Diffuse Roughness"],  
    # subsurface  
    "subsurface_weight":      ["Subsurface Weight"],  
    "subsurface_radius":      ["Subsurface Radius"],  
    "subsurface_scale":       ["Subsurface Scale"],  
    "subsurface_ior":         ["Subsurface IOR"],  
    "subsurface_anisotropy":  ["Subsurface Anisotropy"],  
    # specular  
    "specular_ior_level":  ["Specular IOR Level", "IOR Level", "Specular"],  
    "specular_tint":       ["Specular Tint"],  
    # anisotropic  
    "anisotropic":         ["Anisotropic"],  
    "anisotropic_rotation": ["Anisotropic Rotation"],  
    "tangent":             ["Tangent"],  
    # transmission  
    "transmission_weight": ["Transmission Weight", "Transmission"],  
    # coat  
    "coat_weight":         ["Coat Weight"],  
    "coat_roughness":      ["Coat Roughness"],  
    "coat_ior":            ["Coat IOR"],  
    "coat_tint":           ["Coat Tint"],  
    "coat_normal":         ["Coat Normal"],  
    # sheen  
    "sheen_weight":        ["Sheen Weight"],  
    "sheen_roughness":     ["Sheen Roughness"],  
    "sheen_tint":          ["Sheen Tint"],  
    # emission  
    "emission_color":      ["Emission Color"],  
    "emission_strength":   ["Emission Strength"],  
    # thin film  
    "thin_film_thickness": ["Thin Film Thickness"],  
    "thin_film_ior":       ["Thin Film IOR"],  
}  

# ====== 物理参数丰富的材质库（可自由增改/批量生成）======  
MATERIAL_PRESETS = {  
    "Aluminum": {  
        "base_color": (0.91, 0.92, 0.92, 1),  
        "metallic": 1.0,  "roughness": 0.17,   "specular_ior_level": 0.9,  
        "sheen_weight": 0, "sheen_roughness": 0,  
    },  
    "Copper": {  
        "base_color": (0.95, 0.64, 0.54, 1),  
        "metallic": 1.0,  "roughness": 0.23,   "specular_ior_level": 0.98,  
        "sheen_weight": 0, "sheen_roughness": 0,  
    },  
    "Gold": {  
        "base_color": (1.0, 0.83, 0.45, 1),  
        "metallic": 1.0,  "roughness": 0.20,   "specular_ior_level": 1.0,  
        "sheen_weight": 0, "sheen_roughness": 0,  
    },  
    "Steel": {  
        "base_color": (0.7, 0.72, 0.73, 1),  
        "metallic": 1.0,  "roughness": 0.12,   "specular_ior_level": 0.95,  
        "sheen_weight": 0, "sheen_roughness": 0,  
    },  
    "CarPaint": {  
        "base_color": (0.15, 0.19, 0.75, 1),  
        "metallic": 0.0,  "roughness": 0.07,  "coat_weight": 1.0, "coat_roughness": 0.10,  
        "specular_ior_level": 0.73,  
        "sheen_weight": 0.10, "sheen_roughness": 0.25,  
    },  
    "ABS": {  
        "base_color": (0.1, 0.1, 0.12, 1),  
        "metallic": 0.0, "roughness": 0.37,  "specular_ior_level": 0.56,  
        "sheen_weight": 0.03, "sheen_roughness": 0.27,  
    },  
    "Nylon": {  
        "base_color": (0.75, 0.77, 0.74, 1),  
        "metallic": 0.0, "roughness": 0.44,  "specular_ior_level": 0.52,  
        "sheen_weight": 0.12, "sheen_roughness": 0.32,  
    },  
    "Acrylic": {  
        "base_color": (0.96, 0.97, 1.0, 1),  
        "metallic": 0.0, "roughness": 0.04,  "specular_ior_level": 0.71,  
        "sheen_weight": 0, "sheen_roughness": 0,  
        "transmission_weight": 0.92, "ior": 1.49,  
    },  
    "Rubber": {  
        "base_color": (0.06, 0.06, 0.08, 1), "metallic": 0.0,  
        "roughness": 0.84, "specular_ior_level": 0.06,  
        "sheen_weight": 0.00, "sheen_roughness": 0.55,  
    },  
    "Ceramic": {  
        "base_color": (0.93, 0.91, 0.88, 1),  
        "metallic": 0.0, "roughness": 0.22,  "specular_ior_level": 0.5,  
        "sheen_weight": 0.10, "sheen_roughness": 0.44,  
    },  
    "Wood": {  
        "base_color": (0.41, 0.26, 0.13, 1),  
        "metallic": 0.0, "roughness": 0.62,  "specular_ior_level": 0.19,  
        "sheen_weight": 0.18, "sheen_roughness": 0.26,  
    },  
    "Glass": {  
        "base_color": (0.95, 0.97, 1.0, 1),  
        "metallic": 0.0, "roughness": 0.01, "alpha": 0.0, "transmission_weight": 1.0,  
        "ior": 1.52, "specular_ior_level": 0.91,  
        "sheen_weight": 0, "sheen_roughness": 0,  
    },  
    "Fabric": {  
        "base_color": (0.16, 0.17, 0.22, 1),  
        "metallic": 0.0, "roughness": 0.75,  "specular_ior_level": 0.17,  
        "sheen_weight": 0.63, "sheen_roughness": 0.62, "sheen_tint": 0.56,  
    },  
    "Leather": {  
        "base_color": (0.27, 0.13, 0.07, 1),  
        "metallic": 0.0, "roughness": 0.48,  "specular_ior_level": 0.24,  
        "sheen_weight": 0.37, "sheen_roughness": 0.22, "sheen_tint": 0.40,  
    },  
    "Painted": {  
        "base_color": (0.12, 0.19, 0.70, 1),  
        "metallic": 0.0, "roughness": 0.07,  "specular_ior_level": 0.8,  
        "sheen_weight": 0.23, "sheen_roughness": 0.18,  
    },  
    # 你可继续添加  
}  

# 随机缩放（在对象创建之后调用）  
def random_scale(obj):  
    sx, sy, sz = [random.uniform(0.6, 2.0) for _ in range(3)]  
    obj.scale = (sx, sy, sz)  

# 随机扭转  
def random_twist(obj):  
    twist = obj.modifiers.new("Twist", type='SIMPLE_DEFORM')  
    twist.deform_method = 'TWIST'  
    twist.angle = random.uniform(-2.0, 2.0)  

# 随机锥化  
def random_taper(obj):  
    taper = obj.modifiers.new("Taper", type='SIMPLE_DEFORM')  
    taper.deform_method = 'TAPER'  
    taper.factor = random.uniform(-0.7, 0.7)  

# 随机位移  
def random_displace(obj):  
    disp = obj.modifiers.new("Displace", type="DISPLACE")  
    tex = bpy.data.textures.new("disp_tex", 'CLOUDS')  
    disp.texture = tex  
    disp.strength = random.uniform(0.08, 0.24)  

# 将上述函数组合  
def deform_randomly(obj):  
    # 每次掷骰子决定是否用某种变形  
    if random.random() < 0.8: random_scale(obj)  
    if random.random() < 0.5: random_twist(obj)  
    if random.random() < 0.3: random_taper(obj)  
    if random.random() < 0.7: random_displace(obj) 

##### --- 复杂形变组件 --- #####  
def random_complex_deform(obj):  
    # 1. 强化细长化  
    if random.random() < 0.95:  
        axis = random.choice([0, 1, 2])  
        scales = [random.uniform(0.2, 0.7) for _ in range(3)]  
        scales[axis] = random.uniform(2.5, 6.5)  # 某一轴极拉长  
        obj.scale = tuple(scales)  
    # 2. 多级变形+弯曲  
    if random.random() < 0.7:  
        bend = obj.modifiers.new("Bend", type='SIMPLE_DEFORM')  
        bend.deform_method = 'BEND'  
        bend.angle = random.uniform(-2.2, 2.2)  
        bend.deform_axis = 'Z'
    if random.random() < 0.8:  
        twist = obj.modifiers.new("Twist", type='SIMPLE_DEFORM')  
        twist.deform_method = 'TWIST'  
        twist.angle = random.uniform(-2.5, 2.5)  
        twist.origin = None  
        twist.deform_axis = 'Y'  
    if random.random() < 0.6:  
        taper = obj.modifiers.new("Taper", type='SIMPLE_DEFORM')  
        taper.deform_method = 'TAPER'  
        taper.factor = random.uniform(-0.65, 0.65)  
        taper.origin = None  
        taper.deform_axis = 'X'  
    # 3. 凹陷挖洞（布尔差集，多次）  
    for _ in range(random.randint(1, 2)):  
        if random.random() < 0.8:  
            # 生成随机辅助体（如球/圆柱）  
            shape = random.choice(  
                [bpy.ops.mesh.primitive_uv_sphere_add, bpy.ops.mesh.primitive_cylinder_add]  
            )  
            delta = [random.uniform(-0.3, 0.3) for _ in range(3)]  
            shape(location=(obj.location[0]+delta[0], obj.location[1]+delta[1], obj.location[2]+delta[2]))  
            cutter = bpy.context.active_object  
            cutter.scale = (random.uniform(0.25, 0.95), random.uniform(0.25, 0.95), random.uniform(0.15, 1.6))  
            cutter.rotation_euler = (  
                random.uniform(0, 3.14), random.uniform(0, 3.14), random.uniform(0, 3.14)  
            )  
            # 差集  
            bool_mod = obj.modifiers.new("Boolean", type='BOOLEAN')  
            bool_mod.object = cutter  
            bool_mod.operation = 'DIFFERENCE'  
            bpy.context.view_layer.objects.active = obj  
            bpy.ops.object.modifier_apply(modifier=bool_mod.name)  
            # 清理辅助体  
            bpy.data.objects.remove(cutter)  
    # 4. 表面乱扰动  
    if random.random() < 0.6:  
        disp = obj.modifiers.new("Displace", type="DISPLACE")  
        tex = bpy.data.textures.new("displace_noise", 'CLOUDS')  
        disp.texture = tex  
        disp.strength = random.uniform(-0.22, 0.34)  

    # 5. 更多细分，曲面平滑  
    bpy.context.view_layer.objects.active = obj  
    if random.random() < 0.7:  
        subdiv = obj.modifiers.new("Subd2", type='SUBSURF')  
        subdiv.levels = random.randint(2, 3)  
        subdiv.render_levels = subdiv.levels  
        bpy.ops.object.modifier_apply(modifier=subdiv.name)  
    bpy.ops.object.shade_smooth()  

    # 6. 重算法线  
    if obj.type == 'MESH':  
        bpy.context.view_layer.objects.active = obj  
        bpy.ops.object.editmode_toggle()  
        bpy.ops.mesh.normals_make_consistent(inside=False)  
        bpy.ops.object.editmode_toggle()  


def generate_random_geometry_complex():  
    bpy.ops.object.select_all(action='DESELECT')  
    primitives = [  
        ("cube", bpy.ops.mesh.primitive_cube_add),  
        ("uv_sphere", bpy.ops.mesh.primitive_uv_sphere_add),  
        ("ico_sphere", bpy.ops.mesh.primitive_ico_sphere_add),  
        ("torus", bpy.ops.mesh.primitive_torus_add),  
        ("cylinder", bpy.ops.mesh.primitive_cylinder_add),  
        ("cone", bpy.ops.mesh.primitive_cone_add),  
    ]  
    name, obj_func = random.choice(primitives)  
    obj_func(location=(0,0,0))  
    obj = bpy.context.active_object  
    obj.name = f"Random_{name}"  
    # 初始细分：基础体先细分再复杂变形  
    subdiv = obj.modifiers.new("Subdivision", type='SUBSURF')  
    subdiv.levels = random.randint(2, 4); subdiv.render_levels = subdiv.levels  
    bpy.ops.object.modifier_apply(modifier=subdiv.name)  
    # 复杂级变形  
    random_complex_deform(obj)  
    return obj

def set_input_auto(node, param_key, value):  
    # 自动提取新版port名称，若找不到则跳过  
    ports = PARAM_PORTS.get(param_key)  
    if ports:  
        for port in ports:  
            if port in node.inputs:  
                node.inputs[port].default_value = value  
                return True  
    return False  
    
def create_principled_material_with_normal(name, params):  
    mat = bpy.data.materials.new(name)  
    mat.use_nodes = True  
    nodes, links = mat.node_tree.nodes, mat.node_tree.links  
    for node in nodes: nodes.remove(node)  
    out = nodes.new(type='ShaderNodeOutputMaterial'); out.location = (400, 0)  
    bsdf = nodes.new(type='ShaderNodeBsdfPrincipled'); bsdf.location = (0, 0)  
    links.new(bsdf.outputs[0], out.inputs[0])  
    # 批量赋所有能赋的属性  
    for key, val in params.items():  
        set_input_auto(bsdf, key, val)  
    # 程序化法线  
    noise = nodes.new(type='ShaderNodeTexNoise'); noise.location = (-400,0)  
    noise.inputs[2].default_value = random.uniform(15.0, 45.0) # scale  
    noise.inputs[3].default_value = random.uniform(8, 13) # detail  
    noise.inputs[4].default_value = random.uniform(0.5, 2.2) # distortion  
    bump = nodes.new(type='ShaderNodeBump'); bump.location = (-170,0)  
    bump.inputs[0].default_value = random.uniform(0.06, 0.18) # strength  
    bump.inputs[1].default_value = 1.0  
    links.new(noise.outputs[0], bump.inputs[2])  
    links.new(bump.outputs[0], bsdf.inputs['Normal'])  
    return mat  


def generate_random_geometry():  
    bpy.ops.object.select_all(action='DESELECT')  
    primitives = [  
        ("cube", bpy.ops.mesh.primitive_cube_add),  
        ("uv_sphere", bpy.ops.mesh.primitive_uv_sphere_add),  
        ("ico_sphere", bpy.ops.mesh.primitive_ico_sphere_add),  
        ("torus", bpy.ops.mesh.primitive_torus_add),  
        ("cylinder", bpy.ops.mesh.primitive_cylinder_add),  
        ("cone", bpy.ops.mesh.primitive_cone_add),  
    ]  
    name, obj_func = random.choice(primitives)  
    obj_func(location=(0,0,0))  
    obj = bpy.context.active_object  
    obj.name = f"Random_{name}"  
    subdiv = obj.modifiers.new("Subdivision", type='SUBSURF')  
    subdiv.levels = random.randint(2, 4); subdiv.render_levels = subdiv.levels  
    bpy.context.view_layer.objects.active = obj  
    bpy.ops.object.modifier_apply(modifier=subdiv.name)  
    bpy.ops.object.shade_smooth()  
    return obj  

def assign_random_material(obj):  
    mat_name = random.choice(list(MATERIAL_PRESETS.keys()))  
    mat_params = MATERIAL_PRESETS[mat_name]  
    mat = create_principled_material_with_normal(mat_name, mat_params)  
    if obj.data.materials:  
        obj.data.materials[0] = mat  
    else:  
        obj.data.materials.append(mat)  
    return mat_name  

# ---------- 导出GLB ----------
def export_glb(filepath):
    bpy.ops.export_scene.gltf(
        filepath=filepath,
        export_format='GLB',
        export_animations=False,
        export_apply=True,
        export_materials='EXPORT',
        export_cameras=False,
        export_lights=False
    )

# ---------- 主流程 ----------
# def main():
#     # (1) 清空场景
#     bpy.ops.object.select_all(action='SELECT')
#     bpy.ops.object.delete(use_global=False)
#     # (2) 随机几何
#     obj = generate_random_geometry()
#     deform_randomly(obj)
#     # (3) 材质赋值
#     mat_name = assign_random_material(obj)
#     print(f"生成物体: {obj.name}，覆盖材质: {mat_name}")
#     # (4) 导出GLB（此处以桌面为例，请修改export_path为你指定的位置）
#     home = "export_unips_v1"
#     post_fix = random.randint(1000, 9999)
#     filename = f"primitive_unips_{mat_name}_{obj.name}_{post_fix}.glb"

#     export_path = os.path.join(home, filename)
#     export_glb(export_path)
#     print(f"GLB已导出至: {export_path}")

def main():  
    bpy.ops.object.select_all(action='SELECT')  
    bpy.ops.object.delete(use_global=False)  
    obj = generate_random_geometry_complex()  
    mat_name = assign_random_material(obj)  
    print(f"复杂物体: {obj.name}，材质: {mat_name}")  
    home = "export_unips_v1"  
    post_fix = random.randint(1000, 9999)  
    filename = f"primitive_unips_{mat_name}_{obj.name}_{post_fix}.glb"  
    export_path = os.path.join(home, filename)  
    export_glb(export_path)  
    print(f"GLB已导出至: {export_path}")  

if __name__ == "__main__":
    main()