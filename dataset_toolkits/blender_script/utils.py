import os
import argparse, sys, os, math, re, glob
from typing import *
import bpy
from mathutils import Vector, Matrix, Euler, Quaternion
import numpy as np
import json
import glob

AOV_NAMES_WITHOUT = ['alpha', 'depth', 'normal', 'low_normal', 'albedo', 'glossycol', 'mist', 'Base Color', 'Roughness', 'Metallic', 'position', 'ao']
AOV_NAMES_WITH = ['glossydir', 'diffdir', 'shadow', 'diffind', 'glossyind', 'env', 'image', 'low_image']

def make_output_dir(output_dir: str, name: str):
    output_dir = os.path.join(output_dir, name)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def get_local2world_mat(blender_obj) -> np.ndarray:
    """Returns the pose of the object in the form of a local2world matrix.
    :return: The 4x4 local2world matrix.
    """
    obj = blender_obj
    # Start with local2parent matrix (if obj has no parent, that equals local2world)
    matrix_world = obj.matrix_basis

    # Go up the scene graph along all parents
    while obj.parent is not None:
        # Add transformation to parent frame
        matrix_world = (
            obj.parent.matrix_basis @ obj.matrix_parent_inverse @ matrix_world
        )
        obj = obj.parent

    return np.array(matrix_world)


def init_scene() -> None:
    """Resets the scene to a clean state.

    Returns:
        None
    """
    # delete everything
    for obj in bpy.data.objects:
        bpy.data.objects.remove(obj, do_unlink=True)

    # delete all the materials
    for material in bpy.data.materials:
        bpy.data.materials.remove(material, do_unlink=True)

    # delete all the textures
    for texture in bpy.data.textures:
        bpy.data.textures.remove(texture, do_unlink=True)

    # delete all the images
    for image in bpy.data.images:
        bpy.data.images.remove(image, do_unlink=True)

def del_lighting():
    # Clear existing lights
    bpy.ops.object.select_all(action="DESELECT")
    bpy.ops.object.select_by_type(type="LIGHT")
    bpy.ops.object.delete()

def setup_pointlight_position(light, engery, location, rotation_euler):
    light.data.energy = engery
    light.location = location
    light.rotation_euler = rotation_euler

    return light

def set_trellis_lighting():
    # Create key light
    del_lighting()
    default_light = bpy.data.objects.new("Default_Light", bpy.data.lights.new("Default_Light", type="POINT"))
    bpy.context.collection.objects.link(default_light)
    default_light.data.energy = 1000
    default_light.location = (4, 1, 6)
    default_light.rotation_euler = (0, 0, 0)
    
    # create top light
    top_light = bpy.data.objects.new("Top_Light", bpy.data.lights.new("Top_Light", type="AREA"))
    bpy.context.collection.objects.link(top_light)
    top_light.data.energy = 10000
    top_light.location = (0, 0, 10)
    top_light.scale = (100, 100, 100)
    
    # create bottom light
    bottom_light = bpy.data.objects.new("Bottom_Light", bpy.data.lights.new("Bottom_Light", type="AREA"))
    bpy.context.collection.objects.link(bottom_light)
    bottom_light.data.energy = 1000
    bottom_light.location = (0, 0, -10)
    bottom_light.rotation_euler = (0, 0, 0)
    
    return {
        "default_light": default_light,
        "top_light": top_light,
        "bottom_light": bottom_light
    }

def del_world_tree():
    # Clear existing world nodes and links
    world = bpy.context.scene.world
    world.use_nodes = True
    node_tree = world.node_tree
    nodes = node_tree.nodes
    links = node_tree.links

    nodes.clear()
    links.clear()

    world.use_sun_shadow = True
    world.use_sun_shadow_jitter = True

    return nodes, links


def set_hdri(path_to_hdr_file: str, strength: float = 1.0,
                                 rotation_euler: Union[list, Euler, np.ndarray] = None):
    nodes, links = del_world_tree()

    if rotation_euler is None:
        rotation_euler = [0.0, 0.0, 0.0]

    background_node = bpy.context.scene.world.node_tree.nodes.new(type='ShaderNodeBackground')
    world_output_node = bpy.context.scene.world.node_tree.nodes.new(type='ShaderNodeOutputWorld')
    environtment_texture_node = bpy.context.scene.world.node_tree.nodes.new(type='ShaderNodeTexEnvironment')
    environtment_texture_node.image = bpy.data.images.load(path_to_hdr_file, check_existing=True)
    bpy.context.scene.world.node_tree.links.new(environtment_texture_node.outputs['Color'], background_node.inputs['Color'])
    background_node.inputs['Strength'].default_value = 1
    bpy.context.scene.world.node_tree.links.new(background_node.outputs['Background'], world_output_node.inputs['Surface'])

    texture_coord_node = bpy.context.scene.world.node_tree.nodes.new(type='ShaderNodeTexCoord')
    mapping_node = bpy.context.scene.world.node_tree.nodes.new(type='ShaderNodeMapping')
    bpy.context.scene.world.node_tree.links.new(texture_coord_node.outputs['Generated'], mapping_node.inputs['Vector'])
    bpy.context.scene.world.node_tree.links.new(mapping_node.outputs['Vector'], environtment_texture_node.inputs['Vector'])

    mapping_node.inputs["Rotation"].default_value = rotation_euler
    
    bpy.context.scene.world.use_sun_shadow_jitter = True
    
    return {
        "background_node": background_node,
        "world_output_node": world_output_node,
    }

def set_even_background_lighting_with_strength(strength: float = 0.6):
    nodes, links = del_world_tree()
    background_node = nodes.new(type="ShaderNodeBackground")
    background_node.inputs["Color"].default_value = (1, 1, 1, 1)
    background_node.inputs["Strength"].default_value = strength
    world_output_node = nodes.new(type="ShaderNodeOutputWorld")
    links.new(background_node.outputs["Background"], world_output_node.inputs["Surface"])

    return {
        "background_node": background_node,
        "world_output_node": world_output_node
    }

def setup_5daigc_light_and_world_with_hdri(hdri_file_path, strength=1.0, rotation_euler=None):  

    # 初始化点光源
    default_light = bpy.data.objects.new("Default_Light", bpy.data.lights.new("Default_Light", type="POINT"))
    bpy.context.collection.objects.link(default_light)
    default_light.data.energy = 1000
    default_light.location = (4, 1, 6)
    default_light.rotation_euler = (0, 0, 0)

    # 获取当前场景的 World 节点树  
    nodes, links = del_world_tree()

    # 添加节点  
    texture_node = nodes.new(type="ShaderNodeTexEnvironment")  # 环境纹理节点  
    texture_node.image = bpy.data.images.load(hdri_file_path, check_existing=True)  # 加载 HDRI 文件  
    texture_node.location = (-800, 300)  

    light_path_node1 = nodes.new(type="ShaderNodeLightPath")  # 光程节点  
    light_path_node1.location = (-800, 200)  

    light_path_node2 = nodes.new(type="ShaderNodeLightPath")  # 光程节点  
    light_path_node2.location = (-800, -200)  

    mix_node1 = nodes.new(type="ShaderNodeMixRGB")  # Mix 节点  
    mix_node1.blend_type = 'MIX'  
    mix_node1.inputs[0].default_value = 1.0  # Factor 值  
    mix_node1.inputs[1].default_value = (0.1, 0.1, 0.1, 1)  # Color 1
    mix_node1.inputs[2].default_value = (0, 0, 0, 1)  # Color 2
    mix_node1.location = (-400, 100)  

    mix_node2 = nodes.new(type="ShaderNodeMixRGB")  # Mix 节点  
    mix_node2.blend_type = 'MIX'  
    mix_node2.inputs[0].default_value = 1.0  # Factor 值  
    mix_node2.inputs[1].default_value = (0, 0, 0, 1)  # Color 1
    mix_node2.location = (-400, -100)  

    add_shader_node = nodes.new(type="ShaderNodeAddShader")  # Add Shader 节点  
    add_shader_node.location = (0, 100)  

    world_output = nodes.new(type="ShaderNodeOutputWorld")  # World 输出节点  
    world_output.location = (200, 100)  

    # 链接节点  
    links.new(light_path_node1.outputs['Is Camera Ray'], mix_node1.inputs[0])  # 相机光线 -> Mix 节点 (Factor)  
    links.new(light_path_node2.outputs['Is Camera Ray'], mix_node2.inputs[0])  # 相机光线 -> Mix 节点 (Factor)  
    links.new(texture_node.outputs['Color'], mix_node2.inputs[2])  # 环境纹理 -> Mix 节点 (B)  
    links.new(mix_node1.outputs[0], add_shader_node.inputs[0])  # 次背景 -> Add Shader 输入 1  
    links.new(mix_node2.outputs[0], add_shader_node.inputs[1])  # 主背景 -> Add Shader 输入 2  
    links.new(add_shader_node.outputs['Shader'], world_output.inputs['Surface'])  # Add Shader -> World 输出  

    return {
        "default_light": default_light,
    }


def init_camera():
    cam = bpy.data.objects.new('Camera', bpy.data.cameras.new('Camera'))
    bpy.context.collection.objects.link(cam)
    bpy.context.scene.camera = cam
    cam.data.sensor_height = cam.data.sensor_width = 32
    cam_constraint = cam.constraints.new(type='TRACK_TO')
    cam_constraint.track_axis = 'TRACK_NEGATIVE_Z'
    cam_constraint.up_axis = 'UP_Y'
    cam_empty = bpy.data.objects.new("Empty", None)
    cam_empty.location = (0, 0, 0)
    bpy.context.scene.collection.objects.link(cam_empty)
    cam_constraint.target = cam_empty
    return cam

def init_orthographic_camera(ortho_scale=0.8):
    cam = bpy.data.objects.new('Orthographic_Camera', bpy.data.cameras.new('Camera'))
    bpy.context.collection.objects.link(cam)
    bpy.context.scene.camera = cam
    cam.data.type = 'ORTHO'
    cam.data.ortho_scale = ortho_scale
    cam.location = (0, -2, 0)
    
    cam_constraint = cam.constraints.new(type='TRACK_TO')
    cam_constraint.track_axis = 'TRACK_NEGATIVE_Z'
    cam_constraint.up_axis = 'UP_Y'
    cam_empty = bpy.data.objects.new("Empty", None)
    cam_empty.location = (0, 0, 0)
    bpy.context.scene.collection.objects.link(cam_empty)
    cam_constraint.target = cam_empty
    return cam

def init_render(engine='CYCLES', resolution=512, geo_mode=False, film_transparent=False, color_depth='16'):
    bpy.context.scene.render.engine = engine
    bpy.context.scene.render.resolution_x = resolution
    bpy.context.scene.render.resolution_y = resolution
    bpy.context.scene.render.resolution_percentage = 100
    
    # bpy.context.scene.render.image_settings.file_format = 'OPEN_EXR'
    bpy.context.scene.render.image_settings.file_format = 'PNG'
    bpy.context.scene.render.image_settings.color_depth = color_depth
    bpy.context.scene.render.image_settings.color_mode = 'RGBA'
    bpy.context.scene.render.film_transparent = film_transparent
    
    if engine == 'CYCLES': 
        bpy.context.scene.cycles.samples = 16 if not geo_mode else 1
        bpy.context.scene.cycles.filter_type = 'BOX'
        bpy.context.scene.cycles.filter_width = 1
        # ================== 光程优化配置 ==================
        # bpy.context.scene.cycles.max_bounces = 12            # 总反弹次数[1](@ref)
        bpy.context.scene.cycles.diffuse_bounces = 2         # 漫反射反弹
        bpy.context.scene.cycles.glossy_bounces = 2          # 镜面反射反弹
        # bpy.context.scene.cycles.use_fast_gi = True           # 启用快速全局光照[1](@ref)

        bpy.context.scene.cycles.transparent_max_bounces = 3 if not geo_mode else 0
        bpy.context.scene.cycles.transmission_bounces = 3 if not geo_mode else 1
        
        bpy.context.scene.cycles.use_denoising = True
        # bpy.context.scene.cycles.denoiser = 'OPENIMAGEDENOISE'

        # 设置可见的设备  
        bpy.context.preferences.addons['cycles'].preferences.get_devices()
        
        # 设置计算设备类型为 OPTIX, 并且禁用 CPU
        bpy.context.scene.cycles.device = 'GPU'
        bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'OPTIX'

    elif engine == 'BLENDER_EEVEE':
        bpy.context.scene.eevee.taa_render_samples = 64
        bpy.context.scene.eevee.use_gtao = True
        bpy.context.scene.eevee.use_ssr = True
        bpy.context.scene.eevee.use_bloom = True
        bpy.context.scene.render.use_high_quality_normals = True

    elif engine == 'BLENDER_EEVEE_NEXT': # 4.3 version
        enable_backface_culling(flag=False) # 关闭背面剔除

        bpy.context.scene.eevee.taa_render_samples = 64
        bpy.context.scene.eevee.use_gtao = True
        bpy.context.scene.render.use_high_quality_normals = True
        bpy.context.scene.eevee.use_shadows = True
        bpy.context.scene.eevee.use_shadow_jitter_viewport = True
        bpy.context.scene.eevee.shadow_ray_count = 4
        bpy.context.scene.eevee.use_raytracing = True


"""=============== BLENDER ==============="""

IMPORT_FUNCTIONS: Dict[str, Callable] = {
    "obj": bpy.ops.import_scene.obj,
    "glb": bpy.ops.import_scene.gltf,
    "gltf": bpy.ops.import_scene.gltf,
    "usd": bpy.ops.import_scene.usd,
    "fbx": bpy.ops.import_scene.fbx,
    "stl": bpy.ops.wm.stl_import,
    "usda": bpy.ops.import_scene.usda,
    "dae": bpy.ops.wm.collada_import,
    "ply": bpy.ops.import_mesh.ply,
    "abc": bpy.ops.wm.alembic_import,
    "blend": bpy.ops.wm.append,
}

EXT = {
    'PNG': 'png',
    'JPEG': 'jpg',
    'OPEN_EXR': 'exr',
    'TIFF': 'tiff',
    'BMP': 'bmp',
    'HDR': 'hdr',
    'TARGA': 'tga'
}

def remove_unwanted_objects():
    """
    安全移除场景中的背景平面、灯光及带自发光的物体
    改进特性：
    - 自动去重处理
    - 材质访问安全验证
    - 不依赖上下文删除
    - 错误日志输出
    """
    objs_to_remove = set()

    # 第一阶段：识别待删除对象
    for obj in bpy.data.objects:
        # 背景平面
        if obj.name == 'BackgroundPlane':
            objs_to_remove.add(obj)
            continue

        # 灯光对象
        if obj.type == 'LIGHT':
            objs_to_remove.add(obj)
            continue

        # 带发射材质的物体
        if obj.active_material and obj.active_material.use_nodes:
            material = obj.active_material
            if any(node.type == 'EMISSION' for node in material.node_tree.nodes):
                objs_to_remove.add(obj)

    # 第二阶段：执行删除
    removal_errors = []
    for obj in objs_to_remove:
        try:
            # 解除所有关联
            if obj.users > 0:
                obj.user_clear()
            # 安全删除
            bpy.data.objects.remove(obj, do_unlink=True)
        except Exception as e:
            removal_errors.append((obj.name, str(e)))
            
def load_object(object_path: str, set_origin: bool = False) -> None:
    """Loads a model with a supported file extension into the scene.

    Args:
        object_path (str): Path to the model file.

    Raises:
        ValueError: If the file extension is not supported.

    Returns:
        None
    """
    file_extension = object_path.split(".")[-1].lower()
    if file_extension is None:
        raise ValueError(f"Unsupported file type: {object_path}")

    if file_extension == "usdz":
        # install usdz io package
        dirname = os.path.dirname(os.path.realpath(__file__))
        usdz_package = os.path.join(dirname, "io_scene_usdz.zip")
        bpy.ops.preferences.addon_install(filepath=usdz_package)
        # enable it
        addon_name = "io_scene_usdz"
        bpy.ops.preferences.addon_enable(module=addon_name)
        # import the usdz
        from io_scene_usdz.import_usdz import import_usdzx

        import_usdz(context, filepath=object_path, materials=True, animations=True)
        return None
        # load from existing import functions
    import_function = IMPORT_FUNCTIONS[file_extension]

    print(f"Loading object from {object_path}")
    if file_extension == "blend":
        import_function(directory=object_path, link=False)
    elif file_extension in {"glb", "gltf"}:
        import_function(filepath=object_path, merge_vertices=True, import_shading='NORMALS')
    else:
        import_function(filepath=object_path)

    if set_origin:
        imported_objs = bpy.context.selected_objects
        bpy.ops.object.origin_set(type='GEOMETRY_ORIGIN', center='MEDIAN')

    
def unable_normals_texture_output():
    for material in bpy.data.materials:
        material.use_nodes = True
        node_tree = material.node_tree
        nodes = node_tree.nodes
        if "Principled BSDF" not in nodes.keys():
            continue
        else:
            normal_input = nodes["Principled BSDF"].inputs["Normal"]
            if normal_input.is_linked:
                # 断开该链接
                node_tree.links.remove(normal_input.links[0])

def enable_pbr_output(rl, output_dir='', attr_name='Roughness', color_mode="RGB", file_format="PNG", color_depth="8", file_prefix: str = "", outputs=dict()):

    flag = False

    for material in bpy.data.materials:
        if not material.use_nodes:  
            continue

        node_tree = material.node_tree
        nodes = node_tree.nodes
        if "Principled BSDF" not in nodes.keys():
            flag = True
            break

        roughness_input = nodes["Principled BSDF"].inputs[attr_name]

        if roughness_input.is_linked:
            linked_node = roughness_input.links[0].from_node
            linked_socket = roughness_input.links[0].from_socket

            aov_output = nodes.new("ShaderNodeOutputAOV")
            aov_output.aov_name = attr_name
            
            node_tree.links.new(linked_socket, aov_output.inputs[0])

        else:
            fixed_roughness = roughness_input.default_value
            if isinstance(fixed_roughness, float):
                roughness_value = nodes.new("ShaderNodeValue")
                input_idx = 1
            else:
                roughness_value = nodes.new("ShaderNodeRGB")
                input_idx = 0

            roughness_value.outputs[0].default_value = fixed_roughness

            aov_output = nodes.new("ShaderNodeOutputAOV")
            aov_output.aov_name = attr_name
            node_tree.links.new(roughness_value.outputs[0], aov_output.inputs[0])

    if flag:
        return
    
    tree = bpy.context.scene.node_tree
    links = tree.links
    if "Render Layers" not in tree.nodes:
        rl = tree.nodes.new("CompositorNodeRLayers")
    else:
        rl = tree.nodes["Render Layers"]

    roughness_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
    roughness_file_output.base_path = ''
    roughness_file_output.file_slots[0].use_node_format = True
    roughness_file_output.format.file_format = file_format
    roughness_file_output.format.color_mode = color_mode
    roughness_file_output.format.color_depth = color_depth
    
    # 获取或创建AOV  
    view_layer = bpy.context.view_layer  
    aov = view_layer.aovs.add()
    aov.name = attr_name  

    if color_mode == "RGBA":
        roughness_alpha = tree.nodes.new(type="CompositorNodeSetAlpha")
        tree.links.new(rl.outputs[attr_name], roughness_alpha.inputs["Image"])
        tree.links.new(rl.outputs["Alpha"], roughness_alpha.inputs["Alpha"])
        tree.links.new(roughness_alpha.outputs["Image"], roughness_file_output.inputs["Image"])
    else:
        tree.links.new(rl.outputs[attr_name], roughness_file_output.inputs["Image"])

    outputs[attr_name] = roughness_file_output

    return aov

def init_nodes(mapper_format='Raw', save_image=False, save_low_image=False, save_alpha=False, save_depth=False, save_normal=False, save_low_normal=False, save_albedo=False,  save_glossycol=False, save_mist=False, save_pbr=False, save_env=False, save_pos=False, save_ao=False, save_glossydir=False, save_diffdir=False, save_shadow=False, save_diffind=False, save_glossyind=False):
    if not any([save_image, save_low_image, 
    save_alpha, save_depth, save_normal, save_low_normal, save_albedo, save_glossycol, save_mist, save_pbr, save_env, save_pos, save_ao, save_glossydir, save_diffdir, save_shadow, save_diffind, save_glossyind]):
        return {}, {}, {}
    outputs = {}
    spec_nodes = {}
    aovs = {}

    bpy.context.scene.render.use_compositing = True
    bpy.context.scene.use_nodes = True
    
    # bpy.context.scene.view_settings.view_transform = 'AgX'
    bpy.context.scene.view_settings.view_transform = mapper_format

    nodes = bpy.context.scene.node_tree.nodes
    links = bpy.context.scene.node_tree.links
    for n in nodes:
        nodes.remove(n)
    
    render_layers = nodes.new('CompositorNodeRLayers')

    if save_image:
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

        outputs['image'] = image_file_output
    
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
    
    if save_alpha:
        alpha_file_output = nodes.new('CompositorNodeOutputFile')
        alpha_file_output.base_path = ''
        alpha_file_output.file_slots[0].use_node_format = True
        alpha_file_output.format.file_format = 'PNG'
        alpha_file_output.format.color_mode = 'BW'
        alpha_file_output.format.color_depth = '8'

        links.new(render_layers.outputs['Alpha'], alpha_file_output.inputs[0])

        outputs['alpha'] = alpha_file_output

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
        albedo_file_output.format.file_format = 'PNG'
        albedo_file_output.format.color_mode = 'RGB'
        albedo_file_output.format.color_depth = '8'
        
        links.new(render_layers.outputs['DiffCol'], albedo_file_output.inputs[0])
        
        outputs['albedo'] = albedo_file_output
    
    if save_glossycol:
        bpy.context.view_layer.use_pass_glossy_color = save_glossycol
        glossycolor_file_output = nodes.new('CompositorNodeOutputFile')
        glossycolor_file_output.base_path = ''
        glossycolor_file_output.file_slots[0].use_node_format = True
        glossycolor_file_output.format.file_format = 'PNG'
        glossycolor_file_output.format.color_mode = 'RGB'
        glossycolor_file_output.format.color_depth = '8'

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
        ao_file_output.format.file_format = 'PNG'
        ao_file_output.format.color_mode = 'BW'
        ao_file_output.format.color_depth = '8'

        links.new(render_layers.outputs['AO'], ao_file_output.inputs[0])

        outputs['ao'] = ao_file_output

    if save_diffdir:
        bpy.context.view_layer.use_pass_diffuse_direct = save_diffdir
        diffdir_file_output = nodes.new('CompositorNodeOutputFile')
        diffdir_file_output.base_path = ''
        diffdir_file_output.file_slots[0].use_node_format = True
        diffdir_file_output.format.file_format = 'PNG'
        diffdir_file_output.format.color_mode = 'RGB'
        diffdir_file_output.format.color_depth = '8'

        links.new(render_layers.outputs['DiffDir'], diffdir_file_output.inputs[0])

        outputs['diffdir'] = diffdir_file_output
    
    if save_glossydir:
        bpy.context.view_layer.use_pass_glossy_direct = save_glossydir
        glossydir_file_output = nodes.new('CompositorNodeOutputFile')
        glossydir_file_output.base_path = ''
        glossydir_file_output.file_slots[0].use_node_format = True
        glossydir_file_output.format.file_format = 'PNG'
        glossydir_file_output.format.color_mode = 'RGB'
        glossydir_file_output.format.color_depth = '8'

        links.new(render_layers.outputs['GlossDir'], glossydir_file_output.inputs[0])

        outputs['glossydir'] = glossydir_file_output
    
    if save_shadow:
        bpy.context.view_layer.use_pass_shadow = save_shadow
        shadow_file_output = nodes.new('CompositorNodeOutputFile')
        shadow_file_output.base_path = ''
        shadow_file_output.file_slots[0].use_node_format = True
        shadow_file_output.format.file_format = 'PNG'
        shadow_file_output.format.color_mode = 'RGB'
        shadow_file_output.format.color_depth = '8'

        links.new(render_layers.outputs['Shadow'], shadow_file_output.inputs[0])

        outputs['shadow'] = shadow_file_output

    if save_diffind:
        bpy.context.view_layer.use_pass_diffuse_indirect = save_diffind
        diffind_file_output = nodes.new('CompositorNodeOutputFile')
        diffind_file_output.base_path = ''
        diffind_file_output.file_slots[0].use_node_format = True
        diffind_file_output.format.file_format = 'PNG'
        diffind_file_output.format.color_mode = 'RGB'
        diffind_file_output.format.color_depth = '8'

        links.new(render_layers.outputs['DiffInd'], diffind_file_output.inputs[0])

        outputs['diffind'] = diffind_file_output
    
    if save_glossyind:
        bpy.context.view_layer.use_pass_glossy_indirect = save_glossyind
        glossyind_file_output = nodes.new('CompositorNodeOutputFile')
        glossyind_file_output.base_path = ''
        glossyind_file_output.file_slots[0].use_node_format = True
        glossyind_file_output.format.file_format = 'PNG'
        glossyind_file_output.format.color_mode = 'RGB'
        glossyind_file_output.format.color_depth = '8'

        links.new(render_layers.outputs['GlossInd'], glossyind_file_output.inputs[0])

        outputs['glossyind'] = glossyind_file_output
    
    return outputs, spec_nodes, aovs


# 是否开启背面剔除
def enable_backface_culling(flag: bool = True):
    for material in bpy.data.materials:
        if material.use_nodes:
            material.use_backface_culling = flag

def delete_animation_data():
    # 关闭选中对象的所有动画效果  
    # bpy.ops.object.select_all(action="DESELECT")
    for obj in bpy.context.scene.objects:
        # 删除关键帧  
        if obj.animation_data:  
            obj.animation_data_clear()  
        # 删除物理模拟  
        if obj.rigid_body:  
            bpy.context.view_layer.objects.active = obj  
            bpy.ops.rigidbody.object_remove()  
        # 删除修改器（动态绘画、布料等）  
        if obj.modifiers:  
            for mod in obj.modifiers:  
                if mod.type in ['CLOTH', 'SOFT_BODY', 'FLUID', 'DYNAMIC_PAINT']:  
                    obj.modifiers.remove(mod)  
        # 删除粒子系统  
        if obj.particle_systems:  
            for ps in obj.particle_systems:  
                obj.particle_systems.remove(ps)  

def delete_armature():
    for armature in bpy.data.armatures:
        armature = bpy.data.armatures.get(armature.name)
        bpy.data.armatures.remove(armature)

def delete_gltf_not_imported():
    # delete Icosphere*
    # for obj in bpy.context.scene.objects:
    #     if obj.name.startswith("Icosphere"):
    #         bpy.data.objects.remove(obj)
    collection_name = "glTF_not_exported"  # 替换为你想要删除的集合名称  

    # 获取集合  
    collection = bpy.data.collections.get(collection_name)  

    if collection:  
        # 递归删除集合及其所有内容  
        bpy.data.collections.remove(collection, do_unlink=True)  

def delete_invisible_objects() -> None:
    """Deletes all invisible objects in the scene.

    Returns:
        None
    """
    # bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.select_all(action="DESELECT")
    for obj in bpy.context.scene.objects:
        if obj.hide_viewport or obj.hide_render:
            obj.hide_viewport = False
            obj.hide_render = False
            obj.hide_select = False
            obj.select_set(True)
    bpy.ops.object.delete()

    # Delete invisible collections
    invisible_collections = [col for col in bpy.data.collections if col.hide_viewport]
    for col in invisible_collections:
        bpy.data.collections.remove(col)

def split_mesh_normal():
    bpy.ops.object.select_all(action="DESELECT")
    objs = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    bpy.context.view_layer.objects.active = objs[0]
    for obj in objs:
        obj.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.split_normals()
    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_all(action="DESELECT")
            
def delete_custom_normals():
     for this_obj in bpy.data.objects:
        if this_obj.type == "MESH":
            bpy.context.view_layer.objects.active = this_obj
            bpy.ops.mesh.customdata_custom_splitnormals_clear()
    
def override_material():
    new_mat = bpy.data.materials.new(name="Override0123456789")
    new_mat.use_nodes = True
    new_mat.node_tree.nodes.clear()
    bsdf = new_mat.node_tree.nodes.new('ShaderNodeBsdfDiffuse')
    bsdf.inputs[0].default_value = (0.5, 0.5, 0.5, 1)
    bsdf.inputs[1].default_value = 1
    output = new_mat.node_tree.nodes.new('ShaderNodeOutputMaterial')
    new_mat.node_tree.links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])
    bpy.context.scene.view_layers['View Layer'].material_override = new_mat

def unhide_all_objects() -> None:
    """Unhides all objects in the scene.

    Returns:
        None
    """
    for obj in bpy.context.scene.objects:
        obj.hide_set(False)
        

def convert_to_meshes() -> None:
    """Converts all objects in the scene to meshes.

    Returns:
        None
    """
    bpy.ops.object.select_all(action="DESELECT")
    bpy.context.view_layer.objects.active = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"][0]
    for obj in bpy.context.scene.objects:
        obj.select_set(True)
    bpy.ops.object.convert(target="MESH")

def triangulate_meshes() -> None:
    """Triangulates all meshes in the scene.

    Returns:
        None
    """
    bpy.ops.object.select_all(action="DESELECT")
    objs = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    bpy.context.view_layer.objects.active = objs[0]
    for obj in objs:
        obj.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.reveal()
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.quads_convert_to_tris(quad_method="BEAUTY", ngon_method="BEAUTY")
    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.select_all(action="DESELECT")

def scene_bbox() -> Tuple[Vector, Vector]:
    """Returns the bounding box of the scene.

    Taken from Shap-E rendering script
    (https://github.com/openai/shap-e/blob/main/shap_e/rendering/blender/blender_script.py#L68-L82)

    Returns:
        Tuple[Vector, Vector]: The minimum and maximum coordinates of the bounding box.
    """
    bbox_min = (math.inf,) * 3
    bbox_max = (-math.inf,) * 3
    found = False
    scene_meshes = [obj for obj in bpy.context.scene.objects.values() if isinstance(obj.data, bpy.types.Mesh)]
    for obj in scene_meshes:
        found = True
        for coord in obj.bound_box:
            coord = Vector(coord)
            coord = obj.matrix_world @ coord
            bbox_min = tuple(min(x, y) for x, y in zip(bbox_min, coord))
            bbox_max = tuple(max(x, y) for x, y in zip(bbox_max, coord))
    if not found:
        raise RuntimeError("no objects in scene to compute bounding box for")
    return Vector(bbox_min), Vector(bbox_max)

def normalize_scene() -> Tuple[float, Vector]:
    """Normalizes the scene by scaling and translating it to fit in a unit cube centered
    at the origin.

    Mostly taken from the Point-E / Shap-E rendering script
    (https://github.com/openai/point-e/blob/main/point_e/evals/scripts/blender_script.py#L97-L112),
    but fix for multiple root objects: (see bug report here:
    https://github.com/openai/shap-e/pull/60).

    Returns:
        Tuple[float, Vector]: The scale factor and the offset applied to the scene.
    """
    scene_root_objects = [obj for obj in bpy.context.scene.objects.values() if not obj.parent]
    if len(scene_root_objects) > 1:
        # create an empty object to be used as a parent for all root objects
        scene = bpy.data.objects.new("ParentEmpty", None)
        bpy.context.scene.collection.objects.link(scene)

        # parent all root objects to the empty object
        for obj in scene_root_objects:
            obj.parent = scene
    else:
        scene = scene_root_objects[0]

    bbox_min, bbox_max = scene_bbox()
    scale = 1 / max(bbox_max - bbox_min)
    scene.scale = scene.scale * scale

    # Apply scale to matrix_world.
    bpy.context.view_layer.update()
    bbox_min, bbox_max = scene_bbox()
    offset = -(bbox_min + bbox_max) / 2
    scene.matrix_world.translation += offset
    bpy.ops.object.select_all(action="DESELECT")
    
    return scale, offset

def get_transform_matrix(obj: bpy.types.Object) -> list:
    pos, rt, _ = obj.matrix_world.decompose()
    rt = rt.to_matrix()
    matrix = []
    for ii in range(3):
        a = []
        for jj in range(3):
            a.append(rt[ii][jj])
        a.append(pos[ii])
        matrix.append(a)
    matrix.append([0, 0, 0, 1])
    return matrix


# ===============LOW DISCREPANCY SEQUENCES================

# 质数表（用于 Halton 序列）  
PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71]  

def radical_inverse(base, n):
    val = 0
    inv_base = 1.0 / base
    inv_base_n = inv_base
    while n > 0:
        digit = n % base
        val += digit * inv_base_n
        n //= base
        inv_base_n *= inv_base
    return val

def halton_sequence(dim, n):
    return [radical_inverse(PRIMES[dim], n) for dim in range(dim)]

def hammersley_sequence(dim, n, num_samples):
    return [n / num_samples] + halton_sequence(dim - 1, n)

def sphere_hammersley_sequence(n, num_samples, offset=(0, 0)):
    u, v = hammersley_sequence(2, n, num_samples)
    u += offset[0] / num_samples
    v += offset[1]
    u = 2 * u if u < 0.25 else 2 / 3 * u + 1 / 3
    theta = np.arccos(1 - 2 * u) - np.pi / 2
    phi = v * 2 * np.pi
    return [phi, theta]

def equator_cameras_sequence(i, num_views, allow_theta_fixed=False, return_radians=True):  
    """  
    根据索引 i 生成单个相机的赤道位置，phi 均匀分布，theta 可选随机。  
    
    参数:  
    i (int): 当前相机的索引（从 0 开始）。  
    num_views (int): 相机位置的总数量。  
    allow_theta_random (bool): 是否允许 theta 在 -20 到 40 度之间随机分布。  
    return_radians (bool): 是否返回弧度值（默认返回角度值）。  
    
    返回:  
    tuple: 包含 (phi, theta) 的元组，单位为度或弧度。  
    """  
    phi = (360 / num_views) * i  # 均匀分布 phi  
    if allow_theta_fixed:  
        # theta = random.uniform(-20, 40)  # theta 在 -20 到 40 度之间随机  
        theta = (20 - (-20)) * i / num_views + (-20)
    else:  
        theta = 0  # 固定 theta 为 0 度，表示赤道平面  
    
    if return_radians:  
        # 将角度转换为弧度  
        phi = math.radians(phi)  
        theta = math.radians(theta)  
    
    return [phi, theta]

def cameras_sequence_fixed_26_views():
    theta = [-45, 0, 45]
    phi = list(range(0, 360, 45))

    camera_poses = []

    for t in theta:
        for p in phi:
            camera_poses.append((math.radians(p), math.radians(t)))
    
    # top and bottom
    camera_poses.append((math.radians(0), math.radians(90)))
    camera_poses.append((math.radians(0), math.radians(-90)))
    
    return camera_poses

def sphere_hammersley_sequence(n, num_samples, offset=(0, 0)):
    u, v = hammersley_sequence(2, n, num_samples)
    u += offset[0] / num_samples
    v += offset[1]
    u = 2 * u if u < 0.25 else 2 / 3 * u + 1 / 3
    theta = np.arccos(1 - 2 * u) - np.pi / 2
    phi = v * 2 * np.pi
    return [phi, theta]

def hemisphere_hammersley_sequence(n, num_samples, offset=(0, 0)):  
    """  
    在半球面上生成 Hammersley 序列，支持偏移量  
    :param n: 点的索引  
    :param num_samples: 总采样点数  
    :param offset: 偏移量 (u_offset, v_offset)  
    :return: [phi, theta] (球面坐标)  
    """  
    # 生成 Hammersley 序列  
    u, v = hammersley_sequence(2, n, num_samples)  
    
    # 应用偏移量  
    u += offset[0] / num_samples  
    v += offset[1]  
    
    # 将 u 映射到 theta (pitch)，范围 [-pi/2, pi/2]  
    theta = np.pi * (u - 0.5)  
    
    # 将 v 映射到 phi (yaw)，范围 [-pi/2, pi/2]  
    phi = np.pi * (v - 0.5)  
    
    return [phi, theta]  

def hemisphere_uniform_sequence(range_size=0.75):
    """
    使用 np.uniform
    在半球面上均匀采样，支持偏移量
    :param num_samples: 采样点数

    return: [phi, theta] (球面坐标) shape = (num_samples, 2)
    """
    u = np.random.uniform(0, 1)
    v = np.random.uniform(0, 1)
    # u += offset[0] / num_samples
    # v += offset[1]
    # if top:
        # theta = np.pi * u * 0.5
    # else:
        # theta = np.pi * u * -0.5
    theta = np.pi * (u - 0.5) * range_size
    phi = np.pi * (v - 0.5) * range_size
    return phi, theta

if __name__ == "__main__":
    for i in range(10):
        phi, theta = hemisphere_hammersley_sequence(i, 10)
        phi = math.degrees(phi)
        theta = math.degrees(theta)
        # print(math.degrees(phi), math.degrees(theta))
        location = np.array([math.cos(phi) * math.sin(theta), math.sin(phi) * math.sin(theta), math.cos(theta)])
        print(location)

    print("=====================================")
    
    for i in range(10):
        phi = np.random.uniform(-np.pi/2, np.pi/2)
        theta = np.random.uniform(-np.pi/2, np.pi/2)

        location = np.array([math.cos(phi) * math.sin(theta), math.sin(phi) * math.sin(theta), math.cos(theta)])
        print(location)
