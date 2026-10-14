#!/usr/bin/env python3
"""
Blender ERP全景图渲染脚本

此脚本在Blender内部执行，用于渲染ERP（等距圆柱投影）全景图。

使用方法:
    blender --background --python render_erp_blender.py -- \
        --mesh "path/to/mesh.glb" \
        --output "output/panorama_0000.png" \
        --camera-pos "0.0,0.5,0.0" \
        --camera-rot "0.0,0.0,0.0" \
        --resolution "1024,512"
"""

import bpy
import sys
import os
import json
import math
import argparse
from mathutils import Vector, Euler, Quaternion


def parse_args():
    """解析命令行参数（Blender的--之后的参数）"""
    # 找到'--'之后的参数
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []
    
    parser = argparse.ArgumentParser(description="Blender ERP渲染脚本")
    parser.add_argument("--mesh", type=str, required=True,
                        help="输入mesh文件路径（GLB/GLTF/OBJ）")
    parser.add_argument("--output", type=str, required=True,
                        help="输出图像路径")
    parser.add_argument("--pose-output", type=str, default=None,
                        help="输出位姿JSON路径（默认与图像同名）")
    parser.add_argument("--camera-pos", type=str, default="0.0,0.0,0.0",
                        help="相机位置 x,y,z")
    parser.add_argument("--camera-rot", type=str, default="0.0,0.0,0.0",
                        help="相机旋转 roll,pitch,yaw（弧度）")
    parser.add_argument("--camera-rot-quat", type=str, default=None,
                        help="（已废弃）相机旋转四元数 w,x,y,z。现在主路径使用 --camera-rot Euler角")
    parser.add_argument("--resolution", type=str, default="1024,512",
                        help="渲染分辨率 width,height")
    parser.add_argument("--samples", type=int, default=16,
                        help="渲染采样数（默认16，Emission材质不需要高采样，可大幅提升渲染速度）")
    parser.add_argument("--engine", type=str, default="CYCLES",
                        choices=["BLENDER_EEVEE", "CYCLES"],
                        help="渲染引擎（全景图必须使用CYCLES）")
    parser.add_argument("--frame-id", type=int, default=0,
                        help="帧序号")
    parser.add_argument("--ref-position", type=str, default=None,
                        help="参考帧位置 x,y,z（Y-up坐标系），None表示第一帧")
    parser.add_argument("--ref-quaternion", type=str, default=None,
                        help="参考帧四元数 w,x,y,z，None表示第一帧")
    parser.add_argument("--render-depth", action="store_true",
                        help="是否渲染深度图（保存为.npy格式）")
    parser.add_argument("--depth-output", type=str, default=None,
                        help="深度图输出路径（默认与图像同目录，后缀_depth.npy）")
    
    return parser.parse_args(argv)


def clear_scene():
    """清空当前场景"""
    # 选择所有对象
    bpy.ops.object.select_all(action='SELECT')
    # 删除选中的对象
    bpy.ops.object.delete(use_global=False)
    
    # 清理孤立数据
    for block in bpy.data.meshes:
        if block.users == 0:
            bpy.data.meshes.remove(block)
    for block in bpy.data.materials:
        if block.users == 0:
            bpy.data.materials.remove(block)
    for block in bpy.data.textures:
        if block.users == 0:
            bpy.data.textures.remove(block)
    for block in bpy.data.images:
        if block.users == 0:
            bpy.data.images.remove(block)


def import_mesh(mesh_path):
    """导入mesh文件"""
    ext = os.path.splitext(mesh_path)[1].lower()
    
    if ext in ['.glb', '.gltf']:
        bpy.ops.import_scene.gltf(filepath=mesh_path)
    elif ext == '.obj':
        bpy.ops.wm.obj_import(filepath=mesh_path)
    elif ext == '.fbx':
        bpy.ops.import_scene.fbx(filepath=mesh_path)
    elif ext == '.ply':
        bpy.ops.wm.ply_import(filepath=mesh_path)
    else:
        raise ValueError(f"不支持的文件格式: {ext}")
    
    print(f"[INFO] 导入mesh: {mesh_path}")
    
    # 获取导入的对象
    imported_objects = [obj for obj in bpy.context.selected_objects]
    print(f"[INFO] 导入了 {len(imported_objects)} 个对象")
    
    # 为房间结构添加程序化纹理
    apply_procedural_textures(imported_objects)
    
    return imported_objects


def is_room_structure(obj_name):
    """
    判断对象是否是房间结构（墙面、地板、天花板）
    
    房间结构的常见命名模式：
    1. None.obj - 标准3D-Front房间结构
    2. geometry_N - 无纹理的通用几何体
    3. 纯数字.obj (如 12670.obj) - 数字ID命名的结构
    """
    name_lower = obj_name.lower()
    
    # 模式1: 包含 "none"
    if 'none' in name_lower:
        return True
    
    # 模式2: 以 "geometry_" 开头
    if name_lower.startswith('geometry_') or name_lower.startswith('geometry.'):
        return True
    
    # 模式3: 纯数字命名 (如 "12670.obj", "7319.obj")
    base_name = obj_name.replace('.obj', '').replace('.OBJ', '')
    if base_name.isdigit():
        return True
    
    return False


def apply_procedural_textures(objects):
    """为所有对象添加Emission材质（材质预览模式：显示原始颜色，不受光照影响）"""
    applied_count = 0
    for obj in objects:
        if obj.type != 'MESH':
            continue
        
        # 检查对象是否有材质，或者材质是否为空
        has_material = obj.data.materials and len(obj.data.materials) > 0 and obj.data.materials[0] is not None
        
        # 为所有对象应用Emission材质（材质预览模式）
        # 这样所有对象都会显示原始颜色，不受光照影响
        if not has_material or is_room_structure(obj.name):
            # 没有材质或者是房间结构：应用Emission材质
            print(f"[INFO] 为对象添加Emission材质: {obj.name}")
            apply_room_material(obj)
            applied_count += 1
        else:
            # 有材质：也转换为Emission材质（确保所有对象都使用Emission模式）
            print(f"[INFO] 将对象材质转换为Emission: {obj.name}")
            convert_to_emission_material(obj)
            applied_count += 1
    
    if applied_count == 0:
        print("[WARN] 未找到需要添加材质的对象")
    else:
        print(f"[INFO] 共为 {applied_count} 个对象添加了Emission材质（材质预览模式）")


def apply_room_material(obj):
    """为房间结构应用程序化材质（墙面、地板、天花板）"""
    # 创建新材质
    mat = bpy.data.materials.new(name="RoomProceduralMaterial")
    mat.use_nodes = True
    
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    
    # 清除默认节点
    nodes.clear()
    
    # 创建输出节点
    output = nodes.new('ShaderNodeOutputMaterial')
    output.location = (800, 0)
    
    # 使用几何节点获取法线
    geometry = nodes.new('ShaderNodeNewGeometry')
    geometry.location = (-600, 0)
    
    # 分离法线的Z分量
    separate_xyz = nodes.new('ShaderNodeSeparateXYZ')
    separate_xyz.location = (-400, 0)
    links.new(geometry.outputs['Normal'], separate_xyz.inputs['Vector'])
    
    # === 判断天花板（法线Z < -0.5，朝下的面） ===
    ceiling_check = nodes.new('ShaderNodeMath')
    ceiling_check.operation = 'LESS_THAN'
    ceiling_check.inputs[1].default_value = -0.5
    ceiling_check.location = (-200, 100)
    links.new(separate_xyz.outputs['Z'], ceiling_check.inputs[0])
    
    # === 判断地板（法线Z > 0.5，朝上的面） ===
    floor_check = nodes.new('ShaderNodeMath')
    floor_check.operation = 'GREATER_THAN'
    floor_check.inputs[1].default_value = 0.5
    floor_check.location = (-200, -100)
    links.new(separate_xyz.outputs['Z'], floor_check.inputs[0])
    
    # === 创建三种材质 ===
    floor_shader = create_wood_floor_material(nodes, links)
    floor_shader.location = (0, 300)
    
    wall_shader = create_brick_wall_material(nodes, links)  # 砖墙材质
    wall_shader.location = (0, 0)
    
    ceiling_shader = create_grid_ceiling_material(nodes, links)  # 网格天花板
    ceiling_shader.location = (0, -300)
    
    # === 混合着色器：先混合地板和墙面 ===
    mix_floor_wall = nodes.new('ShaderNodeMixShader')
    mix_floor_wall.location = (300, 100)
    links.new(floor_check.outputs['Value'], mix_floor_wall.inputs['Fac'])
    links.new(wall_shader.outputs['Emission'], mix_floor_wall.inputs[1])  # Emission材质输出
    links.new(floor_shader.outputs['Emission'], mix_floor_wall.inputs[2])  # Emission材质输出
    
    # === 混合着色器：再混合天花板 ===
    mix_final = nodes.new('ShaderNodeMixShader')
    mix_final.location = (500, 0)
    links.new(ceiling_check.outputs['Value'], mix_final.inputs['Fac'])
    links.new(mix_floor_wall.outputs['Shader'], mix_final.inputs[1])
    links.new(ceiling_shader.outputs['Emission'], mix_final.inputs[2])  # Emission材质输出
    
    # 连接输出
    links.new(mix_final.outputs['Shader'], output.inputs['Surface'])
    
    # 应用材质到对象
    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)
    
    print(f"[INFO] Emission材质已应用（地板+砖墙+网格天花板，材质预览模式）")


def convert_to_emission_material(obj):
    """将现有材质转换为Emission材质（材质预览模式）"""
    if not obj.data.materials or len(obj.data.materials) == 0:
        # 如果没有材质，直接应用房间材质
        apply_room_material(obj)
        return
    
    # 获取现有材质
    existing_mat = obj.data.materials[0]
    if existing_mat is None:
        apply_room_material(obj)
        return
    
    # 如果材质已经有节点，尝试提取Base Color并转换为Emission
    if existing_mat.use_nodes:
        nodes = existing_mat.node_tree.nodes
        links = existing_mat.node_tree.links
        
        # 查找Principled BSDF节点
        bsdf_node = None
        for node in nodes:
            if node.type == 'BSDF_PRINCIPLED':
                bsdf_node = node
                break
        
        if bsdf_node and 'Base Color' in bsdf_node.inputs:
            # 找到Base Color输入
            base_color_input = bsdf_node.inputs['Base Color']
            
            # 创建Emission节点
            emission = nodes.new('ShaderNodeEmission')
            emission.name = "Emission"
            emission.location = bsdf_node.location
            
            # 获取Base Color的值或连接
            if base_color_input.is_linked:
                # 如果有连接，连接到Emission
                color_source = base_color_input.links[0].from_node
                color_output = base_color_input.links[0].from_socket
                links.new(color_output, emission.inputs['Color'])
            else:
                # 如果没有连接，使用默认值
                emission.inputs['Color'].default_value = base_color_input.default_value
            
            emission.inputs['Strength'].default_value = 1.0  # Emission强度（材质预览模式，避免过曝）
            
            # 找到输出节点并连接
            output_node = None
            for node in nodes:
                if node.type == 'OUTPUT_MATERIAL':
                    output_node = node
                    break
            
            if output_node:
                # 断开原有连接
                if output_node.inputs['Surface'].is_linked:
                    for link in output_node.inputs['Surface'].links:
                        existing_mat.node_tree.links.remove(link)
                # 连接Emission
                links.new(emission.outputs['Emission'], output_node.inputs['Surface'])
                print(f"[INFO] 已将材质转换为Emission: {obj.name}")
                return
    
    # 如果无法转换，直接应用房间材质
    apply_room_material(obj)


def create_wood_floor_material(nodes, links):
    """创建木地板程序化材质（Emission模式，显示原始颜色）"""
    # 使用Emission材质，直接发光，不受光照影响
    emission = nodes.new('ShaderNodeEmission')
    emission.name = "FloorEmission"
    
    # 木纹噪波纹理
    noise = nodes.new('ShaderNodeTexNoise')
    noise.inputs['Scale'].default_value = 20.0
    noise.inputs['Detail'].default_value = 8.0
    noise.inputs['Roughness'].default_value = 0.6
    noise.location = (-600, 200)
    
    # 波浪纹理（模拟木纹条纹）
    wave = nodes.new('ShaderNodeTexWave')
    wave.wave_type = 'BANDS'
    wave.bands_direction = 'X'
    wave.inputs['Scale'].default_value = 3.0
    wave.inputs['Distortion'].default_value = 5.0
    wave.inputs['Detail'].default_value = 3.0
    wave.location = (-600, 0)
    
    # 颜色渐变（木材颜色）
    color_ramp = nodes.new('ShaderNodeValToRGB')
    color_ramp.color_ramp.elements[0].color = (0.15, 0.08, 0.04, 1.0)  # 深棕色
    color_ramp.color_ramp.elements[1].color = (0.35, 0.20, 0.10, 1.0)  # 浅棕色
    color_ramp.location = (-400, 100)
    
    # 混合噪波和波浪
    mix_rgb = nodes.new('ShaderNodeMix')
    mix_rgb.data_type = 'RGBA'
    mix_rgb.inputs['Factor'].default_value = 0.5
    mix_rgb.location = (-400, 0)
    
    links.new(noise.outputs['Fac'], mix_rgb.inputs['A'])
    links.new(wave.outputs['Fac'], mix_rgb.inputs['B'])
    links.new(mix_rgb.outputs['Result'], color_ramp.inputs['Fac'])
    
    # 连接到Emission材质（直接显示颜色，不受光照影响）
    links.new(color_ramp.outputs['Color'], emission.inputs['Color'])
    emission.inputs['Strength'].default_value = 1.0  # Emission强度（材质预览模式，避免过曝）
    
    return emission


def create_brick_wall_material(nodes, links):
    """创建砖墙程序化材质（Emission模式，显示原始颜色）"""
    # 使用Emission材质，直接发光，不受光照影响
    emission = nodes.new('ShaderNodeEmission')
    emission.name = "BrickWallEmission"
    
    # 使用纹理坐标
    tex_coord = nodes.new('ShaderNodeTexCoord')
    tex_coord.location = (-800, 0)
    
    # 缩放映射（控制砖块大小）
    mapping = nodes.new('ShaderNodeMapping')
    mapping.inputs['Scale'].default_value = (4.0, 8.0, 1.0)  # X方向砖块较宽
    mapping.location = (-600, 0)
    links.new(tex_coord.outputs['Generated'], mapping.inputs['Vector'])
    
    # 砖块纹理
    brick = nodes.new('ShaderNodeTexBrick')
    brick.inputs['Color1'].default_value = (0.6, 0.3, 0.2, 1.0)  # 砖红色
    brick.inputs['Color2'].default_value = (0.5, 0.25, 0.15, 1.0)  # 深砖红色
    brick.inputs['Mortar'].default_value = (0.85, 0.85, 0.8, 1.0)  # 灰白色砂浆
    brick.inputs['Scale'].default_value = 3.0
    brick.inputs['Mortar Size'].default_value = 0.02
    brick.inputs['Mortar Smooth'].default_value = 0.1
    brick.inputs['Bias'].default_value = 0.0
    brick.inputs['Brick Width'].default_value = 0.5
    brick.inputs['Row Height'].default_value = 0.25
    brick.location = (-400, 0)
    links.new(mapping.outputs['Vector'], brick.inputs['Vector'])
    
    # 添加细微噪波增加真实感
    noise = nodes.new('ShaderNodeTexNoise')
    noise.inputs['Scale'].default_value = 50.0
    noise.inputs['Detail'].default_value = 3.0
    noise.location = (-400, -200)
    links.new(mapping.outputs['Vector'], noise.inputs['Vector'])
    
    # 混合砖块颜色和噪波
    mix_color = nodes.new('ShaderNodeMix')
    mix_color.data_type = 'RGBA'
    mix_color.inputs['Factor'].default_value = 0.1
    mix_color.location = (-200, 0)
    links.new(brick.outputs['Color'], mix_color.inputs['A'])
    links.new(noise.outputs['Color'], mix_color.inputs['B'])
    
    # 连接到Emission材质（直接显示颜色，不受光照影响）
    links.new(mix_color.outputs['Result'], emission.inputs['Color'])
    emission.inputs['Strength'].default_value = 2.0  # Emission强度（材质预览模式，避免过曝）
    
    return emission


def create_grid_ceiling_material(nodes, links):
    """创建网格天花板程序化材质（Emission模式，显示原始颜色）"""
    # 使用Emission材质，直接发光，不受光照影响
    emission = nodes.new('ShaderNodeEmission')
    emission.name = "GridCeilingEmission"
    
    # 使用纹理坐标
    tex_coord = nodes.new('ShaderNodeTexCoord')
    tex_coord.location = (-800, -400)
    
    # 缩放映射
    mapping = nodes.new('ShaderNodeMapping')
    mapping.inputs['Scale'].default_value = (5.0, 5.0, 1.0)
    mapping.location = (-600, -400)
    links.new(tex_coord.outputs['Generated'], mapping.inputs['Vector'])
    
    # 分离XY坐标
    separate = nodes.new('ShaderNodeSeparateXYZ')
    separate.location = (-400, -400)
    links.new(mapping.outputs['Vector'], separate.inputs['Vector'])
    
    # X方向网格线（使用正弦波）
    math_sin_x = nodes.new('ShaderNodeMath')
    math_sin_x.operation = 'SINE'
    math_sin_x.location = (-200, -350)
    
    math_mul_x = nodes.new('ShaderNodeMath')
    math_mul_x.operation = 'MULTIPLY'
    math_mul_x.inputs[1].default_value = 6.28  # 2*PI
    math_mul_x.location = (-300, -350)
    links.new(separate.outputs['X'], math_mul_x.inputs[0])
    links.new(math_mul_x.outputs['Value'], math_sin_x.inputs[0])
    
    # Y方向网格线
    math_sin_y = nodes.new('ShaderNodeMath')
    math_sin_y.operation = 'SINE'
    math_sin_y.location = (-200, -500)
    
    math_mul_y = nodes.new('ShaderNodeMath')
    math_mul_y.operation = 'MULTIPLY'
    math_mul_y.inputs[1].default_value = 6.28
    math_mul_y.location = (-300, -500)
    links.new(separate.outputs['Y'], math_mul_y.inputs[0])
    links.new(math_mul_y.outputs['Value'], math_sin_y.inputs[0])
    
    # 取绝对值使线条清晰
    abs_x = nodes.new('ShaderNodeMath')
    abs_x.operation = 'ABSOLUTE'
    abs_x.location = (-100, -350)
    links.new(math_sin_x.outputs['Value'], abs_x.inputs[0])
    
    abs_y = nodes.new('ShaderNodeMath')
    abs_y.operation = 'ABSOLUTE'
    abs_y.location = (-100, -500)
    links.new(math_sin_y.outputs['Value'], abs_y.inputs[0])
    
    # 合并X和Y网格（取最小值形成网格交叉）
    math_min = nodes.new('ShaderNodeMath')
    math_min.operation = 'MINIMUM'
    math_min.location = (0, -425)
    links.new(abs_x.outputs['Value'], math_min.inputs[0])
    links.new(abs_y.outputs['Value'], math_min.inputs[1])
    
    # 颜色渐变：网格线深色，格子浅色
    color_ramp = nodes.new('ShaderNodeValToRGB')
    color_ramp.color_ramp.elements[0].color = (0.3, 0.3, 0.35, 1.0)  # 深灰色网格线
    color_ramp.color_ramp.elements[0].position = 0.0
    color_ramp.color_ramp.elements[1].color = (0.95, 0.95, 0.95, 1.0)  # 白色格子
    color_ramp.color_ramp.elements[1].position = 0.15
    color_ramp.location = (150, -425)
    links.new(math_min.outputs['Value'], color_ramp.inputs['Fac'])
    
    # 连接到Emission材质（直接显示颜色，不受光照影响）
    links.new(color_ramp.outputs['Color'], emission.inputs['Color'])
    emission.inputs['Strength'].default_value = 2.0  # Emission强度（材质预览模式，避免过曝）
    
    return emission


def get_scene_bounds():
    """获取场景中所有物体的边界框"""
    min_coords = [float('inf'), float('inf'), float('inf')]
    max_coords = [float('-inf'), float('-inf'), float('-inf')]
    
    for obj in bpy.context.scene.objects:
        if obj.type == 'MESH':
            # 获取世界坐标下的边界框
            for corner in obj.bound_box:
                world_corner = obj.matrix_world @ Vector(corner)
                for i in range(3):
                    min_coords[i] = min(min_coords[i], world_corner[i])
                    max_coords[i] = max(max_coords[i], world_corner[i])
    
    # 如果没有找到任何mesh，返回默认值
    if min_coords[0] == float('inf'):
        return ([-5, -5, 0], [5, 5, 3])
    
    return (min_coords, max_coords)


def create_erp_camera(name="ERP_Camera"):
    """创建ERP全景相机"""
    # 创建相机数据
    camera_data = bpy.data.cameras.new(name=name)
    
    # 设置为全景相机
    camera_data.type = 'PANO'
    
    # 设置全景类型为等距圆柱投影（EEVEE和Cycles都支持）
    # Blender 5.0 使用 panorama_type
    if hasattr(camera_data, 'panorama_type'):
        camera_data.panorama_type = 'EQUIRECTANGULAR'
    # Cycles相机设置
    if hasattr(camera_data, 'cycles'):
        camera_data.cycles.panorama_type = 'EQUIRECTANGULAR'
    
    # 创建相机对象
    camera_object = bpy.data.objects.new(name, camera_data)
    
    # 链接到场景
    bpy.context.scene.collection.objects.link(camera_object)
    
    print(f"[INFO] 创建ERP相机: {name}")
    
    return camera_object


def setup_camera(camera_object, position, rotation_euler=None, rotation_quat=None):
    """设置相机位置和旋转（Euler 或 Quaternion）"""
    # 设置位置
    camera_object.location = Vector(position)
    
    # 设置旋转
    if rotation_quat is not None:
        # 使用四元数（推荐，避免Euler顺序/分解歧义）
        camera_object.rotation_mode = 'QUATERNION'
        camera_object.rotation_quaternion = Quaternion(rotation_quat)
        print(f"[INFO] 相机位置: {position}")
        print(f"[INFO] 相机旋转(Quaternion wxyz): {list(rotation_quat)}")
    else:
        # 使用欧拉角（Blender使用XYZ顺序的欧拉角）
        camera_object.rotation_mode = 'XYZ'
        camera_object.rotation_euler = Euler(rotation_euler, 'XYZ')
        print(f"[INFO] 相机位置: {position}")
        print(f"[INFO] 相机旋转(Euler XYZ, rad): {rotation_euler}")


def setup_render_settings(resolution, engine, samples):
    """设置渲染参数"""
    scene = bpy.context.scene
    
    # 设置渲染引擎
    scene.render.engine = engine
    print(f"[INFO] 渲染引擎: {engine}")
    
    # 设置分辨率
    scene.render.resolution_x = resolution[0]
    scene.render.resolution_y = resolution[1]
    scene.render.resolution_percentage = 100
    print(f"[INFO] 分辨率: {resolution[0]}x{resolution[1]}")
    
    # 设置输出格式
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGB'
    scene.render.image_settings.color_depth = '8'
    
    # 引擎特定设置
    if engine == 'BLENDER_EEVEE':
        # EEVEE设置
        if hasattr(scene, 'eevee'):
            # 设置采样数（如果属性存在）
            if hasattr(scene.eevee, 'taa_render_samples'):
                scene.eevee.taa_render_samples = samples
            # 软阴影（Blender 5.0可能不支持）
            if hasattr(scene.eevee, 'use_soft_shadows'):
                scene.eevee.use_soft_shadows = True
    elif engine == 'CYCLES':
        # Cycles设置
        scene.cycles.samples = samples
        scene.cycles.use_denoising = True
        
        # 对于Emission材质，需要确保光线反弹足够
        # 但Emission材质本身会发光，不需要太多反弹
        scene.cycles.max_bounces = 4  # 减少反弹次数（Emission材质不需要太多）
        scene.cycles.diffuse_bounces = 2
        scene.cycles.glossy_bounces = 2
        scene.cycles.transmission_bounces = 2
        
        # 尝试使用GPU
        try:
            bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'
            bpy.context.scene.cycles.device = 'GPU'
            print("[INFO] 使用GPU渲染")
        except:
            print("[INFO] 使用CPU渲染")


def setup_lighting(camera_position=None, scene_bounds=None):
    """
    设置照明（材质预览模式：仅使用强环境光，无其他光源，显示材质原始颜色和亮度）
    
    Args:
        camera_position: 相机位置 (x, y, z)（未使用）
        scene_bounds: 场景边界 (min, max)（未使用）
    """
    scene = bpy.context.scene
    
    # 添加环境光（材质预览模式）
    world = bpy.data.worlds.new("World")
    scene.world = world
    world.use_nodes = True
    
    # 获取节点树
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    
    # 清除默认节点
    nodes.clear()
    
    # 创建背景节点 - 使用非常强的环境光（类似材质预览模式）
    # 只使用环境光，无其他光源，确保材质显示原始颜色和亮度，无明暗变化
    background = nodes.new('ShaderNodeBackground')
    background.inputs['Color'].default_value = (1.0, 1.0, 1.0, 1.0)  # 白色背景
    background.inputs['Strength'].default_value = 1.0  # 环境光强度（材质预览模式，避免过曝）
    
    # 创建输出节点
    output = nodes.new('ShaderNodeOutputWorld')
    
    # 连接节点
    links.new(background.outputs['Background'], output.inputs['Surface'])
    
    # === 不添加任何其他光源 ===
    # 只使用环境光，确保整个场景光照完全均匀，无距离衰减，无明暗变化
    # 这样材质会显示其原始颜色和亮度，就像Blender材质预览模式一样
    
    print("[INFO] 设置照明完成（材质预览模式：仅强环境光，无其他光源，显示材质原始颜色和亮度）")


def setup_depth_pass():
    """
    设置深度渲染 pass（Blender 5.0+ API）
    
    在 Blender 中启用 Z pass，用于获取深度信息。
    使用 Blender 5.0 新的 compositing_node_group API。
    """
    scene = bpy.context.scene
    
    # 启用 View Layer 的 Z pass
    view_layer = bpy.context.view_layer
    view_layer.use_pass_z = True
    
    # Blender 5.0: scene.node_tree 已移除，改用 compositing_node_group
    # 创建新的 CompositorNodeTree 并赋给场景
    tree = bpy.data.node_groups.new("DepthCompositor", "CompositorNodeTree")
    scene.compositing_node_group = tree
    nodes = tree.nodes
    links = tree.links
    
    # 创建 Render Layers 节点
    render_layers = nodes.new('CompositorNodeRLayers')
    render_layers.location = (0, 300)
    
    # Blender 5.0: 用 NodeGroupOutput 替代 CompositorNodeComposite
    output = nodes.new('NodeGroupOutput')
    output.location = (400, 300)
    tree.interface.new_socket(name="Image", in_out="OUTPUT", socket_type="NodeSocketColor")
    
    # 连接 RGB 输出
    links.new(render_layers.outputs['Image'], output.inputs['Image'])
    
    # 创建 File Output 节点（用于深度 EXR）
    file_output = nodes.new('CompositorNodeOutputFile')
    file_output.location = (400, 0)
    file_output.directory = ""  # 稍后在渲染时设置（Blender 5.0: 替代 base_path）
    file_output.format.media_type = 'IMAGE'  # Blender 5.0: 必须先设 media_type
    file_output.format.file_format = 'OPEN_EXR'
    file_output.format.color_depth = '32'
    file_output.format.exr_codec = 'ZIP'
    
    # Blender 5.0: file_output_items 替代 file_slots
    file_output.file_output_items.clear()
    file_output.file_output_items.new('FLOAT', "depth")
    
    # 连接深度输出
    links.new(render_layers.outputs['Depth'], file_output.inputs['depth'])
    
    print("[INFO] 深度 pass 已启用（Blender 5.0 API）")
    
    return file_output


def _convert_depth_exr_via_blender_api(exr_path, npy_path):
    """使用 Blender 图像 API 将 EXR 转为 NPY（备用路径，不依赖 OpenEXR）。"""
    import numpy as np
    img = bpy.data.images.load(exr_path)
    width = img.size[0]
    height = img.size[1]
    pixels = np.array(img.pixels[:])
    pixels = pixels.reshape(height, width, -1)
    depth = pixels[:, :, 0]
    depth = np.flipud(depth)
    unit_scale = bpy.context.scene.unit_settings.scale_length
    depth_meters = depth * unit_scale
    max_valid_depth = 1000.0
    depth_meters[depth_meters > max_valid_depth] = np.nan
    depth_meters[depth_meters <= 0] = np.nan
    np.save(npy_path, depth_meters.astype(np.float32))
    bpy.data.images.remove(img)
    os.remove(exr_path)
    print(f"[OK] 深度图保存（备用方法）: {npy_path}")


def convert_depth_exr_to_npy(exr_path, npy_path):
    """
    将 Blender 渲染的深度 EXR 转换为 NPY 格式
    
    Blender ERP 相机的深度是 range depth（射线距离），单位为 Blender 单位（通常是米）
    
    Args:
        exr_path: EXR 文件路径
        npy_path: NPY 输出路径
    """
    import numpy as np
    try:
        import OpenEXR
        import Imath
        
        # 打开 EXR 文件
        exr_file = OpenEXR.InputFile(exr_path)
        
        # 获取图像尺寸
        header = exr_file.header()
        dw = header['dataWindow']
        width = dw.max.x - dw.min.x + 1
        height = dw.max.y - dw.min.y + 1
        
        # 读取深度通道
        # Blender 深度 pass 保存在 'R'、'G'、'B' 或 'V' 通道
        pt = Imath.PixelType(Imath.PixelType.FLOAT)
        
        # 尝试不同的通道名称
        channel_names = ['depth.R', 'R', 'V', 'Z', 'depth.V']
        depth_str = None
        for ch in channel_names:
            if ch in header['channels']:
                depth_str = exr_file.channel(ch, pt)
                print(f"[INFO] 使用深度通道: {ch}")
                break
        
        if depth_str is None:
            # 列出所有可用通道
            available_channels = list(header['channels'].keys())
            print(f"[WARN] 可用通道: {available_channels}")
            # 尝试使用第一个通道
            if available_channels:
                depth_str = exr_file.channel(available_channels[0], pt)
                print(f"[INFO] 使用通道: {available_channels[0]}")
            else:
                raise ValueError("无法找到深度通道")
        
        # 转换为 numpy 数组
        depth = np.frombuffer(depth_str, dtype=np.float32)
        depth = depth.reshape(height, width)
        
        # 获取场景单位比例（转换为米）
        unit_scale = bpy.context.scene.unit_settings.scale_length
        
        # 将深度转换为米
        # Blender 的深度值是场景单位，需要乘以 unit_scale 转换为米
        depth_meters = depth * unit_scale
        
        # 处理无效深度（Blender 用非常大的值表示无穷远）
        # 通常 > 1e9 的值表示背景/无穷远
        max_valid_depth = 1000.0  # 1000 米以上视为无效
        depth_meters[depth_meters > max_valid_depth] = np.nan
        depth_meters[depth_meters <= 0] = np.nan
        
        # 保存为 NPY
        np.save(npy_path, depth_meters.astype(np.float32))
        
        # 删除临时 EXR 文件
        os.remove(exr_path)
        
        # 统计信息
        valid_mask = np.isfinite(depth_meters)
        if np.any(valid_mask):
            print(f"[OK] 深度图保存: {npy_path}")
            print(f"     形状: {depth_meters.shape}")
            print(f"     深度范围: {np.nanmin(depth_meters):.3f} - {np.nanmax(depth_meters):.3f} 米")
            print(f"     有效像素: {np.sum(valid_mask)} / {depth_meters.size} ({100*np.sum(valid_mask)/depth_meters.size:.1f}%)")
        else:
            print(f"[WARN] 深度图全部无效!")
            
    except ImportError:
        print("[ERROR] 需要安装 OpenEXR 库: pip install OpenEXR")
        print("[INFO] 尝试使用 Blender 内置方法...")
        try:
            _convert_depth_exr_via_blender_api(exr_path, npy_path)
        except Exception as e:
            print(f"[ERROR] 备用方法也失败: {e}")
            print(f"[INFO] EXR 文件保留在: {exr_path}")
    except Exception as e:
        print(f"[WARN] OpenEXR 读取失败，尝试 Blender 内置方法: {e}")
        try:
            _convert_depth_exr_via_blender_api(exr_path, npy_path)
        except Exception as e2:
            print(f"[ERROR] 备用方法也失败: {e2}")
            print(f"[INFO] EXR 文件保留在: {exr_path}")


def render_and_save(output_path, render_depth=False, depth_output=None):
    """
    执行渲染并保存
    
    Args:
        output_path: RGB 图像输出路径
        render_depth: 是否渲染深度
        depth_output: 深度图输出路径（.npy 格式）
    """
    # 确保输出目录存在
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 设置输出路径
    bpy.context.scene.render.filepath = output_path
    
    # 执行渲染
    print(f"[INFO] 开始渲染...")
    bpy.ops.render.render(write_still=True)

    # 强校验：必须有实际输出图像，避免上游出现“returncode=0但无文件”
    if (not os.path.exists(output_path)) or os.path.getsize(output_path) <= 0:
        raise RuntimeError(f"渲染完成但输出图像不存在或为空: {output_path}")

    print(f"[OK] 渲染完成: {output_path}")


def save_pose(camera_object, output_path, frame_id=0, ref_position=None, ref_quaternion=None):
    """
    保存相机位姿（绝对位姿，兼容 ERPT 格式）
    
    输出格式：
    - position: 相机中心在世界坐标系的绝对位置（米），[X右, Y上, Z前]
    - rotation_quaternion: [w, x, y, z]，camera->world 旋转 (R_cw)
    
    核心公式：R_cw_erpt = T @ R_blender_obj @ M
    - T: Blender世界(Y前Z上) -> 统一世界(Y上Z前) 坐标轴交换
    - R_blender_obj: Blender相机的旋转矩阵（object local -> world）
    - M: Blender相机本地(-Z前) -> ERPT相机(+Z前) Z轴翻转
    
    Args:
        camera_object: Blender相机对象
        output_path: 输出路径
        frame_id: 帧序号
        ref_position: （保留参数，当前未使用）
        ref_quaternion: （保留参数，当前未使用）
    """
    from mathutils import Matrix
    
    # 获取当前相机的绝对位置和旋转（Blender坐标系：X右, Y前, Z上）
    abs_position_blender = list(camera_object.location)
    abs_quat_blender = camera_object.rotation_euler.to_quaternion()
    
    # === 位置转换 ===
    # Blender世界(X右,Y前,Z上) -> 统一标准(X右,Y上,Z前)
    abs_position_unified = [
        abs_position_blender[0],   # X_unified = X_blender
        abs_position_blender[2],   # Y_unified = Z_blender (上)
        abs_position_blender[1]    # Z_unified = Y_blender (前)
    ]
    
    # === 旋转转换 ===
    # Blender object rotation matrix (local -> world in Blender coords)
    R_obj_blender = abs_quat_blender.to_matrix()
    
    # T: Blender世界坐标 -> 统一世界坐标（交换Y和Z轴）
    T_blender_to_unified = Matrix([
        [1, 0, 0],  # X不变
        [0, 0, 1],  # Y_unified = Z_blender
        [0, 1, 0]   # Z_unified = Y_blender
    ])
    
    # M: Blender相机本地坐标 -> ERPT相机坐标（翻转Z轴）
    # Blender相机沿 -Z_local 看，ERPT相机沿 +Z_camera 看
    # 因此 ERPT_Z = -Blender_Z_local，即 Z 轴翻转
    M_cam = Matrix([
        [1, 0,  0],
        [0, 1,  0],
        [0, 0, -1]
    ])
    
    # 核心公式：R_cw_erpt = T @ R_obj_blender @ M
    # 含义：ERPT相机坐标 -> (M) -> Blender本地 -> (R_obj) -> Blender世界 -> (T) -> 统一世界
    R_cw_erpt = T_blender_to_unified @ R_obj_blender @ M_cam
    
    # 转换为四元数（cam_to_world，ERPT期望的格式）
    quat_cw = R_cw_erpt.to_quaternion()
    abs_quaternion_cw = [quat_cw.w, quat_cw.x, quat_cw.y, quat_cw.z]
    
    # === 输出 ===
    # 绝对位姿，cam_to_world格式，兼容ERPT
    pose_data = {
        "frame_id": frame_id,
        "position": abs_position_unified,
        "rotation_quaternion": abs_quaternion_cw,
        "camera_type": "erp_ray",
        "coordinate_system": "right-handed, Y-up, Z-forward (cam_to_world)",
        "render_method": "blender_cycles"
    }
    
    # 保存JSON
    with open(output_path, 'w') as f:
        json.dump(pose_data, f, indent=2)

    if (not os.path.exists(output_path)) or os.path.getsize(output_path) <= 0:
        raise RuntimeError(f"位姿文件写入失败或为空: {output_path}")
    
    print(f"[OK] 位姿保存: {output_path}")
    print(f"  Position (absolute, meters): {abs_position_unified}")
    print(f"  Rotation (cam_to_world): {abs_quaternion_cw}")
    
    # 返回绝对位姿（统一标准坐标系，cam_to_world）
    return abs_position_unified, abs_quaternion_cw


def main():
    # 解析参数
    args = parse_args()
    
    # 解析相机位置
    camera_pos = [float(x) for x in args.camera_pos.split(',')]
    camera_rot = [float(x) for x in args.camera_rot.split(',')]
    camera_rot_quat = None
    if args.camera_rot_quat:
        camera_rot_quat = [float(x) for x in args.camera_rot_quat.split(',')]
    resolution = [int(x) for x in args.resolution.split(',')]
    
    # 解析参考帧位姿
    ref_position = None
    ref_quaternion = None
    if args.ref_position:
        ref_position = [float(x) for x in args.ref_position.split(',')]
    if args.ref_quaternion:
        ref_quaternion = [float(x) for x in args.ref_quaternion.split(',')]
    
    # 确定位姿输出路径
    if args.pose_output:
        pose_output = args.pose_output
    else:
        pose_output = os.path.splitext(args.output)[0] + '_pose.json'
    
    print("=" * 60)
    print("Blender ERP渲染")
    print("=" * 60)
    
    # 1. 清空场景
    print("\n[1/6] 清空场景...")
    clear_scene()
    
    # 2. 导入mesh
    print("\n[2/6] 导入mesh...")
    import_mesh(args.mesh)
    
    # 3. 创建ERP相机
    print("\n[3/6] 创建ERP相机...")
    camera = create_erp_camera()
    setup_camera(camera, camera_pos, rotation_euler=camera_rot, rotation_quat=camera_rot_quat)
    bpy.context.scene.camera = camera
    
    # 获取场景边界（用于灯光设置）
    scene_bounds = get_scene_bounds()
    print(f"[INFO] 场景边界: min={scene_bounds[0]}, max={scene_bounds[1]}")
    
    # 4. 设置渲染参数
    print("\n[4/6] 设置渲染参数...")
    setup_render_settings(resolution, args.engine, args.samples)
    setup_lighting(camera_position=camera_pos, scene_bounds=scene_bounds)
    
    # 5. 渲染
    print("\n[5/6] 渲染中...")
    render_and_save(args.output)
    
    # 6. 保存位姿（相对于第一帧）
    print("\n[6/6] 保存位姿...")
    abs_pos, abs_quat = save_pose(
        camera, 
        pose_output, 
        frame_id=args.frame_id,
        ref_position=ref_position,
        ref_quaternion=ref_quaternion
    )
    
    # 输出绝对位姿供批量脚本使用
    print(f"[ABS_POSE] {abs_pos[0]},{abs_pos[1]},{abs_pos[2]}|{abs_quat[0]},{abs_quat[1]},{abs_quat[2]},{abs_quat[3]}")
    
    print("\n" + "=" * 60)
    print("渲染完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()

