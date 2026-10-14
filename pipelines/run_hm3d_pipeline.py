#!/usr/bin/env python3
"""
Blend 全流程 Pipeline v5（单 Blender 进程）

v3 → v4 改进:
  - Phase 0+1+2 全部在一个 Blender 进程内完成
  - 不再导出 GLB（用 Blender scene.ray_cast 替代 trimesh）
  - 不再每帧重启 Blender（同进程内移动相机 + 渲染）
  - 去掉 trimesh 外部依赖

v5 → v6 改进:
  - 新增 GLB/GLTF 格式支持（--glb 参数）
  - --blend / --glb 二选一，支持 .blend .glb .gltf 三种格式
  - GLB 导入后与 .blend 流程完全统一

对外接口 100% 兼容 v3/v5:
  - 命令行参数完全一致（新增 --glb 为可选补充）
  - 输出文件名完全一致（panorama_XXXX.png / _depth.npy / pose_XXXX.json）
  - run_full_pipeline.py 零改动

双模式运行:
  1) python run_blend_pipeline.py --blender X --blend Y ...
     python run_blend_pipeline.py --blender X --glb Y ...
     → 检测到 --blender → 启动 blender --python THIS_FILE -- --blend/--glb Y ...
  2) Blender 内部自动进入 in-process 模式
     → Phase 0 (边界) + Phase 1 (撒点+过滤) + Phase 2 (边渲边选)
"""

# =====================================================================
# 检测运行环境
# =====================================================================
try:
    import bpy
    from mathutils import Vector, Euler, Matrix
    IN_BLENDER = True
except ImportError:
    IN_BLENDER = False

import argparse
import json
import math
import os
import subprocess
import sys
import time
import random as _random
from pathlib import Path

import numpy as np


# =====================================================================
# 常量
# =====================================================================

WARP_H = 128
WARP_W = 256
MARGIN = 0.5  # 距墙最小安全距离（防穿模）

DEFAULT_STOP_GAIN = 0.08
DEFAULT_OVERLAP_PENALTY = 0.5
DEFAULT_MIN_DIST = 0.6
DEFAULT_MIN_FRAMES = 5

ROTATION_TYPES = {
    "none":         [0.0, 0.0, 0.0],
    "rotate_x_90":  [math.pi / 2, 0.0, 0.0],
    "rotate_x_180": [math.pi, 0.0, 0.0],
    "rotate_z_90":  [0.0, 0.0, math.pi / 2],
}


def get_camera_rot(rotation_type: str, frame_id: int):
    if rotation_type == "random_yaw":
        yaw = 0.0 if frame_id == 0 else _random.uniform(0, 2 * math.pi)
        return [math.pi / 2, 0.0, yaw]
    return list(ROTATION_TYPES[rotation_type])


#######################################################################
# 渲染设置
#######################################################################
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


# =====================================================================
# 参数解析（兼容两种模式）
# =====================================================================

def parse_args_python():
    """Python 模式: 需要 --blender"""
    parser = argparse.ArgumentParser(description="Blend Pipeline v5（边渲边选）")
    parser.add_argument("--blender", type=str, required=True)
    scene_grp = parser.add_mutually_exclusive_group(required=True)
    scene_grp.add_argument("--blend", type=str, default=None,
                           help=".blend 场景文件路径")
    scene_grp.add_argument("--glb", type=str, default=None,
                           help=".glb / .gltf 场景文件路径")
    # parser.add_argument("--exposure_rate", type=float, default=0.5, help="多次曝光时，前后两次曝光的差距阈值")
    # parser.add_argument("--exposure_count", type=int, default=5, help="最大曝光次数")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--num-frames", type=int, default=30)
    parser.add_argument("--render-depth", action="store_true")
    parser.add_argument("--resolution", type=str, default="2048,1024")
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--engine", type=str, default="CYCLES")
    parser.add_argument("--exposure", type=float, default=0.0)
    parser.add_argument("--grid-spacing", type=float, default=0.5)
    parser.add_argument("--camera-height", type=float, default=None)
    parser.add_argument("--stop-gain", type=float, default=DEFAULT_STOP_GAIN)
    parser.add_argument("--stop-score", type=float, default=-0.3)
    parser.add_argument("--stop-delta", type=float, default=0.08)
    parser.add_argument("--min-frames", type=int, default=DEFAULT_MIN_FRAMES)
    parser.add_argument("--rotation-type", type=str, default="random_yaw",
                        choices=["none", "rotate_x_90", "rotate_x_180",
                                 "rotate_z_90", "random_yaw"])
    parser.add_argument("--gain-curve", action="store_true", default=True)
    parser.add_argument("--no-gain-curve", dest="gain_curve", action="store_false")
    parser.add_argument("--hm3d", type=bool, default=False, help="是否启用 HM3D 数据集的房间搜索和场景加载逻辑")
    return parser.parse_args()


def parse_args_blender():
    """Blender 模式: 不需要 --blender"""
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []
    parser = argparse.ArgumentParser()
    scene_grp = parser.add_mutually_exclusive_group(required=True)
    scene_grp.add_argument("--blend", type=str, default=None,
                           help=".blend 场景文件路径")
    scene_grp.add_argument("--glb", type=str, default=None,
                           help=".glb / .gltf 场景文件路径")
    # parser.add_argument("--exposure_rate", type=float, default=0.5, help="多次曝光时，前后两次曝光的差距阈值")
    # parser.add_argument("--exposure_count", type=int, default=5, help="最大曝光次数")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--num-frames", type=int, default=30)
    parser.add_argument("--resolution", type=str, default="2048,1024")
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--engine", type=str, default="CYCLES")
    parser.add_argument("--exposure", type=float, default=0.0)
    parser.add_argument("--grid-spacing", type=float, default=0.5)
    parser.add_argument("--camera-height", type=float, default=None)
    parser.add_argument("--stop-gain", type=float, default=DEFAULT_STOP_GAIN)
    parser.add_argument("--stop-score", type=float, default=-0.3)
    parser.add_argument("--stop-delta", type=float, default=0.08)
    parser.add_argument("--min-frames", type=int, default=DEFAULT_MIN_FRAMES)
    parser.add_argument("--rotation-type", type=str, default="random_yaw",
                        choices=["none", "rotate_x_90", "rotate_x_180",
                                 "rotate_z_90", "random_yaw"])
    parser.add_argument("--gain-curve", action="store_true", default=True)
    parser.add_argument("--no-gain-curve", dest="gain_curve", action="store_false")
    parser.add_argument("--rooms", type=str, default=None, help="HM3D 房间信息的 JSON 字符串")
    return parser.parse_args(argv)

###############################
#
# 房间搜索相关工具函数
#
###############################
def load_semantic_info(mesh_path):
    """
    加载语义信息（包装函数）
    
    Args:
        mesh_path: GLB文件路径
    
    Returns:
        (room_to_objects, semantic_file_path): 
        - room_to_objects: {room_id: [object_ids]} 字典，如果不存在则返回None
        - semantic_file_path: 使用的语义文件路径，如果不存在则返回None
    """
    from tools.semantic_utils import load_semantic_info as load_semantic_info_util
    return load_semantic_info_util(mesh_path)


def get_scene_bounds(mesh_path, room_id=None, room_to_objects=None):
    """
    获取场景边界框（使用trimesh）
    
    Args:
        mesh_path: mesh文件路径
        room_id: 房间ID（可选），如果提供则只计算该房间的边界框
        room_to_objects: 房间到对象的映射 {room_id: [object_ids]}（可选）
    
    Returns:
        bounds_min, bounds_max: 边界框的最小和最大坐标
    """
    try:
        import trimesh
        import re
        
        scene_or_mesh = trimesh.load(mesh_path, process=False)
        
        # 如果指定了房间ID，需要过滤对象
        object_ids_to_include = None
        if room_id is not None and room_to_objects is not None:
            object_ids_to_include = set(room_to_objects.get(room_id, []))
            if not object_ids_to_include:
                # 如果房间没有对象，返回默认边界框
                return np.array([-1, -1, -1]), np.array([1, 1, 1])
        
        if isinstance(scene_or_mesh, trimesh.Scene):
            # 合并所有几何体的顶点
            all_vertices = []
            for name, geom in scene_or_mesh.geometry.items():
                if isinstance(geom, trimesh.Trimesh):
                    # 如果指定了房间ID，检查对象是否属于该房间
                    if object_ids_to_include is not None:
                        # HM3D的GLB中，对象名称格式为 "chunkXXX_..." 
                        # chunk编号+1对应semantic.txt中的对象ID
                        obj_id = None
                        
                        # 方法1: 尝试从chunk编号提取（chunk000 -> 对象ID 1）
                        match = re.search(r'chunk(\d+)', name)
                        if match:
                            chunk_num = int(match.group(1))
                            obj_id = chunk_num + 1
                        
                        # 方法2: 如果名称是纯数字，直接使用
                        if obj_id is None:
                            try:
                                obj_id = int(name)
                            except (ValueError, TypeError):
                                pass
                        
                        # 如果无法提取对象ID，或者对象ID不在该房间中，跳过
                        if obj_id is None or obj_id not in object_ids_to_include:
                            continue
                    
                    # 获取变换矩阵
                    for node_name in scene_or_mesh.graph.nodes_geometry:
                        if scene_or_mesh.graph[node_name][1] == name:
                            transform = scene_or_mesh.graph[node_name][0]
                            vertices = np.dot(geom.vertices, transform[:3, :3].T) + transform[:3, 3]
                            all_vertices.append(vertices)
                            break
            
            if all_vertices:
                all_vertices = np.vstack(all_vertices)
                bounds_min = all_vertices.min(axis=0)
                bounds_max = all_vertices.max(axis=0)
            else:
                bounds_min = np.array([-1, -1, -1])
                bounds_max = np.array([1, 1, 1])
        else:
            bounds_min = scene_or_mesh.bounds[0]
            bounds_max = scene_or_mesh.bounds[1]
        
        return bounds_min, bounds_max
        
    except ImportError:
        print("[WARN] trimesh未安装，使用默认边界框")
        return np.array([-1, -1, -1]), np.array([1, 1, 1])

def get_valid_object_ids(mesh_path):
    """
    获取GLB文件中所有有效的对象ID
    
    Args:
        mesh_path: mesh文件路径
    
    Returns:
        valid_obj_ids: 有效对象ID的集合
    """
    try:
        import trimesh
        import re
        
        valid_obj_ids = set()
        scene_or_mesh = trimesh.load(mesh_path, process=False)
        
        if isinstance(scene_or_mesh, trimesh.Scene):
            for name, geom in scene_or_mesh.geometry.items():
                if isinstance(geom, trimesh.Trimesh):
                    # 从chunk编号提取对象ID
                    match = re.search(r'chunk(\d+)', name)
                    if match:
                        chunk_num = int(match.group(1))
                        obj_id = chunk_num + 1
                        valid_obj_ids.add(obj_id)
                    else:
                        # 如果名称是纯数字，直接使用
                        try:
                            obj_id = int(name)
                            valid_obj_ids.add(obj_id)
                        except (ValueError, TypeError):
                            pass
        
        return valid_obj_ids
    except Exception as e:
        print(f"[WARN] 获取有效对象ID失败: {e}")
        return set()

def get_valid_object_ids(mesh_path):
    """
    获取GLB文件中所有有效的对象ID
    
    Args:
        mesh_path: mesh文件路径
    
    Returns:
        valid_obj_ids: 有效对象ID的集合
    """
    try:
        import trimesh
        import re
        
        valid_obj_ids = set()
        scene_or_mesh = trimesh.load(mesh_path, process=False)
        
        if isinstance(scene_or_mesh, trimesh.Scene):
            for name, geom in scene_or_mesh.geometry.items():
                if isinstance(geom, trimesh.Trimesh):
                    # 从chunk编号提取对象ID
                    match = re.search(r'chunk(\d+)', name)
                    if match:
                        chunk_num = int(match.group(1))
                        obj_id = chunk_num + 1
                        valid_obj_ids.add(obj_id)
                    else:
                        # 如果名称是纯数字，直接使用
                        try:
                            obj_id = int(name)
                            valid_obj_ids.add(obj_id)
                        except (ValueError, TypeError):
                            pass
        
        return valid_obj_ids
    except Exception as e:
        print(f"[WARN] 获取有效对象ID失败: {e}")
        return set()


def detect_rooms(mesh_path, room_to_objects=None, use_navmesh=True):
    """
    检测场景中的所有房间，计算每个房间的边界框和几何中心
    
    优先使用NavMesh方法，如果NavMesh不可用则使用语义方法
    
    Args:
        mesh_path: mesh文件路径
        room_to_objects: 房间到对象的映射 {room_id: [object_ids]}（可选）
        use_navmesh: 是否优先使用NavMesh方法
    
    Returns:
        rooms: 房间列表，每个元素包含 {"room_id": int, "bounds_min": array, "bounds_max": array, "center": array, "source": str}
    """
    rooms = []
    
    # 尝试使用NavMesh方法
    from tools.navmesh_utils import extract_spaces_from_glb, find_navmesh_file, navmesh_to_3d_spaces
    if use_navmesh:
        navmesh_path = find_navmesh_file(mesh_path)
        if navmesh_path:
            print(f"  [INFO] 找到NavMesh文件: {navmesh_path}")
            print(f"  [INFO] 使用NavMesh方法识别空间...")
            
            # 使用NavMesh结合语义信息提取空间（启用空间聚类）
            navmesh_spaces = navmesh_to_3d_spaces(
                navmesh_path, 
                mesh_path=mesh_path, 
                room_to_objects=room_to_objects,
                use_clustering=True  # 启用空间聚类，识别所有独立空间
            )
            
            if navmesh_spaces:
                print(f"  [INFO] NavMesh方法识别到 {len(navmesh_spaces)} 个空间")
                # 转换格式以保持兼容性
                for space in navmesh_spaces:
                    rooms.append({
                        "room_id": int(space["space_id"]),
                        "bounds_min": space["bounds_min"].tolist(),
                        "bounds_max": space["bounds_max"].tolist(),
                        "center": space["center"].tolist(),
                        "object_count": space.get("object_count", 0),
                        "source": space.get("source", "navmesh")
                    })
                return rooms
            else:
                print(f"  [WARN] NavMesh方法未识别到空间，回退到GLB聚类方法")
        else:
            print(f"  [INFO] 未找到NavMesh文件，尝试直接从GLB文件进行空间聚类...")
    
    # 尝试直接从GLB文件进行空间聚类（不依赖NavMesh和语义信息）
    print(f"  [INFO] 使用GLB空间聚类方法...")
    glb_spaces = extract_spaces_from_glb(mesh_path, room_to_objects, use_clustering=True)
    
    if glb_spaces:
        print(f"  [INFO] GLB聚类方法识别到 {len(glb_spaces)} 个空间")
        for space in glb_spaces:
            rooms.append({
                        "room_id": int(space["space_id"]),
                        "bounds_min": space["bounds_min"].tolist(),
                        "bounds_max": space["bounds_max"].tolist(),
                        "center": space["center"].tolist(),
                        "object_count": space.get("object_count", 0),
                        "source": space.get("source", "navmesh")
                    })
        return rooms
    else:
        print(f"  [WARN] GLB聚类方法未识别到空间，回退到语义方法")
    
    # 回退到语义方法
    if not room_to_objects:
        return rooms
    
    # 获取GLB中有效的对象ID
    valid_obj_ids = get_valid_object_ids(mesh_path)
    
    # 过滤掉无效的对象ID
    filtered_room_to_objects = {}
    for room_id, obj_ids in room_to_objects.items():
        valid_ids = [obj_id for obj_id in obj_ids if obj_id in valid_obj_ids]
        if valid_ids:
            filtered_room_to_objects[room_id] = valid_ids
    
    if not filtered_room_to_objects:
        return rooms
    
    print(f"  过滤后有效房间数: {len(filtered_room_to_objects)}/{len(room_to_objects)}")
    
    for room_id in sorted(filtered_room_to_objects.keys()):
        # 使用过滤后的对象ID获取边界框
        bounds_min, bounds_max = get_scene_bounds(
            mesh_path, 
            room_id=room_id, 
            room_to_objects=filtered_room_to_objects
        )
        
        # 计算几何中心（trimesh Y-up坐标系）
        center = (bounds_min + bounds_max) / 2
        
        # 检查边界框是否有效（不是默认值，且有合理的尺寸）
        if not np.allclose(bounds_min, [-1, -1, -1]) and not np.allclose(bounds_max, [1, 1, 1]):
            size = bounds_max - bounds_min
            # 检查房间是否有合理的尺寸（至少0.1米）
            if np.all(size > 0.1):
                rooms.append({
                    "room_id": room_id,
                    "bounds_min": bounds_min,
                    "bounds_max": bounds_max,
                    "center": center,
                    "object_count": len(filtered_room_to_objects[room_id]),
                    "source": "semantic"
                })
    
    return rooms


# #####################################################################
#
#  Python 模式入口: 启动单个 Blender 进程
#
# #####################################################################
def main_python():
    """Python 调用入口 → 启动一个 Blender 进程执行本脚本"""
    args = parse_args_python()

    # 判断场景格式
    if args.blend:
        scene_path = str(Path(args.blend).resolve())
        scene_flag = "--blend"
        scene_label = f"Blend: {scene_path}"
    else:
        scene_path = str(Path(args.glb).resolve())
        scene_flag = "--glb"
        scene_label = f"GLB:   {scene_path}"

    output_dir = str(Path(args.output_dir).resolve())
    os.makedirs(output_dir, exist_ok=True)

    this_script = str(Path(__file__).resolve())

    # 构建 Blender 命令（把参数透传，去掉 --blender 和 --render-depth）
    cmd = [
        args.blender, "--background",
        "--python", this_script,
        "--",
        scene_flag, scene_path,
        "--output-dir", output_dir,
        "--num-frames", str(args.num_frames),
        "--resolution", args.resolution,
        "--samples", str(args.samples),
        "--engine", args.engine,
        "--exposure", str(args.exposure),
        "--grid-spacing", str(args.grid_spacing),
        "--stop-gain", str(args.stop_gain),
        "--stop-score", str(args.stop_score),
        "--stop-delta", str(args.stop_delta),
        "--min-frames", str(args.min_frames),
        "--rotation-type", args.rotation_type,
        # "--exposure-rate", args.exposure_rate,
        # "--exposure-count", args.exposure_count
    ]
    if args.camera_height is not None:
        cmd += ["--camera-height", str(args.camera_height)]
    if not args.gain_curve:
        cmd += ["--no-gain-curve"]
    
    # [TODO] 如果是hm3d数据集，进行房间搜索并通过添加 `--rooms` 参数传递房间信息（json字符串）给 Blender 内部
    # Blender 内部根据房间列表加载对应的场景并进行渲染。
    if args.hm3d:
        # 记得在 Blender 内部解析 `--rooms` 参数，加载对应的场景并进行渲染。
        # print("\n[INFO] 加载语义信息...")
        room_to_objects, semantic_file_path = load_semantic_info(args.glb)
        rooms = detect_rooms(args.glb, room_to_objects=room_to_objects, use_navmesh=True)
        # 如果NavMesh方法未识别到空间，且语义信息可用，使用纯语义方法
        if len(rooms) == 0 and room_to_objects and semantic_file_path:
            print(f"  [INFO] NavMesh方法未识别到空间，使用语义方法...")
            print(f"  找到语义文件: {semantic_file_path}")
            print(f"  检测到 {len(room_to_objects)} 个房间")
            rooms = detect_rooms(args.glb, room_to_objects=room_to_objects, use_navmesh=False)
            print(f"  有效房间数: {len(rooms)}")
        elif len(rooms) > 0:
            print(f"  [INFO] 成功识别到 {len(rooms)} 个空间")
            # 统计空间来源
            sources = {}
            for room in rooms:
                source = room.get("source", "unknown")
                sources[source] = sources.get(source, 0) + 1
            print(f"  空间来源: {sources}")
        # print("*"*60, json.dumps({"rooms": rooms}), sep="\n")
        # exit(0)
        print(f"  房间数量: {len(rooms)}")
        # input("\n按Enter键继续...")
        cmd += ["--rooms", json.dumps({"rooms": rooms})]
        # print("\n".join(cmd))
        # exit()
        # pass
    print("=" * 60)
    print("ERPT Blend Pipeline v5（单进程边渲边选）")
    print("=" * 60)
    print(f"  {scene_label}")
    print(f"  Output:     {output_dir}")
    print(f"  Max frames: {args.num_frames}")

    # 不设 timeout — 大场景渲染时间不可预测
    proc = subprocess.run(cmd, text=True)
    sys.exit(proc.returncode)


# #####################################################################
#
#  Blender 模式: Phase 0 + 1 + 2 全部在 Blender 内部执行
#
# #####################################################################

# =====================================================================
# Phase 0: 加载场景 + 获取边界
# =====================================================================

def load_scene(mesh_path):
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


# =====================================================================
# Phase 1: 撒点 + 4 层 Blender ray_cast 过滤
# =====================================================================

def compute_camera_heights(floor_z, ceiling_z, manual_height=None, bmin=None, bmax=None):
    """计算相机高度层

    策略:
      - 手动指定 → 只用该高度
      - 多层建筑 → 每层铺固定高度 [0.5, 0.8, 1.2, 1.7, 2.1] + 动态顶层
      - 单层高空间 → 2.5m 以下用固定高度，2.5m 以上阶梯递增:
          +1.0m, +1.5m, +2.0m, +2.5m, +3.0m ...（间距逐步放大）
          最后加动态顶层（天花板 -0.5m）
    """
    CEIL_CLEARANCE = 0.3    # 最高高度：离天花板 0.3m（保留 2.1m 层）
    FIXED_HEIGHTS = [0.5, 0.8, 1.2, 1.7, 2.1]  # 2.5m 以下的固定高度

    if manual_height is not None:
        return [manual_height]

    room_h = ceiling_z - floor_z
    if room_h <= 0:
        return [floor_z + 1.5]

    def _stepped_heights_for_floor(fz, local_ceil):
        """单层高度计算：固定 + 阶梯递增 + 动态顶层"""
        heights = []
        local_h = local_ceil - fz

        # 2.5m 以下: 固定高度
        for eye_h in FIXED_HEIGHTS:
            z = fz + eye_h
            if z < local_ceil - CEIL_CLEARANCE:
                heights.append(z)

        # 2.5m 以上: 阶梯递增（间距从 1.0m 逐步增大到 3.0m）
        if local_h > 3.0:  # 层高 > 3m 才加中间高度
            cur_h = FIXED_HEIGHTS[-1]  # 从 2.1m 开始
            step = 1.0                  # 初始步长 1.0m
            MAX_STEP = 3.0              # 最大步长 3.0m
            STEP_GROW = 0.5             # 每次步长增加 0.5m

            while True:
                cur_h += step
                z = fz + cur_h
                if z >= local_ceil - CEIL_CLEARANCE:
                    break
                heights.append(z)
                step = min(step + STEP_GROW, MAX_STEP)

        # 动态顶层：天花板 - 0.5m（如果比最高已有高度至少高 0.5m）
        top_z = local_ceil - CEIL_CLEARANCE
        if heights:
            if top_z > max(heights) + 0.5:
                heights.append(top_z)
        elif top_z > fz + 0.5:
            heights.append(top_z)

        return heights

    # 先尝试用 Blender raycast 探测楼板
    try:
        floors = _detect_floor_levels(floor_z, ceiling_z, bmin, bmax)
        if floors:
            print(f"  [楼层检测] 发现 {len(floors)} 个楼层: "
                  f"{[f'{z:.2f}m' for z in floors]}")
            heights = []
            for idx, fz in enumerate(floors):
                # 每层的天花板 = 下一层楼板 或 全局天花板
                if idx + 1 < len(floors):
                    local_ceil = floors[idx + 1]
                else:
                    local_ceil = ceiling_z
                heights.extend(_stepped_heights_for_floor(fz, local_ceil))

            if heights:
                result = sorted(set(round(h, 2) for h in heights))
                # 打印高度分布
                for h in result:
                    rel = h - floors[0]
                    print(f"    高度 Z={h:.2f}m (离地 {rel:.2f}m)")
                return result
        else:
            print(f"  [楼层检测] 未检测到楼板，使用启发式")
    except Exception as e:
        print(f"  [楼层检测] 异常: {e}，使用启发式")

    # fallback: 简单启发式（同样用阶梯递增）
    h_list = _stepped_heights_for_floor(floor_z, ceiling_z)
    return sorted(set(round(h, 2) for h in h_list)) if h_list else [floor_z + 1.5]


def _detect_floor_levels(floor_z, ceiling_z, bmin=None, bmax=None):
    """用 raycast 从上往下扫描，检测楼板位置

    在 XY 平面采样若干点，每个点从顶部往下打射线，收集 hit 的 Z 坐标。
    对 Z 坐标做聚类（间距 > 1.5m 算不同楼层），得到各楼层地面高度。

    关键改进:
    1. 采样范围按场景大小缩放（不只中心 ±2m）
    2. 检测到楼板后验证上方有天花板（排除屋顶外表面）
    """
    scene = bpy.context.scene
    depsgraph = bpy.context.evaluated_depsgraph_get()
    dir_down = Vector((0, 0, -1))
    dir_up = Vector((0, 0, 1))

    # 用场景 AABB 的 XY 中心和范围
    if bmin is not None and bmax is not None:
        cx = (bmin[0] + bmax[0]) / 2
        cy = (bmin[1] + bmax[1]) / 2
        # 采样范围: 场景 XY 的 1/4 跨度，至少 2m，最多 20m
        rx = min(20.0, max(2.0, (bmax[0] - bmin[0]) * 0.25))
        ry = min(20.0, max(2.0, (bmax[1] - bmin[1]) * 0.25))
    else:
        cx, cy = 0.0, 0.0
        rx, ry = 2.0, 2.0

    hit_zs = []
    # 3x3 网格采样，按场景大小缩放
    offsets = []
    for fx in [-1, 0, 1]:
        for fy in [-1, 0, 1]:
            offsets.append((fx * rx, fy * ry))

    for dx, dy in offsets:
        origin = Vector((cx + dx, cy + dy, ceiling_z + 1.0))
        # 多次向下 raycast（穿透式：命中后从命中点下方继续）
        cur_z = ceiling_z + 1.0
        for _ in range(10):  # 最多穿 10 层
            hit, loc, norm, *_ = scene.ray_cast(
                depsgraph, Vector((cx + dx, cy + dy, cur_z)), dir_down)
            if not hit:
                break
            # 法线朝上（Z 分量 > 0.5）→ 这是地板/楼板表面
            if norm.z > 0.5:
                hit_zs.append((loc.z, cx + dx, cy + dy))
            cur_z = loc.z - 0.05  # 穿过这个表面继续往下

    if not hit_zs:
        return []

    # 聚类: 排序后间距 > 1.5m 算不同楼层
    hit_zs.sort(key=lambda t: t[0])
    clusters = [[hit_zs[0]]]
    for item in hit_zs[1:]:
        if item[0] - clusters[-1][-1][0] > 1.5:
            clusters.append([item])
        else:
            clusters[-1].append(item)

    # 每个 cluster 验证: 楼板上方是否有天花板
    MAX_CEILING_DIST = 30.0  # 最高天花板距离（超过说明是露天/屋顶外表面）
    floors = []
    for c in clusters:
        fz = sorted(c, key=lambda t: t[0])[len(c) // 2][0]
        if not (floor_z - 0.5 <= fz <= ceiling_z - 1.0):
            continue

        # 验证: 从该楼板上方 1m 处往上打射线，检查是否有天花板
        n_has_ceiling = 0
        n_tested = 0
        for _, px, py in c:
            test_origin = Vector((px, py, fz + 1.0))
            hit_ceil, loc_ceil, norm_ceil, *_ = scene.ray_cast(
                depsgraph, test_origin, dir_up)
            n_tested += 1
            if hit_ceil and (loc_ceil.z - fz) < MAX_CEILING_DIST:
                n_has_ceiling += 1

        # 过半采样点上方有天花板 → 真正的楼板
        if n_tested > 0 and n_has_ceiling / n_tested >= 0.5:
            floors.append(fz)
        else:
            print(f"    [楼层检测] Z={fz:.2f}m 上方无天花板"
                  f"（{n_has_ceiling}/{n_tested}），排除（可能是屋顶外表面）")

    return sorted(floors)


def generate_candidate_grid(bmin, bmax, x_spacing, y_spacing, heights):
    cx = (bmin[0] + bmax[0]) / 2
    cy = (bmin[1] + bmax[1]) / 2
    x_half = int((bmax[0] - cx - MARGIN) / x_spacing)
    y_half = int((bmax[1] - cy - MARGIN) / y_spacing)

    xy_offsets = []
    for ix in range(-x_half, x_half + 1):
        for iy in range(-y_half, y_half + 1):
            x = cx + ix * x_spacing
            y = cy + iy * y_spacing
            if bmin[0] + MARGIN <= x <= bmax[0] - MARGIN and \
               bmin[1] + MARGIN <= y <= bmax[1] - MARGIN:
                xy_offsets.append((ix * ix + iy * iy, x, y))
    xy_offsets.sort(key=lambda t: t[0])

    candidates = []
    for z in heights:
        for _, x, y in xy_offsets:
            candidates.append([float(x), float(y), float(z)])

    n_xy = len(xy_offsets)
    print(f"  网格: {n_xy}点/层 x {len(heights)}层 = {len(candidates)} 个候选")
    print(f"    中心: ({cx:.1f}, {cy:.1f}), X间距={x_spacing:.1f}m, Y间距={y_spacing:.1f}m")
    for i, z in enumerate(heights):
        print(f"    第{i+1}层: Z={z:.2f}m")
    return candidates


def _build_26_directions():
    """26 方向球面采样（mathutils.Vector）"""
    dirs = []
    for i in range(16):
        a = i * (2 * math.pi / 16)
        dirs.append(Vector((math.cos(a), math.sin(a), 0.0)))
    elev = math.pi / 4
    for i in range(5):
        a = i * (2 * math.pi / 5)
        dirs.append(Vector((math.cos(a) * math.cos(elev),
                            math.sin(a) * math.cos(elev),
                            math.sin(elev))))
    for i in range(5):
        a = i * (2 * math.pi / 5)
        dirs.append(Vector((math.cos(a) * math.cos(elev),
                            math.sin(a) * math.cos(elev),
                            -math.sin(elev))))
    return dirs


def raycast_6layer_filter(candidates, room_height, min_wall_dist=1.0):
    """7 层过滤 — 直接用 Blender scene.ray_cast（不需要 trimesh/GLB）

    第 1 层: 室内检测（朝上朝下必须 hit）
    第 2 层: 穿模检测（≥2 方向 < 0.2m）
    第 3 层: 角落检测（>50% 水平方向 < 1.0m）
    第 4 层: 包裹检测（hit_rate≥90% + cv<0.30 + max<8m）
    第 5 层: 墙面间距（最近水平方向 < 0.3m → Blender 渲染会穿模）
    第 6 层: 视野质量（<35% 方向有有效命中 → 太空旷或太闭塞）
    第 7 层: 窄缝检测（对向方向距离之和 < 1.5m → 两面墙夹着）★ 新增

    性能: 用第 1~4 层同样的 26 方向数据，第 5~7 层零额外射线开销
    """
    scene = bpy.context.scene
    depsgraph = bpy.context.evaluated_depsgraph_get()

    N = len(candidates)
    max_up = max(5.0, room_height)
    max_down = max(3.0, room_height)
    dir_up = Vector((0, 0, 1))
    dir_down = Vector((0, 0, -1))
    DIRS_26 = _build_26_directions()
    n26 = len(DIRS_26)

    # 第 5 层阈值: 最近水平墙面距离
    MIN_WALL_CLEARANCE = 0.3  # Blender 渲染安全距离

    # 第 6 层阈值: 有效视野比例
    VIEW_GOOD_MIN = 0.5    # 有效命中距离下限
    VIEW_GOOD_MAX = 20.0   # 有效命中距离上限
    VIEW_GOOD_RATIO = 0.35 # 至少 35% 方向有有效命中

    # 第 7 层阈值: 窄缝检测（对向距离之和）
    MIN_SLIT_WIDTH = 1.5   # 对向墙距之和 < 1.5m → 窄缝

    passed = []
    stats = {"无天花板": 0, "无地板": 0, "穿模": 0, "角落": 0,
             "包裹": 0, "贴墙": 0, "视野差": 0, "窄缝": 0}

    t0 = time.time()
    log_interval = max(1, N // 10)

    for idx, pos in enumerate(candidates):
        if idx % log_interval == 0 and idx > 0:
            print(f"    过滤进度: {idx}/{N} ({idx*100//N}%)", flush=True)

        origin = Vector(pos)

        # ---- 第 1 层: 室内检测（朝上朝下各 1 条射线）----
        hit_up, loc_up, *_ = scene.ray_cast(depsgraph, origin, dir_up)
        if not hit_up or (loc_up - origin).length > max_up:
            stats["无天花板"] += 1
            continue

        hit_dn, loc_dn, *_ = scene.ray_cast(depsgraph, origin, dir_down)
        if not hit_dn or (loc_dn - origin).length > max_down:
            stats["无地板"] += 1
            continue

        # ---- 第 2~6 层: 26 方向球面采样 ----
        dists = []
        for d in DIRS_26:
            hit, loc, *_ = scene.ray_cast(depsgraph, origin, d)
            dists.append((loc - origin).length if hit else float('inf'))

        # 第 2 层: 穿模（≥2 方向 < 0.2m → 在物体内部）
        n_close = sum(1 for d in dists if d < 0.2)
        if n_close >= 2:
            stats["穿模"] += 1
            continue

        # 第 3 层: 角落（水平 16 方向中 > 一半 < 1.0m）
        n_wall = sum(1 for d in dists[:16] if d < min_wall_dist)
        if n_wall > 8:
            stats["角落"] += 1
            continue

        # 第 4 层: 包裹（hit_rate≥90% + CV<0.30 + max<8m）
        finite = [d for d in dists if d < float('inf')]
        hit_rate = len(finite) / n26
        if hit_rate >= 0.90 and len(finite) >= 2:
            mean_d = sum(finite) / len(finite)
            max_d = max(finite)
            if mean_d > 0:
                var = sum((d - mean_d) ** 2 for d in finite) / len(finite)
                cv = var ** 0.5 / mean_d
                if cv < 0.30 and max_d < 8.0:
                    stats["包裹"] += 1
                    continue

        # 第 5 层: 墙面间距（水平 16 方向最近 hit < 0.3m → 贴墙）★ 新增
        horiz_finite = [d for d in dists[:16] if d < float('inf')]
        if horiz_finite and min(horiz_finite) < MIN_WALL_CLEARANCE:
            stats["贴墙"] += 1
            continue

        # 第 6 层: 视野质量（有效方向太少 → 视野差）
        n_good = sum(1 for d in dists
                     if VIEW_GOOD_MIN <= d <= VIEW_GOOD_MAX)
        good_ratio = n_good / n26
        if good_ratio < VIEW_GOOD_RATIO:
            stats["视野差"] += 1
            continue

        # 第 7 层: 窄缝检测（对向水平方向距离之和 < 1.5m → 两面墙夹着）
        # 水平 16 方向中，方向 i 和方向 i+8 是对向的（0°↔180°, 22.5°↔202.5°...）
        in_slit = False
        for i in range(8):
            d_fwd = dists[i] if dists[i] < float('inf') else 999
            d_bwd = dists[i + 8] if dists[i + 8] < float('inf') else 999
            if d_fwd + d_bwd < MIN_SLIT_WIDTH:
                in_slit = True
                break
        if in_slit:
            stats["窄缝"] += 1
            continue

        passed.append(pos)

    dt = time.time() - t0
    print(f"  过滤统计 ({dt:.1f}s): 总计={N}, 通过={len(passed)}")
    for k, v in stats.items():
        print(f"    ❌ {k}: {v} ({v * 100 // max(N, 1)}%)")
    print(f"    阈值: 天花板<{max_up:.1f}m, 地板<{max_down:.1f}m, "
          f"穿模<0.2m, 角落<{min_wall_dist:.1f}m, "
          f"包裹: hit≥90%+cv<0.3+max<8m, "
          f"贴墙<{MIN_WALL_CLEARANCE}m, "
          f"视野: ≥{VIEW_GOOD_RATIO:.0%}方向 {VIEW_GOOD_MIN}-{VIEW_GOOD_MAX}m, "
          f"窄缝<{MIN_SLIT_WIDTH}m")

    if len(passed) < 5 and N > 20:
        print(f"    [诊断] 通过率低 ({len(passed)}/{N})")

    return passed


def setup_erp_camera():
    """创建 ERP 全景相机"""
    for obj in list(bpy.context.scene.objects):
        if obj.type == 'CAMERA':
            bpy.data.objects.remove(obj, do_unlink=True)

    cam_data = bpy.data.cameras.new("ERP_Camera")
    cam_data.type = 'PANO'
    if hasattr(cam_data, 'panorama_type'):
        cam_data.panorama_type = 'EQUIRECTANGULAR'
    if hasattr(cam_data, 'cycles'):
        cam_data.cycles.panorama_type = 'EQUIRECTANGULAR'

    cam_obj = bpy.data.objects.new("ERP_Camera", cam_data)
    bpy.context.scene.collection.objects.link(cam_obj)
    bpy.context.scene.camera = cam_obj
    print(f"  创建 ERP 相机: {cam_obj.name}")
    return cam_obj


def enable_gpu():
    try:
        prefs = bpy.context.preferences.addons['cycles'].preferences
        for dt in ['OPTIX', 'CUDA']:
            try:
                prefs.compute_device_type = dt
                prefs.get_devices()
                gpus = [d for d in prefs.devices if d.type == dt]
                if gpus:
                    for d in prefs.devices:
                        d.use = (d.type == dt)
                    bpy.context.scene.cycles.device = 'GPU'
                    print(f"  GPU 渲染: {gpus[0].name} ({dt})")
                    return True
            except Exception:
                continue
        print("  [WARN] 无可用 GPU，使用 CPU 渲染")
        bpy.context.scene.cycles.device = 'CPU'
    except Exception as e:
        print(f"  [ERROR] GPU 设置异常: {e}")
    return False


def setup_render_settings(resolution, engine, samples, exposure):
    scene = bpy.context.scene
    scene.render.engine = engine
    scene.render.resolution_x = resolution[0]
    scene.render.resolution_y = resolution[1]
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGB'
    scene.render.image_settings.color_depth = '8'
    # 取消曝光，使用强环境光
    # scene.view_settings.exposure = exposure
    # AgX（Blender 4+默认）对室内场景会严重压暗；改用 Standard 线性映射，
    # 颜色准确且更亮，曝光完全由 exposure 参数控制。
    # scene.view_settings.view_transform = 'Standard'
    # scene.view_settings.look = 'None'

    if engine == 'CYCLES':
        scene.cycles.samples = samples
        scene.cycles.use_denoising = True
        scene.cycles.max_bounces = 4
        scene.cycles.diffuse_bounces = 2
        scene.cycles.glossy_bounces = 2
        scene.cycles.transmission_bounces = 2
        scene.cycles.transparent_max_bounces = 8
        enable_gpu()

    print(f"  渲染设置: {engine} {resolution[0]}x{resolution[1]} "
          f"samples={samples} exposure={exposure} view_transform=Standard")


def _world_has_effective_light(world) -> bool:
    """判断 World 节点是否能产生有效的环境光（Strength > 0.05）。
    GLB 导入的场景通常有一个 World 对象，但 Background Strength 可能为 0。
    """
    if world is None:
        return False
    if not world.use_nodes or world.node_tree is None:
        # 没用节点系统：用旧 API 的纯色环境，认为有效
        return True
    for node in world.node_tree.nodes:
        if node.type == 'BACKGROUND':
            strength = node.inputs.get('Strength')
            if strength is not None:
                val = strength.default_value
                # 如果有链接（HDR 贴图等），视为有效
                if strength.is_linked or float(val) > 0.05:
                    return True
    return False


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


# =====================================================================
# 渲染 + 深度转换 + 位姿保存（同进程，只移动相机）
# =====================================================================

def convert_depth_exr_to_npy(exr_path, npy_path):
    """EXR → NPY（Blender 内置 API，不依赖 OpenEXR 库）"""
    img = bpy.data.images.load(exr_path)
    w, h = img.size[0], img.size[1]
    pixels = np.array(img.pixels[:]).reshape(h, w, -1)
    depth = np.flipud(pixels[:, :, 0])

    unit_scale = bpy.context.scene.unit_settings.scale_length
    depth_m = depth * unit_scale
    depth_m[(depth_m > 1000.0) | (depth_m <= 0)] = 0.0

    np.save(npy_path, depth_m.astype(np.float32))
    bpy.data.images.remove(img)
    try:
        os.remove(exr_path)
    except OSError:
        pass


def render_frame_inprocess(cam_obj, frame_id, camera_pos, camera_rot,
                           output_dir, depth_fo):
    """同进程渲染一帧，返回 (rgb_path, depth_path, pose_path)"""
    cam_obj.location = Vector(camera_pos)
    cam_obj.rotation_euler = Euler(camera_rot, 'XYZ')

    base = f"panorama_{frame_id:04d}"
    rgb_path = os.path.join(output_dir, f"{base}.png")
    depth_npy = os.path.join(output_dir, f"{base}_depth.npy")
    pose_path = os.path.join(output_dir, f"pose_{frame_id:04d}.json")

    bpy.context.scene.render.filepath = rgb_path

    abs_dir = os.path.abspath(output_dir)
    depth_fo.directory = abs_dir
    depth_fo.file_name = base + "_"
    depth_exr = os.path.join(abs_dir, base + "_depth.exr")

    bpy.context.scene.frame_set(frame_id)
    bpy.ops.render.render(write_still=True)

    # 深度转换
    if os.path.exists(depth_exr):
        convert_depth_exr_to_npy(depth_exr, depth_npy)
    else:
        import glob
        hits = glob.glob(os.path.join(abs_dir, f"*{base}*depth*.exr"))
        if hits:
            convert_depth_exr_to_npy(hits[0], depth_npy)
        else:
            print(f"    [WARN] 未找到深度 EXR: {depth_exr}")
            depth_npy = None

    # 位姿（与 render_erp_blender.py save_pose 完全一致的格式）
    save_pose(cam_obj, pose_path, frame_id)

    return rgb_path, depth_npy, pose_path


def save_pose(camera_object, output_path, frame_id):
    """保存位姿（绝对位姿，cam_to_world，兼容 ERPT）

    格式与 render_erp_blender.py 的 save_pose 完全一致：
      R_cw_erpt = T @ R_obj_blender @ M
    """
    unit_scale = bpy.context.scene.unit_settings.scale_length

    abs_pos_b = list(camera_object.location)
    abs_quat_b = camera_object.rotation_euler.to_quaternion()

    # Blender(X右,Y前,Z上) → 统一(X右,Y上,Z前)
    abs_pos_u = [
        abs_pos_b[0] * unit_scale,   # X
        abs_pos_b[2] * unit_scale,   # Y_unified = Z_blender
        abs_pos_b[1] * unit_scale,   # Z_unified = Y_blender
    ]

    R_obj = abs_quat_b.to_matrix()
    T = Matrix([[1, 0, 0], [0, 0, 1], [0, 1, 0]])
    M = Matrix([[1, 0, 0], [0, 1, 0], [0, 0, -1]])
    R_cw = T @ R_obj @ M
    q = R_cw.to_quaternion()

    pose_data = {
        "frame_id": frame_id,
        "position": abs_pos_u,
        "rotation_quaternion": [q.w, q.x, q.y, q.z],
        "camera_type": "erp_ray",
        "coordinate_system": "right-handed, Y-up, Z-forward (cam_to_world)",
        "render_method": "blender_cycles",
    }
    with open(output_path, 'w') as f:
        json.dump(pose_data, f, indent=2)


# =====================================================================
# 选帧核心（向量化，内嵌）
# =====================================================================

def build_ray_directions(H=WARP_H, W=WARP_W):
    """向量化构建 ERP 射线方向（Z-up）"""
    i = np.arange(H, dtype=np.float64)
    j = np.arange(W, dtype=np.float64)
    phi = np.pi / 2 - np.pi * (i + 0.5) / H
    theta = 2 * np.pi * (j + 0.5) / W
    phi, theta = np.meshgrid(phi, theta, indexing='ij')
    return np.stack([
        np.cos(phi) * np.cos(theta),
        np.cos(phi) * np.sin(theta),
        np.sin(phi),
    ], axis=-1)


_ray_dirs_cache = {}


def get_ray_dirs(H=WARP_H, W=WARP_W):
    if (H, W) not in _ray_dirs_cache:
        _ray_dirs_cache[(H, W)] = build_ray_directions(H, W)
    return _ray_dirs_cache[(H, W)]


def depth_to_3d_points(position, depth, ray_dirs, max_depth=None):
    valid = depth > 0
    if max_depth is not None:
        valid &= (depth <= max_depth)
    if not np.any(valid):
        return np.empty((0, 3), dtype=np.float64)
    pos = np.array(position, dtype=np.float64)
    return (pos + ray_dirs * depth[..., np.newaxis])[valid]


def project_points_to_coverage(pts, tgt_pos, H=WARP_H, W=WARP_W):
    """把累积点云投影到候选位置的全景图，返回覆盖 mask。"""
    if len(pts) == 0:
        return np.zeros((H, W), dtype=bool)
    tgt = np.array(tgt_pos, dtype=np.float64)
    vecs = pts - tgt
    x, y, z = vecs[:, 0], vecs[:, 1], vecs[:, 2]
    r_xy = np.sqrt(x ** 2 + y ** 2)
    phi = np.arctan2(z, r_xy)
    theta = np.arctan2(y, x) % (2 * np.pi)
    vi = np.clip(((np.pi / 2 - phi) / np.pi * H).astype(np.int32), 0, H - 1)
    uj = np.clip((theta / (2 * np.pi) * W).astype(np.int32), 0, W - 1)
    cov = np.zeros((H, W), dtype=bool)
    cov[vi, uj] = True
    pad = cov.copy()
    pad[1:, :] |= cov[:-1, :]
    pad[:-1, :] |= cov[1:, :]
    pad[:, 1:] |= cov[:, :-1]
    pad[:, :-1] |= cov[:, 1:]
    return pad


# ---- GPU 加速（延迟初始化，Phase 2 第一次选帧时检测）----
_GPU_BACKEND = None
_gpu_lib = None
_gpu_checked = False

def _init_gpu():
    """延迟初始化 GPU，避免模块加载时显存冲突"""
    global _GPU_BACKEND, _gpu_lib, _gpu_checked
    if _gpu_checked:
        return
    _gpu_checked = True

    try:
        import torch
        if torch.cuda.is_available():
            _GPU_BACKEND = "torch"
            _gpu_lib = torch
            print(f"[GPU] torch {torch.__version__} (CUDA)，选帧将使用 GPU 加速")
            return
    except ImportError:
        pass

    try:
        import cupy as cp
        try:
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
        except Exception:
            pass
        cp.zeros(1)
        _GPU_BACKEND = "cupy"
        _gpu_lib = cp
        print(f"[GPU] cupy {cp.__version__}，选帧将使用 GPU 加速")
        return
    except Exception as e:
        print(f"[Warning] cupy 初始化失败: {e}")

    print("[CPU] 未检测到 torch/cupy，选帧使用 CPU")


def _batch_coverage_gpu(pts_np, candidate_positions, remaining_indices, H, W):
    """GPU 批量投影：逐候选在 GPU 上算覆盖数

    返回: dict[ci] -> covered_pixels (int)
    """
    total_px = H * W
    results = {}

    if _GPU_BACKEND == "torch":
        import torch
        device = torch.device("cuda")
        pts_gpu = torch.from_numpy(pts_np).double().to(device)
        PI = torch.pi
        TWO_PI = 2 * torch.pi

        for ci in remaining_indices:
            tgt = torch.tensor(candidate_positions[ci], dtype=torch.float64, device=device)
            vecs = pts_gpu - tgt
            x, y, z = vecs[:, 0], vecs[:, 1], vecs[:, 2]
            r_xy = torch.sqrt(x ** 2 + y ** 2)
            phi = torch.atan2(z, r_xy)
            theta = torch.atan2(y, x) % TWO_PI
            vi = torch.clamp(((PI / 2 - phi) / PI * H).long(), 0, H - 1)
            uj = torch.clamp((theta / TWO_PI * W).long(), 0, W - 1)

            flat = vi * W + uj
            cov = torch.zeros(total_px, dtype=torch.bool, device=device)
            cov[flat] = True
            cov_2d = cov.view(H, W)
            pad = cov_2d.clone()
            pad[1:, :] |= cov_2d[:-1, :]
            pad[:-1, :] |= cov_2d[1:, :]
            pad[:, 1:] |= cov_2d[:, :-1]
            pad[:, :-1] |= cov_2d[:, 1:]
            results[ci] = int(pad.sum().item())

    elif _GPU_BACKEND == "cupy":
        import cupy as cp
        pts_gpu = cp.asarray(pts_np, dtype=cp.float64)
        PI = cp.pi
        TWO_PI = 2 * cp.pi

        for ci in remaining_indices:
            tgt = cp.array(candidate_positions[ci], dtype=cp.float64)
            vecs = pts_gpu - tgt
            x, y, z = vecs[:, 0], vecs[:, 1], vecs[:, 2]
            r_xy = cp.sqrt(x ** 2 + y ** 2)
            phi = cp.arctan2(z, r_xy)
            theta = cp.arctan2(y, x) % TWO_PI
            vi = cp.clip(((PI / 2 - phi) / PI * H).astype(cp.int32), 0, H - 1)
            uj = cp.clip((theta / TWO_PI * W).astype(cp.int32), 0, W - 1)

            flat = vi * W + uj
            cov = cp.zeros(total_px, dtype=cp.bool_)
            cov[flat] = True
            cov_2d = cov.reshape(H, W)
            pad = cov_2d.copy()
            pad[1:, :] |= cov_2d[:-1, :]
            pad[:-1, :] |= cov_2d[1:, :]
            pad[:, 1:] |= cov_2d[:, :-1]
            pad[:, :-1] |= cov_2d[:, 1:]
            results[ci] = int(cp.sum(pad))

    return results


def trim_depth(new_depth, new_pos, existing_pts, ray_dirs):
    H, W = new_depth.shape
    n_orig = int(np.sum(new_depth > 0))
    if len(existing_pts) == 0:
        return new_depth.copy(), n_orig, n_orig
    cov = project_points_to_coverage(existing_pts, new_pos, H, W)
    trimmed = new_depth.copy()
    trimmed[cov] = 0
    return trimmed, n_orig, int(np.sum(trimmed > 0))


def load_depth_downsampled(path, H=WARP_H, W=WARP_W):
    d = np.load(path).astype(np.float32)
    d = np.nan_to_num(d, nan=0.0)
    if d.shape == (H, W):
        return d
    try:
        import cv2
        return cv2.resize(d, (W, H), interpolation=cv2.INTER_AREA)
    except ImportError:
        h, w = d.shape
        bh, bw = h // H, w // W
        if bh < 1 or bw < 1:
            r = np.zeros((H, W), dtype=np.float32)
            r[:min(h, H), :min(w, W)] = d[:min(h, H), :min(w, W)]
            return r
        return d[:bh * H, :bw * W].reshape(H, bh, W, bw).mean(axis=(1, 3))


def select_next_frame(candidates, selected_idx, selected_pos,
                      all_pts, reachable=None):
    """选下一帧：纯贪心，选 score 最高的候选

    reachable: set of candidate indices，可达候选集合。
               None = 不限制。
    cupy 可用时自动 GPU 加速。
    """
    n = len(candidates)
    H, W = WARP_H, WARP_W
    total_px = H * W
    overlap_penalty = DEFAULT_OVERLAP_PENALTY

    remaining = []
    for i in range(n):
        if i in selected_idx:
            continue
        if reachable is not None and i not in reachable:
            continue
        remaining.append(i)

    if not remaining:
        return -1, 0.0, -999.0, 0

    # ---- GPU 路径 ----
    _init_gpu()
    if _GPU_BACKEND and len(all_pts) > 0:
        covered_map = _batch_coverage_gpu(all_pts, candidates, remaining, H, W)
        scores = {}
        for ci in remaining:
            covered = covered_map.get(ci, 0)
            new_r = (total_px - covered) / total_px
            ovl_r = covered / total_px
            scores[ci] = {
                "gain": new_r,
                "overlap": ovl_r,
                "score": new_r - overlap_penalty * ovl_r,
            }
    else:
        # ---- CPU 路径 ----
        scores = {}
        for ci in remaining:
            cov = project_points_to_coverage(all_pts, candidates[ci], H, W)
            covered = int(np.sum(cov))
            new_r = (total_px - covered) / total_px
            ovl_r = covered / total_px
            scores[ci] = {
                "gain": new_r,
                "overlap": ovl_r,
                "score": new_r - overlap_penalty * ovl_r,
            }

    best_ci, best_sc, best_g = -1, -999.0, 0.0
    for ci in remaining:
        if scores[ci]["score"] > best_sc:
            best_sc = scores[ci]["score"]
            best_ci = ci
            best_g = scores[ci]["gain"]

    return best_ci, best_g, best_sc, len(remaining)


def compute_max_depth(candidates):
    pos_arr = np.array(candidates)
    diag = float(np.linalg.norm(pos_arr.max(0) - pos_arr.min(0)))
    return diag * 1.5


# =====================================================================
# Phase 2: 边渲边选主循环
# =====================================================================

def run_phase2(cam_obj, candidates, mesh_center, output_dir,
               max_frames, resolution, depth_fo, args):

    ray_dirs = get_ray_dirs(WARP_H, WARP_W)
    max_depth = compute_max_depth(candidates)

    scene_diag = float(np.linalg.norm(
        np.array(candidates).max(0) - np.array(candidates).min(0)))

    selected_idx = set()
    selected_pos = []
    all_pts = np.empty((0, 3), dtype=np.float64)
    pts_chunks = []
    results = []

    # 可达性
    reachable = set()

    actual_gain_history = []
    delta_history = []

    # ======== 楼层分组（候选按 Z 聚类）========
    z_vals = sorted(set(round(c[2], 2) for c in candidates))
    floors = [[z_vals[0]]]
    for z in z_vals[1:]:
        if z - floors[-1][-1] > 1.0:
            floors.append([z])
        else:
            floors[-1].append(z)

    # 每个候选标记楼层（找 Z 最近的楼层）
    n_floors = len(floors)
    floor_mids = [sum(f) / len(f) for f in floors]  # 每层的 Z 中心
    candidate_floor = []
    for c in candidates:
        cz = c[2]
        fi = min(range(n_floors), key=lambda i: abs(cz - floor_mids[i]))
        candidate_floor.append(fi)

    current_floor = 0

    # 当前楼层的候选索引集合
    def floor_set(fi):
        return set(i for i, f in enumerate(candidate_floor) if f == fi)

    floor_names = [f"楼层{i+1}(Z={min(f):.1f}~{max(f):.1f})" for i, f in enumerate(floors)]

    print(f"\n{'='*60}")
    print(f"[Phase 2] 边渲边选 (候选={len(candidates)}, 固定={max_frames}帧)")
    print(f"{'='*60}")
    print(f"  {n_floors} 个楼层: {floor_names}")
    print(f"  高度层: {['%.2f' % z for z in z_vals]}")
    print(f"  选帧策略: 楼层顺序 + 层内全局最优 (可达优先, 无提前停止)")

    t_total = time.time()

    # 时间统计
    time_select = 0.0
    time_render = 0.0
    time_depth = 0.0
    time_reach = 0.0

    for frame_count in range(max_frames):

        # ======== 选位置 ========
        t_sel = time.time()
        if frame_count == 0:
            # F0: XY 取第一楼层候选的几何中心，Z 取高度层中心
            floor0_candidates = [(i, c) for i, c in enumerate(candidates)
                                 if candidate_floor[i] == 0]
            if floor0_candidates:
                f0_pts = np.array([c for _, c in floor0_candidates])
                xy_center = f0_pts[:, :2].mean(axis=0)  # XY 几何中心
                floor0_zs = sorted(set(c[2] for _, c in floor0_candidates))
                z_target = min(floor0_zs) + 1.2  # 楼板高度 + 1.7m ≈ 人眼高度
                target = np.array([xy_center[0], xy_center[1], z_target])
                dists_to_target = [np.linalg.norm(np.array(c) - target)
                                   for _, c in floor0_candidates]
                best_idx = int(np.argmin(dists_to_target))
                ci = floor0_candidates[best_idx][0]
            else:
                mc = np.array(mesh_center, dtype=np.float64)
                ci = int(np.argmin([np.linalg.norm(np.array(c) - mc)
                                     for c in candidates]))
            gain, score = 1.0, 1.0
            print(f"\n  F{frame_count}: 选候选[{ci}] "
                  f"(楼层中心, Z={candidates[ci][2]:.2f}m) "
                  f"[{floor_names[current_floor]}]")
        else:
            cur_floor_ids = floor_set(current_floor)
            floor_reachable = reachable & cur_floor_ids if reachable else set()

            ci, gain, score, n_remain = select_next_frame(
                candidates, selected_idx, selected_pos, all_pts,
                reachable=floor_reachable if floor_reachable else cur_floor_ids)

            # 当前楼层可达候选不够 → 扩展到当前楼层全局
            if ci < 0:
                ci, gain, score, n_remain = select_next_frame(
                    candidates, selected_idx, selected_pos, all_pts,
                    reachable=cur_floor_ids)

            # 当前楼层无候选 → 切换楼层
            if ci < 0:
                current_floor += 1
                if current_floor < n_floors:
                    print(f"\n  F{frame_count}: {floor_names[current_floor-1]} 候选用尽,"
                          f" 切换到 {floor_names[current_floor]}")
                    ci, gain, score, n_remain = select_next_frame(
                        candidates, selected_idx, selected_pos, all_pts,
                        reachable=floor_set(current_floor))
                    if ci < 0:
                        ci, gain, score, n_remain = select_next_frame(
                            candidates, selected_idx, selected_pos, all_pts,
                            reachable=None)

            # 全局无候选 → 无法继续
            if ci < 0:
                print(f"\n  F{frame_count}: 所有候选已用尽，停止 (已渲染 {len(results)} 帧)")
                break

            print(f"\n  F{frame_count}: 选候选[{ci}]  "
                  f"gain={gain:.1%}  score={score:.3f}  剩余={n_remain}"
                  f"  [Z={candidates[ci][2]:.2f} {floor_names[current_floor]}"
                  f" 可达={len(floor_reachable)}]")

        pos = candidates[ci]
        selected_idx.add(ci)
        selected_pos.append(pos)
        dt_sel = time.time() - t_sel
        time_select += dt_sel
        if frame_count > 0:
            print(f"    [选帧 {dt_sel:.1f}s]")

        # ======== 渲染 ========
        cam_rot = get_camera_rot(args.rotation_type, frame_count)
        print(f"    位置: [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]")
        print(f"    渲染...", end="", flush=True)
        t_r = time.time()

        rgb_path, depth_path, pose_path = render_frame_inprocess(
            cam_obj, frame_count, pos, cam_rot, output_dir, depth_fo)
        dt_r = time.time() - t_r
        time_render += dt_r
        print(f" {dt_r:.1f}s")

        # ======== depth → 3D 点云 ========
        t_dep = time.time()
        actual_gain = 1.0
        delta_ratio = 1.0

        if depth_path and os.path.exists(depth_path):
            depth = load_depth_downsampled(depth_path, WARP_H, WARP_W)
            total_px = WARP_H * WARP_W
            n_valid = int(np.sum(depth > 0))
            valid_ratio = n_valid / total_px

            if frame_count == 0:
                new_pts = depth_to_3d_points(pos, depth, ray_dirs, max_depth)
                pts_chunks.append(new_pts)
                all_pts = new_pts
                actual_gain = valid_ratio
                print(f"    depth: {n_valid}px ({valid_ratio:.0%} 有效)"
                      f" → {len(new_pts)} 个 3D 点 (全部)")
            else:
                # ---- 质量检查 ----
                MIN_VALID_RATIO = 0.30
                if valid_ratio < MIN_VALID_RATIO:
                    print(f"    depth: {n_valid}px ({valid_ratio:.0%} 有效)"
                          f" < {MIN_VALID_RATIO:.0%} → 室外/空壳，跳过此帧")
                    results.append({
                        "frame_id": frame_count,
                        "candidate_idx": ci,
                        "position": pos,
                        "gain": float(gain),
                        "actual_gain": 0.0,
                        "delta_ratio": 0.0,
                        "score": float(score),
                        "skipped": True,
                        "skip_reason": f"valid_ratio={valid_ratio:.1%}",
                    })
                    for fp in [rgb_path, depth_path]:
                        if fp and os.path.exists(fp):
                            try:
                                os.remove(fp)
                            except OSError:
                                pass
                    time_depth += time.time() - t_dep
                    continue

                trimmed, n_orig, n_new = trim_depth(
                    depth, pos, all_pts, ray_dirs)
                new_pts = depth_to_3d_points(pos, trimmed, ray_dirs, max_depth)
                pts_chunks.append(new_pts)
                all_pts = np.concatenate(pts_chunks)
                actual_gain = n_new / total_px
                delta_ratio = (len(new_pts) / len(all_pts)
                               if len(all_pts) > 0 else 1.0)
                print(f"    depth: {n_valid}px ({valid_ratio:.0%} 有效)"
                      f" → trim → {n_new}px 新增"
                      f" → {len(new_pts)} 个新 3D 点 (delta)")
                print(f"    累积点云: {len(all_pts)}")
                print(f"    实际gain: {actual_gain:.1%}, "
                      f"点云增量: {delta_ratio:.1%}")
        else:
            print(f"    [Error] 无 depth 文件！")
            break

        results.append({
            "frame_id": frame_count,
            "candidate_idx": ci,
            "position": pos,
            "gain": float(gain),
            "actual_gain": float(actual_gain),
            "delta_ratio": float(delta_ratio),
            "score": float(score),
        })
        time_depth += time.time() - t_dep

        # ======== 更新可达性 ========
        if IN_BLENDER:
            t_reach = time.time()
            scene = bpy.context.scene
            depsgraph = bpy.context.evaluated_depsgraph_get()
            n_new_reachable = 0
            for ci_check in range(len(candidates)):
                if ci_check in selected_idx or ci_check in reachable:
                    continue
                origin = Vector(pos)
                target = Vector(candidates[ci_check])
                direction = (target - origin).normalized()
                dist_to_target = (target - origin).length

                if dist_to_target < 0.1:
                    reachable.add(ci_check)
                    n_new_reachable += 1
                    continue

                hit, loc, *_ = scene.ray_cast(depsgraph, origin, direction)
                if not hit or (loc - origin).length >= dist_to_target * 0.95:
                    reachable.add(ci_check)
                    n_new_reachable += 1

            dt_reach = time.time() - t_reach
            time_reach += dt_reach
            print(f"    [可达性] 新增 {n_new_reachable} 个可达候选, "
                  f"总可达 {len(reachable)} / {len(candidates)} "
                  f"({dt_reach:.1f}s)")

        if frame_count > 0:
            actual_gain_history.append(actual_gain)
            delta_history.append(delta_ratio)

    # ======== 补帧：候选不足时生成扰动位置补齐到 max_frames ========
    valid_results = [r for r in results if not r.get("skipped")]
    if len(valid_results) < max_frames:
        need = max_frames - len(valid_results)
        print(f"\n  [补帧] 有效帧 {len(valid_results)}/{max_frames}，需补 {need} 帧")

        scene = bpy.context.scene
        depsgraph = bpy.context.evaluated_depsgraph_get()
        frame_count = (results[-1]["frame_id"] + 1) if results else 0
        PERTURB_OFFSETS = [0.2, 0.3, 0.15, 0.25, 0.35]
        perturb_round = 0

        while len(valid_results) < max_frames:
            # 先尝试未使用的原始候选
            ci_sup, gain_sup, score_sup, _ = select_next_frame(
                candidates, selected_idx, selected_pos, all_pts, reachable=None)

            if ci_sup >= 0:
                pos = candidates[ci_sup]
                selected_idx.add(ci_sup)
                selected_pos.append(pos)
                label = f"候选[{ci_sup}]"
            else:
                # 原始候选用尽 → 从已选位置生成扰动位置
                offset = PERTURB_OFFSETS[perturb_round % len(PERTURB_OFFSETS)]
                perturb_round += 1
                found = False

                _random.shuffle(selected_pos)
                for base_pos in selected_pos:
                    for _ in range(20):
                        dx = _random.uniform(-offset, offset)
                        dy = _random.uniform(-offset, offset)
                        new_pos = [base_pos[0] + dx, base_pos[1] + dy, base_pos[2]]

                        # raycast 验证：上下必须 hit（在室内）
                        origin = Vector(new_pos)
                        hit_up, *_ = scene.ray_cast(depsgraph, origin, Vector((0, 0, 1)))
                        hit_dn, *_ = scene.ray_cast(depsgraph, origin, Vector((0, 0, -1)))
                        if hit_up and hit_dn:
                            pos = new_pos
                            selected_pos.append(pos)
                            found = True
                            break
                    if found:
                        break

                if not found:
                    print(f"    无法生成有效扰动位置，停止补帧")
                    break
                label = f"扰动(±{offset:.2f}m)"
                gain_sup, score_sup = 0.0, 0.0

            cam_rot = get_camera_rot(args.rotation_type, frame_count)
            print(f"    补帧 F{frame_count}: {label}"
                  f" [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]")
            print(f"    渲染...", end="", flush=True)
            t_r = time.time()
            rgb_path, depth_path, pose_path = render_frame_inprocess(
                cam_obj, frame_count, pos, cam_rot, output_dir, depth_fo)
            dt_r = time.time() - t_r
            time_render += dt_r
            print(f" {dt_r:.1f}s")

            actual_gain = 0.0
            delta_ratio = 0.0
            skipped = False
            if depth_path and os.path.exists(depth_path):
                depth = load_depth_downsampled(depth_path, WARP_H, WARP_W)
                total_px = WARP_H * WARP_W
                n_valid = int(np.sum(depth > 0))
                valid_ratio = n_valid / total_px
                if valid_ratio < 0.30:
                    skipped = True
                    for fp in [rgb_path, depth_path]:
                        if fp and os.path.exists(fp):
                            try:
                                os.remove(fp)
                            except OSError:
                                pass
                    print(f"    valid={valid_ratio:.0%} < 30% → 跳过")
                else:
                    trimmed, n_orig, n_new = trim_depth(depth, pos, all_pts, ray_dirs)
                    new_pts = depth_to_3d_points(pos, trimmed, ray_dirs, max_depth)
                    pts_chunks.append(new_pts)
                    all_pts = np.concatenate(pts_chunks)
                    actual_gain = n_new / total_px
                    delta_ratio = len(new_pts) / len(all_pts) if len(all_pts) > 0 else 0
                    print(f"    depth: {n_new}px 新增, gain={actual_gain:.1%}")

            result_entry = {
                "frame_id": frame_count,
                "position": pos,
                "gain": float(gain_sup),
                "actual_gain": float(actual_gain),
                "delta_ratio": float(delta_ratio),
                "score": float(score_sup),
                "supplementary": True,
            }
            if skipped:
                result_entry["skipped"] = True
            results.append(result_entry)

            if not skipped:
                valid_results = [r for r in results if not r.get("skipped")]

            frame_count += 1

        valid_results = [r for r in results if not r.get("skipped")]
        print(f"  补帧完成: 有效帧 {len(valid_results)}/{max_frames}")

    dt = time.time() - t_total
    time_other = dt - time_select - time_render - time_depth - time_reach
    print(f"\n  {'─'*50}")
    print(f"  共 {len(results)} 帧, {dt:.1f}s ({dt/60:.1f}min)")
    print(f"  耗时分布:")
    print(f"    选帧:   {time_select:.1f}s ({time_select/max(dt,1)*100:.0f}%)"
          f"  — 点云投影评估候选")
    print(f"    渲染:   {time_render:.1f}s ({time_render/max(dt,1)*100:.0f}%)"
          f"  — Blender Cycles GPU")
    print(f"    深度:   {time_depth:.1f}s ({time_depth/max(dt,1)*100:.0f}%)"
          f"  — depth→点云+trim")
    print(f"    可达性: {time_reach:.1f}s ({time_reach/max(dt,1)*100:.0f}%)"
          f"  — raycast 扫描")
    if time_other > 1:
        print(f"    其他:   {time_other:.1f}s ({time_other/max(dt,1)*100:.0f}%)")

    return results


# =====================================================================
# 自动曝光
# =====================================================================

def auto_adjust_exposure(cam_obj, test_pos, output_dir, initial_exposure):
    """F0 位置低采样快速渲一帧，分析亮度，自动调整 exposure。

    目标：有效像素平均亮度 ≈ 120/255。
    过曝 (>200): 降 EV
    欠曝 (<40):  升 EV
    正常 (40~200): 不动
    """
    TARGET_MEAN = 120.0
    scene = bpy.context.scene
    original_samples = scene.cycles.samples

    # 低采样快速测试
    scene.cycles.samples = 16
    test_path = os.path.join(output_dir, "_exposure_test.png")
    scene.render.filepath = test_path

    cam_obj.location = Vector(test_pos)
    cam_obj.rotation_euler = Euler((math.pi / 2, 0, 0), 'XYZ')

    print(f"\n[自动曝光] 测试渲染 (16 samples, exposure={initial_exposure:.1f})...",
          end="", flush=True)
    t0 = time.time()
    bpy.ops.render.render(write_still=True)
    print(f" {time.time() - t0:.1f}s")

    # 分析亮度
    img = bpy.data.images.load(test_path)
    w, h = img.size[0], img.size[1]
    pixels = np.array(img.pixels[:]).reshape(h, w, -1)
    rgb = pixels[:, :, :3]
    brightness = (0.299 * rgb[:,:,0] + 0.587 * rgb[:,:,1] + 0.114 * rgb[:,:,2]) * 255

    # 只看非纯黑像素（排除天空/无效区域）
    valid_mask = brightness > 1.0
    n_valid = int(np.sum(valid_mask))
    if n_valid > 0:
        mean_b = float(np.mean(brightness[valid_mask]))
        # 过曝比例（亮度 > 250 的像素占比）
        overexposed = float(np.sum(brightness[valid_mask] > 250)) / n_valid
        # 欠曝比例（亮度 < 10 的像素占比）
        underexposed = float(np.sum(brightness[valid_mask] < 10)) / n_valid
    else:
        mean_b = 0.0
        overexposed = 0.0
        underexposed = 1.0

    bpy.data.images.remove(img)
    try:
        os.remove(test_path)
    except OSError:
        pass

    print(f"  亮度分析: 平均={mean_b:.0f}/255, "
          f"过曝={overexposed:.0%}, 欠曝={underexposed:.0%}, "
          f"有效像素={n_valid}/{h*w}")

    # 调整
    new_exposure = initial_exposure
    if mean_b < 1.0:
        new_exposure = initial_exposure + 4.0
        print(f"  [严重欠曝] exposure: {initial_exposure:.1f} → {new_exposure:.1f} (+4.0 EV)")
    elif mean_b < 40:
        ev_adj = min(4.0, math.log2(TARGET_MEAN / max(mean_b, 1.0)))
        new_exposure = initial_exposure + ev_adj + 1.0  # 额外 +1
        print(f"  [欠曝] exposure: {initial_exposure:.1f} → {new_exposure:.1f} (+{ev_adj:.1f} EV)")
    elif mean_b > 200:
        ev_adj = max(-4.0, math.log2(TARGET_MEAN / mean_b))
        new_exposure = initial_exposure + ev_adj
        print(f"  [过曝] exposure: {initial_exposure:.1f} → {new_exposure:.1f} ({ev_adj:.1f} EV)")
    elif overexposed > 0.3:
        # 平均还行但大面积过曝
        new_exposure = initial_exposure - 1.5
        print(f"  [局部过曝 {overexposed:.0%}] exposure: {initial_exposure:.1f} → {new_exposure:.1f} (-1.5 EV)")
    else:
        print(f"  [正常] 曝光无需调整")

    # 限幅
    new_exposure = max(-2.0, min(12.0, new_exposure))

    scene.view_settings.exposure = new_exposure
    scene.cycles.samples = original_samples
    return new_exposure


# =====================================================================
# 有效天花板检测（忽略塔尖/天线等异常高点）
# =====================================================================

def _detect_effective_ceiling(bmin, bmax, floor_z, ceiling_z_raw):
    """用 raycast 从多个 XY 采样点往上打，统计天花板高度的 75% 分位数。

    塔尖、天线等只有少量采样点能 hit 到，被分位数过滤掉。
    """
    scene = bpy.context.scene
    depsgraph = bpy.context.evaluated_depsgraph_get()
    dir_up = Vector((0, 0, 1))

    cx = (bmin[0] + bmax[0]) / 2
    cy = (bmin[1] + bmax[1]) / 2
    x_range = bmax[0] - bmin[0]
    y_range = bmax[1] - bmin[1]

    # 5x5 网格采样
    ceil_hits = []
    for ix in range(5):
        for iy in range(5):
            x = bmin[0] + x_range * (ix + 0.5) / 5
            y = bmin[1] + y_range * (iy + 0.5) / 5
            origin = Vector((x, y, floor_z + 0.5))
            hit, loc, *_ = scene.ray_cast(depsgraph, origin, dir_up)
            if hit:
                ceil_hits.append(loc.z)

    if not ceil_hits:
        print(f"  [天花板] 无 hit，使用 AABB: {ceiling_z_raw:.2f}m")
        return ceiling_z_raw

    ceil_hits.sort()
    # 75% 分位数：忽略最高的 25%（塔尖/天线）
    p75_idx = int(len(ceil_hits) * 0.75)
    effective_ceil = ceil_hits[min(p75_idx, len(ceil_hits) - 1)]

    # 至少保留 AABB 高度的合理范围（不能比中位数还低太多）
    median_ceil = ceil_hits[len(ceil_hits) // 2]
    effective_ceil = max(effective_ceil, median_ceil)

    # 不能比最低的 hit 还低（安全下限）
    effective_ceil = max(effective_ceil, floor_z + 2.5)

    if effective_ceil < ceiling_z_raw - 1.0:
        print(f"  [天花板] AABB={ceiling_z_raw:.2f}m → 有效={effective_ceil:.2f}m"
              f" (忽略 {ceiling_z_raw - effective_ceil:.1f}m 塔尖/天线)")
    else:
        print(f"  [天花板] {effective_ceil:.2f}m")

    return effective_ceil


# =====================================================================
# Blender 模式主函数
# =====================================================================

def main_blender():
    args = parse_args_blender()

    # 统一 scene_path
    if args.blend:
        scene_path = os.path.abspath(args.blend)
    else:
        scene_path = os.path.abspath(args.glb)

    output_dir = os.path.abspath(args.output_dir)
    resolution = tuple(int(x) for x in args.resolution.split(","))
    os.makedirs(output_dir, exist_ok=True)
    

    scene_ext = Path(scene_path).suffix.lower()

    print("=" * 60)
    print("ERPT Blend Pipeline v5（单进程边渲边选）")
    print("=" * 60)
    print(f"  Scene:      {scene_path}  [{scene_ext}]")
    print(f"  Output:     {output_dir}")
    print(f"  Max frames: {args.num_frames}")
    print(f"  Resolution: {resolution[0]}x{resolution[1]}")
    t_start = time.time()

    # ===== Phase 0: 加载场景 =====
    load_scene(scene_path)

    # ===== 渲染设置（只做一次） =====
    print(f"\n[Setup] 渲染配置")
    cam_obj = setup_erp_camera()
    setup_render_settings(resolution, args.engine, args.samples, args.exposure)
    setup_lighting()
    depth_fo = setup_depth_pass()

    rooms = json.loads(args.rooms) if args.rooms else {"rooms": []}
    for room in rooms["rooms"]:
        print(f"  房间: {room['room_id']}")
        # input("\n按Enter键继续...")
        bmin, bmax = np.array(room['bounds_min']), np.array(room['bounds_max'])

        # ===== Phase 1: 撒点 + 过滤 =====
        print(f"\n{'='*60}")
        print("[Phase 1] 多层撒点 + 4层过滤")
        print(f"{'='*60}")

        floor_z_raw, ceiling_z_raw = bmin[2], bmax[2]

        # 有效天花板检测：用 raycast 忽略塔尖等异常高点
        ceiling_z = _detect_effective_ceiling(bmin, bmax, floor_z_raw, ceiling_z_raw)
        floor_z = floor_z_raw

        heights = compute_camera_heights(floor_z, ceiling_z, args.camera_height,
                                        bmin=bmin, bmax=bmax)
        print(f"  场景 Z 范围: {floor_z:.2f} ~ {ceiling_z:.2f}m (总高 {ceiling_z - floor_z:.2f}m)")
        print(f"  相机层数: {len(heights)}")
        for i, z in enumerate(heights):
            print(f"    第{i+1}层: Z={z:.2f}m (离地 {z - floor_z:.2f}m)")

        x_range = bmax[0] - bmin[0]
        y_range = bmax[1] - bmin[1]
        n_layers = len(heights)
        scene_diag = math.sqrt(x_range ** 2 + y_range ** 2)

        x_sp = max(0.5, x_range / 20)
        y_sp = max(0.5, y_range / 20)
        nx = max(1, int((x_range - 2 * MARGIN) / args.grid_spacing))
        ny = max(1, int((y_range - 2 * MARGIN) / args.grid_spacing))
        total_user = nx * ny * n_layers

        if total_user <= 10000:
            x_sp = args.grid_spacing
            y_sp = args.grid_spacing
            print(f"  间距: {args.grid_spacing}m (候选≈{total_user}个)")
        else:
            nx_auto = max(1, int((x_range - 2 * MARGIN) / x_sp))
            ny_auto = max(1, int((y_range - 2 * MARGIN) / y_sp))
            total_auto = nx_auto * ny_auto * n_layers
            print(f"  [自适应] 场景 {x_range:.0f}x{y_range:.0f}m, "
                f"X间距={x_sp:.1f}m, Y间距={y_sp:.1f}m "
                f"(候选≈{total_auto})")

        candidates = generate_candidate_grid(bmin, bmax, x_sp, y_sp, heights)
        if not candidates:
            print("  [Error] 没有候选点")
            # sys.exit(1)

        room_height = ceiling_z - floor_z
        candidates = raycast_6layer_filter(candidates, room_height)
        if not candidates:
            print("  [Warning] 全部被过滤，使用 mesh 中心")
            cx = (bmin[0] + bmax[0]) / 2
            cy = (bmin[1] + bmax[1]) / 2
            candidates = [[cx, cy, heights[0]]]

        sel_dir = os.path.join(output_dir, "frame_selection")
        os.makedirs(sel_dir, exist_ok=True)
        np.save(os.path.join(sel_dir, "candidates_filtered.npy"),
                np.array(candidates))

        # ===== 自动曝光：用候选中心点快速测试 =====
        mesh_center = [(bmin[0] + bmax[0]) / 2,
                    (bmin[1] + bmax[1]) / 2,
                    (bmin[2] + bmax[2]) / 2]
        # # 选最靠近中心的候选作为测试点
        # mc = np.array(mesh_center)
        # test_dists = [np.linalg.norm(np.array(c) - mc) for c in candidates]
        # test_pos = candidates[int(np.argmin(test_dists))]
        space_output = os.path.join(output_dir, f"space_{room['room_id']:02d}")
        # # final_exposure = auto_adjust_exposure(cam_obj, test_pos, output_dir, depth_fo, args.exposure)
        # exposure_rate = 0.5
        # exposure_count = 5
        # exposure1 = auto_adjust_exposure(cam_obj, test_pos, space_output, args.exposure)
        # exposure2 = auto_adjust_exposure(cam_obj, test_pos, space_output, exposure1)
        # count = 2
        # while abs(exposure1 - exposure2) > exposure_rate and count <= exposure_count:
        #     exposure1 = exposure2
        #     exposure2 = auto_adjust_exposure(cam_obj, test_pos, space_output, exposure2)
        #     count += 1
        # print(f"  最终曝光: {exposure2:.1f} (共调整 {count-1} 次)")
        # ===== Phase 2: 边渲边选 =====
        results = run_phase2(
            cam_obj, candidates, mesh_center, space_output,
            args.num_frames, resolution, depth_fo, args)

        # ===== 保存选帧摘要 =====
        summary = {
            "scene": os.path.basename(scene_path),
            "scene_format": scene_ext,
            "total_frames": len(results),
            "candidates_count": len(candidates),
            "frames": [{
                "frame_id": r["frame_id"],
                "position": r["position"],
                "gain": r["gain"],
                "actual_gain": r["actual_gain"],
                "delta_ratio": r["delta_ratio"],
                "score": r["score"],
            } for r in results],
        }
        
        with open(os.path.join(sel_dir, "selected_frames.json"), "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        dt = time.time() - t_start
        print(f"\n{'='*60}")
        print(f"完成! {len(results)} 帧, {dt:.1f}s ({dt/60:.1f}min)")
        print(f"{'='*60}")
        print(f"输出目录: {output_dir}/")
        for r in results:
            fid = r["frame_id"]
            print(f"  panorama_{fid:04d}.png  +  _depth.npy  +  pose_{fid:04d}.json")


# =====================================================================
# 入口
# =====================================================================

if __name__ == "__main__":
    if IN_BLENDER:
        main_blender()
    else:
        main_python()
