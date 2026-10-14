#!/usr/bin/env python3
"""
获取.blend文件的场景信息（相机位置、边界框）

此脚本在Blender内部执行，用于获取.blend文件的场景信息。
输出JSON格式的数据供batch_render.py使用。

智能检测策略：
1. 如果场景中有相机，使用相机位置作为第一帧参考
2. 如果没有相机，使用室内对象（Floor、Wall、Room等）的边界框计算中心

使用方法:
    blender --background --python get_blend_bounds.py -- \
        --blend "path/to/scene.blend"

输出格式:
    [BOUNDS] {"bounds_min": [...], "bounds_max": [...], "center": [...], "camera": {...}}
"""

import bpy
import sys
import json
import argparse
from mathutils import Vector


# 室内对象关键词
INDOOR_KEYWORDS = [
    'room', 'interior', 'floor', 'wall', 'ceiling', 'indoor',
    'kitchen', 'bathroom', 'bedroom', 'living', 'dining',
    'door', 'window', 'lamp', 'light', 'sofa', 'table', 'chair',
    'bed', 'desk', 'cabinet', 'shelf', 'carpet', 'curtain',
    'stair', 'corridor', 'hallway', 'closet', 'wardrobe'
]


def parse_args():
    """解析命令行参数"""
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []
    
    parser = argparse.ArgumentParser(description="获取.blend文件场景信息")
    parser.add_argument("--blend", type=str, required=True,
                        help=".blend文件路径")
    
    return parser.parse_args(argv)


def is_indoor_object(obj_name):
    """判断对象是否是室内对象"""
    name_lower = obj_name.lower()
    return any(kw in name_lower for kw in INDOOR_KEYWORDS)


def get_scene_cameras():
    """
    获取场景中的所有相机
    
    Returns:
        cameras: 相机信息列表 [{"name": str, "position": [x,y,z], "rotation": [x,y,z]}, ...]
    """
    cameras = []
    for obj in bpy.context.scene.objects:
        if obj.type == 'CAMERA':
            loc = obj.location
            rot = obj.rotation_euler
            cameras.append({
                "name": obj.name,
                "position": [loc.x, loc.y, loc.z],
                "rotation": [rot.x, rot.y, rot.z]
            })
    return cameras


def get_scene_bounds(indoor_only=True):
    """
    获取场景中mesh物体的边界框（Blender坐标系：X右, Y前, Z上）
    
    Args:
        indoor_only: 是否仅考虑室内对象（默认True）
    
    Returns:
        bounds_min: 边界框最小坐标 [x, y, z]
        bounds_max: 边界框最大坐标 [x, y, z]
        center: 几何中心 [x, y, z]
        indoor_count: 室内对象数量
    """
    min_coords = [float('inf'), float('inf'), float('inf')]
    max_coords = [float('-inf'), float('-inf'), float('-inf')]
    
    mesh_count = 0
    indoor_count = 0
    
    for obj in bpy.context.scene.objects:
        if obj.type == 'MESH':
            mesh_count += 1
            
            # 如果启用室内过滤，检查对象名称
            if indoor_only and not is_indoor_object(obj.name):
                continue
            
            indoor_count += 1
            
            # 获取世界坐标下的边界框
            for corner in obj.bound_box:
                world_corner = obj.matrix_world @ Vector(corner)
                for i in range(3):
                    min_coords[i] = min(min_coords[i], world_corner[i])
                    max_coords[i] = max(max_coords[i], world_corner[i])
    
    # 如果没有找到任何符合条件的mesh
    if min_coords[0] == float('inf'):
        if indoor_only:
            print(f"[WARN] 未找到室内对象，回退到全场景边界框", file=sys.stderr)
            return get_scene_bounds(indoor_only=False)
        else:
            print(f"[WARN] 未找到任何mesh对象，使用默认边界框", file=sys.stderr)
            min_coords = [-5, -5, 0]
            max_coords = [5, 5, 3]
            indoor_count = 0
    
    # 计算几何中心
    center = [
        (min_coords[0] + max_coords[0]) / 2,
        (min_coords[1] + max_coords[1]) / 2,
        (min_coords[2] + max_coords[2]) / 2
    ]
    
    mode_str = "室内对象" if indoor_only else "全场景"
    print(f"[INFO] 边界框模式: {mode_str}", file=sys.stderr)
    print(f"[INFO] 找到 {indoor_count}/{mesh_count} 个对象", file=sys.stderr)
    print(f"[INFO] 边界框: min={[f'{x:.2f}' for x in min_coords]}, max={[f'{x:.2f}' for x in max_coords]}", file=sys.stderr)
    print(f"[INFO] 几何中心: [{center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}]", file=sys.stderr)
    
    return min_coords, max_coords, center, indoor_count


def main():
    args = parse_args()
    
    print(f"[INFO] 打开.blend文件: {args.blend}", file=sys.stderr)
    
    # 打开.blend文件
    bpy.ops.wm.open_mainfile(filepath=args.blend)
    
    # 1. 检测场景中的相机
    cameras = get_scene_cameras()
    print(f"[INFO] 检测到 {len(cameras)} 个相机", file=sys.stderr)
    
    camera_info = None
    first_frame_position = None
    position_source = "none"
    
    if cameras:
        # 使用第一个相机的位置作为参考
        cam = cameras[0]
        camera_info = cam
        first_frame_position = cam["position"]
        position_source = "camera"
        print(f"[INFO] 使用相机 '{cam['name']}' 的位置作为第一帧参考", file=sys.stderr)
        print(f"[INFO] 相机位置: [{cam['position'][0]:.2f}, {cam['position'][1]:.2f}, {cam['position'][2]:.2f}]", file=sys.stderr)
    
    # 2. 获取室内对象边界框
    bounds_min, bounds_max, center, indoor_count = get_scene_bounds(indoor_only=True)
    
    # 如果没有相机，使用室内边界框中心
    if first_frame_position is None:
        first_frame_position = center
        position_source = "indoor_bounds_center"
        print(f"[INFO] 无相机，使用室内边界框中心作为第一帧位置", file=sys.stderr)
    
    # 3. 获取场景单位比例（用于将米转换为场景单位）
    unit_scale = bpy.context.scene.unit_settings.scale_length
    unit_system = bpy.context.scene.unit_settings.system
    print(f"[INFO] 场景单位: scale={unit_scale}, system={unit_system}", file=sys.stderr)
    
    # 输出JSON格式（供batch_render.py解析）
    result = {
        "bounds_min": bounds_min,
        "bounds_max": bounds_max,
        "center": center,
        "first_frame_position": first_frame_position,
        "position_source": position_source,
        "camera": camera_info,
        "cameras_count": len(cameras),
        "indoor_objects_count": indoor_count,
        "unit_scale": unit_scale,  # 用于米->场景单位转换
        "coordinate_system": "blender_z_up"  # X右, Y前, Z上
    }
    
    # 使用特殊前缀，便于batch_render.py解析
    print(f"[BOUNDS] {json.dumps(result)}")


if __name__ == "__main__":
    main()
