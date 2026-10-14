#!/usr/bin/env python3
"""
HM3D语义信息读取工具

用于从semantic.txt或semantic.glb文件中读取房间语义信息
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np


def parse_semantic_txt(semantic_txt_path: str) -> Dict[int, List[int]]:
    """
    解析semantic.txt文件，提取房间ID到对象ID的映射
    
    semantic.txt格式：
    ID,颜色,类别名,房间ID
    
    Args:
        semantic_txt_path: semantic.txt文件路径
    
    Returns:
        room_to_objects: {room_id: [object_ids]} 字典
    """
    room_to_objects: Dict[int, List[int]] = {}
    
    if not os.path.exists(semantic_txt_path):
        return room_to_objects
    
    with open(semantic_txt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
        # 跳过第一行标题（如果有）
        start_idx = 0
        if len(lines) > 0 and 'HM3D Semantic Annotations' in lines[0]:
            start_idx = 1
        
        for line in lines[start_idx:]:
            line = line.strip()
            if not line:
                continue
            
            # 解析格式: ID,颜色,类别名,房间ID
            parts = line.split(',')
            if len(parts) >= 4:
                try:
                    object_id = int(parts[0].strip())
                    room_id_str = parts[3].strip()
                    
                    # 房间ID可能是数字或空字符串
                    if room_id_str.isdigit():
                        room_id = int(room_id_str)
                        if room_id not in room_to_objects:
                            room_to_objects[room_id] = []
                        room_to_objects[room_id].append(object_id)
                except (ValueError, IndexError):
                    continue
    
    return room_to_objects


def load_semantic_from_glb(semantic_glb_path: str) -> Dict[int, List[int]]:
    """
    从semantic.glb文件中读取语义信息
    
    Args:
        semantic_glb_path: semantic.glb文件路径
    
    Returns:
        room_to_objects: {room_id: [object_ids]} 字典
    """
    room_to_objects: Dict[int, List[int]] = {}
    
    if not os.path.exists(semantic_glb_path):
        return room_to_objects
    
    try:
        import trimesh
        
        scene = trimesh.load(semantic_glb_path, process=False)
        
        if isinstance(scene, trimesh.Scene):
            # 遍历场景中的所有几何体
            for name, geom in scene.geometry.items():
                if isinstance(geom, trimesh.Trimesh):
                    # 尝试从对象名称或材质中提取房间ID
                    # HM3D的semantic.glb中，对象名称可能包含房间信息
                    # 或者通过材质索引来区分房间
                    # 这里需要根据实际格式调整
                    # 暂时返回空字典，因为semantic.txt更可靠
                    pass
    except ImportError:
        print("[WARN] trimesh未安装，无法从GLB读取语义信息")
    except Exception as e:
        print(f"[WARN] 从GLB读取语义信息失败: {e}")
    
    return room_to_objects


def find_semantic_file(mesh_path: str) -> Optional[str]:
    """
    根据GLB文件路径查找对应的语义文件
    
    查找顺序：
    1. semantic.txt文件（优先）
    2. semantic.glb文件
    
    Args:
        mesh_path: GLB文件路径
    
    Returns:
        语义文件路径，如果不存在则返回None
    """
    mesh_path_obj = Path(mesh_path)
    mesh_dir = mesh_path_obj.parent
    mesh_stem = mesh_path_obj.stem  # 不含扩展名的文件名
    
    # 尝试查找semantic.txt
    # 路径模式：从 hm3d-example-glb-v0.2/... 到 hm3d-example-semantic-annots-v0.2/...
    semantic_txt_path = None
    semantic_glb_path = None
    
    # 方法1: 在同一目录下查找
    semantic_txt_candidate = mesh_dir / f"{mesh_stem}.semantic.txt"
    if semantic_txt_candidate.exists():
        semantic_txt_path = str(semantic_txt_candidate)
    
    semantic_glb_candidate = mesh_dir / f"{mesh_stem}.semantic.glb"
    if semantic_glb_candidate.exists():
        semantic_glb_path = str(semantic_glb_candidate)
    
    # 方法2: 在语义标注目录中查找
    # 从 glb-v0.2 推断到 semantic-annots-v0.2
    if 'glb-v0.2' in str(mesh_dir):
        semantic_dir = str(mesh_dir).replace('glb-v0.2', 'semantic-annots-v0.2')
        semantic_dir_obj = Path(semantic_dir)
        
        if semantic_dir_obj.exists():
            semantic_txt_candidate = semantic_dir_obj / f"{mesh_stem}.semantic.txt"
            if semantic_txt_candidate.exists():
                semantic_txt_path = str(semantic_txt_candidate)
            
            semantic_glb_candidate = semantic_dir_obj / f"{mesh_stem}.semantic.glb"
            if semantic_glb_candidate.exists():
                semantic_glb_path = str(semantic_glb_candidate)
    
    # 方法3: 在上级目录的语义标注目录中查找
    if semantic_txt_path is None and semantic_glb_path is None:
        # 尝试在dataset_HM3D目录下查找
        dataset_root = mesh_dir.parent.parent if 'glb-v0.2' in str(mesh_dir) else mesh_dir.parent
        semantic_annots_dir = dataset_root / "hm3d-example-semantic-annots-v0.2"
        
        if semantic_annots_dir.exists():
            # 查找场景目录
            for scene_dir in semantic_annots_dir.iterdir():
                if scene_dir.is_dir():
                    semantic_txt_candidate = scene_dir / f"{mesh_stem}.semantic.txt"
                    if semantic_txt_candidate.exists():
                        semantic_txt_path = str(semantic_txt_candidate)
                        break
                    
                    semantic_glb_candidate = scene_dir / f"{mesh_stem}.semantic.glb"
                    if semantic_glb_candidate.exists():
                        semantic_glb_path = str(semantic_glb_candidate)
                        break
    
    # 优先返回semantic.txt
    if semantic_txt_path:
        return semantic_txt_path
    elif semantic_glb_path:
        return semantic_glb_path
    else:
        return None


def load_semantic_info(mesh_path: str) -> Tuple[Optional[Dict[int, List[int]]], Optional[str]]:
    """
    加载语义信息（优先使用semantic.txt，如果不存在则使用semantic.glb）
    
    Args:
        mesh_path: GLB文件路径
    
    Returns:
        (room_to_objects, semantic_file_path): 
        - room_to_objects: {room_id: [object_ids]} 字典，如果不存在则返回None
        - semantic_file_path: 使用的语义文件路径，如果不存在则返回None
    """
    semantic_file_path = find_semantic_file(mesh_path)
    
    if semantic_file_path is None:
        return None, None
    
    # 根据文件类型选择解析方法
    if semantic_file_path.endswith('.txt'):
        room_to_objects = parse_semantic_txt(semantic_file_path)
        return room_to_objects, semantic_file_path
    elif semantic_file_path.endswith('.glb'):
        room_to_objects = load_semantic_from_glb(semantic_file_path)
        return room_to_objects, semantic_file_path
    else:
        return None, None
