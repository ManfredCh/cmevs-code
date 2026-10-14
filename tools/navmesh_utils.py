#!/usr/bin/env python3
"""
NavMesh解析工具

用于解析HM3D数据集的NavMesh文件，提取可导航区域（islands）
"""

import os
import struct
import numpy as np
from typing import List, Dict, Optional, Tuple
from collections import defaultdict


def parse_navmesh_binary(navmesh_path: str) -> Optional[Dict]:
    """
    解析Recast Navigation格式的NavMesh二进制文件
    
    NavMesh格式（基于分析）：
    - 文件头：'TESM' (4字节)
    - 版本/标志：uint32, uint32
    - 边界框：float32x3 (min), float32x3 (max)
    - 顶点数据、三角形数据、区域数据等
    
    Args:
        navmesh_path: NavMesh文件路径
    
    Returns:
        navmesh_data: 包含顶点、三角形、区域信息的字典，如果解析失败返回None
    """
    if not os.path.exists(navmesh_path):
        return None
    
    try:
        with open(navmesh_path, 'rb') as f:
            # 读取文件头
            header = f.read(4)
            if header != b'TESM':
                print(f"[WARN] NavMesh文件头不匹配: {header}")
                return None
            
            # 读取版本/标志（可能是版本号和标志）
            version = struct.unpack('<I', f.read(4))[0]
            flags = struct.unpack('<I', f.read(4))[0]
            
            # 读取边界框（可能是场景边界框）
            bbox_min = struct.unpack('<fff', f.read(12))
            bbox_max = struct.unpack('<fff', f.read(12))
            
            # 尝试读取更多数据
            # Recast格式可能包含：
            # - 顶点数量
            # - 顶点数据
            # - 多边形数量
            # - 多边形数据
            # - 区域信息
            
            # 由于格式复杂，我们采用更实用的方法：
            # 1. 尝试读取所有剩余数据
            # 2. 或者使用其他方法（如结合语义信息）
            
            # 暂时返回基本信息
            return {
                'header': header,
                'version': version,
                'flags': flags,
                'bbox_min': np.array(bbox_min),
                'bbox_max': np.array(bbox_max),
                'raw_data': f.read()  # 剩余数据
            }
    
    except Exception as e:
        print(f"[ERROR] 解析NavMesh文件失败: {e}")
        return None


def extract_spaces_from_semantic_glb(semantic_glb_path: str, 
                                     room_to_objects: Dict[int, List[int]] = None,
                                     use_clustering: bool = True,
                                     cluster_eps: float = 0.8,
                                     cluster_min_samples: int = 2) -> List[Dict]:
    """
    从semantic.glb中提取所有空间，使用空间聚类方法
    
    策略：
    1. 提取所有对象的中心点
    2. 使用DBSCAN聚类识别空间
    3. 如果提供了语义标注，可以结合使用
    
    Args:
        semantic_glb_path: semantic.glb文件路径
        room_to_objects: 房间到对象的映射 {room_id: [object_ids]}（可选）
        use_clustering: 是否使用空间聚类（默认True）
        cluster_eps: DBSCAN聚类半径（米），默认2.0
        cluster_min_samples: DBSCAN最小样本数，默认3
    
    Returns:
        spaces: 空间列表，每个元素包含空间信息
    """
    spaces = []
    
    if not os.path.exists(semantic_glb_path):
        return spaces
    
    try:
        import trimesh
        
        scene = trimesh.load(semantic_glb_path, process=False)
        if not isinstance(scene, trimesh.Scene):
            return spaces
        
        # 提取所有对象的信息（中心点和顶点）
        obj_info_list = []
        obj_index_to_info = {}
        
        for idx, (name, geom) in enumerate(scene.geometry.items(), 1):
            if isinstance(geom, trimesh.Trimesh):
                # 获取变换矩阵
                transform = np.eye(4)
                for node_name in scene.graph.nodes_geometry:
                    if scene.graph[node_name][1] == name:
                        transform = scene.graph[node_name][0]
                        break
                
                # 计算对象中心点（世界坐标）
                local_center = geom.bounds.mean(axis=0)
                world_center = np.dot(transform[:3, :3], local_center) + transform[:3, 3]
                
                # 获取对象的所有顶点（世界坐标）
                world_vertices = np.dot(geom.vertices, transform[:3, :3].T) + transform[:3, 3]
                
                obj_info = {
                    'index': idx,
                    'name': name,
                    'geom': geom,
                    'center': world_center,
                    'vertices': world_vertices,
                    'transform': transform
                }
                obj_info_list.append(obj_info)
                obj_index_to_info[idx] = obj_info
        
        if not obj_info_list:
            return spaces
        
        # 方法1：如果使用聚类，基于对象中心点进行空间聚类
        if use_clustering:
            try:
                from sklearn.cluster import DBSCAN
                
                # 提取所有对象的中心点
                centers = np.array([info['center'] for info in obj_info_list])
                
                # 使用DBSCAN聚类
                # 方法1：只考虑X和Y坐标（适合单层场景）
                # 方法2：考虑3D坐标，但Z轴权重较小（适合多层场景）
                # 这里先尝试2D，如果空间太少再尝试3D
                use_3d = len(obj_info_list) > 50  # 如果对象很多，可能有多层
                
                if use_3d:
                    # 3D聚类，但Z轴权重较小（高度差异通常比水平距离大）
                    centers_3d = centers.copy()
                    centers_3d[:, 2] = centers_3d[:, 2] * 0.5  # Z轴权重减半
                    clustering = DBSCAN(eps=cluster_eps, min_samples=cluster_min_samples)
                    cluster_labels = clustering.fit_predict(centers_3d)
                else:
                    # 2D聚类（只使用X和Y）
                    centers_2d = centers[:, :2]
                    clustering = DBSCAN(eps=cluster_eps, min_samples=cluster_min_samples)
                    cluster_labels = clustering.fit_predict(centers_2d)
                
                # 统计聚类结果
                unique_labels = set(cluster_labels)
                if -1 in unique_labels:
                    unique_labels.remove(-1)  # 移除噪声点
                
                print(f"[INFO] DBSCAN聚类识别到 {len(unique_labels)} 个空间（eps={cluster_eps}, min_samples={cluster_min_samples}）")
                
                # 为每个聚类计算边界框
                for cluster_id in sorted(unique_labels):
                    # 获取该聚类的所有对象
                    cluster_objects = [obj_info_list[i] for i in range(len(obj_info_list)) 
                                      if cluster_labels[i] == cluster_id]
                    
                    if not cluster_objects:
                        continue
                    
                    # 合并所有对象的顶点
                    all_vertices = np.vstack([obj['vertices'] for obj in cluster_objects])
                    
                    bounds_min = all_vertices.min(axis=0)
                    bounds_max = all_vertices.max(axis=0)
                    center = (bounds_min + bounds_max) / 2
                    size = bounds_max - bounds_min
                    
                    # 检查空间是否有效（至少0.2米，降低阈值以识别更多空间）
                    # 注意：太小的空间可能是噪声，但用户希望识别所有空间
                    if np.all(size > 0.2):
                        spaces.append({
                            "space_id": cluster_id,
                            "bounds_min": bounds_min,
                            "bounds_max": bounds_max,
                            "center": center,
                            "object_count": len(cluster_objects),
                            "source": "spatial_clustering"
                        })
                
            except ImportError:
                print("[WARN] sklearn未安装，无法使用空间聚类，回退到语义标注方法")
                use_clustering = False
        
        # 方法2：如果提供了语义标注且未使用聚类，使用语义标注
        if not use_clustering and room_to_objects:
            # 为每个房间计算边界框（使用semantic.glb中存在的对象）
            for room_id in sorted(room_to_objects.keys()):
                obj_ids = room_to_objects[room_id]
                # 只使用在semantic.glb中存在的对象ID（1-206）
                valid_obj_ids = [obj_id for obj_id in obj_ids if obj_id in obj_index_to_info]
                
                if valid_obj_ids:
                    # 收集这些对象的所有顶点
                    all_vertices = []
                    for obj_id in valid_obj_ids:
                        info = obj_index_to_info[obj_id]
                        all_vertices.append(info['vertices'])
                    
                    if all_vertices:
                        all_vertices = np.vstack(all_vertices)
                        bounds_min = all_vertices.min(axis=0)
                        bounds_max = all_vertices.max(axis=0)
                        center = (bounds_min + bounds_max) / 2
                        size = bounds_max - bounds_min
                        
                        # 检查空间是否有效（至少0.1米）
                        if np.all(size > 0.1):
                            spaces.append({
                                "space_id": room_id,
                                "bounds_min": bounds_min,
                                "bounds_max": bounds_max,
                                "center": center,
                                "object_count": len(valid_obj_ids),
                                "total_objects_in_annotation": len(obj_ids),
                                "source": "semantic_glb"
                            })
        
    except Exception as e:
        print(f"[WARN] 从semantic.glb提取空间失败: {e}")
        import traceback
        traceback.print_exc()
    
    return spaces


def extract_spaces_from_glb(glb_path: str, 
                             room_to_objects: Dict[int, List[int]] = None,
                             use_clustering: bool = True,
                             cluster_eps: float = 0.8,
                             cluster_min_samples: int = 2) -> List[Dict]:
    """
    从原始GLB文件中提取所有空间，使用空间聚类方法
    
    当没有semantic.glb文件时，使用此方法作为回退方案
    
    Args:
        glb_path: GLB文件路径
        room_to_objects: 房间到对象的映射 {room_id: [object_ids]}（可选）
        use_clustering: 是否使用空间聚类（默认True）
        cluster_eps: DBSCAN聚类半径（米），默认0.8
        cluster_min_samples: DBSCAN最小样本数，默认2
    
    Returns:
        spaces: 空间列表，每个元素包含空间信息
    """
    spaces = []
    
    if not os.path.exists(glb_path):
        return spaces
    
    try:
        import trimesh
        
        scene = trimesh.load(glb_path, process=False)
        if not isinstance(scene, trimesh.Scene):
            return spaces
        
        # 提取所有对象的信息（中心点和顶点）
        obj_info_list = []
        obj_index_to_info = {}
        
        for idx, (name, geom) in enumerate(scene.geometry.items(), 1):
            if isinstance(geom, trimesh.Trimesh):
                # 获取变换矩阵
                transform = np.eye(4)
                for node_name in scene.graph.nodes_geometry:
                    if scene.graph[node_name][1] == name:
                        transform = scene.graph[node_name][0]
                        break
                
                # 计算对象中心点（世界坐标）
                local_center = geom.bounds.mean(axis=0)
                world_center = np.dot(transform[:3, :3], local_center) + transform[:3, 3]
                
                # 获取对象的所有顶点（世界坐标）
                world_vertices = np.dot(geom.vertices, transform[:3, :3].T) + transform[:3, 3]
                
                obj_info = {
                    'index': idx,
                    'name': name,
                    'geom': geom,
                    'center': world_center,
                    'vertices': world_vertices,
                    'transform': transform
                }
                obj_info_list.append(obj_info)
                obj_index_to_info[idx] = obj_info
        
        if not obj_info_list:
            return spaces
        
        # 使用聚类，基于对象中心点进行空间聚类
        if use_clustering:
            try:
                from sklearn.cluster import DBSCAN
                
                # 提取所有对象的中心点
                centers = np.array([info['center'] for info in obj_info_list])
                
                # 使用DBSCAN聚类
                # 根据对象数量决定使用2D还是3D聚类
                use_3d = len(obj_info_list) > 50  # 如果对象很多，可能有多层
                
                if use_3d:
                    # 3D聚类，但Z轴权重较小（高度差异通常比水平距离大）
                    centers_3d = centers.copy()
                    centers_3d[:, 2] = centers_3d[:, 2] * 0.5  # Z轴权重减半
                    clustering = DBSCAN(eps=cluster_eps, min_samples=cluster_min_samples)
                    cluster_labels = clustering.fit_predict(centers_3d)
                else:
                    # 2D聚类（只使用X和Y）
                    centers_2d = centers[:, :2]
                    clustering = DBSCAN(eps=cluster_eps, min_samples=cluster_min_samples)
                    cluster_labels = clustering.fit_predict(centers_2d)
                
                # 统计聚类结果
                unique_labels = set(cluster_labels)
                if -1 in unique_labels:
                    unique_labels.remove(-1)  # 移除噪声点
                
                print(f"[INFO] DBSCAN聚类识别到 {len(unique_labels)} 个空间（eps={cluster_eps}, min_samples={cluster_min_samples}）")
                
                # 为每个聚类计算边界框
                for cluster_id in sorted(unique_labels):
                    # 获取该聚类的所有对象
                    cluster_objects = [obj_info_list[i] for i in range(len(obj_info_list)) 
                                      if cluster_labels[i] == cluster_id]
                    
                    if not cluster_objects:
                        continue
                    
                    # 合并所有对象的顶点
                    all_vertices = np.vstack([obj['vertices'] for obj in cluster_objects])
                    
                    bounds_min = all_vertices.min(axis=0)
                    bounds_max = all_vertices.max(axis=0)
                    center = (bounds_min + bounds_max) / 2
                    size = bounds_max - bounds_min
                    
                    # 检查空间是否有效（至少0.2米）
                    if np.all(size > 0.2):
                        spaces.append({
                            "space_id": cluster_id,
                            "bounds_min": bounds_min,
                            "bounds_max": bounds_max,
                            "center": center,
                            "object_count": len(cluster_objects),
                            "source": "glb_clustering"
                        })
                
            except ImportError:
                print("[WARN] sklearn未安装，无法使用空间聚类")
            except Exception as e:
                print(f"[WARN] 空间聚类失败: {e}")
        
    except Exception as e:
        print(f"[WARN] 从GLB提取空间失败: {e}")
        import traceback
        traceback.print_exc()
    
    return spaces


def extract_navmesh_islands_from_semantic(navmesh_path: str, mesh_path: str, 
                                         room_to_objects: Dict[int, List[int]]) -> List[Dict]:
    """
    结合语义信息从NavMesh提取空间区域
    
    现在改为使用semantic.glb直接计算，不依赖NavMesh的island分割
    
    Args:
        navmesh_path: NavMesh文件路径（保留参数以保持兼容性）
        mesh_path: GLB文件路径
        room_to_objects: 房间到对象的映射 {room_id: [object_ids]}
    
    Returns:
        spaces: 空间列表
    """
    # 查找semantic.glb文件（优先查找.glb，而不是.txt）
    from semantic_utils import find_semantic_file
    from pathlib import Path
    import os
    
    # 先尝试直接查找semantic.glb
    mesh_path_obj = Path(mesh_path)
    mesh_dir = mesh_path_obj.parent
    mesh_stem = mesh_path_obj.stem
    
    semantic_glb_path = None
    
    # 方法1: 在同一目录下查找
    semantic_glb_candidate = mesh_dir / f"{mesh_stem}.semantic.glb"
    if semantic_glb_candidate.exists():
        semantic_glb_path = str(semantic_glb_candidate)
    
    # 方法2: 在语义标注目录中查找
    if not semantic_glb_path and 'glb-v0.2' in str(mesh_dir):
        semantic_dir = str(mesh_dir).replace('glb-v0.2', 'semantic-annots-v0.2')
        semantic_dir_obj = Path(semantic_dir)
        
        if semantic_dir_obj.exists():
            semantic_glb_candidate = semantic_dir_obj / f"{mesh_stem}.semantic.glb"
            if semantic_glb_candidate.exists():
                semantic_glb_path = str(semantic_glb_candidate)
    
    # 方法3: 在上级目录的语义标注目录中查找
    if not semantic_glb_path:
        dataset_root = mesh_dir.parent.parent if 'glb-v0.2' in str(mesh_dir) else mesh_dir.parent
        semantic_annots_dir = dataset_root / "hm3d-example-semantic-annots-v0.2"
        
        if semantic_annots_dir.exists():
            # 查找场景目录
            for scene_dir in semantic_annots_dir.iterdir():
                if scene_dir.is_dir():
                    semantic_glb_candidate = scene_dir / f"{mesh_stem}.semantic.glb"
                    if semantic_glb_candidate.exists():
                        semantic_glb_path = str(semantic_glb_candidate)
                        break
    
    if semantic_glb_path and os.path.exists(semantic_glb_path):
        # 直接使用semantic.glb + 空间聚类
        return extract_spaces_from_semantic_glb(semantic_glb_path, room_to_objects, use_clustering=True)
    elif mesh_path and os.path.exists(mesh_path):
        # 如果没有semantic.glb，尝试直接从原始GLB文件进行空间聚类
        print(f"  [INFO] 未找到semantic.glb，尝试从原始GLB文件进行空间聚类...")
        return extract_spaces_from_glb(mesh_path, room_to_objects, use_clustering=True)
    else:
        # 回退到原有方法
        return []


def get_room_bounds_from_mesh(mesh_path: str, object_ids: List[int], 
                               scene=None) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    从mesh中获取指定对象ID列表的边界框
    
    Args:
        mesh_path: mesh文件路径
        object_ids: 对象ID列表
        scene: 已加载的场景（可选，避免重复加载）
    
    Returns:
        bounds_min, bounds_max: 边界框的最小和最大坐标
    """
    try:
        import trimesh
        import re
        
        if scene is None:
            scene = trimesh.load(mesh_path, process=False)
        
        if not isinstance(scene, trimesh.Scene):
            return None, None
        
        object_ids_set = set(object_ids)
        all_vertices = []
        
        for name, geom in scene.geometry.items():
            if isinstance(geom, trimesh.Trimesh):
                # 从对象名称提取ID
                obj_id = None
                match = re.search(r'chunk(\d+)', name)
                if match:
                    chunk_num = int(match.group(1))
                    obj_id = chunk_num + 1
                
                if obj_id is None or obj_id not in object_ids_set:
                    continue
                
                # 获取变换矩阵
                for node_name in scene.graph.nodes_geometry:
                    if scene.graph[node_name][1] == name:
                        transform = scene.graph[node_name][0]
                        vertices = np.dot(geom.vertices, transform[:3, :3].T) + transform[:3, 3]
                        all_vertices.append(vertices)
                        break
        
        if all_vertices:
            all_vertices = np.vstack(all_vertices)
            bounds_min = all_vertices.min(axis=0)
            bounds_max = all_vertices.max(axis=0)
            return bounds_min, bounds_max
        
        return None, None
    
    except Exception as e:
        print(f"[WARN] 获取房间边界框失败: {e}")
        return None, None


def extract_navmesh_islands_connectivity(navmesh_path: str) -> List[Dict]:
    """
    通过连通性分析提取NavMesh中的独立区域（islands）
    
    由于直接解析Recast格式复杂，此函数尝试：
    1. 解析NavMesh的基本几何信息
    2. 使用连通性分析识别独立区域
    3. 计算每个区域的边界框和中心
    
    Args:
        navmesh_path: NavMesh文件路径
    
    Returns:
        islands: 独立区域列表，每个元素包含边界框和中心信息
    """
    # 由于NavMesh格式复杂且habitat-sim不可用，
    # 这里返回空列表，实际使用时会结合语义信息
    # 如果将来需要直接解析，可以在这里实现
    
    print("[INFO] NavMesh直接解析暂未实现，使用语义信息结合方法")
    return []


def navmesh_to_3d_spaces(navmesh_path: str, mesh_path: str = None, 
                         room_to_objects: Dict[int, List[int]] = None,
                         use_clustering: bool = True) -> List[Dict]:
    """
    将NavMesh区域映射到3D空间
    
    优先策略：
    1. 如果提供了语义信息，使用semantic.glb + 空间聚类识别所有空间
    2. 否则尝试直接解析NavMesh
    
    Args:
        navmesh_path: NavMesh文件路径（保留参数以保持兼容性）
        mesh_path: GLB文件路径（可选，用于查找semantic.glb）
        room_to_objects: 房间到对象的映射（可选，用于结合语义信息）
        use_clustering: 是否使用空间聚类（默认True）
    
    Returns:
        spaces: 空间列表，每个元素包含空间ID、边界框、中心等信息
    """
    # 优先使用semantic.glb + 空间聚类方法（可以识别所有空间）
    if mesh_path:
        from pathlib import Path
        import os
        
        # 直接查找semantic.glb文件（不依赖find_semantic_file，因为它优先返回.txt）
        mesh_path_obj = Path(mesh_path)
        mesh_dir = mesh_path_obj.parent
        mesh_stem = mesh_path_obj.stem
        
        semantic_glb_path = None
        
        # 方法1: 在同一目录下查找
        semantic_glb_candidate = mesh_dir / f"{mesh_stem}.semantic.glb"
        if semantic_glb_candidate.exists():
            semantic_glb_path = str(semantic_glb_candidate)
        
        # 方法2: 在语义标注目录中查找
        if not semantic_glb_path and 'glb-v0.2' in str(mesh_dir):
            semantic_dir = str(mesh_dir).replace('glb-v0.2', 'semantic-annots-v0.2')
            semantic_dir_obj = Path(semantic_dir)
            
            if semantic_dir_obj.exists():
                semantic_glb_candidate = semantic_dir_obj / f"{mesh_stem}.semantic.glb"
                if semantic_glb_candidate.exists():
                    semantic_glb_path = str(semantic_glb_candidate)
        
        # 方法3: 在上级目录的语义标注目录中查找
        if not semantic_glb_path:
            dataset_root = mesh_dir.parent.parent if 'glb-v0.2' in str(mesh_dir) else mesh_dir.parent
            semantic_annots_dir = dataset_root / "hm3d-example-semantic-annots-v0.2"
            
            if semantic_annots_dir.exists():
                # 查找场景目录
                for scene_dir in semantic_annots_dir.iterdir():
                    if scene_dir.is_dir():
                        semantic_glb_candidate = scene_dir / f"{mesh_stem}.semantic.glb"
                        if semantic_glb_candidate.exists():
                            semantic_glb_path = str(semantic_glb_candidate)
                            break
        
        if semantic_glb_path and os.path.exists(semantic_glb_path):
            # 使用semantic.glb + 空间聚类识别所有空间
            return extract_spaces_from_semantic_glb(semantic_glb_path, room_to_objects, 
                                                   use_clustering=use_clustering)
        elif use_clustering and mesh_path and os.path.exists(mesh_path):
            # 如果没有semantic.glb，尝试直接从原始GLB文件进行空间聚类
            print(f"  [INFO] 未找到semantic.glb，尝试从原始GLB文件进行空间聚类...")
            return extract_spaces_from_glb(mesh_path, room_to_objects, 
                                         use_clustering=True)
        elif room_to_objects:
            # 回退到原有方法
            return extract_navmesh_islands_from_semantic(navmesh_path, mesh_path, room_to_objects)
    
    # 否则尝试直接解析NavMesh（如果实现）
    return extract_navmesh_islands_connectivity(navmesh_path)


def find_navmesh_file(mesh_path: str) -> Optional[str]:
    """
    根据GLB文件路径查找对应的NavMesh文件
    
    路径模式：
    - GLB: hm3d-example-glb-v0.2/{scene_id}/{scene_id}.glb
    - NavMesh: hm3d-example-habitat-v0.2/{scene_id}/{scene_id}.basis.navmesh
    
    Args:
        mesh_path: GLB文件路径
    
    Returns:
        NavMesh文件路径，如果不存在则返回None
    """
    from pathlib import Path
    
    mesh_path_obj = Path(mesh_path)
    mesh_dir = mesh_path_obj.parent
    mesh_stem = mesh_path_obj.stem  # 不含扩展名的文件名
    
    # 方法1: 在同一目录下查找
    navmesh_candidate = mesh_dir / f"{mesh_stem}.basis.navmesh"
    if navmesh_candidate.exists():
        return str(navmesh_candidate)
    
    # 方法2: 在habitat目录中查找
    # 从 glb-v0.2 推断到 habitat-v0.2
    if 'glb-v0.2' in str(mesh_dir):
        habitat_dir = str(mesh_dir).replace('glb-v0.2', 'habitat-v0.2')
        habitat_dir_obj = Path(habitat_dir)
        
        if habitat_dir_obj.exists():
            navmesh_candidate = habitat_dir_obj / f"{mesh_stem}.basis.navmesh"
            if navmesh_candidate.exists():
                return str(navmesh_candidate)
    
    # 方法3: 在上级目录的habitat目录中查找
    dataset_root = mesh_dir.parent.parent if 'glb-v0.2' in str(mesh_dir) else mesh_dir.parent
    habitat_dir = dataset_root / "hm3d-example-habitat-v0.2"
    
    if habitat_dir.exists():
        # 查找场景目录
        for scene_dir in habitat_dir.iterdir():
            if scene_dir.is_dir():
                navmesh_candidate = scene_dir / f"{mesh_stem}.basis.navmesh"
                if navmesh_candidate.exists():
                    return str(navmesh_candidate)
                
                # 也可能文件名不同，尝试查找所有.navmesh文件
                navmesh_files = list(scene_dir.glob("*.navmesh"))
                if navmesh_files:
                    # 返回第一个找到的（通常只有一个）
                    return str(navmesh_files[0])
    
    return None
