"""
位姿工具模块

位姿加载、验证和转换。

ERPT_native 位姿格式：
{
    "frame_id": int,
    "position": [x, y, z],          # 相机中心在世界坐标系的位置（米）
    "rotation_quaternion": [w, x, y, z]  # camera->world 旋转 (R_cw)
}
"""

import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from .io_utils import load_json

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.coordinate import (
    quat_wxyz_to_rotation_matrix,
    validate_rotation_matrix,
    orthonormalize_rotation,
)


@dataclass
class Pose:
    """位姿数据类"""
    frame_id: int
    position: np.ndarray        # (3,) 相机中心在世界坐标系的位置
    R_cw: np.ndarray            # (3, 3) camera->world 旋转矩阵
    
    @property
    def R_wc(self) -> np.ndarray:
        """world->camera 旋转矩阵"""
        return self.R_cw.T
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        from core.coordinate import rotation_matrix_to_quat_wxyz
        quat = rotation_matrix_to_quat_wxyz(self.R_cw)
        return {
            "frame_id": self.frame_id,
            "position": self.position.tolist(),
            "rotation_quaternion": quat.tolist(),
        }


def load_pose(path: Path) -> Pose:
    """
    加载位姿文件
    
    Args:
        path: 位姿 JSON 文件路径
        
    Returns:
        Pose 实例
    """
    data = load_json(path)
    return parse_pose(data)


def parse_pose(data: Dict[str, Any]) -> Pose:
    """
    解析位姿数据
    
    Args:
        data: 位姿 JSON 数据
        
    Returns:
        Pose 实例
    """
    # 必需字段
    if "position" not in data:
        raise ValueError("Pose requires 'position' field")
    if "rotation_quaternion" not in data:
        raise ValueError("Pose requires 'rotation_quaternion' field")
    
    frame_id = data.get("frame_id", 0)
    position = np.array(data["position"], dtype=np.float64)
    quat = np.array(data["rotation_quaternion"], dtype=np.float64)
    
    # 验证
    if position.shape != (3,):
        raise ValueError(f"position shape {position.shape}, expected (3,)")
    if quat.shape != (4,):
        raise ValueError(f"rotation_quaternion shape {quat.shape}, expected (4,)")
    
    # 四元数归一化
    quat_norm = np.linalg.norm(quat)
    if quat_norm < 1e-9:
        raise ValueError(f"Quaternion norm too small: {quat_norm}")
    quat = quat / quat_norm
    
    # 转换为旋转矩阵
    R_cw = quat_wxyz_to_rotation_matrix(quat)
    
    return Pose(
        frame_id=int(frame_id),
        position=position.astype(np.float32),
        R_cw=R_cw.astype(np.float32),
    )


def validate_pose(pose: Pose, strict: bool = True) -> Tuple[bool, str]:
    """
    验证位姿有效性
    
    Args:
        pose: Pose 实例
        strict: 严格模式
        
    Returns:
        (is_valid, message)
    """
    # 检查旋转矩阵
    is_valid, msg = validate_rotation_matrix(pose.R_cw)
    if not is_valid:
        if strict:
            return False, msg
        else:
            # 尝试修复
            pose.R_cw = orthonormalize_rotation(pose.R_cw).astype(np.float32)
    
    # 检查位置是否有限
    if not np.all(np.isfinite(pose.position)):
        return False, f"Position contains non-finite values: {pose.position}"
    
    return True, "Valid pose"


def load_all_poses(pose_dir: Path) -> List[Pose]:
    """
    加载目录中所有位姿文件
    
    Args:
        pose_dir: 位姿目录
        
    Returns:
        poses: 按 frame_id 排序的位姿列表
    """
    pose_files = sorted(pose_dir.glob("pose_*.json"))
    
    poses = []
    for path in pose_files:
        try:
            pose = load_pose(path)
            poses.append(pose)
        except Exception as e:
            print(f"[Warning] Failed to load {path}: {e}")
    
    # 按 frame_id 排序
    poses.sort(key=lambda p: p.frame_id)
    
    return poses


def compute_relative_pose(
    pose_src: Pose,
    pose_tgt: Pose,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算从源视角到目标视角的相对位姿
    
    Args:
        pose_src: 源位姿
        pose_tgt: 目标位姿
        
    Returns:
        R_rel: (3, 3) 相对旋转 R_tgt_src
        t_rel: (3,) 相对平移
    """
    # R_tgt_src = R_wc_tgt @ R_cw_src
    R_rel = pose_tgt.R_wc @ pose_src.R_cw
    
    # t_rel = R_wc_tgt @ (t_src - t_tgt)
    t_rel = pose_tgt.R_wc @ (pose_src.position - pose_tgt.position)
    
    return R_rel, t_rel


def compute_translation_distance(pose1: Pose, pose2: Pose) -> float:
    """
    计算两个位姿之间的平移距离
    
    Args:
        pose1, pose2: Pose 实例
        
    Returns:
        distance: 欧氏距离（米）
    """
    return float(np.linalg.norm(pose1.position - pose2.position))


def compute_rotation_angle(pose1: Pose, pose2: Pose) -> float:
    """
    计算两个位姿之间的旋转角度
    
    Args:
        pose1, pose2: Pose 实例
        
    Returns:
        angle: 旋转角度（度）
    """
    R_rel = pose1.R_wc @ pose2.R_cw
    trace = np.trace(R_rel)
    # cos(θ) = (trace - 1) / 2
    cos_theta = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
    angle_rad = np.arccos(cos_theta)
    return float(np.degrees(angle_rad))
