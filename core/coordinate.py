"""
坐标系约定和四元数工具

ERPT_native 坐标系标准：
- 世界坐标系：右手系 [X右, Y上, Z前]
- 满足：X × Y = Z（右手法则）
- ERP投影约定：
  - lon = atan2(x, z)：经度，范围 [-π, π]
  - lat = asin(y)：纬度，范围 [-π/2, π/2]
  - 图像中心（u=W/2, v=H/2）看向 +Z 方向
  - 图像顶部是 +Y 方向（上）

位姿格式：
- position: [x, y, z]，相机中心在世界坐标系的位置（米）
- rotation_quaternion: [w, x, y, z]，表示 camera->world 旋转 (R_cw)

数学约定：
- P_world = R_cw @ P_cam + t（相机坐标系到世界坐标系）
- P_cam = R_wc @ (P_world - t)（世界坐标系到相机坐标系）
- R_wc = R_cw^T
"""

import numpy as np
from typing import Tuple


def quat_wxyz_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """
    四元数转旋转矩阵
    
    输入四元数表示 camera->world 旋转 (R_cw)
    
    Args:
        q: (4,) 四元数 [w, x, y, z]，需归一化
        
    Returns:
        R: (3, 3) 旋转矩阵 R_cw
    """
    q = np.asarray(q, dtype=np.float64).flatten()
    assert q.shape == (4,), f"Expected shape (4,), got {q.shape}"
    
    # 归一化
    norm = np.linalg.norm(q)
    if norm < 1e-9:
        raise ValueError(f"Quaternion norm too small: {norm}")
    q = q / norm
    
    w, x, y, z = q[0], q[1], q[2], q[3]
    
    # 旋转矩阵公式
    R = np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)]
    ], dtype=np.float64)
    
    return R


def rotation_matrix_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    """
    旋转矩阵转四元数
    
    Args:
        R: (3, 3) 旋转矩阵
        
    Returns:
        q: (4,) 四元数 [w, x, y, z]
    """
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    
    # 确保正交性（SVD正交化）
    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    
    # Shepperd's method
    trace = np.trace(R)
    
    if trace > 0:
        s = 2.0 * np.sqrt(trace + 1.0)
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    
    q = np.array([w, x, y, z], dtype=np.float64)
    
    # 归一化
    q = q / np.linalg.norm(q)
    
    # 确保 w >= 0（唯一性）
    if q[0] < 0:
        q = -q
    
    return q


def R_cw_to_R_wc(R_cw: np.ndarray) -> np.ndarray:
    """
    camera->world 旋转矩阵转换为 world->camera
    
    R_wc = R_cw^T
    
    Args:
        R_cw: (3, 3) camera->world 旋转矩阵
        
    Returns:
        R_wc: (3, 3) world->camera 旋转矩阵
    """
    return R_cw.T


def R_wc_to_R_cw(R_wc: np.ndarray) -> np.ndarray:
    """
    world->camera 旋转矩阵转换为 camera->world
    
    R_cw = R_wc^T
    
    Args:
        R_wc: (3, 3) world->camera 旋转矩阵
        
    Returns:
        R_cw: (3, 3) camera->world 旋转矩阵
    """
    return R_wc.T


def validate_rotation_matrix(R: np.ndarray, tol: float = 1e-5) -> Tuple[bool, str]:
    """
    验证旋转矩阵的有效性
    
    Args:
        R: (3, 3) 待验证的矩阵
        tol: 容差
        
    Returns:
        (is_valid, message)
    """
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    
    # 检查正交性：R^T @ R = I
    I = R.T @ R
    orth_err = np.max(np.abs(I - np.eye(3)))
    if orth_err > tol:
        return False, f"Orthogonality error: {orth_err:.6e} > {tol}"
    
    # 检查行列式：det(R) = +1
    det = np.linalg.det(R)
    if np.abs(det - 1.0) > tol:
        return False, f"Determinant error: det(R)={det:.6f}, expected 1.0"
    
    return True, "Valid rotation matrix"


def orthonormalize_rotation(R: np.ndarray) -> np.ndarray:
    """
    使用SVD正交化旋转矩阵
    
    Args:
        R: (3, 3) 近似旋转矩阵
        
    Returns:
        R_orth: (3, 3) 正交化后的旋转矩阵
    """
    U, _, Vt = np.linalg.svd(R)
    R_orth = U @ Vt
    if np.linalg.det(R_orth) < 0:
        U[:, -1] *= -1
        R_orth = U @ Vt
    return R_orth
