"""
ERP 投影模块

ERPT_native ERP投影约定：
- 经度：lon = atan2(x, z)，范围 [-π, π]
- 纬度：lat = asin(y)，范围 [-π/2, π/2]
- 像素坐标：u ∈ [0, W), v ∈ [0, H)
- 图像中心 (u=W/2, v=H/2) 对应 (lon=0, lat=0)，看向 +Z
- 图像顶部 (v=0) 对应 lat=+π/2，看向 +Y（上）
- 图像底部 (v=H-1) 对应 lat=-π/2，看向 -Y（下）
- 图像右边 lon增加，对应 +X 方向

像素到经纬度映射：
  lon = (u / W) * 2π - π
  lat = π/2 - (v / (H-1)) * π

方向向量（相机坐标系，也是世界坐标系当无旋转时）：
  x = sin(lon) * cos(lat)   # 右
  y = sin(lat)              # 上
  z = cos(lon) * cos(lat)   # 前
"""

import math
import numpy as np
import torch
from typing import Tuple, Union


def erp_to_lonlat(
    u: Union[np.ndarray, torch.Tensor],
    v: Union[np.ndarray, torch.Tensor],
    H: int,
    W: int,
) -> Tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]]:
    """
    ERP像素坐标转经纬度
    
    Args:
        u: 水平像素坐标，范围 [0, W)
        v: 垂直像素坐标，范围 [0, H)
        H: 图像高度
        W: 图像宽度
        
    Returns:
        lon: 经度，范围 [-π, π]
        lat: 纬度，范围 [-π/2, π/2]
    """
    # lon = (u / W) * 2π - π
    lon = (u / float(W)) * (2.0 * math.pi) - math.pi
    
    # lat = π/2 - (v / (H-1)) * π
    lat = (math.pi / 2.0) - (v / float(H - 1)) * math.pi
    
    return lon, lat


def lonlat_to_erp(
    lon: Union[np.ndarray, torch.Tensor],
    lat: Union[np.ndarray, torch.Tensor],
    H: int,
    W: int,
) -> Tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]]:
    """
    经纬度转ERP像素坐标
    
    Args:
        lon: 经度，范围 [-π, π]
        lat: 纬度，范围 [-π/2, π/2]
        H: 图像高度
        W: 图像宽度
        
    Returns:
        u: 水平像素坐标
        v: 垂直像素坐标
    """
    # u = (lon + π) / (2π) * W
    u = (lon + math.pi) / (2.0 * math.pi) * float(W)
    
    # v = (π/2 - lat) / π * (H-1)
    v = (math.pi / 2.0 - lat) / math.pi * float(H - 1)
    
    return u, v


def lonlat_to_direction(
    lon: Union[np.ndarray, torch.Tensor],
    lat: Union[np.ndarray, torch.Tensor],
) -> Union[np.ndarray, torch.Tensor]:
    """
    经纬度转方向向量（单位向量）
    
    坐标系：[X右, Y上, Z前]
    
    Args:
        lon: 经度
        lat: 纬度
        
    Returns:
        dirs: (..., 3) 单位方向向量 [x, y, z]
    """
    if isinstance(lon, torch.Tensor):
        cos_lat = torch.cos(lat)
        x = torch.sin(lon) * cos_lat  # 右
        y = torch.sin(lat)            # 上
        z = torch.cos(lon) * cos_lat  # 前
        dirs = torch.stack([x, y, z], dim=-1)
    else:
        cos_lat = np.cos(lat)
        x = np.sin(lon) * cos_lat
        y = np.sin(lat)
        z = np.cos(lon) * cos_lat
        dirs = np.stack([x, y, z], axis=-1)
    
    return dirs


def direction_to_lonlat(
    dirs: Union[np.ndarray, torch.Tensor],
) -> Tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]]:
    """
    方向向量转经纬度
    
    Args:
        dirs: (..., 3) 方向向量 [x, y, z]
        
    Returns:
        lon: 经度
        lat: 纬度
    """
    x = dirs[..., 0]
    y = dirs[..., 1]
    z = dirs[..., 2]
    
    if isinstance(dirs, torch.Tensor):
        # 归一化
        norm = torch.norm(dirs, dim=-1, keepdim=False)
        norm = torch.clamp(norm, min=1e-9)
        
        # lon = atan2(x, z)
        lon = torch.atan2(x, z)
        
        # lat = asin(y / norm)
        y_normalized = torch.clamp(y / norm, -1.0, 1.0)
        lat = torch.asin(y_normalized)
    else:
        norm = np.linalg.norm(dirs, axis=-1)
        norm = np.maximum(norm, 1e-9)
        
        lon = np.arctan2(x, z)
        y_normalized = np.clip(y / norm, -1.0, 1.0)
        lat = np.arcsin(y_normalized)
    
    return lon, lat


def erp_to_direction(
    u: Union[np.ndarray, torch.Tensor],
    v: Union[np.ndarray, torch.Tensor],
    H: int,
    W: int,
) -> Union[np.ndarray, torch.Tensor]:
    """
    ERP像素坐标转方向向量
    
    Args:
        u: 水平像素坐标
        v: 垂直像素坐标
        H: 图像高度
        W: 图像宽度
        
    Returns:
        dirs: (..., 3) 单位方向向量 [x, y, z]
    """
    lon, lat = erp_to_lonlat(u, v, H, W)
    return lonlat_to_direction(lon, lat)


def direction_to_erp(
    dirs: Union[np.ndarray, torch.Tensor],
    H: int,
    W: int,
) -> Tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]]:
    """
    方向向量转ERP像素坐标
    
    Args:
        dirs: (..., 3) 方向向量 [x, y, z]
        H: 图像高度
        W: 图像宽度
        
    Returns:
        u: 水平像素坐标
        v: 垂直像素坐标
    """
    lon, lat = direction_to_lonlat(dirs)
    return lonlat_to_erp(lon, lat, H, W)


def create_erp_grid(
    H: int,
    W: int,
    device: torch.device = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    创建ERP像素网格
    
    Args:
        H: 图像高度
        W: 图像宽度
        device: 计算设备
        
    Returns:
        uu: (H, W) 水平坐标网格
        vv: (H, W) 垂直坐标网格
    """
    if device is None:
        device = torch.device("cpu")
    
    us = torch.arange(W, device=device, dtype=torch.float32)
    vs = torch.arange(H, device=device, dtype=torch.float32)
    vv, uu = torch.meshgrid(vs, us, indexing="ij")
    
    return uu, vv


def create_direction_grid(
    H: int,
    W: int,
    device: torch.device = None,
) -> torch.Tensor:
    """
    创建ERP方向向量网格
    
    Args:
        H: 图像高度
        W: 图像宽度
        device: 计算设备
        
    Returns:
        dirs: (H, W, 3) 单位方向向量
    """
    uu, vv = create_erp_grid(H, W, device)
    return erp_to_direction(uu, vv, H, W)


def wrap_u(u: Union[np.ndarray, torch.Tensor], W: int) -> Union[np.ndarray, torch.Tensor]:
    """
    水平坐标环绕处理（ERP在水平方向是周期性的）
    
    Args:
        u: 水平像素坐标
        W: 图像宽度
        
    Returns:
        u_wrapped: 环绕后的坐标，范围 [0, W)
    """
    if isinstance(u, torch.Tensor):
        return torch.remainder(u, float(W))
    else:
        return np.mod(u, float(W))


def clamp_v(v: Union[np.ndarray, torch.Tensor], H: int) -> Union[np.ndarray, torch.Tensor]:
    """
    垂直坐标裁剪处理
    
    Args:
        v: 垂直像素坐标
        H: 图像高度
        
    Returns:
        v_clamped: 裁剪后的坐标，范围 [0, H-1]
    """
    if isinstance(v, torch.Tensor):
        return torch.clamp(v, 0.0, float(H - 1))
    else:
        return np.clip(v, 0.0, float(H - 1))
