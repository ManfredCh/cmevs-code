"""
ERP -> Tangent 切片生成模块（移植自原版 ERPT）

功能：
1. 生成 icosahedron 20 面的相机朝向
2. 生成 north/south pole 额外切片（使用更大 FOV）
3. 从 ERP 采样生成透视切片（支持 seam wrap）
4. 输出切片 RGB 和元数据

关键算法：
- icosahedron 面法向计算
- 相机坐标系构建（look-at）
- ERP -> 透视投影（grid_sample with seam wrap）
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class TangentSlice:
    """切片规格"""
    slice_id: str                    # 切片 ID（如 "face_00", "north", "south"）
    slice_type: str                  # 类型："face" | "pole_north" | "pole_south"
    center_dir: np.ndarray           # 切片中心方向（世界坐标，单位向量）
    R_cw: np.ndarray                 # 相机到世界的旋转矩阵 (3,3)
    fov_deg: float                   # 视场角（度）
    resolution: int                  # 输出分辨率（像素，正方形）
    K: np.ndarray                    # 相机内参 (3,3)
    f_px: float                      # 焦距（像素）

    def to_dict(self) -> Dict[str, Any]:
        """转换为可 JSON 序列化的字典"""
        return {
            "slice_id": self.slice_id,
            "slice_type": self.slice_type,
            "center_dir": self.center_dir.tolist(),
            "R_cw": self.R_cw.tolist(),
            "fov_deg": float(self.fov_deg),
            "resolution": int(self.resolution),
            "K": self.K.tolist(),
            "f_px": float(self.f_px),
        }


def _compute_icosahedron_face_centers() -> List[np.ndarray]:
    """
    计算正二十面体 20 个面的中心方向（单位向量）

    正二十面体有 12 个顶点、20 个面、30 条边。
    每个面是等边三角形，面中心 = (v0 + v1 + v2) / 3 归一化

    Returns:
        20 个单位向量的列表，每个指向一个面的中心
    """
    # 黄金比例
    phi = (1.0 + math.sqrt(5.0)) / 2.0

    # 正二十面体 12 个顶点（坐标已归一化）
    vertices = np.array([
        [-1,  phi, 0],
        [ 1,  phi, 0],
        [-1, -phi, 0],
        [ 1, -phi, 0],
        [0, -1,  phi],
        [0,  1,  phi],
        [0, -1, -phi],
        [0,  1, -phi],
        [ phi, 0, -1],
        [ phi, 0,  1],
        [-phi, 0, -1],
        [-phi, 0,  1],
    ], dtype=np.float64)

    # 归一化顶点
    vertices = vertices / np.linalg.norm(vertices, axis=1, keepdims=True)

    # 20 个面的顶点索引
    faces = [
        (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
        (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
        (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
        (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1),
    ]

    centers = []
    for i0, i1, i2 in faces:
        center = vertices[i0] + vertices[i1] + vertices[i2]
        center = center / np.linalg.norm(center)
        centers.append(center.astype(np.float32))

    return centers


def _look_at_rotation(forward: np.ndarray, up_hint: Optional[np.ndarray] = None) -> np.ndarray:
    """
    构建从相机坐标系到世界坐标系的旋转矩阵

    相机坐标系约定：
    - +Z: 前向（forward）
    - +Y: 上方（up）
    - +X: 右方（right = up × forward）

    Args:
        forward: 相机前向方向（世界坐标，单位向量）
        up_hint: 上方提示（默认世界 Y 轴）

    Returns:
        R_cw: (3,3) 旋转矩阵，v_world = R_cw @ v_cam
    """
    f = np.asarray(forward, dtype=np.float64).reshape(3)
    f = f / (np.linalg.norm(f) + 1e-12)

    if up_hint is None:
        up_hint = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    u = np.asarray(up_hint, dtype=np.float64).reshape(3)
    u = u / (np.linalg.norm(u) + 1e-12)

    # 如果 forward 与 up_hint 几乎平行，换一个 up_hint
    if abs(np.dot(f, u)) > 0.95:
        u = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    # 右方向 = up × forward
    r = np.cross(u, f)
    r = r / (np.linalg.norm(r) + 1e-12)

    # 真正的上方向 = forward × right
    u2 = np.cross(f, r)
    u2 = u2 / (np.linalg.norm(u2) + 1e-12)

    # 旋转矩阵的列是相机坐标轴在世界坐标系中的表示
    R_cw = np.stack([r, u2, f], axis=1)
    return R_cw.astype(np.float32)


def _compute_intrinsics(resolution: int, fov_deg: float) -> Tuple[np.ndarray, float]:
    """
    计算针孔相机内参

    Args:
        resolution: 图像分辨率（正方形）
        fov_deg: 水平视场角（度）

    Returns:
        K: (3,3) 内参矩阵
        f_px: 焦距（像素）
    """
    fov_rad = np.deg2rad(fov_deg)
    f_px = 0.5 * resolution / np.tan(0.5 * fov_rad)

    cx = (resolution - 1) * 0.5
    cy = (resolution - 1) * 0.5

    K = np.array([
        [f_px, 0.0,  cx],
        [0.0,  f_px, cy],
        [0.0,  0.0,  1.0]
    ], dtype=np.float32)

    return K, float(f_px)


def build_icosahedron_slices(cfg: Dict[str, Any]) -> List[TangentSlice]:
    """
    根据配置构建 icosahedron + poles 切片列表

    360MonoDepth 风格：使用 padding_factor 而非 overlap_pad_deg
    有效 FOV = base_fov * padding_factor

    Args:
        cfg: 配置字典（包含 tangent 配置）

    Returns:
        切片规格列表
    """
    tcfg = cfg.get("tangent", {})

    # 基本参数
    face_resolution = int(tcfg.get("face_resolution", 768))
    fov_deg = float(tcfg.get("fov_deg", 90.0))

    # 360MonoDepth 风格 padding（优先使用 padding_factor）
    padding_factor = float(tcfg.get("padding_factor", 1.3))
    overlap_pad_deg = float(tcfg.get("overlap_pad_deg", 0.0))  # 向后兼容

    # 计算有效 FOV
    if padding_factor > 1.0:
        effective_fov = fov_deg * padding_factor
    else:
        effective_fov = fov_deg + overlap_pad_deg

    # 限制最大 FOV 避免极端畸变
    effective_fov = min(effective_fov, 170.0)

    # 极区参数（增强覆盖）
    add_poles = bool(tcfg.get("add_poles", True))
    pole_fov_deg = float(tcfg.get("pole_fov_deg", 150.0))  # 默认更大
    pole_resolution = int(tcfg.get("pole_resolution", face_resolution))
    pole_extra_rings = int(tcfg.get("pole_extra_rings", 0))  # 额外极区密采样

    slices = []

    # 1. 添加 20 个 icosahedron 面
    face_centers = _compute_icosahedron_face_centers()
    for i, center in enumerate(face_centers):
        R_cw = _look_at_rotation(center)
        K, f_px = _compute_intrinsics(face_resolution, effective_fov)

        slices.append(TangentSlice(
            slice_id=f"face_{i:02d}",
            slice_type="face",
            center_dir=center,
            R_cw=R_cw,
            fov_deg=effective_fov,
            resolution=face_resolution,
            K=K,
            f_px=f_px,
        ))

    # 2. 添加极区切片
    if add_poles:
        # 北极（+Y）
        north_dir = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        R_north = _look_at_rotation(north_dir, up_hint=np.array([0.0, 0.0, -1.0]))
        K_north, f_north = _compute_intrinsics(pole_resolution, pole_fov_deg)

        slices.append(TangentSlice(
            slice_id="north",
            slice_type="pole_north",
            center_dir=north_dir,
            R_cw=R_north,
            fov_deg=pole_fov_deg,
            resolution=pole_resolution,
            K=K_north,
            f_px=f_north,
        ))

        # 南极（-Y）
        south_dir = np.array([0.0, -1.0, 0.0], dtype=np.float32)
        R_south = _look_at_rotation(south_dir, up_hint=np.array([0.0, 0.0, 1.0]))
        K_south, f_south = _compute_intrinsics(pole_resolution, pole_fov_deg)

        slices.append(TangentSlice(
            slice_id="south",
            slice_type="pole_south",
            center_dir=south_dir,
            R_cw=R_south,
            fov_deg=pole_fov_deg,
            resolution=pole_resolution,
            K=K_south,
            f_px=f_south,
        ))

        # 3. 额外极区密采样环（可选）
        if pole_extra_rings > 0:
            _add_polar_ring_slices(
                slices, pole_extra_rings, pole_resolution, pole_fov_deg * 0.8
            )

    return slices


def _add_polar_ring_slices(
    slices: List[TangentSlice],
    num_rings: int,
    resolution: int,
    fov_deg: float,
) -> None:
    """
    添加额外的极区密采样切片（环状分布在极区附近）
    """
    latitudes = [math.radians(75)]
    if num_rings > 1:
        latitudes = [math.radians(60 + 25 * i / (num_rings - 1)) for i in range(num_rings)]

    K, f_px = _compute_intrinsics(resolution, fov_deg)

    for ring_idx, lat in enumerate(latitudes):
        num_slices_per_ring = 6
        for lon_idx in range(num_slices_per_ring):
            lon = lon_idx * 2 * math.pi / num_slices_per_ring

            # 北极附近
            x_n = math.cos(lat) * math.sin(lon)
            y_n = math.sin(lat)
            z_n = math.cos(lat) * math.cos(lon)
            dir_n = np.array([x_n, y_n, z_n], dtype=np.float32)
            R_n = _look_at_rotation(dir_n)

            slices.append(TangentSlice(
                slice_id=f"pole_ring_n_{ring_idx}_{lon_idx}",
                slice_type="pole_ring",
                center_dir=dir_n,
                R_cw=R_n,
                fov_deg=fov_deg,
                resolution=resolution,
                K=K,
                f_px=f_px,
            ))

            # 南极附近
            y_s = -math.sin(lat)
            dir_s = np.array([x_n, y_s, z_n], dtype=np.float32)
            R_s = _look_at_rotation(dir_s)

            slices.append(TangentSlice(
                slice_id=f"pole_ring_s_{ring_idx}_{lon_idx}",
                slice_type="pole_ring",
                center_dir=dir_s,
                R_cw=R_s,
                fov_deg=fov_deg,
                resolution=resolution,
                K=K,
                f_px=f_px,
            ))


def _build_sample_grid(
    slice_spec: TangentSlice,
    erp_h: int,
    erp_w: int,
    device: torch.device,
) -> torch.Tensor:
    """
    构建从 ERP 采样到切片的网格

    对于切片的每个像素 (u, v):
    1. 反投影到相机坐标系射线方向
    2. 旋转到世界坐标系
    3. 计算球面经纬度
    4. 映射到 ERP 像素坐标
    """
    res = slice_spec.resolution
    K = slice_spec.K
    R_cw = slice_spec.R_cw

    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])

    # 切片像素坐标
    xs = torch.arange(res, device=device, dtype=torch.float32)
    ys = torch.arange(res, device=device, dtype=torch.float32)
    yv, xv = torch.meshgrid(ys, xs, indexing="ij")  # (H, W)

    # 反投影到相机坐标系
    x_cam = (xv - cx) / fx
    y_cam = -(yv - cy) / fy  # 图像 y 向下，相机 y 向上
    z_cam = torch.ones_like(x_cam)

    # 归一化射线方向
    dirs_cam = torch.stack([x_cam, y_cam, z_cam], dim=-1)  # (H, W, 3)
    dirs_cam = dirs_cam / torch.clamp(torch.norm(dirs_cam, dim=-1, keepdim=True), min=1e-9)

    # 旋转到世界坐标系
    R = torch.tensor(R_cw, device=device, dtype=torch.float32)
    dirs_world = torch.einsum("ij,hwj->hwi", R, dirs_cam)  # (H, W, 3)

    # 计算球面坐标
    x = dirs_world[..., 0]
    y = dirs_world[..., 1]
    z = dirs_world[..., 2]

    lon = torch.atan2(x, z)
    lat = torch.asin(torch.clamp(y, -1.0, 1.0))

    # 映射到 ERP 像素坐标
    u = (lon + math.pi) / (2.0 * math.pi) * float(erp_w)
    v = (math.pi / 2.0 - lat) / math.pi * float(erp_h - 1)

    # Seam wrap: ERP 在 x 方向扩展 3 倍，采样时从中间段采样
    u_padded = u + float(erp_w)
    erp_w_padded = erp_w * 3

    x_norm = (u_padded / float(erp_w_padded - 1)) * 2.0 - 1.0
    y_norm = (v / float(erp_h - 1)) * 2.0 - 1.0

    grid = torch.stack([x_norm, y_norm], dim=-1).unsqueeze(0)  # (1, H, W, 2)
    return grid


@torch.no_grad()
def extract_tangent_from_erp(
    erp_rgb: torch.Tensor,
    slice_spec: TangentSlice,
    device: torch.device,
) -> np.ndarray:
    """
    从 ERP 提取单个切片

    Args:
        erp_rgb: (1, 3, H, W) ERP 图像
        slice_spec: 切片规格
        device: 计算设备

    Returns:
        tangent_rgb: (H, W, 3) uint8 numpy array
    """
    erp_h, erp_w = erp_rgb.shape[2], erp_rgb.shape[3]

    # Seam wrap: 扩展 ERP 宽度
    erp_padded = torch.cat([erp_rgb, erp_rgb, erp_rgb], dim=-1)  # (1, 3, H, 3W)

    # 构建采样网格
    grid = _build_sample_grid(slice_spec, erp_h, erp_w, device)

    # 采样
    tangent = F.grid_sample(
        erp_padded,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )  # (1, 3, res, res)

    # 转换为 numpy
    tangent_np = (tangent.squeeze(0).permute(1, 2, 0).clamp(0, 1) * 255.0).byte().cpu().numpy()
    return tangent_np


@torch.no_grad()
def extract_all_tangents(
    erp_rgb_np: np.ndarray,
    slices: List[TangentSlice],
    device: torch.device,
) -> Dict[str, np.ndarray]:
    """
    从 ERP 提取所有切片

    Args:
        erp_rgb_np: (H, W, 3) ERP 图像 numpy array
        slices: 切片规格列表
        device: 计算设备

    Returns:
        字典 {slice_id: tangent_rgb}
    """
    erp_t = torch.from_numpy(erp_rgb_np).to(device).permute(2, 0, 1).float() / 255.0
    erp_t = erp_t.unsqueeze(0)  # (1, 3, H, W)

    results = {}
    for s in slices:
        tangent = extract_tangent_from_erp(erp_t, s, device)
        results[s.slice_id] = tangent

    return results


def compute_ray_directions_for_slice(
    slice_spec: TangentSlice,
    device: torch.device,
) -> torch.Tensor:
    """
    计算切片每个像素对应的世界坐标系射线方向（融合时使用）

    Returns:
        dirs_world: (H, W, 3) 单位方向向量
    """
    res = slice_spec.resolution
    K = slice_spec.K
    R_cw = slice_spec.R_cw

    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])

    xs = torch.arange(res, device=device, dtype=torch.float32)
    ys = torch.arange(res, device=device, dtype=torch.float32)
    yv, xv = torch.meshgrid(ys, xs, indexing="ij")

    x_cam = (xv - cx) / fx
    y_cam = -(yv - cy) / fy
    z_cam = torch.ones_like(x_cam)

    dirs_cam = torch.stack([x_cam, y_cam, z_cam], dim=-1)
    dirs_cam = dirs_cam / torch.clamp(torch.norm(dirs_cam, dim=-1, keepdim=True), min=1e-9)

    R = torch.tensor(R_cw, device=device, dtype=torch.float32)
    dirs_world = torch.einsum("ij,hwj->hwi", R, dirs_cam)

    return dirs_world


@torch.no_grad()
def compute_coverage_mask(
    slices: List[TangentSlice],
    erp_h: int,
    erp_w: int,
    device: torch.device,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    计算 ERP 覆盖率掩码（纯几何计算）

    Returns:
        coverage_mask: (H, W) uint8, 255=covered, 0=uncovered
        stats: 覆盖率统计字典
    """
    coverage = torch.zeros(erp_h, erp_w, device=device, dtype=torch.float32)

    for s in slices:
        res = s.resolution
        K = s.K
        R_cw = s.R_cw

        fx, fy = float(K[0, 0]), float(K[1, 1])
        cx, cy = float(K[0, 2]), float(K[1, 2])

        xs = torch.arange(res, device=device, dtype=torch.float32)
        ys = torch.arange(res, device=device, dtype=torch.float32)
        yv, xv = torch.meshgrid(ys, xs, indexing="ij")

        x_cam = (xv - cx) / fx
        y_cam = -(yv - cy) / fy
        z_cam = torch.ones_like(x_cam)

        dirs_cam = torch.stack([x_cam, y_cam, z_cam], dim=-1)
        dirs_cam = dirs_cam / torch.clamp(torch.norm(dirs_cam, dim=-1, keepdim=True), min=1e-9)

        R = torch.tensor(R_cw, device=device, dtype=torch.float32)
        dirs_world = torch.einsum("ij,hwj->hwi", R, dirs_cam)

        x = dirs_world[..., 0]
        y = dirs_world[..., 1]
        z = dirs_world[..., 2]

        lon = torch.atan2(x, z)
        lat = torch.asin(torch.clamp(y, -1.0, 1.0))

        u = (lon + math.pi) / (2.0 * math.pi) * float(erp_w)
        v = (math.pi / 2.0 - lat) / math.pi * float(erp_h - 1)

        u_int = torch.round(u).to(torch.int64)
        v_int = torch.round(v).to(torch.int64)

        u_int = torch.clamp(u_int % erp_w, 0, erp_w - 1)
        v_int = torch.clamp(v_int, 0, erp_h - 1)

        idx = v_int * erp_w + u_int
        idx = idx.reshape(-1)

        coverage_flat = coverage.reshape(-1)
        coverage_flat.scatter_add_(0, idx, torch.ones_like(idx, dtype=torch.float32))

    covered = coverage > 0
    coverage_mask = (covered.float() * 255).byte().cpu().numpy()

    total_pixels = erp_h * erp_w
    covered_pixels = int(covered.sum().item())

    pole_rows = int(erp_h * 0.1)
    north_covered = covered[:pole_rows, :].float().mean().item()
    south_covered = covered[-pole_rows:, :].float().mean().item()

    stats = {
        "total_coverage": covered_pixels / total_pixels * 100,
        "uncovered_pixels": total_pixels - covered_pixels,
        "north_pole_coverage": north_covered * 100,
        "south_pole_coverage": south_covered * 100,
    }

    return coverage_mask, stats
