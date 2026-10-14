#!/usr/bin/env python3
"""
PLY 全流程 Pipeline（纯 Python，无需 Blender）

从 .ply 点云/网格文件出发，完成：
  Phase 0: 加载场景 + 获取 AABB 边界
  Phase 1: 多高度层撒点 + 7 层过滤（trimesh ray_cast 替代 bpy）
  Phase 2: 边渲边选（Open3D ERP 点云渲染 + 深度图）

输出格式与 run_blend_pipeline.py 完全一致:
  panorama_XXXX.png   + panorama_XXXX_depth.npy  + pose_XXXX.json

坐标系: ERPT_native 右手系 [X右, Y上, Z前]
  PLY 坐标系通常为 Z-up，渲染前统一转换为 Y-up。

运行:
    python run_ply_pipeline.py \\
        --ply /path/to/scene.ply \\
        --output-dir /path/to/output \\
        --num-frames 30 \\
        --resolution 2048,1024

依赖:
    pip install open3d trimesh numpy opencv-python pillow
"""

import argparse
import json
import math
import os
import random as _random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import numpy as np
import trimesh
import open3d as o3d
import cv2

# ── GPU 支持检测 ──────────────────────────────────────────────────────────────
try:
    import torch
    import torch.nn.functional as _F
    _CUDA_AVAILABLE = torch.cuda.is_available()
    _TORCH_DEVICE = torch.device("cuda") if _CUDA_AVAILABLE else torch.device("cpu")
    if _CUDA_AVAILABLE:
        print(f"[GPU] CUDA 可用: {torch.cuda.get_device_name(0)}")
    else:
        print("[GPU] CUDA 不可用，使用 CPU 渲染")
except ImportError:
    torch = None
    _CUDA_AVAILABLE = False
    _TORCH_DEVICE = None
    print("[GPU] torch 未安装，使用 CPU 渲染")


WARP_H = 128
WARP_W = 256
MARGIN = 0.2          # 距边界最小安全距离（PLY 场景通常比 blend 精度低，稍微宽松）

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


def parse_args():
    parser = argparse.ArgumentParser(description="PLY Pipeline（边渲边选）")
    parser.add_argument("--ply", type=str, required=True,
                        help=".ply 文件路径")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="输出目录")
    parser.add_argument("--num-frames", type=int, default=30)
    parser.add_argument("--resolution", type=str, default="2048,1024",
                        help="渲染分辨率 width,height")
    parser.add_argument("--grid-spacing", type=float, default=0.5,
                        help="候选点网格间距（米）")
    parser.add_argument("--camera-height", type=float, default=None,
                        help="固定相机高度（米），None=自动多层")
    parser.add_argument("--stop-gain", type=float, default=DEFAULT_STOP_GAIN)
    parser.add_argument("--stop-score", type=float, default=-0.3)
    parser.add_argument("--stop-delta", type=float, default=0.08)
    parser.add_argument("--min-frames", type=int, default=DEFAULT_MIN_FRAMES)
    parser.add_argument("--rotation-type", type=str, default="random_yaw",
                        choices=["none", "rotate_x_90", "rotate_x_180",
                                 "rotate_z_90", "random_yaw"])
    parser.add_argument("--point-size", type=float, default=2.0,
                        help="点云渲染点径（像素）")
    parser.add_argument("--z-up", action="store_true", default=True,
                        help="PLY 坐标系为 Z-up（默认 True，转为 Y-up）")
    parser.add_argument("--no-z-up", dest="z_up", action="store_false")
    return parser.parse_args()


def load_ply_scene(ply_path: str, z_up: bool = True):
    """加载 PLY，可选将 Z-up 转为 Y-up（ERPT_native）

    PLY 常见坐标系:
      Z-up:  X右, Y前, Z上  → 转换: X'=X, Y'=Z, Z'=Y（ERPT_native）
      Y-up:  X右, Y上, Z前  → 直接使用

    Returns:
        mesh_or_pc: trimesh 对象（Mesh 或 PointCloud）
        pts_world:  np.ndarray (N,3) Y-up 世界点坐标
        bmin, bmax: AABB (3,) float
        is_mesh:    bool, True=Trimesh Mesh（支持 ray_cast）
        faces:      np.ndarray (F,3) int 或 None（纯点云时为 None）
    """
    print(f"\n[Phase 0] 加载场景: {ply_path}")
    scene_or_mesh = trimesh.load(ply_path, process=False, force=None)

    # trimesh 可能返回 Scene（多个 mesh 合并）
    if isinstance(scene_or_mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(
            [g for g in scene_or_mesh.geometry.values()
             if isinstance(g, trimesh.Trimesh)]
        )
        is_mesh = True
    elif isinstance(scene_or_mesh, trimesh.Trimesh):
        mesh = scene_or_mesh
        is_mesh = True
    elif isinstance(scene_or_mesh, trimesh.PointCloud):
        mesh = scene_or_mesh
        is_mesh = False
    else:
        # 尝试强制为 PointCloud
        mesh = trimesh.load(ply_path, process=False, force='mesh')
        is_mesh = isinstance(mesh, trimesh.Trimesh)

    # 获取顶点坐标和面数据
    pts_raw = np.array(mesh.vertices, dtype=np.float64)
    faces = np.array(mesh.faces, dtype=np.int32) if is_mesh else None

    print(f"  点数: {len(pts_raw)}, 面数: {len(faces) if faces is not None else 0}, is_mesh={is_mesh}")

    # 坐标系转换 Z-up → Y-up（ERPT_native）
    if z_up:
        pts_world = pts_raw[:, [0, 2, 1]].copy()
    else:
        pts_world = pts_raw.copy()

    bmin = pts_world.min(axis=0)
    bmax = pts_world.max(axis=0)

    print(f"  AABB (Y-up): min=[{bmin[0]:.2f}, {bmin[1]:.2f}, {bmin[2]:.2f}]  "
          f"max=[{bmax[0]:.2f}, {bmax[1]:.2f}, {bmax[2]:.2f}]")

    return mesh, pts_world, bmin, bmax, is_mesh, faces


class RayCaster:
    """封装 trimesh RayMeshIntersector，提供与 Blender ray_cast 相同的接口。

    对于纯点云（非 mesh）场景，降级为"无碰撞"模式（所有射线无 hit），
    只能依靠 AABB 做粗略过滤。
    """

    def __init__(self, mesh, pts_world: np.ndarray, bmin, bmax,
                 is_mesh: bool, z_up: bool = True):
        self.is_mesh = is_mesh
        self.pts_world = pts_world
        self.bmin = np.array(bmin)
        self.bmax = np.array(bmax)
        self.z_up = z_up
        self._intersector = None

        if is_mesh and isinstance(mesh, trimesh.Trimesh):
            if z_up:
                # 需要把 mesh 顶点也转为 Y-up
                verts = np.array(mesh.vertices, dtype=np.float64)
                verts_yup = verts[:, [0, 2, 1]]
                import copy
                m2 = copy.deepcopy(mesh)
                m2.vertices = verts_yup
                self._intersector = trimesh.ray.ray_pyembree.RayMeshIntersector(m2) \
                    if hasattr(trimesh.ray, 'ray_pyembree') \
                    else trimesh.ray.ray_triangle.RayMeshIntersector(m2)
            else:
                self._intersector = trimesh.ray.ray_pyembree.RayMeshIntersector(mesh) \
                    if hasattr(trimesh.ray, 'ray_pyembree') \
                    else trimesh.ray.ray_triangle.RayMeshIntersector(mesh)
            print("  [RayCaster] 使用 trimesh RayMeshIntersector")
        else:
            print("  [RayCaster] 非 Mesh 场景，使用 AABB 降级模式")

    def cast_ray(self, origin: np.ndarray, direction: np.ndarray):
        """单条射线，返回 (hit: bool, dist: float)

        hit=True 时 dist 为交点距离（米）。
        hit=False 时 dist=inf。
        """
        if self._intersector is None:
            return False, float('inf')

        o = np.array(origin, dtype=np.float64)[np.newaxis]  # (1,3)
        d = np.array(direction, dtype=np.float64)[np.newaxis]
        d = d / (np.linalg.norm(d) + 1e-12)

        try:
            locs, idx_ray, idx_tri = self._intersector.intersects_location(
                o, d, multiple_hits=True)
        except Exception:
            return False, float('inf')

        if len(locs) == 0:
            return False, float('inf')

        dists = np.linalg.norm(locs - origin, axis=1)
        # 过滤极近距离（防止自交）
        valid = dists > 1e-4
        if not np.any(valid):
            return False, float('inf')

        min_dist = float(dists[valid].min())
        return True, min_dist

    def cast_rays_batch(self, origin: np.ndarray,
                        directions: np.ndarray) -> np.ndarray:
        """批量射线，返回 dist 数组 (N,)，无 hit 为 inf。"""
        if self._intersector is None:
            return np.full(len(directions), float('inf'))

        origins = np.tile(origin[np.newaxis], (len(directions), 1))
        dirs = directions / (np.linalg.norm(directions, axis=1, keepdims=True) + 1e-12)

        try:
            locs, idx_ray, idx_tri = self._intersector.intersects_location(
                origins, dirs, multiple_hits=True)
        except Exception:
            return np.full(len(directions), float('inf'))

        dists_out = np.full(len(directions), float('inf'))
        if len(locs) == 0:
            return dists_out

        # 每条射线取最近交点
        for i, (loc, ir) in enumerate(zip(locs, idx_ray)):
            d = float(np.linalg.norm(loc - origin))
            if d > 1e-4 and d < dists_out[ir]:
                dists_out[ir] = d

        return dists_out


def compute_camera_heights(floor_y: float, ceiling_y: float,
                           manual_height=None):
    """计算相机高度层（Y-up 坐标系，Y=高度）"""
    CEIL_CLEARANCE = 0.3
    FIXED_HEIGHTS = [0.5, 0.8, 1.2, 1.7, 2.1]

    if manual_height is not None:
        return [manual_height]

    room_h = ceiling_y - floor_y
    if room_h <= 0:
        return [floor_y + 1.5]

    heights = []
    for eye_h in FIXED_HEIGHTS:
        z = floor_y + eye_h
        if z < ceiling_y - CEIL_CLEARANCE:
            heights.append(z)

    if room_h > 3.0:
        cur_h = FIXED_HEIGHTS[-1]
        step = 1.0
        while True:
            cur_h += step
            z = floor_y + cur_h
            if z >= ceiling_y - CEIL_CLEARANCE:
                break
            heights.append(z)
            step = min(step + 0.5, 3.0)

    top_y = ceiling_y - CEIL_CLEARANCE
    if heights and top_y > max(heights) + 0.5:
        heights.append(top_y)
    elif not heights and top_y > floor_y + 0.5:
        heights.append(top_y)

    return sorted(set(round(h, 2) for h in heights)) if heights else [floor_y + 1.5]


def generate_candidate_grid(bmin, bmax, x_spacing, z_spacing, heights):
    """生成候选点网格（Y-up 坐标系：X=右, Y=高, Z=前）

    heights 是 Y 方向的高度值列表。
    """
    cx = (bmin[0] + bmax[0]) / 2
    cz = (bmin[2] + bmax[2]) / 2

    x_half = int((bmax[0] - cx - MARGIN) / x_spacing)
    z_half = int((bmax[2] - cz - MARGIN) / z_spacing)

    xz_offsets = []
    for ix in range(-x_half, x_half + 1):
        for iz in range(-z_half, z_half + 1):
            x = cx + ix * x_spacing
            z = cz + iz * z_spacing
            if (bmin[0] + MARGIN <= x <= bmax[0] - MARGIN and
                    bmin[2] + MARGIN <= z <= bmax[2] - MARGIN):
                xz_offsets.append((ix * ix + iz * iz, x, z))
    xz_offsets.sort(key=lambda t: t[0])

    candidates = []
    for y in heights:
        for _, x, z in xz_offsets:
            candidates.append([float(x), float(y), float(z)])

    n_xz = len(xz_offsets)
    print(f"  网格: {n_xz}点/层 x {len(heights)}层 = {len(candidates)} 个候选")
    return candidates


def _build_26_directions_yup():
    """26 方向球面采样（Y-up 坐标系：水平面=XZ，竖直=Y）"""
    dirs = []
    # 水平 16 方向（XZ 平面）
    for i in range(16):
        a = i * (2 * math.pi / 16)
        dirs.append(np.array([math.cos(a), 0.0, math.sin(a)]))
    # 上方 5 方向
    elev = math.pi / 4
    for i in range(5):
        a = i * (2 * math.pi / 5)
        dirs.append(np.array([
            math.cos(a) * math.cos(elev),
            math.sin(elev),
            math.sin(a) * math.cos(elev),
        ]))
    # 下方 5 方向
    for i in range(5):
        a = i * (2 * math.pi / 5)
        dirs.append(np.array([
            math.cos(a) * math.cos(elev),
            -math.sin(elev),
            math.sin(a) * math.cos(elev),
        ]))
    return dirs


def raycast_filter(candidates, raycaster: RayCaster, room_height: float,
                   min_wall_dist: float = 1.0):
    """7 层候选点过滤（Y-up 坐标系）

    第 1 层: 室内检测（朝+Y/-Y 各一条射线，距离合理）
    第 2 层: 穿模检测（≥2 方向 < 0.2m）
    第 3 层: 角落检测（>50% 水平方向 < 1.0m）
    第 4 层: 包裹检测（hit_rate≥90% + CV<0.30 + max<8m）
    第 5 层: 贴墙检测（水平 16 方向最近 < 0.3m）
    第 6 层: 视野质量（有效命中比例 < 35%）
    第 7 层: 窄缝检测（对向水平距离之和 < 1.5m）

    非 Mesh 场景（降级模式）：跳过射线过滤，仅做 AABB 内判断。
    """
    if raycaster._intersector is None:
        print("  [过滤] 非 Mesh 场景，跳过射线过滤，返回所有 AABB 内候选")
        bmin, bmax = raycaster.bmin, raycaster.bmax
        passed = [c for c in candidates
                  if all(bmin[i] + MARGIN <= c[i] <= bmax[i] - MARGIN
                         for i in [0, 2])]
        print(f"  过滤统计: 总计={len(candidates)}, 通过={len(passed)}")
        return passed

    DIRS_26 = _build_26_directions_yup()
    n26 = len(DIRS_26)

    dir_up = np.array([0.0, 1.0, 0.0])
    dir_down = np.array([0.0, -1.0, 0.0])
    max_up = max(5.0, room_height)
    max_down = max(3.0, room_height)

    MIN_WALL_CLEARANCE = 0.3
    VIEW_GOOD_MIN = 0.5
    VIEW_GOOD_MAX = 20.0
    VIEW_GOOD_RATIO = 0.35
    MIN_SLIT_WIDTH = 1.5

    N = len(candidates)
    passed = []
    stats = {"无天花板": 0, "无地板": 0, "穿模": 0, "角落": 0,
             "包裹": 0, "贴墙": 0, "视野差": 0, "窄缝": 0}

    t0 = time.time()
    log_interval = max(1, N // 10)

    for idx, pos in enumerate(candidates):
        if idx % log_interval == 0 and idx > 0:
            print(f"    过滤进度: {idx}/{N} ({idx*100//N}%)", flush=True)

        origin = np.array(pos, dtype=np.float64)

        # 第 1 层: 室内检测（Y-up：朝上=+Y，朝下=-Y）
        hit_up, d_up = raycaster.cast_ray(origin, dir_up)
        if not hit_up or d_up > max_up:
            stats["无天花板"] += 1
            continue

        hit_dn, d_dn = raycaster.cast_ray(origin, dir_down)
        if not hit_dn or d_dn > max_down:
            stats["无地板"] += 1
            continue

        # 第 2~7 层: 26 方向采样
        dists = raycaster.cast_rays_batch(origin, np.array(DIRS_26))

        # 第 2 层: 穿模
        n_close = int(np.sum(dists < 0.2))
        if n_close >= 2:
            stats["穿模"] += 1
            continue

        # 第 3 层: 角落（水平 16 方向）
        n_wall = int(np.sum(dists[:16] < min_wall_dist))
        if n_wall > 8:
            stats["角落"] += 1
            continue

        # 第 4 层: 包裹
        finite = dists[np.isfinite(dists)]
        hit_rate = len(finite) / n26
        if hit_rate >= 0.90 and len(finite) >= 2:
            mean_d = float(finite.mean())
            max_d = float(finite.max())
            if mean_d > 0:
                cv = float(finite.std()) / mean_d
                if cv < 0.30 and max_d < 8.0:
                    stats["包裹"] += 1
                    continue

        # 第 5 层: 贴墙
        horiz_finite = dists[:16][np.isfinite(dists[:16])]
        if len(horiz_finite) > 0 and float(horiz_finite.min()) < MIN_WALL_CLEARANCE:
            stats["贴墙"] += 1
            continue

        # 第 6 层: 视野质量
        n_good = int(np.sum((dists >= VIEW_GOOD_MIN) & (dists <= VIEW_GOOD_MAX)))
        if n_good / n26 < VIEW_GOOD_RATIO:
            stats["视野差"] += 1
            continue

        # 第 7 层: 窄缝
        in_slit = False
        for i in range(8):
            d_fwd = dists[i] if np.isfinite(dists[i]) else 999
            d_bwd = dists[i + 8] if np.isfinite(dists[i + 8]) else 999
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
        if v > 0:
            print(f"    ❌ {k}: {v} ({v * 100 // max(N, 1)}%)")

    return passed


def _euler_to_rotation_matrix(rx: float, ry: float, rz: float) -> np.ndarray:
    """XYZ 欧拉角 → 旋转矩阵（用于 ERP 相机朝向，Y-up 坐标系）"""
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)

    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def render_erp_pointcloud(pts_world: np.ndarray,
                          cam_pos: np.ndarray,
                          cam_rot_euler: list,
                          width: int,
                          height: int,
                          point_size: float = 2.0):
    """将点云渲染为 ERP 全景图（等距圆柱投影）

    算法:
      1. 将所有世界点变换到相机坐标系
      2. 计算每个点的方位角 (lon) 和仰角 (lat)（Y-up 右手系）
      3. 投影到 ERP 像素坐标
      4. 用 Z-buffer 填充 RGB + depth 图，splat_radius=point_size

    坐标系 (ERPT_native, Y-up):
      相机前向 = +Z_cam, 上方 = +Y_cam, 右方 = +X_cam
      lon = atan2(x_cam, z_cam)  （正前方=0，右=+）
      lat = atan2(y_cam, sqrt(x^2+z^2))（上=+π/2）

    Returns:
        rgb   : np.ndarray (H, W, 3) uint8
        depth : np.ndarray (H, W) float32，range depth（米），0=无效
    """
    o3d = o3d

    cam_pos = np.array(cam_pos, dtype=np.float64)
    # 旋转矩阵：world → camera
    # 相机默认朝向 +Z，根据欧拉角旋转
    R_cw = _euler_to_rotation_matrix(*cam_rot_euler)  # cam_to_world
    R_wc = R_cw.T  # world_to_cam

    # 变换点云到相机坐标系
    vecs = pts_world - cam_pos  # (N, 3)
    pts_cam = (R_wc @ vecs.T).T  # (N, 3)

    x_c = pts_cam[:, 0]
    y_c = pts_cam[:, 1]
    z_c = pts_cam[:, 2]

    # 计算经纬度（ERPT_native 约定）
    lon = np.arctan2(x_c, z_c)                          # [-π, π]
    r_xz = np.sqrt(x_c ** 2 + z_c ** 2)
    lat = np.arctan2(y_c, r_xz)                         # [-π/2, π/2]

    # 转为像素坐标
    u = ((lon / (2 * math.pi) + 0.5) * width).astype(np.float32)
    v = ((0.5 - lat / math.pi) * height).astype(np.float32)
    u = np.clip(u, 0, width - 1).astype(np.int32)
    v = np.clip(v, 0, height - 1).astype(np.int32)

    # range depth = 射线距离（米）
    dist = np.sqrt(x_c ** 2 + y_c ** 2 + z_c ** 2).astype(np.float32)

    # 尝试获取点云颜色
    has_colors = hasattr(pts_world, '_colors')
    colors_rgb = None

    # 初始化图像缓冲区
    rgb_buf = np.zeros((height, width, 3), dtype=np.uint8)
    depth_buf = np.full((height, width), np.inf, dtype=np.float32)

    # Z-buffer 渲染（每点 splat_radius 像素）
    radius = max(1, int(round(point_size / 2)))

    # 为效率起见，用 numpy 向量化做单像素填充，然后 dilate
    # 先做精确 Z-buffer（单像素）
    for i in np.argsort(dist)[::-1]:    # 从远到近，近的覆盖远的
        ui, vi = u[i], v[i]
        di = dist[i]
        if di <= 0 or not np.isfinite(di):
            continue
        if di < depth_buf[vi, ui]:
            depth_buf[vi, ui] = di
            if colors_rgb is not None:
                rgb_buf[vi, ui] = colors_rgb[i]
            else:
                # 无颜色时用伪彩色（深度着色）
                c = int(np.clip(255 * (1.0 - di / 20.0), 0, 255))
                rgb_buf[vi, ui] = [c, c, c]

    # 如果有 Open3D 点云颜色，补充颜色
    # （此处暂用灰度，完整颜色在下方 _render_with_colors 中处理）
    depth_out = np.where(np.isfinite(depth_buf), depth_buf, 0.0).astype(np.float32)
    return rgb_buf, depth_out


def _gpu_align_u(u_ref, u_other, W: int):
    """把 u_other 对齐到与 u_ref 最近的 ERP 循环副本（GPU tensor）"""
    half_w = float(W) / 2.0
    diff = u_other - u_ref
    u_other = torch.where(diff >  half_w, u_other - W, u_other)
    u_other = torch.where(diff < -half_w, u_other + W, u_other)
    return u_other


def _gpu_raster_batch(u0, v0, u1, v1, u2, v2,
                      c0, c1, c2, d0b, d1b, d2b,
                      rgb_flat, depth_flat, H: int, W: int):
    """完全向量化批量光栅化（无 Python for 循环）。

    将所有三角面的包围盒像素展开成一个大 tensor，一次性完成
    重心坐标计算和 scatter_reduce Z-buffer 写入。

    Args:
        u0/v0, u1/v1, u2/v2: (B,) float32，三顶点 ERP 像素坐标
        c0/c1/c2:            (B,3) float32，三顶点颜色 [0,255]
        d0b/d1b/d2b:         (B,) float32，三顶点距离
        rgb_flat:   (H*W, 3) 输出颜色缓冲（就地修改）
        depth_flat: (H*W,)   输出深度缓冲（就地修改）
    """
    dev = u0.device

    # ── 包围盒 ──────────────────────────────────────────────────────────────
    u_lo = torch.clamp(torch.floor(torch.minimum(torch.minimum(u0, u1), u2)).long(), 0, W - 1)
    u_hi = torch.clamp(torch.ceil (torch.maximum(torch.maximum(u0, u1), u2)).long(), 0, W - 1)
    v_lo = torch.clamp(torch.floor(torch.minimum(torch.minimum(v0, v1), v2)).long(), 0, H - 1)
    v_hi = torch.clamp(torch.ceil (torch.maximum(torch.maximum(v0, v1), v2)).long(), 0, H - 1)

    du = u_hi - u_lo + 1
    dv = v_hi - v_lo + 1
    bbox_px = du * dv

    # 过滤退化面 & 超大面
    valid = (u_hi >= u_lo) & (v_hi >= v_lo) & (bbox_px <= 128 * 128)
    if not valid.any():
        return

    idx = valid.nonzero(as_tuple=False).squeeze(1)
    u0v = u0[idx]; v0v = v0[idx]
    u1v = u1[idx]; v1v = v1[idx]
    u2v = u2[idx]; v2v = v2[idx]
    c0v = c0[idx]; c1v = c1[idx]; c2v = c2[idx]
    d0v = d0b[idx]; d1v = d1b[idx]; d2v = d2b[idx]
    u_lv = u_lo[idx]; u_hv = u_hi[idx]
    v_lv = v_lo[idx]; v_hv = v_hi[idx]
    duv = u_hv - u_lv + 1
    dvv = v_hv - v_lv + 1
    npx = duv * dvv

    Bp = int(npx.sum().item())
    if Bp == 0:
        return

    # ── 展开：repeat_interleave 把面 id 重复 npx[i] 次 ──────────────────────
    face_id = torch.repeat_interleave(
        torch.arange(len(idx), device=dev, dtype=torch.long), npx)

    cumsum = torch.zeros(len(idx) + 1, dtype=torch.long, device=dev)
    cumsum[1:] = torch.cumsum(npx, 0)
    local_flat = torch.arange(Bp, device=dev, dtype=torch.long) - cumsum[face_id]

    local_u = local_flat % duv[face_id]
    local_v = local_flat // duv[face_id]

    uu = (u_lv[face_id] + local_u).float()
    vv = (v_lv[face_id] + local_v).float()

    # ── 重心坐标（完全向量化） ───────────────────────────────────────────────
    ax = u0v[face_id]; ay = v0v[face_id]
    bx = u1v[face_id]; by = v1v[face_id]
    cx = u2v[face_id]; cy = v2v[face_id]

    denom = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
    safe  = denom.abs() > 1e-8
    inv_d = torch.where(safe, 1.0 / denom, torch.zeros_like(denom))
    w0 = ((by - cy) * (uu - cx) + (cx - bx) * (vv - cy)) * inv_d
    w1 = ((cy - ay) * (uu - cx) + (ax - cx) * (vv - cy)) * inv_d
    w2 = 1.0 - w0 - w1

    inside = safe & (w0 >= -0.01) & (w1 >= -0.01) & (w2 >= -0.01)
    if not inside.any():
        return

    fi  = face_id[inside]
    uui = uu[inside].long()
    vvi = vv[inside].long()
    w0i = w0[inside]; w1i = w1[inside]; w2i = w2[inside]

    # ── Z-buffer scatter_reduce(amin) ───────────────────────────────────────
    di  = w0i * d0v[fi] + w1i * d1v[fi] + w2i * d2v[fi]
    lin = vvi * W + uui
    depth_flat.scatter_reduce_(0, lin, di, reduce='amin', include_self=True)

    # ── 颜色写入（near-wins） ────────────────────────────────────────────────
    cur_d  = depth_flat[lin]
    winner = (di - cur_d).abs() < 1e-4
    w0e = w0i[winner].unsqueeze(1)
    w1e = w1i[winner].unsqueeze(1)
    w2e = w2i[winner].unsqueeze(1)
    fie = fi[winner]
    col_i = torch.clamp(w0e * c0v[fie] + w1e * c1v[fie] + w2e * c2v[fie], 0, 255)
    rgb_flat.scatter_(0, lin[winner].unsqueeze(1).expand(-1, 3), col_i)


def _gpu_raster_mesh(u_f, v_f, dist, col_t, f_t,
                     rgb_flat, depth_flat, H: int, W: int,
                     batch_size: int = 65536):
    """ERP 网格光栅化主流程（含接缝三副本 + OOM 自动降级）。

    操作 rgb_flat / depth_flat 缓冲（就地修改）。
    """
    d0t = dist[f_t[:, 0]]; d1t = dist[f_t[:, 1]]; d2t = dist[f_t[:, 2]]
    avg_d = (d0t + d1t + d2t) / 3.0
    order = torch.argsort(avg_d, descending=True)
    f_ord = f_t[order]
    d0t, d1t, d2t = dist[f_ord[:, 0]], dist[f_ord[:, 1]], dist[f_ord[:, 2]]
    c0t = col_t[f_ord[:, 0]]; c1t = col_t[f_ord[:, 1]]; c2t = col_t[f_ord[:, 2]]
    u0r = u_f[f_ord[:, 0]]; u1r = u_f[f_ord[:, 1]]; u2r = u_f[f_ord[:, 2]]
    v0r = v_f[f_ord[:, 0]]; v1r = v_f[f_ord[:, 1]]; v2r = v_f[f_ord[:, 2]]

    valid_f = (d0t > 1e-4) & (d1t > 1e-4) & (d2t > 1e-4)
    f_idx = valid_f.nonzero(as_tuple=False).squeeze(1)

    def _process_batch(bi):
        u0b = u0r[bi]; v0b = v0r[bi]
        u1b = _gpu_align_u(u0b, u1r[bi], W)
        u2b = _gpu_align_u(u0b, u2r[bi], W)
        v1b = v1r[bi]; v2b = v2r[bi]
        c0b = c0t[bi]; c1b = c1t[bi]; c2b = c2t[bi]
        d0b_ = d0t[bi]; d1b_ = d1t[bi]; d2b_ = d2t[bi]
        # 三副本 concat：主 + 左(u-W) + 右(u+W)，一次送入减少 kernel launch
        _gpu_raster_batch(
            torch.cat([u0b, u0b - W, u0b + W]),
            torch.cat([v0b, v0b,     v0b    ]),
            torch.cat([u1b, u1b - W, u1b + W]),
            torch.cat([v1b, v1b,     v1b    ]),
            torch.cat([u2b, u2b - W, u2b + W]),
            torch.cat([v2b, v2b,     v2b    ]),
            torch.cat([c0b, c0b, c0b]),
            torch.cat([c1b, c1b, c1b]),
            torch.cat([c2b, c2b, c2b]),
            torch.cat([d0b_, d0b_, d0b_]),
            torch.cat([d1b_, d1b_, d1b_]),
            torch.cat([d2b_, d2b_, d2b_]),
            rgb_flat, depth_flat, H, W,
        )

    try:
        _process_batch(f_idx)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        print(f"  [WARN] OOM（{len(f_idx)} 面），自动降级分批 batch={batch_size}")
        for start in range(0, len(f_idx), batch_size):
            _process_batch(f_idx[start: start + batch_size])


def _gpu_splat_pointcloud(u_f, v_f, dist, col_t,
                          rgb_flat, depth_flat, H: int, W: int,
                          point_size: float = 2.0):
    """点云 scatter Z-buffer splatting（就地修改 rgb_flat / depth_flat）"""
    valid  = dist > 1e-4
    u_i    = torch.clamp(u_f[valid].long(), 0, W - 1)
    v_i    = torch.clamp(v_f[valid].long(), 0, H - 1)
    dist_v = dist[valid]
    col_v  = col_t[valid]
    lin    = v_i * W + u_i
    radius = max(0, int(round(point_size / 2)) - 1)

    if radius == 0:
        depth_flat.scatter_reduce_(0, lin, dist_v, reduce='amin', include_self=True)
        sort_idx = torch.argsort(dist_v)
        rgb_flat.scatter_(0, lin[sort_idx].unsqueeze(1).expand(-1, 3), col_v[sort_idx])
    else:
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                v_nb = torch.clamp(v_i + dr, 0, H - 1)
                u_nb = torch.clamp(u_i + dc, 0, W - 1)
                depth_flat.scatter_reduce_(0, v_nb * W + u_nb, dist_v,
                                           reduce='amin', include_self=True)
        sort_idx = torch.argsort(dist_v, descending=True)
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                v_nb = torch.clamp(v_i[sort_idx] + dr, 0, H - 1)
                u_nb = torch.clamp(u_i[sort_idx] + dc, 0, W - 1)
                rgb_flat.scatter_(0, (v_nb * W + u_nb).unsqueeze(1).expand(-1, 3),
                                  col_v[sort_idx])


def _gpu_fill_holes(rgb_2d, depth_out):
    """GPU 空洞填充：大核 max_pool2d + 残余小孔迭代收尾。

    Args:
        rgb_2d:    (H,W,3) float32 GPU tensor
        depth_out: (H,W)   float32 GPU tensor（0=空洞）

    Returns:
        rgb_2d: (H,W,3) float32，填充后
    """
    hole = (depth_out == 0)
    if not hole.any():
        return rgb_2d

    fill_radius = 32
    k = fill_radius * 2 + 1   # 65

    rgb_f   = rgb_2d.permute(2, 0, 1).unsqueeze(0).float()       # (1,3,H,W)
    valid_m = (~hole).float().unsqueeze(0).unsqueeze(0)           # (1,1,H,W)

    rgb_masked = rgb_f * valid_m
    expanded   = _F.max_pool2d(rgb_masked, kernel_size=k, stride=1, padding=fill_radius)
    valid_exp  = _F.max_pool2d(valid_m,    kernel_size=k, stride=1, padding=fill_radius) > 0

    fill_mask = hole.unsqueeze(0).unsqueeze(0) & valid_exp
    rgb_f = torch.where(fill_mask.expand_as(rgb_f), expanded, rgb_f)

    # 残余大孔洞：最多 8 轮 3×3 迭代收尾
    hole2 = hole & ~fill_mask.squeeze(0).squeeze(0)
    if hole2.any():
        valid_f2 = (~hole2).float().unsqueeze(0).unsqueeze(0)
        rgb_f2   = rgb_f
        for _ in range(8):
            if not hole2.any():
                break
            r2m  = rgb_f2 * valid_f2
            exp2 = _F.max_pool2d(r2m,     kernel_size=3, stride=1, padding=1)
            vd2  = _F.max_pool2d(valid_f2, kernel_size=3, stride=1, padding=1) > 0
            nw2  = hole2.unsqueeze(0).unsqueeze(0) & vd2
            if not nw2.any():
                break
            rgb_f2   = torch.where(nw2.expand_as(rgb_f2), exp2, rgb_f2)
            valid_f2 = torch.where(nw2, torch.ones_like(valid_f2), valid_f2)
            hole2    = hole2 & ~nw2.squeeze(0).squeeze(0)
        rgb_f = rgb_f2

    return rgb_f.squeeze(0).permute(1, 2, 0)   # (H,W,3)


def _render_erp_gpu(pts_world: np.ndarray,
                    colors_world,
                    cam_pos,
                    R_wc: np.ndarray,
                    width: int,
                    height: int,
                    faces: np.ndarray = None,
                    point_size: float = 2.0,
                    batch_size: int = 65536) -> tuple:
    """GPU（CUDA）加速的 ERP 全景渲染。

    所有计算在 GPU tensor 上完成：
      1. 顶点变换 + ERP 投影（全量向量化）
      2. 网格/点云光栅化（委托给共享辅助函数）
      3. 空洞填充：max_pool2d 大核膨胀（纯 GPU，无 cv2）

    Returns:
        rgb   (H, W, 3) uint8 numpy
        depth (H, W)    float32 numpy
    """
    dev = _TORCH_DEVICE

    # ── 1. 顶点变换到相机坐标系 ──────────────────────────────────────────────
    pts = torch.from_numpy(pts_world.astype(np.float32)).to(dev)
    cp  = torch.from_numpy(cam_pos.astype(np.float32)).to(dev)
    R   = torch.from_numpy(R_wc.astype(np.float32)).to(dev)

    pts_cam = (R @ (pts - cp).T).T
    x_c, y_c, z_c = pts_cam[:, 0], pts_cam[:, 1], pts_cam[:, 2]

    # ── 2. ERP 投影 ──────────────────────────────────────────────────────────
    lon  = torch.atan2(x_c, z_c)
    r_xz = torch.sqrt(x_c ** 2 + z_c ** 2)
    lat  = torch.atan2(y_c, r_xz)
    u_f  = (lon  / (2 * math.pi) + 0.5) * width
    v_f  = (0.5  - lat / math.pi) * height
    dist = torch.sqrt(x_c ** 2 + y_c ** 2 + z_c ** 2)

    # ── 3. 顶点颜色 ──────────────────────────────────────────────────────────
    if colors_world is not None:
        col_np = colors_world if colors_world.dtype == np.uint8 \
            else (np.clip(colors_world, 0, 1) * 255).astype(np.uint8)
        col_t = torch.from_numpy(col_np.astype(np.float32)).to(dev)
    else:
        d_norm = torch.clamp(dist / max(float(dist.max()), 1.0), 0, 1)
        g = torch.clamp((1.0 - d_norm) * 200 + 30, 0, 255)
        col_t = g.unsqueeze(1).expand(-1, 3)

    H, W = height, width
    INF  = 1e9
    rgb_flat   = torch.zeros(H * W, 3, dtype=torch.float32, device=dev)
    depth_flat = torch.full((H * W,), INF, dtype=torch.float32, device=dev)

    # ── 4. 光栅化 ────────────────────────────────────────────────────────────
    if faces is not None and len(faces) > 0:
        f_t = torch.from_numpy(faces.astype(np.int64)).to(dev)
        _gpu_raster_mesh(u_f, v_f, dist, col_t, f_t,
                         rgb_flat, depth_flat, H, W, batch_size)
    else:
        _gpu_splat_pointcloud(u_f, v_f, dist, col_t,
                              rgb_flat, depth_flat, H, W, point_size)

    # ── 5. reshape ───────────────────────────────────────────────────────────
    depth_2d  = depth_flat.reshape(H, W)
    rgb_2d    = rgb_flat.reshape(H, W, 3)
    depth_out = torch.where(depth_2d < INF / 2, depth_2d, torch.zeros_like(depth_2d))

    # ── 6. 空洞填充 ──────────────────────────────────────────────────────────
    if faces is not None and len(faces) > 0:
        rgb_2d = _gpu_fill_holes(rgb_2d, depth_out)

    # ── 7. 回传 numpy ────────────────────────────────────────────────────────
    rgb_np   = rgb_2d.clamp(0, 255).byte().cpu().numpy()
    depth_np = depth_out.cpu().numpy().astype(np.float32)
    return rgb_np, depth_np


def render_erp_batch_gpu(pts_world: np.ndarray,
                         colors_world,
                         cam_poses: list,
                         cam_rots: list,
                         width: int,
                         height: int,
                         faces: np.ndarray = None,
                         point_size: float = 2.0) -> list:
    """并行批量渲染多个相机视角（共用同一场景，减少重复数据传输）。

    将场景数据（pts_world, colors_world, faces）只上传一次到 GPU，
    然后依次渲染 len(cam_poses) 个视角，显著减少 PCIe 传输开销。

    Args:
        cam_poses: list of [x, y, z]，各帧相机位置
        cam_rots:  list of [rx, ry, rz]，各帧相机欧拉角

    Returns:
        list of (rgb_np, depth_np) 与输入顺序对应
    """
    if not (_CUDA_AVAILABLE and torch is not None):
        return [
            render_erp_from_ply(pts_world, colors_world, pos, rot,
                                 width, height, point_size, faces)
            for pos, rot in zip(cam_poses, cam_rots)
        ]

    dev = _TORCH_DEVICE

    # ── 场景数据上传（只做一次）──────────────────────────────────────────────
    pts_t = torch.from_numpy(pts_world.astype(np.float32)).to(dev)

    if colors_world is not None:
        col_np = colors_world if colors_world.dtype == np.uint8 \
            else (np.clip(colors_world, 0, 1) * 255).astype(np.uint8)
        col_t_scene = torch.from_numpy(col_np.astype(np.float32)).to(dev)
    else:
        col_t_scene = None

    f_t = None
    if faces is not None and len(faces) > 0:
        f_t = torch.from_numpy(faces.astype(np.int64)).to(dev)

    # ── 逐帧渲染（场景数据复用）──────────────────────────────────────────────
    results = []
    for cam_pos, cam_rot in zip(cam_poses, cam_rots):
        cam_pos_np = np.array(cam_pos, dtype=np.float64)
        R_wc = _euler_to_rotation_matrix(*cam_rot).T

        cp = torch.from_numpy(cam_pos_np.astype(np.float32)).to(dev)
        R  = torch.from_numpy(R_wc.astype(np.float32)).to(dev)

        pts_cam = (R @ (pts_t - cp).T).T
        x_c, y_c, z_c = pts_cam[:, 0], pts_cam[:, 1], pts_cam[:, 2]
        lon  = torch.atan2(x_c, z_c)
        r_xz = torch.sqrt(x_c ** 2 + z_c ** 2)
        lat  = torch.atan2(y_c, r_xz)
        u_f  = (lon  / (2 * math.pi) + 0.5) * width
        v_f  = (0.5  - lat / math.pi) * height
        dist = torch.sqrt(x_c ** 2 + y_c ** 2 + z_c ** 2)

        if col_t_scene is None:
            d_norm = torch.clamp(dist / max(float(dist.max()), 1.0), 0, 1)
            g = torch.clamp((1.0 - d_norm) * 200 + 30, 0, 255)
            col_frame = g.unsqueeze(1).expand(-1, 3)
        else:
            col_frame = col_t_scene

        try:
            rgb_np, depth_np = _render_erp_gpu_from_projected(
                u_f, v_f, dist, col_frame, f_t, width, height, point_size)
        except Exception as e:
            print(f"  [WARN] batch GPU 渲染帧失败，回退单帧: {e}")
            rgb_np, depth_np = _render_erp_gpu(
                pts_world, colors_world, cam_pos_np, R_wc,
                width, height, faces=faces, point_size=point_size)
        results.append((rgb_np, depth_np))

    return results


def _render_erp_gpu_from_projected(
        u_f, v_f, dist, col_t, f_t,
        width: int, height: int, point_size: float = 2.0) -> tuple:
    """内部函数：从已投影的 GPU tensor 直接光栅化，省去重复的顶点变换。

    Args:
        u_f, v_f:  (N,) float32 GPU tensor，ERP 像素浮点坐标
        dist:      (N,) float32 GPU tensor，各顶点到相机距离
        col_t:     (N,3) float32 GPU tensor，顶点颜色 [0,255]
        f_t:       (F,3) int64  GPU tensor 或 None
        width, height: 输出分辨率
    """
    dev = u_f.device
    H, W = height, width
    INF  = 1e9

    rgb_flat   = torch.zeros(H * W, 3, dtype=torch.float32, device=dev)
    depth_flat = torch.full((H * W,), INF, dtype=torch.float32, device=dev)

    if f_t is not None and len(f_t) > 0:
        _gpu_raster_mesh(u_f, v_f, dist, col_t, f_t,
                         rgb_flat, depth_flat, H, W)
    else:
        _gpu_splat_pointcloud(u_f, v_f, dist, col_t,
                              rgb_flat, depth_flat, H, W, point_size)

    depth_2d  = depth_flat.reshape(H, W)
    rgb_2d    = rgb_flat.reshape(H, W, 3)
    depth_out = torch.where(depth_2d < INF / 2, depth_2d, torch.zeros_like(depth_2d))

    if f_t is not None and len(f_t) > 0:
        rgb_2d = _gpu_fill_holes(rgb_2d, depth_out)

    rgb_np   = rgb_2d.clamp(0, 255).byte().cpu().numpy()
    depth_np = depth_out.cpu().numpy().astype(np.float32)
    return rgb_np, depth_np


def render_erp_from_ply(pts_world: np.ndarray,
                        colors_world,
                        cam_pos: list,
                        cam_rot_euler: list,
                        width: int,
                        height: int,
                        point_size: float = 2.0,
                        faces: np.ndarray = None):
    """带颜色的 ERP 全景图渲染（自动 dispatch 到 GPU/CPU）

    Args:
        pts_world:     (N,3) float64，Y-up 世界坐标
        colors_world:  (N,3) uint8 或 float32，RGB 颜色，None=伪彩
        cam_pos:       [x, y, z] 相机位置（Y-up）
        cam_rot_euler: [rx, ry, rz] 相机欧拉角（弧度）
        width, height: 输出分辨率
        point_size:    splat 直径（像素），仅纯点云模式有效
        faces:         (F,3) int，三角面顶点索引；有面时走面光栅化

    Returns:
        rgb:   (H, W, 3) uint8
        depth: (H, W) float32，range depth（米），0=无效
    """
    cam_pos = np.array(cam_pos, dtype=np.float64)
    R_cw = _euler_to_rotation_matrix(*cam_rot_euler)
    R_wc = R_cw.T

    # ── GPU dispatch ──────────────────────────────────────────────────────
    if _CUDA_AVAILABLE and torch is not None:
        try:
            return _render_erp_gpu(pts_world, colors_world,
                                   cam_pos, R_wc,
                                   width, height,
                                   faces=faces, point_size=point_size)
        except Exception as _gpu_err:
            print(f"  [WARN] GPU 渲染失败，回退 CPU: {_gpu_err}")

    # ── CPU 路径 ──────────────────────────────────────────────────────────
    # 所有顶点变换到相机坐标系
    vecs = pts_world - cam_pos
    pts_cam = (R_wc @ vecs.T).T  # (N, 3)

    x_c = pts_cam[:, 0].astype(np.float32)
    y_c = pts_cam[:, 1].astype(np.float32)
    z_c = pts_cam[:, 2].astype(np.float32)

    # ERP 投影：每顶点 → (u_f, v_f, dist)
    lon = np.arctan2(x_c, z_c)
    r_xz = np.sqrt(x_c ** 2 + z_c ** 2)
    lat = np.arctan2(y_c, r_xz)
    u_f = (lon / (2 * math.pi) + 0.5) * width    # float 像素坐标
    v_f = (0.5 - lat / math.pi) * height
    dist = np.sqrt(x_c ** 2 + y_c ** 2 + z_c ** 2)

    # 顶点颜色
    if colors_world is not None:
        col_all = colors_world
        if col_all.dtype != np.uint8:
            col_all = (np.clip(col_all, 0, 1) * 255).astype(np.uint8)
    else:
        d_norm = np.clip(dist / max(float(dist.max()), 1.0), 0, 1)
        g = np.clip((1.0 - d_norm) * 200 + 30, 0, 255).astype(np.uint8)
        col_all = np.stack([g, g, g], axis=1)

    rgb_buf = np.zeros((height, width, 3), dtype=np.uint8)
    depth_buf = np.full((height, width), np.inf, dtype=np.float32)

    if faces is not None and len(faces) > 0:
        # ── 网格模式：三角面光栅化 ──────────────────────────────────────
        f = faces.astype(np.int32)
        d0, d1, d2 = dist[f[:, 0]], dist[f[:, 1]], dist[f[:, 2]]
        face_dist = (d0 + d1 + d2) / 3.0
        valid_face = (d0 > 1e-4) & (d1 > 1e-4) & (d2 > 1e-4)
        f = f[valid_face]
        face_dist = face_dist[valid_face]

        # 从远到近排序，近面覆盖远面
        order = np.argsort(face_dist)[::-1]
        f = f[order]
        face_dist = face_dist[order]

        c0s = col_all[f[:, 0]].astype(np.float32)
        c1s = col_all[f[:, 1]].astype(np.float32)
        c2s = col_all[f[:, 2]].astype(np.float32)
        d0s = dist[f[:, 0]]
        d1s = dist[f[:, 1]]
        d2s = dist[f[:, 2]]

        # 三顶点的 ERP 浮点坐标
        u0s_raw = u_f[f[:, 0]]
        u1s_raw = u_f[f[:, 1]]
        u2s_raw = u_f[f[:, 2]]
        v0s = v_f[f[:, 0]]
        v1s = v_f[f[:, 1]]
        v2s = v_f[f[:, 2]]

        def _raster_triangle(u0v, v0v, u1v, v1v, u2v, v2v,
                             c0, c1, c2, dep0, dep1, dep2):
            """将单个三角面光栅化写入 rgb_buf / depth_buf（闭包）"""
            v_min = max(0, int(math.floor(min(v0v, v1v, v2v))))
            v_max = min(height - 1, int(math.ceil(max(v0v, v1v, v2v))))
            u_min = max(0, int(math.floor(min(u0v, u1v, u2v))))
            u_max = min(width - 1, int(math.ceil(max(u0v, u1v, u2v))))
            if v_max < v_min or u_max < u_min:
                return
            vs_arr = np.arange(v_min, v_max + 1)
            us_arr = np.arange(u_min, u_max + 1)
            uu, vv = np.meshgrid(us_arr, vs_arr)
            uu = uu.flatten().astype(np.float32)
            vv = vv.flatten().astype(np.float32)
            denom = ((v1v - v2v) * (u0v - u2v) + (u2v - u1v) * (v0v - v2v))
            if abs(denom) < 1e-8:
                return
            inv_d = 1.0 / denom
            w0 = ((v1v - v2v) * (uu - u2v) + (u2v - u1v) * (vv - v2v)) * inv_d
            w1 = ((v2v - v0v) * (uu - u2v) + (u0v - u2v) * (vv - v2v)) * inv_d
            w2 = 1.0 - w0 - w1
            inside = (w0 >= -0.01) & (w1 >= -0.01) & (w2 >= -0.01)
            if not inside.any():
                return
            uu_in = uu[inside].astype(np.int32)
            vv_in = vv[inside].astype(np.int32)
            w0_in = w0[inside][:, None]
            w1_in = w1[inside][:, None]
            w2_in = w2[inside][:, None]
            di = w0_in[:, 0] * dep0 + w1_in[:, 0] * dep1 + w2_in[:, 0] * dep2
            ci_rgb = np.clip(w0_in * c0 + w1_in * c1 + w2_in * c2, 0, 255).astype(np.uint8)
            closer = di < depth_buf[vv_in, uu_in]
            if closer.any():
                depth_buf[vv_in[closer], uu_in[closer]] = di[closer]
                rgb_buf[vv_in[closer], uu_in[closer]] = ci_rgb[closer]

        half_w = width / 2.0

        for i in range(len(f)):
            u0v, v0v = float(u0s_raw[i]), float(v0s[i])
            u1v, v1v = float(u1s_raw[i]), float(v1s[i])
            u2v, v2v = float(u2s_raw[i]), float(v2s[i])
            c0, c1, c2 = c0s[i], c1s[i], c2s[i]
            dep0, dep1, dep2 = float(d0s[i]), float(d1s[i]), float(d2s[i])

            # 检测是否跨越 ERP 左右边界（u 坐标差 > width/2）
            us_tri = np.array([u0v, u1v, u2v])
            u_span = float(us_tri.max() - us_tri.min())

            if u_span > half_w:
                # 跨边界：以 u0 为基准，把 u1/u2 对齐到与 u0 最近的循环副本
                def _align(u_ref, u_other):
                    diff = u_other - u_ref
                    if diff > half_w:
                        return u_other - width
                    elif diff < -half_w:
                        return u_other + width
                    return u_other

                u1_a = _align(u0v, u1v)
                u2_a = _align(u0v, u2v)

                # 主渲染（对齐后坐标，_raster_triangle 内部 clip 到 [0, width-1]）
                _raster_triangle(u0v, v0v, u1_a, v1v, u2_a, v2v, c0, c1, c2, dep0, dep1, dep2)
                # 循环副本（偏移 ±width 处理左右两侧黑边）
                _raster_triangle(u0v - width, v0v, u1_a - width, v1v, u2_a - width, v2v,
                                 c0, c1, c2, dep0, dep1, dep2)
                _raster_triangle(u0v + width, v0v, u1_a + width, v1v, u2_a + width, v2v,
                                 c0, c1, c2, dep0, dep1, dep2)
            else:
                _raster_triangle(u0v, v0v, u1v, v1v, u2v, v2v, c0, c1, c2, dep0, dep1, dep2)

    else:
        # ── 纯点云模式：Z-buffer Splatting ────────────────────────────
        valid_mask = dist > 1e-4
        u_i = np.clip(u_f.astype(np.int32), 0, width - 1)[valid_mask]
        v_i = np.clip(v_f.astype(np.int32), 0, height - 1)[valid_mask]
        dist_v = dist[valid_mask]
        col_v = col_all[valid_mask]

        order = np.argsort(dist_v)[::-1]
        radius = max(0, int(round(point_size / 2)) - 1)

        if radius == 0:
            near_order = np.argsort(dist_v)
            for idx in near_order:
                vi, ui, di = v_i[idx], u_i[idx], dist_v[idx]
                if di < depth_buf[vi, ui]:
                    depth_buf[vi, ui] = di
                    rgb_buf[vi, ui] = col_v[idx]
        else:
            for idx in order:
                vi, ui, di = int(v_i[idx]), int(u_i[idx]), float(dist_v[idx])
                if not np.isfinite(di):
                    continue
                v0 = max(0, vi - radius)
                v1 = min(height, vi + radius + 1)
                u0 = max(0, ui - radius)
                u1 = min(width, ui + radius + 1)
                region = depth_buf[v0:v1, u0:u1]
                mask = di < region
                region[mask] = di
                depth_buf[v0:v1, u0:u1] = region
                rgb_region = rgb_buf[v0:v1, u0:u1]
                rgb_region[mask] = col_v[idx]
                rgb_buf[v0:v1, u0:u1] = rgb_region

    depth_out = np.where(np.isfinite(depth_buf), depth_buf, 0.0).astype(np.float32)

    # ── 空洞填充：迭代最近邻复制，每轮向外扩 1 像素，不修改已有有效像素 ──────
    if faces is not None and len(faces) > 0:
        hole_mask = (depth_out == 0)  # True=空洞
        if hole_mask.any():
            kernel = np.ones((3, 3), np.uint8)
            valid_u8 = (~hole_mask).astype(np.uint8)
            for _ in range(32):
                if not hole_mask.any():
                    break
                dilated_valid = cv2.dilate(valid_u8, kernel)
                newly = hole_mask & (dilated_valid > 0)
                if not newly.any():
                    break
                for c in range(3):
                    src = rgb_buf[:, :, c]
                    # dilate 在 valid 区域的颜色，传播到邻近空洞（取邻域最大值近似最近邻）
                    expanded = cv2.dilate(src * valid_u8, kernel)
                    rgb_buf[:, :, c] = np.where(newly, expanded, src)
                hole_mask[newly] = False
                valid_u8[newly] = 1

    return rgb_buf, depth_out


def _extract_ply_colors(mesh_or_pc) -> Optional[np.ndarray]:
    """尝试从 trimesh 对象中提取顶点颜色 (N,3) uint8"""
    try:
        if isinstance(mesh_or_pc, trimesh.Trimesh):
            if mesh_or_pc.visual is not None:
                if hasattr(mesh_or_pc.visual, 'vertex_colors'):
                    vc = mesh_or_pc.visual.vertex_colors
                    if vc is not None and len(vc) > 0:
                        return np.array(vc[:, :3], dtype=np.uint8)
        elif isinstance(mesh_or_pc, trimesh.PointCloud):
            if mesh_or_pc.colors is not None and len(mesh_or_pc.colors) > 0:
                return np.array(mesh_or_pc.colors[:, :3], dtype=np.uint8)
    except Exception as e:
        print(f"  [WARN] 提取颜色失败: {e}")
    return None


def save_pose(cam_pos_yup: list, cam_rot_euler: list,
              output_path: str, frame_id: int):
    """保存位姿 JSON（ERPT 格式，Y-up 坐标系，cam_to_world）

    cam_pos_yup: [x, y, z]，Y-up 世界坐标（已是 ERPT_native）
    cam_rot_euler: [rx, ry, rz] 弧度，XYZ 顺序
    """
    R_cw = _euler_to_rotation_matrix(*cam_rot_euler)

    # 旋转矩阵 → 四元数（XYZW 顺序，转为 WXYZ）
    # Shepperd 方法
    m = R_cw
    t = m[0, 0] + m[1, 1] + m[2, 2]
    if t > 0:
        s = 0.5 / math.sqrt(t + 1.0)
        w = 0.25 / s
        x = (m[2, 1] - m[1, 2]) * s
        y = (m[0, 2] - m[2, 0]) * s
        z = (m[1, 0] - m[0, 1]) * s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s

    pose_data = {
        "frame_id": frame_id,
        "position": [float(v) for v in cam_pos_yup],
        "rotation_quaternion": [float(w), float(x), float(y), float(z)],
        "camera_type": "erp_ray",
        "coordinate_system": "right-handed, Y-up, Z-forward (cam_to_world)",
        "render_method": "ply_erp",
    }
    with open(output_path, 'w') as f:
        json.dump(pose_data, f, indent=2)


def build_ray_directions(H=WARP_H, W=WARP_W):
    """ERP 射线方向（Y-up 坐标系）"""
    i = np.arange(H, dtype=np.float64)
    j = np.arange(W, dtype=np.float64)
    lat = np.pi / 2 - np.pi * (i + 0.5) / H   # [-π/2, π/2]
    lon = 2 * np.pi * (j + 0.5) / W             # [0, 2π]
    lat, lon = np.meshgrid(lat, lon, indexing='ij')
    r_xz = np.cos(lat)
    return np.stack([
        r_xz * np.sin(lon),   # X
        np.sin(lat),           # Y (up)
        r_xz * np.cos(lon),   # Z (front)
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
    if len(pts) == 0:
        return np.zeros((H, W), dtype=bool)
    tgt = np.array(tgt_pos, dtype=np.float64)
    vecs = pts - tgt
    x, y, z = vecs[:, 0], vecs[:, 1], vecs[:, 2]
    r_xz = np.sqrt(x ** 2 + z ** 2)
    lat = np.arctan2(y, r_xz)
    lon = np.arctan2(x, z) % (2 * np.pi)
    vi = np.clip(((np.pi / 2 - lat) / np.pi * H).astype(np.int32), 0, H - 1)
    uj = np.clip((lon / (2 * np.pi) * W).astype(np.int32), 0, W - 1)
    cov = np.zeros((H, W), dtype=bool)
    cov[vi, uj] = True
    pad = cov.copy()
    pad[1:, :] |= cov[:-1, :]
    pad[:-1, :] |= cov[1:, :]
    pad[:, 1:] |= cov[:, :-1]
    pad[:, :-1] |= cov[:, 1:]
    return pad


def select_next_frame(candidates, selected_idx, selected_pos, all_pts,
                      reachable=None):
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


def load_depth_downsampled(path, H=WARP_H, W=WARP_W):
    d = np.load(path).astype(np.float32)
    d = np.nan_to_num(d, nan=0.0)
    if d.shape == (H, W):
        return d
    try:
        return cv2.resize(d, (W, H), interpolation=cv2.INTER_AREA)
    except Exception:
        h, w = d.shape
        bh, bw = h // H, w // W
        if bh < 1 or bw < 1:
            r = np.zeros((H, W), dtype=np.float32)
            r[:min(h, H), :min(w, W)] = d[:min(h, H), :min(w, W)]
            return r
        return d[:bh * H, :bw * W].reshape(H, bh, W, bw).mean(axis=(1, 3))


def trim_depth(new_depth, new_pos, existing_pts, ray_dirs):
    n_orig = int(np.sum(new_depth > 0))
    if len(existing_pts) == 0:
        return new_depth.copy(), n_orig, n_orig
    cov = project_points_to_coverage(existing_pts, new_pos,
                                     new_depth.shape[0], new_depth.shape[1])
    trimmed = new_depth.copy()
    trimmed[cov] = 0
    return trimmed, n_orig, int(np.sum(trimmed > 0))


def update_reachability(current_pos: np.ndarray, candidates, selected_idx,
                        reachable: set, raycaster: RayCaster):
    """从当前位置出发，检测哪些候选点可达（无遮挡直线视线）"""
    if raycaster._intersector is None:
        # 非 Mesh：所有候选都"可达"
        for ci, c in enumerate(candidates):
            if ci not in selected_idx:
                reachable.add(ci)
        return 0

    n_new = 0
    for ci, cand in enumerate(candidates):
        if ci in selected_idx or ci in reachable:
            continue
        target = np.array(cand, dtype=np.float64)
        dist_to_target = float(np.linalg.norm(target - current_pos))
        if dist_to_target < 0.1:
            reachable.add(ci)
            n_new += 1
            continue
        direction = (target - current_pos) / dist_to_target
        hit, hit_dist = raycaster.cast_ray(current_pos, direction)
        if not hit or hit_dist >= dist_to_target * 0.95:
            reachable.add(ci)
            n_new += 1

    return n_new


def run_phase2(pts_world: np.ndarray,
               colors_world,
               faces,
               candidates,
               mesh_center,
               raycaster: RayCaster,
               output_dir: str,
               max_frames: int,
               resolution,
               args):
    """边选帧边渲染主循环（PLY 版本，逻辑与 Blender 版本对齐）"""
    W_render, H_render = resolution
    ray_dirs = get_ray_dirs(WARP_H, WARP_W)
    max_depth = compute_max_depth(candidates)

    selected_idx = set()
    selected_pos = []
    all_pts = np.empty((0, 3), dtype=np.float64)
    pts_chunks = []
    results = []
    reachable = set()

    stop_score = args.stop_score
    stop_delta = args.stop_delta
    min_frames = args.min_frames

    ACTUAL_GAIN_WINDOW = 3
    ACTUAL_GAIN_FLOOR = args.stop_gain
    actual_gain_history = []
    delta_history = []

    # ── 异步 I/O 线程池：写盘与 GPU 渲染并行 ─────────────────────────────
    _io_executor = ThreadPoolExecutor(max_workers=2)
    _pending_io  = []   # list of Future，用于等待上一帧写盘完成

    def _save_frame_async(rgb, depth, rgb_path, depth_npy, pos, cam_rot,
                          pose_path, frame_id):
        """在线程池中异步保存 PNG + npy + json"""
        def _do_save():
            cv2.imwrite(rgb_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            np.save(depth_npy, depth.astype(np.float32))
            save_pose(pos, cam_rot, pose_path, frame_id)
        return _io_executor.submit(_do_save)
    consecutive_skips = 0
    MAX_CONSECUTIVE_SKIPS = 3

    # 楼层分组（按 Y 坐标聚类）
    y_vals = sorted(set(round(c[1], 2) for c in candidates))
    floors = [[y_vals[0]]]
    for y in y_vals[1:]:
        if y - floors[-1][-1] > 1.0:
            floors.append([y])
        else:
            floors[-1].append(y)

    n_floors = len(floors)
    floor_mids = [sum(f) / len(f) for f in floors]
    candidate_floor = [
        min(range(n_floors), key=lambda i: abs(c[1] - floor_mids[i]))
        for c in candidates
    ]
    current_floor = 0

    def floor_set(fi):
        return set(i for i, f in enumerate(candidate_floor) if f == fi)

    floor_names = [f"楼层{i+1}(Y={min(f):.1f}~{max(f):.1f})" for i, f in enumerate(floors)]

    print(f"\n{'='*60}")
    print(f"[Phase 2] 边渲边选 (候选={len(candidates)}, 最多={max_frames}帧)")
    print(f"{'='*60}")
    print(f"  {n_floors} 个楼层: {floor_names}")

    t_total = time.time()
    time_select = time_render = time_depth = time_reach = 0.0

    for frame_count in range(max_frames):

        # ---- 选位置 ----
        t_sel = time.time()
        if frame_count == 0:
            floor0_cands = [(i, c) for i, c in enumerate(candidates)
                            if candidate_floor[i] == 0]
            if floor0_cands:
                f0_pts = np.array([c for _, c in floor0_cands])
                xz_center = np.array([f0_pts[:, 0].mean(), f0_pts[:, 2].mean()])
                floor0_ys = sorted(set(c[1] for _, c in floor0_cands))
                y_target = min(floor0_ys) + 1.2
                target = np.array([xz_center[0], y_target, xz_center[1]])
                dists = [np.linalg.norm(np.array(c) - target)
                         for _, c in floor0_cands]
                ci = floor0_cands[int(np.argmin(dists))][0]
            else:
                mc = np.array(mesh_center, dtype=np.float64)
                ci = int(np.argmin([np.linalg.norm(np.array(c) - mc)
                                    for c in candidates]))
            gain, score = 1.0, 1.0
            print(f"\n  F{frame_count}: 选候选[{ci}] (楼层中心, Y={candidates[ci][1]:.2f}m) "
                  f"[{floor_names[current_floor]}]")
        else:
            cur_floor_ids = floor_set(current_floor)
            floor_reachable = reachable & cur_floor_ids if reachable else set()

            ci, gain, score, n_remain = select_next_frame(
                candidates, selected_idx, selected_pos, all_pts,
                reachable=floor_reachable if floor_reachable else cur_floor_ids)

            expand = False
            if ci < 0 or score < stop_score:
                ci2, gain2, score2, n2 = select_next_frame(
                    candidates, selected_idx, selected_pos, all_pts,
                    reachable=cur_floor_ids)
                if ci2 >= 0 and (ci < 0 or score2 > score):
                    ci, gain, score, n_remain = ci2, gain2, score2, n2
                    expand = True

            if ci < 0 or (score < stop_score and gain < ACTUAL_GAIN_FLOOR):
                reason = "无可选候选" if ci < 0 else f"gain={gain:.1%} score={score:.3f}"
                current_floor += 1
                if current_floor < n_floors:
                    print(f"\n  F{frame_count}: {reason}"
                          f" → 切换到 {floor_names[current_floor]}")
                    continue
                else:
                    print(f"\n  F{frame_count}: {reason} → 所有楼层拍满，停止")
                    break

            tag = "[扩展]" if expand else ""
            print(f"\n  F{frame_count}: 选候选[{ci}]  "
                  f"gain={gain:.1%}  score={score:.3f}  剩余={n_remain}"
                  f"  [Y={candidates[ci][1]:.2f} {floor_names[current_floor]}]{tag}")

        pos = candidates[ci]
        selected_idx.add(ci)
        selected_pos.append(pos)
        dt_sel = time.time() - t_sel
        time_select += dt_sel

        # ---- 渲染（GPU 共享场景数据，减少重复上传）----
        cam_rot = get_camera_rot(args.rotation_type, frame_count)
        base = f"panorama_{frame_count:04d}"
        rgb_path = os.path.join(output_dir, f"{base}.png")
        depth_npy = os.path.join(output_dir, f"{base}_depth.npy")
        pose_path = os.path.join(output_dir, f"pose_{frame_count:04d}.json")

        print(f"    位置: [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]")
        print(f"    渲染...", end="", flush=True)
        t_r = time.time()

        # 等待上一帧异步 I/O 完成（保证不超过 2 帧待写）
        while len(_pending_io) >= 2:
            _pending_io.pop(0).result()

        # 使用 render_erp_batch_gpu：场景数据只上传一次
        batch_results = render_erp_batch_gpu(
            pts_world, colors_world,
            cam_poses=[pos],
            cam_rots=[cam_rot],
            width=W_render,
            height=H_render,
            faces=faces,
            point_size=args.point_size,
        )
        rgb, depth = batch_results[0]

        # 异步写盘（与下一帧 GPU 渲染并行）
        fut = _save_frame_async(
            rgb, depth, rgb_path, depth_npy,
            pos, cam_rot, pose_path, frame_count)
        _pending_io.append(fut)

        dt_r = time.time() - t_r
        time_render += dt_r
        print(f" {dt_r:.1f}s")

        # ---- depth → 3D 点云 ----
        t_dep = time.time()
        actual_gain = 1.0
        delta_ratio = 1.0

        # 直接使用内存中的 depth（depth_npy 由异步线程写入，可能尚未落盘）
        _d = depth.astype(np.float32)
        _d = np.nan_to_num(_d, nan=0.0)
        if _d.shape == (WARP_H, WARP_W):
            depth_small = _d
        else:
            try:
                depth_small = cv2.resize(_d, (WARP_W, WARP_H), interpolation=cv2.INTER_AREA)
            except Exception:
                h, w = _d.shape
                bh, bw = h // WARP_H, w // WARP_W
                if bh < 1 or bw < 1:
                    depth_small = np.zeros((WARP_H, WARP_W), dtype=np.float32)
                    depth_small[:min(h, WARP_H), :min(w, WARP_W)] = _d[:min(h, WARP_H), :min(w, WARP_W)]
                else:
                    depth_small = _d[:bh*WARP_H, :bw*WARP_W].reshape(WARP_H, bh, WARP_W, bw).mean(axis=(1, 3))
        total_px = WARP_H * WARP_W
        n_valid = int(np.sum(depth_small > 0))
        valid_ratio = n_valid / total_px

        if frame_count == 0:
            new_pts = depth_to_3d_points(pos, depth_small, ray_dirs, max_depth)
            pts_chunks.append(new_pts)
            all_pts = new_pts
            actual_gain = valid_ratio
            print(f"    depth: {n_valid}px ({valid_ratio:.0%} 有效)"
                  f" → {len(new_pts)} 个 3D 点 (全部)")
        else:
            MIN_VALID_RATIO = 0.10   # PLY 点云空洞较多，阈值适当降低
            if valid_ratio < MIN_VALID_RATIO:
                print(f"    depth: {n_valid}px ({valid_ratio:.0%} 有效) < "
                      f"{MIN_VALID_RATIO:.0%} → 跳过此帧")
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
                for fp in [rgb_path, depth_npy]:
                    if os.path.exists(fp):
                        try:
                            os.remove(fp)
                        except OSError:
                            pass
                consecutive_skips += 1
                if consecutive_skips >= MAX_CONSECUTIVE_SKIPS:
                    current_floor += 1
                    consecutive_skips = 0
                    if current_floor < n_floors:
                        print(f"    连续 {MAX_CONSECUTIVE_SKIPS} 帧空洞"
                              f" → 切换到 {floor_names[current_floor]}")
                    else:
                        print(f"    连续 {MAX_CONSECUTIVE_SKIPS} 帧空洞，停止")
                        break
                time_depth += time.time() - t_dep
                continue

            trimmed, n_orig, n_new = trim_depth(depth_small, pos, all_pts, ray_dirs)
            new_pts = depth_to_3d_points(pos, trimmed, ray_dirs, max_depth)
            pts_chunks.append(new_pts)
            all_pts = np.concatenate(pts_chunks)
            actual_gain = n_new / total_px
            delta_ratio = len(new_pts) / len(all_pts) if len(all_pts) > 0 else 1.0
            print(f"    depth: {n_valid}px ({valid_ratio:.0%} 有效)"
                  f" → trim → {n_new}px 新增 → {len(new_pts)} 个新 3D 点")
            print(f"    累积点云: {len(all_pts)}, 实际gain: {actual_gain:.1%}")
            consecutive_skips = 0

        time_depth += time.time() - t_dep

        results.append({
            "frame_id": frame_count,
            "candidate_idx": ci,
            "position": pos,
            "gain": float(gain),
            "actual_gain": float(actual_gain),
            "delta_ratio": float(delta_ratio),
            "score": float(score),
        })

        # ---- 可达性更新 ----
        t_reach = time.time()
        n_new_r = update_reachability(
            np.array(pos), candidates, selected_idx, reachable, raycaster)
        dt_reach = time.time() - t_reach
        time_reach += dt_reach
        print(f"    [可达性] 新增 {n_new_r} 个，总 {len(reachable)}/{len(candidates)} "
              f"({dt_reach:.1f}s)")

        # ---- 停止条件 ----
        if frame_count > 0:
            actual_gain_history.append(actual_gain)
            delta_history.append(delta_ratio)

        if frame_count > 0 and frame_count >= min_frames:
            if len(actual_gain_history) >= ACTUAL_GAIN_WINDOW:
                recent_gain = actual_gain_history[-ACTUAL_GAIN_WINDOW:]
                recent_delta = delta_history[-ACTUAL_GAIN_WINDOW:]
                gain_exhausted = all(g < ACTUAL_GAIN_FLOOR for g in recent_gain)
                delta_exhausted = all(d < stop_delta for d in recent_delta)

                if gain_exhausted or delta_exhausted:
                    current_floor += 1
                    if current_floor < n_floors:
                        reason = (f"gain<{ACTUAL_GAIN_FLOOR:.0%}" if gain_exhausted
                                  else f"delta<{stop_delta:.0%}")
                        print(f"    连续 {ACTUAL_GAIN_WINDOW} 帧 {reason}"
                              f" → 切换到 {floor_names[current_floor]}")
                    else:
                        print(f"    所有楼层拍满，停止")
                        break

    # 补帧：确保 4n+1
    while len(results) > 1 and (len(results) - 1) % 4 != 0:
        frame_count = results[-1]["frame_id"] + 1
        if frame_count >= max_frames + 3:
            break
        print(f"\n  [补帧] 当前 {len(results)} 帧，需补至 4n+1")
        ci, gain, score, n_remain = select_next_frame(
            candidates, selected_idx, selected_pos, all_pts, reachable=None)
        if ci < 0:
            break

        pos = candidates[ci]
        selected_idx.add(ci)
        selected_pos.append(pos)

        cam_rot = get_camera_rot(args.rotation_type, frame_count)
        base = f"panorama_{frame_count:04d}"
        rgb_path = os.path.join(output_dir, f"{base}.png")
        depth_npy = os.path.join(output_dir, f"{base}_depth.npy")

        batch_res = render_erp_batch_gpu(
            pts_world, colors_world,
            cam_poses=[pos], cam_rots=[cam_rot],
            width=W_render, height=H_render,
            faces=faces, point_size=args.point_size,
        )
        rgb, depth = batch_res[0]
        try:
            cv2.imwrite(rgb_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        except Exception:
            from PIL import Image
            Image.fromarray(rgb).save(rgb_path)
        np.save(depth_npy, depth.astype(np.float32))
        save_pose(pos, cam_rot, os.path.join(output_dir, f"pose_{frame_count:04d}.json"),
                  frame_count)

        depth_small = load_depth_downsampled(depth_npy, WARP_H, WARP_W)
        trimmed, n_orig, n_new = trim_depth(depth_small, pos, all_pts, ray_dirs)
        new_pts = depth_to_3d_points(pos, trimmed, ray_dirs, max_depth)
        pts_chunks.append(new_pts)
        all_pts = np.concatenate(pts_chunks)
        actual_gain = n_new / total_px
        results.append({
            "frame_id": frame_count,
            "candidate_idx": ci,
            "position": pos,
            "gain": float(gain),
            "actual_gain": float(actual_gain),
            "delta_ratio": float(len(new_pts) / max(len(all_pts), 1)),
            "score": float(score),
            "supplementary": True,
        })
        print(f"    补帧 F{frame_count}: gain={actual_gain:.1%}")

    # ── 等待所有异步 I/O 完成，关闭线程池 ────────────────────────────────
    for fut in _pending_io:
        try:
            fut.result()
        except Exception as e:
            print(f"  [WARN] 异步写盘失败: {e}")
    _io_executor.shutdown(wait=False)

    dt = time.time() - t_total
    print(f"\n  {'─'*50}")
    print(f"  共 {len(results)} 帧, {dt:.1f}s ({dt/60:.1f}min)")
    print(f"  耗时: 选帧={time_select:.1f}s  渲染={time_render:.1f}s  "
          f"深度={time_depth:.1f}s  可达={time_reach:.1f}s")
    return results


def main():
    args = parse_args()

    ply_path = str(Path(args.ply).resolve())
    output_dir = str(Path(args.output_dir).resolve())
    resolution = tuple(int(x) for x in args.resolution.split(","))
    os.makedirs(output_dir, exist_ok=True)
    sel_dir = os.path.join(output_dir, "frame_selection")
    os.makedirs(sel_dir, exist_ok=True)

    print("=" * 60)
    print("ERPT PLY Pipeline（边渲边选）")
    print("=" * 60)
    print(f"  PLY:        {ply_path}")
    print(f"  Output:     {output_dir}")
    print(f"  Max frames: {args.num_frames}")
    print(f"  Resolution: {resolution[0]}x{resolution[1]}")
    t_start = time.time()

    # ===== Phase 0: 加载场景 =====
    mesh_obj, pts_world, bmin, bmax, is_mesh, faces = load_ply_scene(
        ply_path, z_up=args.z_up)

    # 提取颜色
    colors_world = _extract_ply_colors(mesh_obj)
    if colors_world is not None:
        print(f"  颜色: {len(colors_world)} 个顶点颜色")
    else:
        print(f"  颜色: 无顶点颜色，使用深度伪彩")

    # 构建射线检测器
    raycaster = RayCaster(mesh_obj, pts_world, bmin, bmax, is_mesh, z_up=args.z_up)

    # ===== Phase 1: 撒点 + 过滤 =====
    print(f"\n{'='*60}")
    print("[Phase 1] 多层撒点 + 7 层过滤")
    print(f"{'='*60}")

    # Y-up 坐标系：Y=高度，floor=bmin[1]，ceiling=bmax[1]
    floor_y = float(bmin[1])
    ceiling_y = float(bmax[1])
    print(f"  场景 Y 范围: {floor_y:.2f} ~ {ceiling_y:.2f}m (总高 {ceiling_y-floor_y:.2f}m)")

    heights = compute_camera_heights(floor_y, ceiling_y, args.camera_height)
    print(f"  相机高度层: {[f'{h:.2f}m' for h in heights]}")

    x_range = float(bmax[0] - bmin[0])
    z_range = float(bmax[2] - bmin[2])
    x_sp = args.grid_spacing
    z_sp = args.grid_spacing

    # 候选点过多时自适应增大间距
    n_xy = max(1, int((x_range - 2 * MARGIN) / x_sp)) * \
           max(1, int((z_range - 2 * MARGIN) / z_sp))
    total_est = n_xy * len(heights)
    if total_est > 10000:
        scale = math.sqrt(total_est / 10000)
        x_sp = min(x_sp * scale, x_range / 4)
        z_sp = min(z_sp * scale, z_range / 4)
        print(f"  [自适应] 候选≈{total_est}个，间距调整为 X={x_sp:.1f}m Z={z_sp:.1f}m")

    candidates = generate_candidate_grid(bmin, bmax, x_sp, z_sp, heights)
    if not candidates:
        print("  [Error] 没有候选点")
        sys.exit(1)

    room_height = ceiling_y - floor_y
    candidates = raycast_filter(candidates, raycaster, room_height)
    if not candidates:
        print("  [Warning] 全部被过滤，使用 AABB 中心")
        cx = float((bmin[0] + bmax[0]) / 2)
        cy = float(heights[0])
        cz = float((bmin[2] + bmax[2]) / 2)
        candidates = [[cx, cy, cz]]

    np.save(os.path.join(sel_dir, "candidates_filtered.npy"),
            np.array(candidates))
    print(f"  最终候选点: {len(candidates)} 个")

    mesh_center = [
        float((bmin[0] + bmax[0]) / 2),
        float((bmin[1] + bmax[1]) / 2),
        float((bmin[2] + bmax[2]) / 2),
    ]

    # ===== Phase 2: 边渲边选 =====
    results = run_phase2(
        pts_world=pts_world,
        colors_world=colors_world,
        faces=faces,
        candidates=candidates,
        mesh_center=mesh_center,
        raycaster=raycaster,
        output_dir=output_dir,
        max_frames=args.num_frames,
        resolution=resolution,
        args=args,
    )

    # ===== 保存选帧摘要 =====
    summary = {
        "scene": os.path.basename(ply_path),
        "total_frames": len(results),
        "candidates_count": len(candidates),
        "frames": [{
            "frame_id": r["frame_id"],
            "position": r["position"],
            "gain": r["gain"],
            "actual_gain": r["actual_gain"],
            "delta_ratio": r["delta_ratio"],
            "score": r["score"],
        } for r in results if not r.get("skipped")],
    }
    with open(os.path.join(sel_dir, "selected_frames.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    dt = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"完成! {len(results)} 帧, {dt:.1f}s ({dt/60:.1f}min)")
    print(f"{'='*60}")
    print(f"输出目录: {output_dir}/")
    for r in results:
        if not r.get("skipped"):
            fid = r["frame_id"]
            print(f"  panorama_{fid:04d}.png  +  _depth.npy  +  pose_{fid:04d}.json")


if __name__ == "__main__":
    main()
