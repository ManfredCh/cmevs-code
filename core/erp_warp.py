"""
ERP Forward Warp 模块（移植自原版 ERPT erp_softsplat.py）

使用锁定的投影/坐标系接口：
- core.erp_projection: erp_to_direction, direction_to_erp, wrap_u, clamp_v
- utils.pose_utils: Pose (R_cw, R_wc, position)

算法流程：
1. 对每个 src ERP 像素，通过 erp_to_direction 获取射线方向
2. 根据深度计算 3D 点，变换到目标相机坐标系
3. 通过 direction_to_erp 投影到目标 ERP
4. Forward splatting 累积 RGB（softmax / zbuffer / point）

支持的 splatting 方法：
- softmax_splatting（默认）：自适应半径 + 高斯核 + softmax 深度竞争
- zbuffer_splatting：两遍 z-buffer 硬遮挡
- zbuffer_point：最近邻投影
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import torch

from .erp_projection import (
    erp_to_direction,
    direction_to_erp,
    wrap_u,
    create_erp_grid,
)

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.pose_utils import Pose


@dataclass
class WarpResult:
    """Warp 结果"""
    warped_rgb: np.ndarray                      # (H, W, 3) uint8
    valid_mask: np.ndarray                      # (H, W) uint8, 1=valid, 0=invalid
    flow: Optional[np.ndarray]                  # (H, W, 2) float32, optical flow
    weight_sum: np.ndarray                      # (H, W) float32
    warped_depth: Optional[np.ndarray] = None   # (H, W) float32, NaN=invalid


# =============================================================================
# Forward Projection（坐标变换）
# =============================================================================

@torch.no_grad()
def _forward_project(
    src_depth_t: torch.Tensor,
    src_pose: Pose,
    tgt_pose: Pose,
    erp_h: int,
    erp_w: int,
    device: torch.device,
    uu: Optional[torch.Tensor] = None,
    vv: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    将源 ERP 像素投影到目标 ERP

    使用锁定的 erp_projection 接口进行坐标变换。

    Returns:
        u_tgt, v_tgt: (H, W) 目标像素坐标
        range_tgt: (H, W) 目标 range depth
        dirs_tgt: (H, W, 3) 目标方向向量
    """
    if uu is None or vv is None:
        uu, vv = create_erp_grid(erp_h, erp_w, device)

    # 1. 源像素 -> 方向（源相机坐标系）
    dirs_src = erp_to_direction(uu, vv, erp_h, erp_w)  # (H, W, 3)

    # 2. 方向 * 深度 -> 源相机坐标系 3D 点
    P_cam_src = dirs_src * src_depth_t.unsqueeze(-1)    # (H, W, 3)

    # 3. 源相机 -> 世界
    R_cw_src = torch.tensor(src_pose.R_cw, device=device, dtype=torch.float32)
    t_src = torch.tensor(src_pose.position, device=device, dtype=torch.float32)
    P_world = torch.einsum("ij,hwj->hwi", R_cw_src, P_cam_src) + t_src

    # 4. 世界 -> 目标相机
    R_wc_tgt = torch.tensor(tgt_pose.R_wc, device=device, dtype=torch.float32)
    t_tgt = torch.tensor(tgt_pose.position, device=device, dtype=torch.float32)
    P_cam_tgt = torch.einsum("ij,hwj->hwi", R_wc_tgt, P_world - t_tgt)

    # 5. 目标 range depth 和方向
    range_tgt = torch.norm(P_cam_tgt, dim=-1)
    dirs_tgt = P_cam_tgt / torch.clamp(range_tgt.unsqueeze(-1), min=1e-9)

    # 6. 方向 -> 目标 ERP 像素
    u_tgt, v_tgt = direction_to_erp(dirs_tgt, erp_h, erp_w)
    u_tgt = wrap_u(u_tgt, erp_w)

    return u_tgt, v_tgt, range_tgt, dirs_tgt


# =============================================================================
# Adaptive Softmax Splatting
# =============================================================================

def _adaptive_splat_rgb(
    erp_h: int,
    erp_w: int,
    u: torch.Tensor,
    v: torch.Tensor,
    rgb: torch.Tensor,
    depth_compete: torch.Tensor,
    valid: torch.Tensor,
    alpha: float,
    radius: torch.Tensor,
    occlusion_gate: Optional[Dict[str, Any]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    自适应半径 softmax splatting

    - 高斯核加权
    - softmax(alpha * inv_depth) 深度竞争
    - 可选 occlusion gate（近似 z-buffer 门控）
    """
    device = u.device
    u_flat = u.reshape(-1)
    v_flat = v.reshape(-1)
    rgb_flat = rgb.reshape(-1, 3)
    d_flat = depth_compete.reshape(-1)
    valid_flat = valid.reshape(-1)
    r_flat = radius.reshape(-1)

    # 安全深度
    safe_d = torch.where(
        valid_flat & torch.isfinite(d_flat) & (d_flat > 0),
        d_flat, torch.ones_like(d_flat),
    )

    # Softmax 权重 = exp(alpha * inv_depth)
    inv_d = 1.0 / torch.clamp(safe_d, min=0.1)
    valid_inv = inv_d[valid_flat]
    inv_max = valid_inv.max() if len(valid_inv) > 0 else inv_d.max()
    exp_w = torch.exp(alpha * (inv_d - inv_max))

    # 可选 occlusion gate
    gate_enabled = False
    min_d_flat: Optional[torch.Tensor] = None
    gate_abs = 0.0
    gate_rel = 0.0
    if occlusion_gate and bool(occlusion_gate.get("enabled", False)):
        gate_enabled = True
        gate_abs = float(occlusion_gate.get("abs_eps_m", 0.05))
        gate_rel = float(occlusion_gate.get("rel_eps", 0.05))
        u_nn = torch.round(u_flat).to(torch.long)
        v_nn = torch.round(v_flat).to(torch.long)
        u_nn = torch.remainder(u_nn, erp_w)
        v_ok = (v_nn >= 0) & (v_nn < erp_h)
        v_nn_c = torch.clamp(v_nn, 0, erp_h - 1)
        idx_nn = v_nn_c * erp_w + u_nn
        min_d_flat = torch.full((erp_h * erp_w,), float("inf"), device=device)
        d_nn = torch.where(valid_flat & v_ok & torch.isfinite(d_flat),
                           d_flat, torch.full_like(d_flat, float("inf")))
        min_d_flat.scatter_reduce_(0, idx_nn, d_nn, reduce="amin", include_self=True)

    accum_rgb = torch.zeros(erp_h, erp_w, 3, device=device, dtype=torch.float32)
    accum_w = torch.zeros(erp_h, erp_w, device=device, dtype=torch.float32)
    accum_hit = torch.zeros(erp_h, erp_w, device=device, dtype=torch.float32)
    accum_d = torch.zeros(erp_h, erp_w, device=device, dtype=torch.float32)

    u0 = torch.floor(u_flat).to(torch.int64)
    v0 = torch.floor(v_flat).to(torch.int64)
    du = (u_flat - u0.float()).clamp(0, 1)
    dv = (v_flat - v0.float()).clamp(0, 1)

    # Splat 范围
    valid_radii = r_flat[valid_flat & torch.isfinite(r_flat)]
    max_r = min(int(valid_radii.max().item()) + 1, 5) if len(valid_radii) > 0 else 2

    def _add(u_idx, v_idx, bw):
        v_ok = (v_idx >= 0) & (v_idx < erp_h)
        m = valid_flat & v_ok & torch.isfinite(d_flat)
        u_safe = torch.where(m, u_idx, torch.zeros_like(u_idx))
        v_safe = torch.where(m, v_idx, torch.zeros_like(v_idx))
        idx = v_safe * erp_w + u_safe

        if gate_enabled and min_d_flat is not None:
            md = min_d_flat.gather(0, idx)
            gate = d_flat <= (md * (1.0 + gate_rel) + gate_abs)
            mm = m & gate
        else:
            mm = m

        final_w = torch.where(mm, bw * exp_w, torch.zeros_like(bw))
        hit_w = torch.where(mm, bw, torch.zeros_like(bw))
        accum_w.view(-1).scatter_add_(0, idx, final_w)
        accum_hit.view(-1).scatter_add_(0, idx, hit_w)
        accum_rgb.view(-1, 3).scatter_add_(
            0, idx.unsqueeze(-1).expand(-1, 3),
            (final_w.unsqueeze(-1) * rgb_flat).float(),
        )
        accum_d.view(-1).scatter_add_(0, idx, (final_w * d_flat).float())

    for di in range(-max_r, max_r + 1):
        for dj in range(-max_r, max_r + 1):
            dist_ij = math.sqrt(di * di + dj * dj)
            if dist_ij > max_r + 0.5:
                continue
            dx = float(di) - du
            dy = float(dj) - dv
            dist = torch.sqrt(dx * dx + dy * dy)
            within = dist <= (r_flat + 0.5)
            gauss_w = torch.where(
                within,
                torch.exp(-0.5 * (dist / r_flat.clamp(min=0.5)) ** 2),
                torch.zeros_like(r_flat),
            )
            u_off = torch.remainder(u0 + di, erp_w)
            v_off = v0 + dj
            _add(u_off, v_off, gauss_w)

    return accum_rgb, accum_w, accum_hit, accum_d


# =============================================================================
# Z-Buffer Splatting
# =============================================================================

def _zbuffer_splat_rgb(
    erp_h: int, erp_w: int,
    u: torch.Tensor, v: torch.Tensor,
    rgb: torch.Tensor, depth_compete: torch.Tensor, valid: torch.Tensor,
    eps_abs_m: float, eps_rel: float, min_w: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Z-buffer 硬遮挡 forward splatting（两遍法）"""
    device = u.device
    u_flat, v_flat = u.reshape(-1), v.reshape(-1)
    d_flat = depth_compete.reshape(-1)
    rgb_flat = rgb.reshape(-1, 3)
    valid_flat = valid.reshape(-1)

    m0 = valid_flat & torch.isfinite(u_flat) & torch.isfinite(v_flat) & \
         torch.isfinite(d_flat) & (d_flat > 0.0)

    u0 = torch.floor(u_flat).to(torch.int64)
    v0 = torch.floor(v_flat).to(torch.int64)
    du = (u_flat - u0.float()).clamp(0, 1)
    dv = (v_flat - v0.float()).clamp(0, 1)
    u0w = torch.remainder(u0, erp_w)
    u1w = torch.remainder(u0 + 1, erp_w)
    v1 = v0 + 1
    w00 = (1 - du) * (1 - dv)
    w10 = du * (1 - dv)
    w01 = (1 - du) * dv
    w11 = du * dv

    # Pass A: min depth
    min_depth = torch.full((erp_h * erp_w,), float("inf"), device=device)

    def _amin(ui, vi, w):
        m = m0 & (vi >= 0) & (vi < erp_h) & (w >= min_w)
        us = torch.where(m, ui, torch.zeros_like(ui))
        vs = torch.where(m, vi, torch.zeros_like(vi))
        idx = vs * erp_w + us
        cand = torch.where(m, d_flat, torch.full_like(d_flat, float("inf")))
        min_depth.scatter_reduce_(0, idx, cand, reduce="amin", include_self=True)

    _amin(u0w, v0, w00); _amin(u1w, v0, w10)
    _amin(u0w, v1, w01); _amin(u1w, v1, w11)

    # Pass B: accumulate near-front
    accum_rgb = torch.zeros(erp_h, erp_w, 3, device=device)
    accum_w = torch.zeros(erp_h, erp_w, device=device)
    accum_hit = torch.zeros(erp_h, erp_w, device=device)
    accum_d = torch.zeros(erp_h, erp_w, device=device)

    def _acc(ui, vi, w):
        m = m0 & (vi >= 0) & (vi < erp_h) & (w >= min_w)
        us = torch.where(m, ui, torch.zeros_like(ui))
        vs = torch.where(m, vi, torch.zeros_like(vi))
        idx = vs * erp_w + us
        md = min_depth.gather(0, idx)
        gate = d_flat <= (md * (1 + eps_rel) + eps_abs_m)
        mm = m & gate
        wf = torch.where(mm, w, torch.zeros_like(w))
        accum_w.view(-1).scatter_add_(0, idx, wf)
        accum_hit.view(-1).scatter_add_(0, idx, wf)
        accum_rgb.view(-1, 3).scatter_add_(
            0, idx.unsqueeze(-1).expand(-1, 3),
            (wf.unsqueeze(-1) * rgb_flat).float(),
        )
        accum_d.view(-1).scatter_add_(0, idx, (wf * d_flat).float())

    _acc(u0w, v0, w00); _acc(u1w, v0, w10)
    _acc(u0w, v1, w01); _acc(u1w, v1, w11)

    return accum_rgb, accum_w, accum_hit, accum_d


# =============================================================================
# Z-Buffer Point
# =============================================================================

def _zbuffer_point_rgb(
    erp_h: int, erp_w: int,
    u: torch.Tensor, v: torch.Tensor,
    rgb: torch.Tensor, depth_compete: torch.Tensor, valid: torch.Tensor,
    eps_abs_m: float, eps_rel: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Z-buffer 点渲染（radius=0, winner-take-all）"""
    device = u.device
    u_flat, v_flat = u.reshape(-1), v.reshape(-1)
    d_flat = depth_compete.reshape(-1)
    rgb_flat = rgb.reshape(-1, 3)
    valid_flat = valid.reshape(-1)

    m0 = valid_flat & torch.isfinite(u_flat) & torch.isfinite(v_flat) & \
         torch.isfinite(d_flat) & (d_flat > 0.0)

    u_nn = torch.remainder(torch.round(u_flat).to(torch.int64), erp_w)
    v_nn = torch.round(v_flat).to(torch.int64)
    v_ok = (v_nn >= 0) & (v_nn < erp_h)
    m = m0 & v_ok
    us = torch.where(m, u_nn, torch.zeros_like(u_nn))
    vs = torch.where(m, v_nn, torch.zeros_like(v_nn))
    idx = vs * erp_w + us

    # Pass A: min depth
    min_depth = torch.full((erp_h * erp_w,), float("inf"), device=device)
    cand = torch.where(m, d_flat, torch.full_like(d_flat, float("inf")))
    min_depth.scatter_reduce_(0, idx, cand, reduce="amin", include_self=True)

    # Pass B
    md = min_depth.gather(0, idx)
    gate = d_flat <= (md * (1 + eps_rel) + eps_abs_m)
    mm = m & gate
    wf = torch.where(mm, torch.ones_like(d_flat), torch.zeros_like(d_flat))

    accum_rgb = torch.zeros(erp_h, erp_w, 3, device=device)
    accum_w = torch.zeros(erp_h, erp_w, device=device)
    accum_hit = torch.zeros(erp_h, erp_w, device=device)
    accum_d = torch.zeros(erp_h, erp_w, device=device)

    accum_w.view(-1).scatter_add_(0, idx, wf)
    accum_hit.view(-1).scatter_add_(0, idx, wf)
    accum_rgb.view(-1, 3).scatter_add_(
        0, idx.unsqueeze(-1).expand(-1, 3),
        (wf.unsqueeze(-1) * rgb_flat).float(),
    )
    accum_d.view(-1).scatter_add_(0, idx, (wf * d_flat).float())

    return accum_rgb, accum_w, accum_hit, accum_d


# =============================================================================
# Hole Fill
# =============================================================================

def _edge_aware_hole_fill(
    rgb: np.ndarray, mask: np.ndarray,
    max_hole_px: int = 5,
    inpaint_radius: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    """小洞填充（只填充极小洞，避免 disocclusion 被错误填充）"""
    holes = (mask == 0).astype(np.uint8)
    if holes.sum() == 0:
        return rgb, mask

    num, labels, stats, _ = cv2.connectedComponentsWithStats(holes, connectivity=8)
    fill_mask = np.zeros_like(holes)
    max_area = max_hole_px * max_hole_px

    for i in range(1, num):
        area = stats[i, cv2.CC_STAT_AREA]
        if area <= max_area:
            fill_mask[labels == i] = 1

    if fill_mask.sum() == 0:
        return rgb, mask

    rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    filled = cv2.inpaint(rgb_bgr, fill_mask, inpaint_radius, cv2.INPAINT_TELEA)
    filled_rgb = cv2.cvtColor(filled, cv2.COLOR_BGR2RGB)

    rgb_out = rgb.copy()
    mask_out = mask.copy()
    fill_bool = fill_mask > 0
    rgb_out[fill_bool] = filled_rgb[fill_bool]
    mask_out[fill_bool] = 1

    return rgb_out, mask_out


# =============================================================================
# 主函数
# =============================================================================

@torch.no_grad()
def warp_erp_to_target(
    src_rgb: np.ndarray,
    src_depth: np.ndarray,
    src_pose: Pose,
    tgt_pose: Pose,
    cfg: Dict[str, Any],
    device: torch.device,
) -> WarpResult:
    """
    从源 ERP 视角 warp 到目标 ERP 视角

    使用锁定的 erp_projection.py 进行坐标变换，
    使用锁定的 pose_utils.Pose 进行位姿处理。

    Args:
        src_rgb: (H, W, 3) uint8 源 RGB
        src_depth: (H, W) float32 源 range depth（米）
        src_pose: 源相机位姿（Pose 实例）
        tgt_pose: 目标相机位姿（Pose 实例）
        cfg: 配置字典
        device: 计算设备

    Returns:
        WarpResult
    """
    warp_cfg = cfg.get("warp", {})
    method = str(warp_cfg.get("method", "softmax_splatting"))
    alpha = float(warp_cfg.get("alpha", 2.0))
    min_weight_sum = float(warp_cfg.get("min_weight_sum", 1e-4))
    output_flow = bool(warp_cfg.get("output_flow", True))
    output_depth = bool(warp_cfg.get("output_depth", True))
    depth_scale_factor = float(warp_cfg.get("depth_scale_factor", 1.0))

    # Z-buffer 参数
    z_eps_abs = float(warp_cfg.get("zbuffer_eps_abs_m", 0.03))
    z_eps_rel = float(warp_cfg.get("zbuffer_eps_rel", 0.03))
    z_min_w = float(warp_cfg.get("zbuffer_min_weight", 1e-3))

    # 自适应半径参数
    base_radius = float(warp_cfg.get("splat_radius_px", 1.5))
    radius_min = float(warp_cfg.get("radius_min_px", 0.6))
    radius_max_eq = float(warp_cfg.get("radius_max_px", 2.2))
    radius_max_pole = float(warp_cfg.get("radius_max_pole_px", 3.4))
    pole_radius_scale = float(warp_cfg.get("pole_radius_scale", 3.0))
    pole_lat_threshold = float(warp_cfg.get("pole_lat_threshold", 60.0)) * math.pi / 180.0
    depth_radius_scale = bool(warp_cfg.get("depth_radius_scale", False))
    depth_ref = float(warp_cfg.get("depth_ref_m", 2.0))
    depth_edge_aware = bool(warp_cfg.get("depth_edge_aware", True))
    depth_edge_threshold = float(warp_cfg.get("depth_edge_threshold", 0.3))
    depth_edge_min_scale = float(warp_cfg.get("depth_edge_min_scale", 0.12))

    # Hole fill
    hole_fill = bool(warp_cfg.get("hole_fill_enabled", False)) and method not in ("zbuffer_splatting", "zbuffer_point")
    max_hole_px = int(warp_cfg.get("max_hole_px", 16))

    erp_h, erp_w = src_rgb.shape[:2]

    # 转 tensor
    src_rgb_t = torch.from_numpy(src_rgb.astype(np.float32)).to(device) / 255.0
    src_depth_t = torch.from_numpy(src_depth.astype(np.float32)).to(device)
    if depth_scale_factor != 1.0:
        src_depth_t *= depth_scale_factor

    valid = torch.isfinite(src_depth_t) & (src_depth_t > 0.0)

    # --- 深度边缘掩码 ---
    depth_edge_scale = torch.ones_like(src_depth_t)
    if depth_edge_aware:
        from torch.nn.functional import conv2d
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                                dtype=torch.float32, device=device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                                dtype=torch.float32, device=device).view(1, 1, 3, 3)
        safe_d = torch.where(valid, src_depth_t, src_depth_t[valid].median() if valid.any() else torch.ones_like(src_depth_t))
        log_d = torch.log(torch.clamp(safe_d, min=0.1)).unsqueeze(0).unsqueeze(0)
        gx = conv2d(log_d, sobel_x, padding=1).squeeze()
        gy = conv2d(log_d, sobel_y, padding=1).squeeze()
        grad = torch.sqrt(gx ** 2 + gy ** 2)
        gmax = grad.max()
        if gmax > 1e-6:
            gnorm = grad / gmax
        else:
            gnorm = torch.zeros_like(grad)
        depth_edge_scale = torch.clamp(
            1.0 - gnorm / max(depth_edge_threshold, 1e-6),
            min=depth_edge_min_scale, max=1.0,
        )
        depth_edge_scale = torch.where(torch.isfinite(depth_edge_scale),
                                        depth_edge_scale, torch.ones_like(depth_edge_scale))

    # --- ERP 网格 ---
    uu, vv = create_erp_grid(erp_h, erp_w, device)

    # --- Forward project ---
    u_tgt, v_tgt, range_tgt, dirs_tgt = _forward_project(
        src_depth_t, src_pose, tgt_pose, erp_h, erp_w, device, uu, vv,
    )

    # --- 自适应半径 ---
    lat_tgt = torch.asin(torch.clamp(dirs_tgt[..., 1], -1.0, 1.0))
    abs_lat = torch.abs(lat_tgt)
    pole_factor = torch.clamp(
        (abs_lat - pole_lat_threshold) / (0.5 * math.pi - pole_lat_threshold),
        min=0.0, max=1.0,
    )
    lat_scale = 1.0 + pole_factor * (pole_radius_scale - 1.0)

    if depth_radius_scale:
        safe_range = torch.where(valid, range_tgt, torch.full_like(range_tgt, depth_ref))
        d_scale = 1.0 / (1.0 + safe_range / depth_ref)
    else:
        d_scale = torch.ones_like(range_tgt)

    adaptive_radius = base_radius * lat_scale * d_scale * depth_edge_scale
    adaptive_radius = torch.where(valid, adaptive_radius, torch.full_like(adaptive_radius, base_radius))
    radius_max_local = radius_max_eq + pole_factor * (radius_max_pole - radius_max_eq)
    adaptive_radius = torch.clamp(adaptive_radius, min=radius_min)
    adaptive_radius = torch.minimum(adaptive_radius, radius_max_local)

    # --- Splatting ---
    if method == "zbuffer_splatting":
        _rgb, _w, _hit, _d = _zbuffer_splat_rgb(
            erp_h, erp_w, u_tgt, v_tgt, src_rgb_t, range_tgt, valid,
            z_eps_abs, z_eps_rel, z_min_w,
        )
    elif method == "zbuffer_point":
        _rgb, _w, _hit, _d = _zbuffer_point_rgb(
            erp_h, erp_w, u_tgt, v_tgt, src_rgb_t, range_tgt, valid,
            z_eps_abs, z_eps_rel,
        )
    else:
        _rgb, _w, _hit, _d = _adaptive_splat_rgb(
            erp_h, erp_w, u_tgt, v_tgt, src_rgb_t, range_tgt, valid,
            alpha, adaptive_radius, warp_cfg.get("occlusion_gate", None),
        )

    # --- 归一化 ---
    denom = _w > 0.0
    out_rgb = torch.zeros_like(_rgb)
    out_rgb[denom] = _rgb[denom] / _w[denom].unsqueeze(-1)

    min_hit = float(warp_cfg.get("min_hit_sum", 1e-6))
    valid_mask = _hit > min_hit

    warped_np = (out_rgb.clamp(0, 1) * 255).byte().cpu().numpy()
    mask_np = valid_mask.cpu().numpy().astype(np.uint8)
    weight_np = _hit.cpu().numpy().astype(np.float32)

    # --- Warped depth ---
    warped_depth_np = None
    if output_depth:
        out_d = torch.full((erp_h, erp_w), float("nan"), device=device)
        out_d[denom] = _d[denom] / torch.clamp(_w[denom], min=1e-9)
        out_d[~valid_mask] = float("nan")
        warped_depth_np = out_d.cpu().numpy().astype(np.float32)

    # --- Hole fill ---
    if hole_fill:
        warped_np, mask_np = _edge_aware_hole_fill(warped_np, mask_np, max_hole_px)

    # --- Optical flow ---
    flow_np = None
    if output_flow:
        du = u_tgt - uu
        du = (du + 0.5 * erp_w) % erp_w - 0.5 * erp_w
        dv = v_tgt - vv
        flow_np = torch.stack([du, dv], dim=-1).cpu().numpy().astype(np.float32)

    return WarpResult(
        warped_rgb=warped_np,
        valid_mask=mask_np,
        flow=flow_np,
        weight_sum=weight_np,
        warped_depth=warped_depth_np,
    )


def create_comparison_image(
    warped_rgb: np.ndarray,
    valid_mask: np.ndarray,
    gt_rgb: Optional[np.ndarray] = None,
) -> np.ndarray:
    """创建对比图（warped | GT），如无 GT 则只返回 warped"""
    vis = warped_rgb.copy()
    vis[valid_mask == 0] = 0

    if gt_rgb is not None:
        return np.concatenate([vis, gt_rgb], axis=0)
    return vis
