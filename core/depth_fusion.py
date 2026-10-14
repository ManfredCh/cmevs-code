"""
Tangent Depth -> ERP Depth 融合模块（完整移植自原版 ERPT）

核心功能：
1. 将每个切片的深度回投影到 ERP
2. 使用 cosine 权重实现平滑融合（无块状边界）
3. 使用 softmin(1/depth) 处理重叠区深度竞争
4. 极区增强处理
5. Multiband 金字塔融合（消除接缝）
6. Pole consistency 极区深度对齐
7. Z-buffer 门控投影（保持边缘锐利）

关键算法：
- Cosine 权重: w_face = max(0, dot(ray, face_center))^k
- Depth 竞争: softmin(1/depth) 确保近处优先且平滑过渡
- Forward splatting 将切片像素投影到 ERP
- Multiband: Gaussian/Laplacian 金字塔融合

输出：
- depth_range: ERP range depth (float32, meters)
- weight_sum: 权重和（用于 debug）
- valid_mask: 有效掩码
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from .tangent_extraction import TangentSlice


# =============================================================================
# 基础工具函数
# =============================================================================

def compute_cosine_weight(
    ray_dirs: torch.Tensor,
    face_center: torch.Tensor,
    k: float = 4.0,
) -> torch.Tensor:
    """
    计算 cosine 权重: w = max(0, dot(ray, face_center))^k
    """
    dots = torch.sum(ray_dirs * face_center.view(1, 1, 3), dim=-1)
    weights = torch.clamp(dots, min=0.0) ** k
    return weights


def _dirs_to_erp_uv(
    dirs_world: torch.Tensor,
    erp_h: int,
    erp_w: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """将世界坐标方向转换为 ERP 像素坐标"""
    x = dirs_world[..., 0]
    y = dirs_world[..., 1]
    z = dirs_world[..., 2]

    lon = torch.atan2(x, z)
    lat = torch.asin(torch.clamp(y, -1.0, 1.0))

    u = (lon + math.pi) / (2.0 * math.pi) * float(erp_w)
    u = torch.remainder(u, float(erp_w))
    v = (math.pi / 2.0 - lat) / math.pi * float(erp_h - 1)
    v = torch.clamp(v, 0.0, float(erp_h - 1))

    return u, v


# =============================================================================
# 极区权重处理
# =============================================================================

def _apply_pole_weights(
    slice_type: str,
    dirs_world: torch.Tensor,
    base_weight: torch.Tensor,
    fusion_cfg: Dict[str, Any],
) -> torch.Tensor:
    """极区权重门控与增强"""
    # pole_ring gating
    pole_ring_cfg = fusion_cfg.get("pole_ring", {})
    pole_ring_enabled = bool(pole_ring_cfg.get("enabled", True))
    pole_ring_min_lat_deg = float(pole_ring_cfg.get("min_latitude_deg", 60.0))
    pole_ring_ramp_deg = float(pole_ring_cfg.get("ramp_deg", 10.0))

    if slice_type == "pole_ring":
        if not pole_ring_enabled:
            return torch.zeros_like(base_weight)
        lat = torch.asin(torch.clamp(dirs_world[..., 1], -1.0, 1.0)) * (180.0 / math.pi)
        abs_lat = torch.abs(lat)
        ramp = torch.clamp(
            (abs_lat - pole_ring_min_lat_deg) / max(pole_ring_ramp_deg, 1e-3),
            min=0.0, max=1.0,
        )
        return base_weight * ramp

    # pole caps progressive boost
    pole_boost = bool(fusion_cfg.get("pole_boost", True))
    pole_boost_factor = float(fusion_cfg.get("pole_boost_factor", 1.5))
    pole_latitude_deg = float(fusion_cfg.get("pole_latitude_deg", 75.0))
    pole_ramp_deg = float(fusion_cfg.get("pole_ramp_deg", 10.0))

    if pole_boost and slice_type in ("pole_north", "pole_south"):
        lat = torch.asin(torch.clamp(dirs_world[..., 1], -1.0, 1.0)) * (180.0 / math.pi)
        abs_lat = torch.abs(lat)
        ramp = torch.clamp(
            (abs_lat - pole_latitude_deg) / max(pole_ramp_deg, 1e-3),
            min=0.0, max=1.0,
        )
        mult = 1.0 + ramp * (pole_boost_factor - 1.0)
        return base_weight * mult

    # faces 在极区衰减
    face_pole_cfg = fusion_cfg.get("face_pole_suppress", {})
    if slice_type == "face" and bool(face_pole_cfg.get("enabled", True)):
        min_lat = float(face_pole_cfg.get("min_latitude_deg", 70.0))
        ramp_deg = float(face_pole_cfg.get("ramp_deg", 10.0))
        min_scale = float(face_pole_cfg.get("min_scale", 0.4))
        lat = torch.asin(torch.clamp(dirs_world[..., 1], -1.0, 1.0)) * (180.0 / math.pi)
        abs_lat = torch.abs(lat)
        t = torch.clamp((abs_lat - min_lat) / max(ramp_deg, 1e-3), 0.0, 1.0)
        scale = 1.0 - t * (1.0 - min_scale)
        return base_weight * scale

    return base_weight


# =============================================================================
# Forward splatting（softmin_invdepth 模式用）
# =============================================================================

def _forward_splat(
    erp_h: int,
    erp_w: int,
    u: torch.Tensor,
    v: torch.Tensor,
    range_depth: torch.Tensor,
    weight: torch.Tensor,
    accum_weighted_invdepth: torch.Tensor,
    accum_weight: torch.Tensor,
    depth_competition: str,
    softmin_alpha: float,
    pole_boost: bool,
    pole_boost_factor: float,
    pole_latitude_deg: float,
) -> None:
    """Forward splatting with bilinear interpolation"""
    u_flat = u.reshape(-1)
    v_flat = v.reshape(-1)
    d_flat = range_depth.reshape(-1)
    w_flat = weight.reshape(-1)

    valid = torch.isfinite(d_flat) & (d_flat > 0.0) & torch.isfinite(w_flat) & (w_flat > 0.0)

    u0 = torch.floor(u_flat).to(torch.int64)
    v0 = torch.floor(v_flat).to(torch.int64)
    du = (u_flat - u0.to(u_flat.dtype)).clamp(0.0, 1.0)
    dv = (v_flat - v0.to(v_flat.dtype)).clamp(0.0, 1.0)

    u0_wrap = torch.remainder(u0, erp_w)
    u1_wrap = torch.remainder(u0 + 1, erp_w)
    v1 = v0 + 1

    w00 = (1.0 - du) * (1.0 - dv)
    w10 = du * (1.0 - dv)
    w01 = (1.0 - du) * dv
    w11 = du * dv

    if depth_competition == "softmin_invdepth":
        inv_depth = 1.0 / torch.clamp(d_flat, min=1e-6)
        value_to_splat = inv_depth
    elif depth_competition == "softmax_negdepth":
        exp_weight = torch.exp(-softmin_alpha * d_flat)
        w_flat = w_flat * exp_weight
        value_to_splat = d_flat
    else:
        value_to_splat = d_flat

    def _add(u_idx, v_idx, bilinear_w):
        v_ok = (v_idx >= 0) & (v_idx < erp_h)
        m = valid & v_ok
        u_safe = torch.where(m, u_idx, torch.zeros_like(u_idx))
        v_safe = torch.where(m, v_idx, torch.zeros_like(v_idx))
        idx = v_safe * erp_w + u_safe
        final_w = torch.where(m, bilinear_w * w_flat, torch.zeros_like(bilinear_w))
        final_val = torch.where(m, bilinear_w * w_flat * value_to_splat, torch.zeros_like(bilinear_w))
        accum_weight.scatter_add_(0, idx, final_w)
        accum_weighted_invdepth.scatter_add_(0, idx, final_val)

    _add(u0_wrap, v0, w00)
    _add(u1_wrap, v0, w10)
    _add(u0_wrap, v1, w01)
    _add(u1_wrap, v1, w11)


# =============================================================================
# Z-buffer 门控投影（multiband 模式用）
# =============================================================================

def _project_slice_to_erp_disp_weight_zbuffer(
    depth_t: torch.Tensor,
    slice_spec: TangentSlice,
    cfg: Dict[str, Any],
    erp_h: int,
    erp_w: int,
    depth_def: str,
    k: float,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    将单个切片投影到 ERP，输出 disparity(1/range) 与 weight。
    采用 per-slice z-buffer（min depth）避免同一 slice 内的边缘被平均糊掉。
    """
    fusion_cfg = cfg.get("fusion", {})
    weight_mode = str(fusion_cfg.get("weight_mode", "cosine"))

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
    ray_len = torch.norm(dirs_cam, dim=-1, keepdim=True).clamp(min=1e-9)
    dirs_cam = dirs_cam / ray_len

    R = torch.tensor(R_cw, device=device, dtype=torch.float32)
    dirs_world = torch.einsum("ij,hwj->hwi", R, dirs_cam)

    # range depth
    if depth_def == "z":
        range_depth = depth_t * ray_len.squeeze(-1)
    else:
        range_depth = depth_t

    u, v = _dirs_to_erp_uv(dirs_world, erp_h, erp_w)

    if weight_mode == "cosine":
        face_center = torch.tensor(slice_spec.center_dir, device=device, dtype=torch.float32)
        base_w = compute_cosine_weight(dirs_world, face_center, k=k)
    else:
        base_w = torch.ones_like(range_depth)

    base_w = _apply_pole_weights(slice_spec.slice_type, dirs_world, base_w, fusion_cfg)

    u_flat = u.reshape(-1)
    v_flat = v.reshape(-1)
    d_flat = range_depth.reshape(-1)
    w_flat = base_w.reshape(-1)

    valid = torch.isfinite(d_flat) & (d_flat > 0.0) & torch.isfinite(w_flat) & (w_flat > 0.0)

    u0 = torch.floor(u_flat).to(torch.int64)
    v0 = torch.floor(v_flat).to(torch.int64)
    du = (u_flat - u0.float()).clamp(0.0, 1.0)
    dv = (v_flat - v0.float()).clamp(0.0, 1.0)

    u0w = torch.remainder(u0, erp_w)
    u1w = torch.remainder(u0 + 1, erp_w)
    v1 = v0 + 1

    bw00 = (1.0 - du) * (1.0 - dv)
    bw10 = du * (1.0 - dv)
    bw01 = (1.0 - du) * dv
    bw11 = du * dv

    # Pass A: min depth
    min_depth = torch.full((erp_h * erp_w,), float("inf"), device=device, dtype=torch.float32)

    def _amin(ui, vi, bw):
        m = valid & (vi >= 0) & (vi < erp_h)
        ui_safe = torch.where(m, ui, torch.zeros_like(ui))
        vi_safe = torch.where(m, vi, torch.zeros_like(vi))
        idx = vi_safe * erp_w + ui_safe
        cand = torch.where(m, d_flat, torch.full_like(d_flat, float("inf")))
        min_depth.scatter_reduce_(0, idx, cand, reduce="amin", include_self=True)

    _amin(u0w, v0, bw00)
    _amin(u1w, v0, bw10)
    _amin(u0w, v1, bw01)
    _amin(u1w, v1, bw11)

    # Pass B: accumulate disparity near min depth
    disp_acc = torch.zeros(erp_h * erp_w, device=device, dtype=torch.float32)
    w_acc = torch.zeros(erp_h * erp_w, device=device, dtype=torch.float32)

    eps_abs = float(fusion_cfg.get("project_zbuffer_eps_abs_m", 0.02))
    eps_rel = float(fusion_cfg.get("project_zbuffer_eps_rel", 0.02))

    inv_d = 1.0 / torch.clamp(d_flat, min=1e-6)

    def _acc(ui, vi, bw):
        m = valid & (vi >= 0) & (vi < erp_h)
        ui_safe = torch.where(m, ui, torch.zeros_like(ui))
        vi_safe = torch.where(m, vi, torch.zeros_like(vi))
        idx = vi_safe * erp_w + ui_safe
        md = min_depth.gather(0, idx)
        gate = d_flat <= (md * (1.0 + eps_rel) + eps_abs)
        mm = m & gate
        w_here = torch.where(mm, bw * w_flat, torch.zeros_like(bw))
        disp_here = torch.where(mm, w_here * inv_d, torch.zeros_like(w_here))
        w_acc.scatter_add_(0, idx, w_here)
        disp_acc.scatter_add_(0, idx, disp_here)

    _acc(u0w, v0, bw00)
    _acc(u1w, v0, bw10)
    _acc(u0w, v1, bw01)
    _acc(u1w, v1, bw11)

    w_map = w_acc.view(erp_h, erp_w)
    disp_map = torch.zeros_like(w_map)
    m = w_map > 1e-9
    disp_map[m] = disp_acc.view(erp_h, erp_w)[m] / w_map[m]
    return disp_map, w_map


# =============================================================================
# Multiband 金字塔工具
# =============================================================================

def _pad_circular_w(x: torch.Tensor, pad: int) -> torch.Tensor:
    if pad <= 0:
        return x
    return torch.cat([x[..., -pad:], x, x[..., :pad]], dim=-1)


def _gauss5_kernel(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    k1 = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0], device=device, dtype=dtype)
    k1 = k1 / k1.sum()
    k2 = (k1[:, None] * k1[None, :]).view(1, 1, 5, 5)
    return k2


def _blur_circular_w(x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    import torch.nn.functional as F
    pad = kernel.shape[-1] // 2
    xw = _pad_circular_w(x, pad)
    xwh = F.pad(xw, (0, 0, pad, pad), mode="reflect")
    return F.conv2d(xwh, kernel)


def _down2(x: torch.Tensor) -> torch.Tensor:
    return x[..., ::2, ::2]


def _upsample2_circular_w(x: torch.Tensor, out_h: int, out_w: int) -> torch.Tensor:
    import torch.nn.functional as F
    x3 = torch.cat([x, x, x], dim=-1)
    y3 = F.interpolate(x3, size=(out_h, out_w * 3), mode="bilinear", align_corners=False)
    return y3[..., out_w: 2 * out_w]


# =============================================================================
# 主融合函数
# =============================================================================

@torch.no_grad()
def fuse_tangent_depths_to_erp(
    tangent_depths: Dict[str, np.ndarray],
    slices: List[TangentSlice],
    cfg: Dict[str, Any],
    device: torch.device,
    debug_dir: Optional[Path] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    将所有切片深度融合为 ERP range depth

    Args:
        tangent_depths: {slice_id: depth_array}
        slices: 切片规格列表
        cfg: 配置字典
        device: 计算设备

    Returns:
        depth_range: (erp_h, erp_w) ERP range depth, float32
        weight_sum: (erp_h, erp_w) 权重和
        valid_mask: (erp_h, erp_w) uint8
    """
    erp_cfg = cfg.get("erp", {})
    erp_h = int(erp_cfg.get("height", 1024))
    erp_w = int(erp_cfg.get("width", 2048))

    fusion_cfg = cfg.get("fusion", {})
    blend_mode = str(fusion_cfg.get("blend_mode", "softmin_invdepth"))

    if blend_mode == "multiband":
        depth_np, weight_np, valid_np = _fuse_multiband(
            tangent_depths, slices, cfg, device, erp_h, erp_w, debug_dir,
        )
    else:
        depth_np, weight_np, valid_np = _fuse_softmin(
            tangent_depths, slices, cfg, device, erp_h, erp_w,
        )

    # output_scale 校正
    output_scale = float(fusion_cfg.get("output_scale", 1.0))
    if output_scale != 1.0:
        valid = np.isfinite(depth_np) & (depth_np > 0)
        depth_np[valid] *= output_scale

    return depth_np, weight_np, valid_np


def _fuse_softmin(
    tangent_depths: Dict[str, np.ndarray],
    slices: List[TangentSlice],
    cfg: Dict[str, Any],
    device: torch.device,
    erp_h: int,
    erp_w: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """softmin_invdepth 模式融合"""
    fusion_cfg = cfg.get("fusion", {})
    weight_mode = str(fusion_cfg.get("weight_mode", "cosine"))
    k = float(fusion_cfg.get("k", 4.0))
    depth_competition = str(fusion_cfg.get("depth_competition", "softmin_invdepth"))
    softmin_alpha = float(fusion_cfg.get("softmin_alpha", 10.0))
    min_weight_sum = float(fusion_cfg.get("min_weight_sum", 1e-6))
    pole_boost = bool(fusion_cfg.get("pole_boost", True))
    pole_boost_factor = float(fusion_cfg.get("pole_boost_factor", 1.5))
    pole_latitude_deg = float(fusion_cfg.get("pole_latitude_deg", 75.0))
    pole_ring_cfg = fusion_cfg.get("pole_ring", {})
    pole_ring_enabled = bool(pole_ring_cfg.get("enabled", True))
    pole_ring_min_lat_deg = float(pole_ring_cfg.get("min_latitude_deg", 60.0))
    pole_ring_ramp_deg = float(pole_ring_cfg.get("ramp_deg", 5.0))
    depth_def = str(cfg.get("depth_pro", {}).get("depth_def", "z"))

    accum_weighted_invdepth = torch.zeros(erp_h * erp_w, device=device, dtype=torch.float32)
    accum_weight = torch.zeros(erp_h * erp_w, device=device, dtype=torch.float32)

    for s in slices:
        if s.slice_id not in tangent_depths:
            continue
        depth_np = tangent_depths[s.slice_id]
        depth_t = torch.from_numpy(depth_np.astype(np.float32)).to(device)

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

        if depth_def == "z":
            ray_length = torch.sqrt(x_cam ** 2 + y_cam ** 2 + 1.0)
            range_depth = depth_t * ray_length
        else:
            range_depth = depth_t

        u, v = _dirs_to_erp_uv(dirs_world, erp_h, erp_w)

        if weight_mode == "cosine":
            face_center = torch.tensor(s.center_dir, device=device, dtype=torch.float32)
            base_weight = compute_cosine_weight(dirs_world, face_center, k=k)
        else:
            base_weight = torch.ones_like(range_depth)

        if s.slice_type == "pole_ring":
            if not pole_ring_enabled:
                base_weight = torch.zeros_like(base_weight)
            else:
                lat = torch.asin(torch.clamp(dirs_world[..., 1], -1.0, 1.0)) * (180.0 / math.pi)
                abs_lat = torch.abs(lat)
                ramp = torch.clamp(
                    (abs_lat - pole_ring_min_lat_deg) / max(pole_ring_ramp_deg, 1e-3),
                    min=0.0, max=1.0,
                )
                base_weight = base_weight * ramp

        if pole_boost and s.slice_type in ("pole_north", "pole_south"):
            base_weight = base_weight * pole_boost_factor

        _forward_splat(
            erp_h, erp_w, u, v, range_depth, base_weight,
            accum_weighted_invdepth, accum_weight,
            depth_competition, softmin_alpha,
            pole_boost, pole_boost_factor, pole_latitude_deg,
        )

    valid_mask_t = accum_weight > min_weight_sum

    if depth_competition == "softmin_invdepth":
        avg_invdepth = torch.zeros_like(accum_weighted_invdepth)
        avg_invdepth[valid_mask_t] = accum_weighted_invdepth[valid_mask_t] / accum_weight[valid_mask_t]
        depth_out = torch.zeros_like(avg_invdepth)
        depth_out[valid_mask_t] = 1.0 / torch.clamp(avg_invdepth[valid_mask_t], min=1e-6)
    else:
        depth_out = torch.zeros_like(accum_weighted_invdepth)
        depth_out[valid_mask_t] = accum_weighted_invdepth[valid_mask_t] / accum_weight[valid_mask_t]

    depth_out[~valid_mask_t] = float("nan")

    depth_out = depth_out.reshape(erp_h, erp_w)
    weight_sum = accum_weight.reshape(erp_h, erp_w)
    valid_mask = valid_mask_t.reshape(erp_h, erp_w)

    return (
        depth_out.cpu().numpy().astype(np.float32),
        weight_sum.cpu().numpy().astype(np.float32),
        valid_mask.cpu().numpy().astype(np.uint8),
    )


def _fuse_multiband(
    tangent_depths: Dict[str, np.ndarray],
    slices: List[TangentSlice],
    cfg: Dict[str, Any],
    device: torch.device,
    erp_h: int,
    erp_w: int,
    debug_dir: Optional[Path] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Multiband 金字塔融合"""
    fusion_cfg = cfg.get("fusion", {})
    mb_cfg = fusion_cfg.get("multiband", {})
    levels = int(mb_cfg.get("levels", 6))
    highfreq_levels = int(mb_cfg.get("highfreq_levels", 2))
    eps = float(mb_cfg.get("eps", 1e-6))
    min_weight_sum = float(fusion_cfg.get("min_weight_sum", 1e-6))

    depth_def = str(cfg.get("depth_pro", {}).get("depth_def", "z"))
    k = float(fusion_cfg.get("k", 4.0))

    # Pole consistency 配置
    pole_cons_cfg = fusion_cfg.get("pole_consistency", {})
    if not isinstance(pole_cons_cfg, dict):
        pole_cons_cfg = {}
    pole_cons_enabled = bool(pole_cons_cfg.get("enabled", False))
    pole_cons_min_lat_deg = float(pole_cons_cfg.get("min_latitude_deg", 60.0))
    pole_cons_min_overlap = int(pole_cons_cfg.get("min_overlap_pixels", 4000))
    pole_cons_max_abs_log_shift = float(pole_cons_cfg.get("max_abs_log_shift", 0.7))
    pole_cons_ref_types = [str(x) for x in pole_cons_cfg.get("ref_slice_types", ["face", "pole_ring"])]
    pole_cons_target_types = [str(x) for x in pole_cons_cfg.get("target_slice_types", ["pole_north", "pole_south"])]

    top_v_max = int(math.floor((90.0 - pole_cons_min_lat_deg) / 180.0 * float(max(erp_h - 1, 1))))
    bot_v_min = int(math.ceil((90.0 + pole_cons_min_lat_deg) / 180.0 * float(max(erp_h - 1, 1))))
    top_v_max = max(0, min(erp_h - 1, top_v_max))
    bot_v_min = max(0, min(erp_h - 1, bot_v_min))

    ref_num_top = ref_den_top = ref_num_bot = ref_den_bot = None
    pole_pending: List[TangentSlice] = []
    if pole_cons_enabled:
        ref_num_top = torch.zeros((top_v_max + 1, erp_w), device=device, dtype=torch.float32)
        ref_den_top = torch.zeros_like(ref_num_top)
        ref_num_bot = torch.zeros((erp_h - bot_v_min, erp_w), device=device, dtype=torch.float32)
        ref_den_bot = torch.zeros_like(ref_num_bot)

    # Per-level accumulators
    kernel = _gauss5_kernel(device=device, dtype=torch.float32)

    Hs = [erp_h]
    Ws = [erp_w]
    for _ in range(1, levels):
        Hs.append(max(1, Hs[-1] // 2))
        Ws.append(max(1, Ws[-1] // 2))

    fused_lap: List[torch.Tensor] = []
    best_w: List[torch.Tensor] = []
    sum_w: List[torch.Tensor] = []
    sum_w_lap: List[torch.Tensor] = []

    for l in range(levels):
        shape = (1, 1, Hs[l], Ws[l])
        if l < highfreq_levels:
            fused_lap.append(torch.zeros(shape, device=device, dtype=torch.float32))
            best_w.append(torch.zeros(shape, device=device, dtype=torch.float32))
        else:
            fused_lap.append(torch.zeros(shape, device=device, dtype=torch.float32))
            sum_w.append(torch.zeros(shape, device=device, dtype=torch.float32))
            sum_w_lap.append(torch.zeros(shape, device=device, dtype=torch.float32))

    weight_sum0 = torch.zeros(erp_h, erp_w, device=device, dtype=torch.float32)

    def _process_one_slice(s: TangentSlice, depth_np: np.ndarray):
        depth_t = torch.from_numpy(depth_np.astype(np.float32)).to(device)

        disp0, w0 = _project_slice_to_erp_disp_weight_zbuffer(
            depth_t, s, cfg, erp_h, erp_w, depth_def, k, device,
        )
        return disp0, w0

    def _blend_into_pyramid(disp0: torch.Tensor, w0: torch.Tensor):
        nonlocal weight_sum0
        weight_sum0 += w0

        disp_pyr = [disp0.unsqueeze(0).unsqueeze(0)]
        w_pyr = [w0.unsqueeze(0).unsqueeze(0)]

        for l in range(1, levels):
            num = _blur_circular_w(disp_pyr[l - 1] * w_pyr[l - 1], kernel)
            den = _blur_circular_w(w_pyr[l - 1], kernel)
            num_ds = _down2(num)
            den_ds = _down2(den)
            disp_ds = num_ds / torch.clamp(den_ds, min=eps)
            disp_pyr.append(disp_ds)
            w_pyr.append(den_ds)

        lap_pyr: List[torch.Tensor] = []
        for l in range(levels - 1):
            up = _upsample2_circular_w(disp_pyr[l + 1], Hs[l], Ws[l])
            lap_pyr.append(disp_pyr[l] - up)
        lap_pyr.append(disp_pyr[-1])

        for l in range(levels):
            wl = w_pyr[l]
            Ll = lap_pyr[l]
            if l < highfreq_levels:
                better = wl > best_w[l]
                fused_lap[l] = torch.where(better, Ll, fused_lap[l])
                best_w[l] = torch.where(better, wl, best_w[l])
            else:
                idx = l - highfreq_levels
                sum_w_lap[idx] += wl * Ll
                sum_w[idx] += wl

    # Process non-pole slices first
    for s in slices:
        if s.slice_id not in tangent_depths:
            continue
        if pole_cons_enabled and (s.slice_type in pole_cons_target_types):
            pole_pending.append(s)
            continue

        disp0, w0 = _process_one_slice(s, tangent_depths[s.slice_id])

        # Reference accumulation for pole consistency
        if pole_cons_enabled and (s.slice_type in pole_cons_ref_types):
            if ref_num_top is not None and top_v_max >= 0:
                ref_num_top += disp0[:top_v_max + 1] * w0[:top_v_max + 1]
                ref_den_top += w0[:top_v_max + 1]
            if ref_num_bot is not None and bot_v_min < erp_h:
                ref_num_bot += disp0[bot_v_min:] * w0[bot_v_min:]
                ref_den_bot += w0[bot_v_min:]

        _blend_into_pyramid(disp0, w0)

    # Pole consistency pass
    if pole_cons_enabled and pole_pending and ref_num_top is not None:
        ref_disp_top = ref_num_top / torch.clamp(ref_den_top, min=eps)
        ref_disp_bot = ref_num_bot / torch.clamp(ref_den_bot, min=eps)

        for s in pole_pending:
            disp0, w0 = _process_one_slice(s, tangent_depths[s.slice_id])

            try:
                if s.slice_type == "pole_north":
                    disp_other = disp0[:top_v_max + 1]
                    w_other = w0[:top_v_max + 1]
                    disp_ref = ref_disp_top
                    den_ref = ref_den_top
                else:
                    disp_other = disp0[bot_v_min:]
                    w_other = w0[bot_v_min:]
                    disp_ref = ref_disp_bot
                    den_ref = ref_den_bot

                overlap = (w_other > 1e-9) & (den_ref > 1e-9) & (disp_other > eps) & (disp_ref > eps)
                n_overlap = int(overlap.sum().item())
                if n_overlap >= pole_cons_min_overlap:
                    log_ref = -torch.log(disp_ref[overlap].clamp(min=eps))
                    log_other = -torch.log(disp_other[overlap].clamp(min=eps))
                    shift = float(torch.median(log_ref - log_other).item())
                    shift = max(-pole_cons_max_abs_log_shift, min(pole_cons_max_abs_log_shift, shift))
                    disp0 = disp0 * float(math.exp(-shift))
            except Exception:
                pass

            _blend_into_pyramid(disp0, w0)

    # Finalize lowfreq levels
    for l in range(highfreq_levels, levels):
        idx = l - highfreq_levels
        fused_lap[l] = sum_w_lap[idx] / torch.clamp(sum_w[idx], min=eps)

    # Reconstruct fused disparity
    disp = fused_lap[-1]
    for l in range(levels - 2, -1, -1):
        disp = _upsample2_circular_w(disp, Hs[l], Ws[l]) + fused_lap[l]

    disp0_fused = disp.squeeze(0).squeeze(0)
    depth = torch.zeros_like(disp0_fused)
    m = disp0_fused > eps
    depth[m] = 1.0 / disp0_fused[m]
    depth[~m] = float("nan")

    weight_np = weight_sum0.detach().cpu().numpy().astype(np.float32)
    depth_np = depth.detach().cpu().numpy().astype(np.float32)
    valid_np = (weight_np > min_weight_sum).astype(np.uint8)

    return depth_np, weight_np, valid_np


# =============================================================================
# 可视化函数
# =============================================================================

def visualize_depth(
    depth: np.ndarray,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> np.ndarray:
    """
    可视化深度图（percentile + TURBO colormap）

    Returns:
        vis: (H, W, 3) uint8 RGB
    """
    d = depth.astype(np.float32).copy()
    valid = np.isfinite(d) & (d > 0)

    if not np.any(valid):
        return np.zeros((d.shape[0], d.shape[1], 3), dtype=np.uint8)

    if vmin is None:
        vmin = float(np.percentile(d[valid], 2))
    if vmax is None:
        vmax = float(np.percentile(d[valid], 98))
    vmax = max(vmax, vmin + 1e-6)

    d_norm = (np.clip(d, vmin, vmax) - vmin) / (vmax - vmin)
    d_norm[~valid] = 0.0
    d_u8 = (d_norm * 255).astype(np.uint8)

    try:
        import cv2
        cm = cv2.applyColorMap(d_u8, cv2.COLORMAP_TURBO)
        return cv2.cvtColor(cm, cv2.COLOR_BGR2RGB)
    except ImportError:
        return np.stack([d_u8, d_u8, d_u8], axis=-1)


def save_depth_visualization(
    depth: np.ndarray,
    output_path: Path,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> None:
    """保存深度可视化图像"""
    import cv2
    vis = visualize_depth(depth, vmin=vmin, vmax=vmax)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
