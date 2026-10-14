#!/usr/bin/env python3
"""
ERPT Pipeline 主入口（Forward Warp，深度估计可选）

使用方法：
    # 默认：使用已有深度真值做 warp（不加载深度估计权重）
    python run_pipeline.py --stage warp_only --data_dir /path/to/scene

    # 强制完整流程（深度估计 + warp）
    python run_pipeline.py --stage all

    # 仅深度估计
    python run_pipeline.py --stage depth_only
"""

import argparse
import re
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

import yaml
import numpy as np
import torch
import cv2

# 添加模块路径
import sys
sys.path.insert(0, str(Path(__file__).parent))

# Warp 相关（始终加载）
from core.erp_warp import warp_erp_to_target, WarpResult, create_comparison_image
from utils.io_utils import load_image, save_image, load_json, save_json, save_depth
from utils.pose_utils import Pose, load_pose

# 深度估计相关（延迟加载，仅 depth_only / all 模式才 import）
_depth_modules_loaded = False


def _load_depth_modules():
    """延迟加载深度估计模块（避免 warp_only 模式加载大模型权重）"""
    global _depth_modules_loaded
    if _depth_modules_loaded:
        return
    global build_icosahedron_slices, extract_all_tangents, compute_coverage_mask
    global estimate_all_tangent_depths
    global fuse_tangent_depths_to_erp, save_depth_visualization, visualize_depth

    from core.tangent_extraction import (
        build_icosahedron_slices,
        extract_all_tangents,
        compute_coverage_mask,
    )
    from core.depth_estimation import estimate_all_tangent_depths
    from core.depth_fusion import (
        fuse_tangent_depths_to_erp,
        save_depth_visualization,
        visualize_depth,
    )
    _depth_modules_loaded = True
    print("[Depth] 深度估计模块已加载")


# =============================================================================
# 数据发现
# =============================================================================

def discover_image_files(directory: Path) -> dict:
    """自动发现目录中的全景图文件"""
    image_extensions = ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']
    image_files = []
    for ext in image_extensions:
        image_files.extend(directory.glob(f"*{ext}"))
    image_files = sorted(image_files)
    if not image_files:
        return {}

    result = {}
    for img_path in image_files:
        stem = img_path.stem
        match = re.search(r'[_-](\d+)$', stem)
        if match:
            result[int(match.group(1))] = img_path
            continue
        if stem.isdigit():
            result[int(stem)] = img_path

    if not result:
        for idx, img_path in enumerate(image_files):
            result[idx] = img_path

    return result


def discover_pose_files(directory: Path) -> dict:
    """自动发现目录中的位姿文件"""
    pose_files = sorted(directory.glob("*.json"))
    result = {}
    for pose_path in pose_files:
        stem = pose_path.stem
        if stem in ['meta', 'config', 'stats', 'cameras', 'render_meta', 'description']:
            continue
        match = re.search(r'[_-](\d+)$', stem)
        if match:
            result[int(match.group(1))] = pose_path
            continue
        if stem.isdigit():
            result[int(stem)] = pose_path

    return result


# =============================================================================
# 配置加载
# =============================================================================

def load_config(config_path: Path) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_paths(cfg: Dict[str, Any], config_dir: Path) -> Dict[str, Any]:
    """解析相对路径为绝对路径"""
    data_cfg = cfg.get("data", {})
    for key in ["data_dir", "output_dir", "depth_dir"]:
        if key in data_cfg and data_cfg[key]:
            path = Path(data_cfg[key])
            if not path.is_absolute():
                data_cfg[key] = str(config_dir / path)

    depth_pro_cfg = cfg.get("depth_pro", {})
    if "repo_dir" in depth_pro_cfg and depth_pro_cfg["repo_dir"]:
        rp = Path(depth_pro_cfg["repo_dir"])
        if not rp.is_absolute():
            depth_pro_cfg["repo_dir"] = str(config_dir / rp)

    cfg["_project_root"] = str(config_dir)
    return cfg


# =============================================================================
# 深度估计流程（仅 all / depth_only 模式调用）
# =============================================================================

def run_depth_pipeline(
    center_rgb: np.ndarray,
    cfg: Dict[str, Any],
    device: torch.device,
    output_dir: Path,
    erp_h: int,
    erp_w: int,
    frame_id: int = 0,
) -> np.ndarray:
    """运行深度估计全流程：切片 -> 推理 -> 融合"""
    _load_depth_modules()

    depth_out_dir = output_dir / "depth_erp"
    depth_out_dir.mkdir(parents=True, exist_ok=True)

    # --- Step 1: 构建切片规格 ---
    print(f"\n{'='*60}")
    print(f"[Step 1] Building tangent slices (frame {frame_id})")
    print(f"{'='*60}")

    if "erp" not in cfg:
        cfg["erp"] = {}
    cfg["erp"]["height"] = erp_h
    cfg["erp"]["width"] = erp_w

    slices = build_icosahedron_slices(cfg)
    print(f"  Total slices: {len(slices)}")
    for s in slices:
        if s.slice_type != "face":
            print(f"    {s.slice_id}: type={s.slice_type}, fov={s.fov_deg:.1f}°")

    coverage_mask, coverage_stats = compute_coverage_mask(slices, erp_h, erp_w, device)
    print(f"  Coverage: {coverage_stats['total_coverage']:.2f}%")

    dbg_dir = output_dir / "debug"
    dbg_dir.mkdir(parents=True, exist_ok=True)
    save_image(np.stack([coverage_mask] * 3, axis=-1), dbg_dir / "coverage_mask.png")

    # --- Step 2: 提取切片 ---
    print(f"\n{'='*60}")
    print(f"[Step 2] Extracting tangent slices (frame {frame_id})")
    print(f"{'='*60}")

    t0 = time.time()
    tangent_rgbs = extract_all_tangents(center_rgb, slices, device)
    print(f"  Extracted {len(tangent_rgbs)} slices in {time.time()-t0:.2f}s")

    if cfg.get("run", {}).get("save_intermediates", False):
        tangent_dir = output_dir / "tangents"
        tangent_dir.mkdir(parents=True, exist_ok=True)
        for slice_id, rgb in tangent_rgbs.items():
            save_image(rgb, tangent_dir / f"{slice_id}_rgb.png")

    # --- Step 3: Depth Pro 推理 ---
    print(f"\n{'='*60}")
    print(f"[Step 3] Running Depth Pro inference (frame {frame_id})")
    print(f"{'='*60}")

    dp_cfg = cfg.get("depth_pro", {})
    if not bool(dp_cfg.get("enabled", True)):
        print("  [Warning] Depth Pro disabled, using dummy depth")
        tangent_depths = {}
        for sid, rgb in tangent_rgbs.items():
            tangent_depths[sid] = np.full(rgb.shape[:2], 5.0, dtype=np.float32)
    else:
        t0 = time.time()
        tangent_depths = estimate_all_tangent_depths(
            tangent_rgbs, slices, cfg, device,
        )
        print(f"  Estimated {len(tangent_depths)} depths in {time.time()-t0:.2f}s")

    if cfg.get("run", {}).get("save_intermediates", False):
        tangent_dir = output_dir / "tangents"
        for sid, depth in tangent_depths.items():
            save_depth(depth, tangent_dir / f"{sid}_depth.npy")

    # --- Step 4: 融合到 ERP ---
    print(f"\n{'='*60}")
    print(f"[Step 4] Fusing tangent depths to ERP (frame {frame_id})")
    print(f"{'='*60}")

    t0 = time.time()
    depth_erp, weight_sum, valid_mask = fuse_tangent_depths_to_erp(
        tangent_depths, slices, cfg, device,
        debug_dir=dbg_dir if cfg.get("run", {}).get("save_intermediates", False) else None,
    )
    print(f"  Fused in {time.time()-t0:.2f}s")

    valid_ratio = np.sum(valid_mask > 0) / (erp_h * erp_w)
    valid_depths = depth_erp[np.isfinite(depth_erp) & (depth_erp > 0)]
    if len(valid_depths) > 0:
        print(f"  Valid depth ratio: {valid_ratio * 100:.2f}%")
        print(f"  Depth range: [{valid_depths.min():.2f}, {valid_depths.max():.2f}] m")

    # --- Step 5: 保存结果 ---
    save_depth(depth_erp, depth_out_dir / f"depth_{frame_id:04d}.npy")
    save_depth_visualization(depth_erp, depth_out_dir / f"depth_{frame_id:04d}_vis.png")
    cv2.imwrite(str(depth_out_dir / f"depth_{frame_id:04d}_valid_mask.png"), valid_mask * 255)

    return depth_erp


# =============================================================================
# Warp 流程
# =============================================================================

def run_warp_pipeline(
    center_rgb: np.ndarray,
    depth_erp: np.ndarray,
    center_frame: int,
    image_files: dict,
    pose_files: dict,
    cfg: Dict[str, Any],
    device: torch.device,
    output_dir: Path,
    erp_h: int,
    erp_w: int,
) -> None:
    """运行 warp 全流程：遍历目标帧，执行 forward splatting"""
    warp_cfg = cfg.get("warp", {})
    output_depth = bool(warp_cfg.get("output_depth", True))

    # 确定目标帧列表
    available_targets = sorted([fid for fid in pose_files.keys() if fid != center_frame])

    cfg_targets = warp_cfg.get("target_frames", None)
    if cfg_targets is not None and cfg_targets != "auto":
        cfg_set = set(int(t) for t in cfg_targets)
        target_frames = [fid for fid in available_targets if fid in cfg_set]
    else:
        target_frames = available_targets

    print(f"\n{'='*60}")
    print(f"[Warp] Forward splatting from frame {center_frame}")
    print(f"{'='*60}")
    print(f"  Method: {warp_cfg.get('method', 'softmax_splatting')}")
    print(f"  Available targets with pose: {available_targets}")
    print(f"  Will warp: {target_frames}")

    # 加载中心帧位姿
    if center_frame not in pose_files:
        print(f"  [Error] Center pose not found for frame {center_frame}")
        return
    src_pose = load_pose(pose_files[center_frame])
    print(f"  Source pose: position={src_pose.position.tolist()}")

    # 输出目录
    warp_rgb_dir = output_dir / "warp_rgb"
    warp_rgb_dir.mkdir(parents=True, exist_ok=True)
    if output_depth:
        warp_depth_dir = output_dir / "warp_depth"
        warp_depth_dir.mkdir(parents=True, exist_ok=True)

    total_warp = len(target_frames)
    for idx, tgt_id in enumerate(target_frames):
        if tgt_id not in pose_files:
            print(f"  [{idx+1}/{total_warp}] Frame {tgt_id}: pose not found, skip")
            continue

        tgt_pose = load_pose(pose_files[tgt_id])
        print(f"  [{idx+1}/{total_warp}] Frame {center_frame} -> {tgt_id} ...", end="", flush=True)

        t0 = time.time()
        result = warp_erp_to_target(
            src_rgb=center_rgb,
            src_depth=depth_erp,
            src_pose=src_pose,
            tgt_pose=tgt_pose,
            cfg=cfg,
            device=device,
        )
        dt = time.time() - t0

        valid_pct = result.valid_mask.sum() / result.valid_mask.size * 100
        print(f" done ({dt:.2f}s, valid={valid_pct:.1f}%)")

        prefix = f"pano{center_frame:04d}_to_pano{tgt_id:04d}"

        # 保存 warped RGB
        save_image(result.warped_rgb, warp_rgb_dir / f"{prefix}_rgb.png")

        # 保存 valid mask
        cv2.imwrite(str(warp_rgb_dir / f"{prefix}_mask.png"), result.valid_mask * 255)

        # 保存 warped depth
        if output_depth and result.warped_depth is not None:
            save_depth(result.warped_depth, warp_depth_dir / f"{prefix}_depth_range.npy")

    print(f"  Warp complete. Output saved to: {warp_rgb_dir}")


# =============================================================================
# 主函数
# =============================================================================

def main():
    _script_dir = Path(__file__).parent
    _default_config = _script_dir / "config.yaml"

    parser = argparse.ArgumentParser(description="ERPT Pipeline")
    parser.add_argument("--config", type=str,
                        default=str(_default_config) if _default_config.exists() else None,
                        help="Config file path")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Data directory (overrides config)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (overrides config)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--stage", type=str, default="warp_only",
                        choices=["all", "depth_only", "warp_only"])
    parser.add_argument("--center_frame", type=int, default=None,
                        help="Center frame ID (overrides config)")
    args = parser.parse_args()

    # 加载配置
    if args.config:
        config_path = Path(args.config)
        cfg = load_config(config_path)
        cfg = resolve_paths(cfg, config_path.parent)
    else:
        cfg = {
            "data": {},
            "erp": {"auto_size": True},
            "tangent": {},
            "depth_pro": {"enabled": True, "precision": "fp16", "pass_f_px": True},
            "fusion": {"blend_mode": "multiband", "output_scale": 1.10, "k": 4},
            "run": {"save_intermediates": False},
        }

    # 命令行覆盖
    if args.data_dir:
        cfg["data"]["data_dir"] = str(Path(args.data_dir).resolve())
    if args.output_dir:
        cfg["data"]["output_dir"] = args.output_dir

    data_dir = Path(cfg["data"].get("data_dir", "inputs"))
    output_dir = Path(cfg["data"].get("output_dir", "outputs"))

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    print(f"Using device: {device}")

    center_frame = args.center_frame or int(cfg.get("warp", {}).get("center_frame", 0))

    print(f"\n{'='*60}")
    print("ERPT Pipeline")
    print(f"{'='*60}")
    print(f"Stage: {args.stage}")
    print(f"Data dir: {data_dir}")
    print(f"Output dir: {output_dir}")

    t_start = time.time()

    # --- 加载数据 ---
    print(f"\n{'='*60}")
    print("[Loading data]")
    print(f"{'='*60}")

    image_files = discover_image_files(data_dir)
    pose_files = discover_pose_files(data_dir)
    print(f"  Found {len(image_files)} images, {len(pose_files)} poses")

    if not image_files:
        raise FileNotFoundError(f"No image files found in: {data_dir}")

    if center_frame not in image_files:
        center_frame = sorted(image_files.keys())[0]
        print(f"  Using frame {center_frame} as center")

    center_rgb = load_image(image_files[center_frame])
    print(f"  Center image: {image_files[center_frame].name}")
    print(f"  Shape: {center_rgb.shape}")

    erp_cfg = cfg.get("erp", {})
    if bool(erp_cfg.get("auto_size", True)):
        erp_h, erp_w = center_rgb.shape[:2]
        print(f"  Auto size: {erp_w}x{erp_h}")
    else:
        erp_h = int(erp_cfg.get("height", 2048))
        erp_w = int(erp_cfg.get("width", 4096))

    # --- 深度加载 / 估计 ---
    depth_erp = None

    if args.stage == "all":
        print(f"\n  [Stage: all] 强制执行深度估计")
        depth_erp = run_depth_pipeline(
            center_rgb, cfg, device, output_dir, erp_h, erp_w, center_frame,
        )

    elif args.stage == "depth_only":
        depth_erp = run_depth_pipeline(
            center_rgb, cfg, device, output_dir, erp_h, erp_w, center_frame,
        )

    elif args.stage == "warp_only":
        # 搜索已有深度（真值 > 已估计结果），不回退到深度估计
        depth_candidates = []

        if center_frame in image_files:
            stem = image_files[center_frame].stem
            depth_candidates.append(data_dir / f"{stem}_depth.npy")
            depth_candidates.append(data_dir / f"{stem}_depth.exr")
            depth_candidates.append(data_dir / f"{stem}.npy")
        depth_candidates.append(data_dir / f"depth_{center_frame:04d}.npy")
        depth_candidates.append(output_dir / "depth_erp" / f"depth_{center_frame:04d}.npy")

        for dp in depth_candidates:
            if dp.exists():
                if dp.suffix == ".exr":
                    depth_erp = cv2.imread(str(dp), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
                    if depth_erp is not None and depth_erp.ndim == 3:
                        depth_erp = depth_erp[:, :, 0]
                    depth_erp = depth_erp.astype(np.float32) if depth_erp is not None else None
                else:
                    depth_erp = np.load(str(dp)).astype(np.float32)
                if depth_erp is not None:
                    print(f"  Loaded depth from {dp}")
                    break

        # 尺寸校验
        if depth_erp is not None and depth_erp.shape != (erp_h, erp_w):
            old_shape = depth_erp.shape
            depth_erp = cv2.resize(depth_erp, (erp_w, erp_h), interpolation=cv2.INTER_LINEAR)
            print(f"  [Warning] Depth resized: {old_shape} -> ({erp_h}, {erp_w})")

        # 没找到深度 → 报错（不回退到深度估计）
        if depth_erp is None:
            tried = "\n    ".join(str(p) for p in depth_candidates)
            raise FileNotFoundError(
                f"[warp_only] 未找到深度文件，无法执行 warp。\n"
                f"已搜索路径:\n    {tried}\n"
                f"如需深度估计请使用 --stage all"
            )

    # --- Warp 阶段 ---
    warp_cfg = cfg.get("warp", {})
    warp_enabled = bool(warp_cfg.get("enabled", True))

    if args.stage in ("all", "warp_only") and warp_enabled:
        run_warp_pipeline(
            center_rgb, depth_erp, center_frame,
            image_files, pose_files,
            cfg, device, output_dir, erp_h, erp_w,
        )

    # --- 完成 ---
    total_time = time.time() - t_start
    print(f"\n{'='*60}")
    print("Pipeline Complete")
    print(f"{'='*60}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Output saved to: {output_dir}")


if __name__ == "__main__":
    main()
