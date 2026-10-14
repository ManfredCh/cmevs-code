"""
Depth Pro Wrapper（移植自原版 ERPT）

封装 Apple Depth Pro 单目深度估计模型。

API 使用说明:
    1. 使用 depth_pro.create_model_and_transforms() 创建模型和预处理 transforms
    2. 输入 RGB 图像 (PIL Image 或 numpy array)
    3. 调用 model.infer(image, f_px=focal_length) 得到深度
    4. 输出 depth 单位为米 (m)

深度定义:
    - Depth Pro 输出的是透视相机的 z-depth (沿相机前向轴的深度)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from .tangent_extraction import TangentSlice

# 模型缓存，避免重复加载
_MODEL_CACHE: Dict[str, Tuple[torch.nn.Module, Any]] = {}


def _get_precision(cfg: Dict[str, Any]) -> torch.dtype:
    """获取计算精度"""
    prec = cfg.get("depth_pro", {}).get("precision", "fp16")
    if prec == "fp16":
        return torch.float16
    elif prec == "bf16":
        return torch.bfloat16
    return torch.float32


def _load_depthpro_model(
    cfg: Dict[str, Any],
    device: torch.device,
) -> Tuple[torch.nn.Module, Any]:
    """
    加载 Depth Pro 模型和 transforms

    Depth Pro 默认从 ./checkpoints/depth_pro.pt 加载权重，
    因此需要切换到 repo 目录加载模型。
    """
    dcfg = cfg.get("depth_pro", {})

    # 获取 Depth Pro 仓库目录
    repo_dir = Path(dcfg.get("repo_dir", "third_party/ml-depth-pro"))
    if not repo_dir.is_absolute():
        root = Path(str(cfg.get("_project_root", Path.cwd())))
        repo_dir = root / repo_dir

    checkpoint_path = repo_dir / "checkpoints" / "depth_pro.pt"

    precision = _get_precision(cfg)
    cache_key = f"{checkpoint_path}_{device}_{precision}"

    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    # 添加 Depth Pro 路径到 sys.path
    if repo_dir.exists():
        src_path = str(repo_dir / "src")
        if src_path not in sys.path:
            sys.path.insert(0, src_path)
        if str(repo_dir) not in sys.path:
            sys.path.insert(0, str(repo_dir))

    try:
        import depth_pro
    except ImportError as e:
        raise RuntimeError(
            f"Failed to import depth_pro module. "
            f"Please ensure ml-depth-pro is installed at {repo_dir}\n"
            f"Error: {e}"
        ) from e

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Depth Pro checkpoint not found: {checkpoint_path}\n"
            f"Please place depth_pro.pt in {checkpoint_path.parent}"
        )

    print(f"[DepthPro] Loading model from {checkpoint_path}")
    print(f"[DepthPro] Device: {device}, Precision: {precision}")

    # 保存当前目录并切换到 repo_dir（Depth Pro 默认从 ./checkpoints 加载）
    original_cwd = os.getcwd()
    try:
        os.chdir(repo_dir)

        # 使用官方 API 加载模型
        model, transform = depth_pro.create_model_and_transforms(
            device=device,
            precision=precision,
        )

        model.eval()
        print(f"[DepthPro] Model loaded successfully")

    finally:
        os.chdir(original_cwd)

    _MODEL_CACHE[cache_key] = (model, transform)
    return model, transform


class DepthEstimator:
    """
    Depth Pro 深度估计器封装类

    提供统一的接口用于批量深度估计。
    """

    def __init__(self, cfg: Dict[str, Any], device: torch.device):
        self.cfg = cfg
        self.device = device
        self.model, self.transform = _load_depthpro_model(cfg, device)
        self.pass_f_px = bool(cfg.get("depth_pro", {}).get("pass_f_px", True))

    @torch.no_grad()
    def predict_single(self, rgb: np.ndarray, f_px: Optional[float] = None) -> np.ndarray:
        """
        单张图像深度预测

        Args:
            rgb: (H, W, 3) uint8 numpy array
            f_px: 可选的 focal length (像素)

        Returns:
            (H, W) float32 numpy array, 单位米
        """
        pil_img = Image.fromarray(rgb.astype(np.uint8))
        img_tensor = self.transform(pil_img)

        f_px_tensor = None
        if f_px is not None and self.pass_f_px:
            f_px_tensor = torch.tensor([f_px], device=self.device)

        prediction = self.model.infer(img_tensor, f_px=f_px_tensor)
        return prediction["depth"].detach().cpu().float().numpy().astype(np.float32)


def estimate_all_tangent_depths(
    tangent_rgbs: Dict[str, np.ndarray],
    slices: List[TangentSlice],
    cfg: Dict[str, Any],
    device: torch.device,
) -> Dict[str, np.ndarray]:
    """
    对所有切片估计深度

    Args:
        tangent_rgbs: {slice_id: rgb_array} 字典
        slices: 切片规格列表
        cfg: 配置字典
        device: 计算设备

    Returns:
        tangent_depths: {slice_id: depth_array} 字典
    """
    estimator = DepthEstimator(cfg, device)

    # 建立 slice_id -> f_px 映射
    f_px_map = {s.slice_id: s.f_px for s in slices}

    results = {}
    total = len(tangent_rgbs)

    for i, (slice_id, rgb) in enumerate(tangent_rgbs.items()):
        f_px = f_px_map.get(slice_id)
        depth = estimator.predict_single(rgb, f_px=f_px)
        results[slice_id] = depth

        print(f"  [{i+1}/{total}] {slice_id}: "
              f"depth range [{depth.min():.2f}, {depth.max():.2f}] m")

    return results
