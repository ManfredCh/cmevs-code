"""
IO 工具模块

图像和 JSON 文件的读写。
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, Union
import cv2


def load_image(path: Union[str, Path]) -> np.ndarray:
    """
    加载图像
    
    Args:
        path: 图像路径
        
    Returns:
        img: (H, W, 3) RGB uint8 图像
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    
    # OpenCV 读取 BGR
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to load image: {path}")
    
    # BGR -> RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    return img


def save_image(img: np.ndarray, path: Union[str, Path]) -> None:
    """
    保存图像
    
    Args:
        img: (H, W, 3) RGB uint8 图像
        path: 输出路径
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # RGB -> BGR
    if len(img.shape) == 3 and img.shape[2] == 3:
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    else:
        img_bgr = img
    
    cv2.imwrite(str(path), img_bgr)


def load_json(path: Union[str, Path]) -> Dict[str, Any]:
    """
    加载 JSON 文件
    
    Args:
        path: JSON 文件路径
        
    Returns:
        data: 解析后的字典
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSON not found: {path}")
    
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Dict[str, Any], path: Union[str, Path], indent: int = 2) -> None:
    """
    保存 JSON 文件
    
    Args:
        data: 要保存的数据
        path: 输出路径
        indent: 缩进
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def load_depth(path: Union[str, Path]) -> np.ndarray:
    """
    加载深度图（.npy 格式）
    
    Args:
        path: 深度图路径
        
    Returns:
        depth: (H, W) float32 深度图
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Depth not found: {path}")
    
    depth = np.load(str(path))
    return depth.astype(np.float32)


def save_depth(depth: np.ndarray, path: Union[str, Path]) -> None:
    """
    保存深度图（.npy 格式）
    
    Args:
        depth: (H, W) 深度图
        path: 输出路径
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    np.save(str(path), depth.astype(np.float32))


def list_files(
    directory: Union[str, Path],
    pattern: str = "*",
    sort: bool = True,
) -> list:
    """
    列出目录中的文件
    
    Args:
        directory: 目录路径
        pattern: glob 模式
        sort: 是否排序
        
    Returns:
        files: 文件路径列表
    """
    directory = Path(directory)
    if not directory.exists():
        return []
    
    files = list(directory.glob(pattern))
    if sort:
        files = sorted(files)
    
    return files
