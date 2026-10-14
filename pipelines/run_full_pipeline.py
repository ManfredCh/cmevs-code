#!/usr/bin/env python3
"""
全流程 Pipeline: .blend/.glb/.gltf/.ply → 边渲边选 → ERPT Warp

支持六种模式:

1. 单 Blend 场景:
    python run_full_pipeline.py \
        --blender /path/to/blender \
        --blend /path/to/scene.blend \
        --scene-name my_scene \
        --output-root ./dataset

2. 批量 Blend（扫描 input-dir 下所有 .blend）:
    python run_full_pipeline.py \
        --blender /path/to/blender \
        --input-dir /path/to/blend_files/ \
        --output-root ./dataset

3. 单 GLB/GLTF 场景:
    python run_full_pipeline.py \
        --blender /path/to/blender \
        --glb /path/to/scene.glb \
        --scene-name my_scene \
        --output-root ./dataset

4. 批量 GLB（扫描 input-dir 下所有 .glb/.gltf）:
    python run_full_pipeline.py \
        --blender /path/to/blender \
        --input-dir /path/to/glb_files/ \
        --output-root ./dataset

5. 单 PLY 场景（无需 Blender）:
    python run_full_pipeline.py \
        --ply /path/to/scene.ply \
        --scene-name my_scene \
        --output-root ./dataset

6. 批量 PLY（扫描 input-dir 下所有 .ply）:
    python run_full_pipeline.py \
        --input-dir /path/to/ply_files/ \
        --output-root ./dataset

    加 --dry-run 预览要跑哪些场景
    已跑完的场景自动跳过（--no-skip-done 强制重跑）
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


def run_step1_blend_pipeline(
    blender_exe: str,
    scene_path: str,
    temp_dir: str,
    num_frames: int,
    resolution: str,
    samples: int,
    engine: str,
    exposure: float,
    grid_spacing: float,
    camera_height,
    stop_gain: float,
    stop_score: float,
    stop_delta: float,
    min_frames: int,
    rotation_type: str = "random_yaw",
    gain_curve: bool = True,
    scene_flag: str = "--blend",
) -> int:
    """步骤 1 (Blend/GLB): 调 run_blend_pipeline.py 边渲边选。
    scene_flag: "--blend" 或 "--glb"
    """
    script = Path(__file__).parent / "run_blend_pipeline.py"
    if not script.exists():
        raise FileNotFoundError(f"找不到 run_blend_pipeline.py: {script}")

    cmd = [
        sys.executable, str(script),
        "--blender", blender_exe,
        scene_flag, scene_path,
        "--output-dir", temp_dir,
        "--num-frames", str(num_frames),
        "--render-depth",
        "--resolution", resolution,
        "--samples", str(samples),
        "--engine", engine,
        "--exposure", str(exposure),
        "--grid-spacing", str(grid_spacing),
        "--stop-gain", str(stop_gain),
        "--stop-score", str(stop_score),
        "--stop-delta", str(stop_delta),
        "--min-frames", str(min_frames),
        "--rotation-type", rotation_type,
    ]
    if camera_height is not None:
        cmd += ["--camera-height", str(camera_height)]
    if not gain_curve:
        cmd += ["--no-gain-curve"]

    print(f"\n{'='*60}")
    print("[Step 1] 边渲边选 (Blender Cycles)")
    print(f"{'='*60}")

    proc = subprocess.run(cmd, text=True)
    if proc.returncode != 0:
        print(f"  [Error] run_blend_pipeline 退出码: {proc.returncode}")
        return proc.returncode

    n = sum(1 for f in Path(temp_dir).glob("panorama_*.png"))
    print(f"  渲染完成: {n} 帧")
    return 0


def run_step1_ply_pipeline(
    ply_path: str,
    temp_dir: str,
    num_frames: int,
    resolution: str,
    grid_spacing: float,
    camera_height,
    stop_gain: float,
    stop_score: float,
    stop_delta: float,
    min_frames: int,
    rotation_type: str = "random_yaw",
    point_size: float = 2.0,
    z_up: bool = True,
) -> int:
    """步骤 1 (PLY): 调 run_ply_pipeline.py 边渲边选（无需 Blender）"""
    script = Path(__file__).parent / "run_ply_pipeline.py"
    if not script.exists():
        raise FileNotFoundError(f"找不到 run_ply_pipeline.py: {script}")

    cmd = [
        sys.executable, str(script),
        "--ply", ply_path,
        "--output-dir", temp_dir,
        "--num-frames", str(num_frames),
        "--resolution", resolution,
        "--grid-spacing", str(grid_spacing),
        "--stop-gain", str(stop_gain),
        "--stop-score", str(stop_score),
        "--stop-delta", str(stop_delta),
        "--min-frames", str(min_frames),
        "--rotation-type", rotation_type,
        "--point-size", str(point_size),
    ]
    if camera_height is not None:
        cmd += ["--camera-height", str(camera_height)]
    if not z_up:
        cmd += ["--no-z-up"]

    print(f"\n{'='*60}")
    print("[Step 1] 边渲边选 (PLY 点云)")
    print(f"{'='*60}")

    proc = subprocess.run(cmd, text=True)
    if proc.returncode != 0:
        print(f"  [Error] run_ply_pipeline 退出码: {proc.returncode}")
        return proc.returncode

    n = sum(1 for f in Path(temp_dir).glob("panorama_*.png"))
    print(f"  渲染完成: {n} 帧")
    return 0


def run_step1_hm3d_pipeline(
    blender_exe: str,
    scene_path: str,
    temp_dir: str,
    num_frames: int,
    resolution: str,
    samples: int,
    engine: str,
    exposure: float,
    grid_spacing: float,
    camera_height,
    stop_gain: float,
    stop_score: float,
    stop_delta: float,
    min_frames: int,
    rotation_type: str = "random_yaw",
    gain_curve: bool = True,
) -> int:
    """步骤 1 (HM3D GLB): 调 run_hm3d_pipeline.py 边渲边选。"""
    script = Path(__file__).parent / "run_hm3d_pipeline.py"
    if not script.exists():
        raise FileNotFoundError(f"找不到 run_hm3d_pipeline.py: {script}")

    cmd = [
        sys.executable, str(script),
        "--blender", blender_exe,
        "--glb", scene_path,
        "--output-dir", temp_dir,
        "--num-frames", str(num_frames),
        "--render-depth",
        "--resolution", resolution,
        "--samples", str(samples),
        "--engine", engine,
        "--exposure", str(exposure),
        "--grid-spacing", str(grid_spacing),
        "--stop-gain", str(stop_gain),
        "--stop-score", str(stop_score),
        "--stop-delta", str(stop_delta),
        "--min-frames", str(min_frames),
        "--rotation-type", rotation_type,
        "--hm3d", "True",
    ]
    if camera_height is not None:
        cmd += ["--camera-height", str(camera_height)]
    if not gain_curve:
        cmd += ["--no-gain-curve"]

    print(f"\n{'='*60}")
    print("[Step 1] 边渲边选 (HM3D GLB)")
    print(f"{'='*60}")

    proc = subprocess.run(cmd, text=True)
    if proc.returncode != 0:
        print(f"  [Error] run_hm3d_pipeline 退出码: {proc.returncode}")
        return proc.returncode

    n = sum(1 for f in Path(temp_dir).rglob("panorama_*.png"))
    print(f"  渲染完成: {n} 帧")
    return 0


def run_step2_organize_hm3d(temp_dir: str, scene_dir: str) -> int:
    """步骤 2 (HM3D): 整理多空间目录结构

    temp_dir 里有:
        frame_selection/
        space_00/ (panorama_*.png, *_depth.npy, pose_*.json)
        space_01/
        ...

    整理成（每个 space 一个独立目录）:
        scene_dir/space_00/input/   → 中心帧 RGB + depth + 所有 pose
        scene_dir/space_00/output/  → 所有帧 RGB + depth（GT 真值）
        scene_dir/space_01/input/
        scene_dir/space_01/output/
        ...
        scene_dir/frame_selection/  → 选帧信息
    """
    temp = Path(temp_dir)

    print(f"\n{'='*60}")
    print("[Step 2] 整理目录结构 (HM3D 多空间)")
    print(f"{'='*60}")

    space_dirs = sorted(
        [d for d in temp.iterdir() if d.is_dir() and d.name.startswith("space_")]
    )
    if not space_dirs:
        print("  [Error] 没有找到 space_XX 目录")
        return 1

    print(f"  共 {len(space_dirs)} 个空间")

    for space_d in space_dirs:
        space_name = space_d.name

        rgb_files = sorted(space_d.glob("panorama_*.png"))
        if not rgb_files:
            print(f"  {space_name}: 无渲染结果，跳过")
            continue

        n_frames = len(rgb_files)
        out_space_dir = Path(scene_dir) / space_name
        inp_dir = out_space_dir / "input"
        out_dir = out_space_dir / "output"
        inp_dir.mkdir(parents=True, exist_ok=True)
        out_dir.mkdir(parents=True, exist_ok=True)

        for rgb_path in rgb_files:
            shutil.copy2(str(rgb_path), str(out_dir / rgb_path.name))
            depth_path = space_d / rgb_path.name.replace(".png", "_depth.npy")
            if depth_path.exists():
                shutil.copy2(str(depth_path), str(out_dir / depth_path.name))

        center_rgb = space_d / "panorama_0000.png"
        center_depth = space_d / "panorama_0000_depth.npy"
        if center_rgb.exists():
            shutil.copy2(str(center_rgb), str(inp_dir / center_rgb.name))
        if center_depth.exists():
            shutil.copy2(str(center_depth), str(inp_dir / center_depth.name))

        n_pose = 0
        for pose_path in sorted(space_d.glob("pose_*.json")):
            shutil.copy2(str(pose_path), str(inp_dir / pose_path.name))
            n_pose += 1

        print(f"  {space_name}: {n_frames} 帧 → output/, 中心帧 + {n_pose} pose → input/")

    sel_dir = Path(scene_dir) / "frame_selection"
    sel_dir.mkdir(parents=True, exist_ok=True)

    sel_json = temp / "frame_selection" / "selected_frames.json"
    if sel_json.exists():
        shutil.copy2(str(sel_json), str(sel_dir / "selected_frames.json"))

    cand_npy = temp / "frame_selection" / "candidates_filtered.npy"
    if cand_npy.exists():
        shutil.copy2(str(cand_npy), str(sel_dir / "candidates_filtered.npy"))

    return 0


def run_step2_organize(temp_dir: str, scene_dir: str) -> int:
    """步骤 2: 整理目录结构

    temp_dir 里有:
        panorama_0000.png, panorama_0000_depth.npy, pose_0000.json, ...

    整理成:
        scene_dir/input/   → 中心帧 RGB + depth + 所有 pose（供 ERPT warp 使用）
        scene_dir/output/  → 所有帧 RGB + depth（GT 真值）
    """
    temp = Path(temp_dir)
    inp_dir = Path(scene_dir) / "input"
    out_dir = Path(scene_dir) / "output"
    inp_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("[Step 2] 整理目录结构")
    print(f"{'='*60}")

    # 找所有帧
    rgb_files = sorted(temp.glob("panorama_*.png"))
    if not rgb_files:
        print("  [Error] 没有找到渲染的全景图")
        return 1

    n_frames = len(rgb_files)
    print(f"  共 {n_frames} 帧")

    # output/: 复制所有帧的 RGB + depth（GT 真值）
    for rgb_path in rgb_files:
        shutil.copy2(str(rgb_path), str(out_dir / rgb_path.name))
        # depth
        depth_path = temp / rgb_path.name.replace(".png", "_depth.npy")
        if depth_path.exists():
            shutil.copy2(str(depth_path), str(out_dir / depth_path.name))

    print(f"  output/: {n_frames} 帧 RGB + depth")

    # input/: 中心帧 RGB + depth + 所有 pose
    center_rgb = temp / "panorama_0000.png"
    center_depth = temp / "panorama_0000_depth.npy"

    if center_rgb.exists():
        shutil.copy2(str(center_rgb), str(inp_dir / center_rgb.name))
    if center_depth.exists():
        shutil.copy2(str(center_depth), str(inp_dir / center_depth.name))

    # 所有 pose
    n_pose = 0
    for pose_path in sorted(temp.glob("pose_*.json")):
        shutil.copy2(str(pose_path), str(inp_dir / pose_path.name))
        n_pose += 1

    print(f"  input/: 中心帧 + {n_pose} 个 pose")

    # 复制选帧信息（供参考）
    sel_dir = inp_dir / "frame_selection"
    sel_dir.mkdir(parents=True, exist_ok=True)

    sel_json = temp / "frame_selection" / "selected_frames.json"
    if sel_json.exists():
        shutil.copy2(str(sel_json), str(sel_dir / "selected_frames.json"))

    cand_npy = temp / "frame_selection" / "candidates_filtered.npy"
    if cand_npy.exists():
        shutil.copy2(str(cand_npy), str(sel_dir / "candidates_filtered.npy"))

    # 增益曲线（从 selected_frames.json 读数据，用 PIL 画）
    if sel_json.exists():
        try:
            draw_gain_curve(str(sel_dir / "selected_frames.json"),
                            str(sel_dir / "gain_curve.jpg"))
            print(f"  增益曲线: {sel_dir}/gain_curve.jpg")
        except Exception as e:
            print(f"  [跳过] 画增益曲线失败: {e}")

    return 0


def draw_gain_curve(json_path, output_path):
    """画增益曲线（优先 matplotlib，fallback PIL）"""
    with open(json_path) as f:
        data = json.load(f)

    frames = [fr for fr in data["frames"] if not fr.get("skipped")]
    if len(frames) < 2:
        return

    fids = [fr["frame_id"] for fr in frames]
    pred_gains = [fr["gain"] for fr in frames]
    actual_gains = [fr["actual_gain"] for fr in frames]
    scores = [fr["score"] for fr in frames]
    deltas = [fr["delta_ratio"] for fr in frames]

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

        ax1.plot(fids, pred_gains, 'o-', color='#2196F3', label='predicted', markersize=3, linewidth=1.5)
        ax1.plot(fids, actual_gains, 'o-', color='#FF9800', label='actual', markersize=3, linewidth=1.5)
        ax1.axhline(y=0.05, color='red', linestyle='--', alpha=0.5, label='stop_gain=5%')
        ax1.set_ylabel('gain')
        ax1.set_ylim(-0.05, 1.05)
        ax1.legend(loc='upper right', fontsize=9)
        ax1.set_title(f'Gain Curve ({len(frames)} frames)', fontsize=11)
        ax1.grid(True, alpha=0.3)
        # 标注首末值
        ax1.annotate(f'{actual_gains[0]:.0%}', (fids[0], actual_gains[0]),
                     textcoords="offset points", xytext=(5, 5), fontsize=7, color='#FF9800')
        ax1.annotate(f'{actual_gains[-1]:.0%}', (fids[-1], actual_gains[-1]),
                     textcoords="offset points", xytext=(-25, 5), fontsize=7, color='#FF9800')

        ax2.plot(fids, scores, 'D-', color='#4CAF50', label='score', markersize=3, linewidth=1.5)
        ax2.plot(fids, deltas, 's-', color='#9C27B0', label='delta', markersize=2, linewidth=1.2)
        ax2.axhline(y=-0.33, color='red', linestyle='--', alpha=0.5, label='stop_score=-0.33')
        ax2.axhline(y=0.01, color='#9C27B0', linestyle=':', alpha=0.4, label='stop_delta=1%')
        ax2.set_ylabel('value')
        ax2.set_xlabel('frame')
        ax2.legend(loc='upper right', fontsize=9)
        ax2.grid(True, alpha=0.3)
        # 标注首末值
        ax2.annotate(f'{deltas[0]:.1%}', (fids[0], deltas[0]),
                     textcoords="offset points", xytext=(5, 5), fontsize=7, color='#9C27B0')
        ax2.annotate(f'{deltas[-1]:.1%}', (fids[-1], deltas[-1]),
                     textcoords="offset points", xytext=(-25, -10), fontsize=7, color='#9C27B0')

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        return

    except ImportError:
        pass

    # ---- fallback: PIL ----
    try:
        from PIL import Image, ImageDraw, ImageFont

        W, H = 800, 500
        ML, MR, MT, MB = 50, 20, 30, 25
        MID = H // 2
        pw = W - ML - MR

        img = Image.new("RGB", (W, H), "white")
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 9)
        except Exception:
            font = ImageFont.load_default()

        n = len(frames)

        def px(i, v, y0, y1, vmin, vmax):
            x = ML + int(i / max(n-1,1) * pw)
            y = y0 + int((1 - (v-vmin)/(vmax-vmin)) * (y1-y0))
            return x, max(y0, min(y1, y))

        def line(pts, color, y0, y1, vmin, vmax):
            for j in range(len(pts)-1):
                draw.line([px(j,pts[j],y0,y1,vmin,vmax), px(j+1,pts[j+1],y0,y1,vmin,vmax)], fill=color, width=2)

        line(pred_gains, "#2196F3", MT, MID-5, 0, 1.05)
        line(actual_gains, "#FF9800", MT, MID-5, 0, 1.05)
        line(scores, "#4CAF50", MID+10, H-MB, -0.6, 1.05)
        line(deltas, "#9C27B0", MID+10, H-MB, -0.6, 1.05)

        draw.text((ML, MT-12), f"Gain ({n} frames)", fill="black", font=font)
        draw.text((ML, MID+2), "Score / Delta", fill="black", font=font)
        img.save(output_path, quality=90)

    except ImportError:
        pass


def run_step3_erpt_warp_hm3d(scene_dir: str, device: str = "cuda") -> int:
    """步骤 3 (HM3D): 对每个空间调 run_pipeline.py 执行 ERPT warp

    遍历 scene_dir/space_XX/input/，对帧数 >= 2 的空间生成 warp 文件
    """
    script = Path(__file__).parent / "run_pipeline.py"
    if not script.exists():
        raise FileNotFoundError(f"找不到 run_pipeline.py: {script}")

    print(f"\n{'='*60}")
    print("[Step 3] ERPT Warp (HM3D 多空间)")
    print(f"{'='*60}")

    scene_path = Path(scene_dir)
    space_dirs = sorted(
        [d for d in scene_path.iterdir()
         if d.is_dir() and d.name.startswith("space_")]
    )

    if not space_dirs:
        print("  [Error] 没有找到 space_XX 目录")
        return 1

    total_ret = 0
    n_warped = 0
    n_skipped = 0

    for space_d in space_dirs:
        inp_dir = space_d / "input"
        if not inp_dir.exists():
            continue

        n_poses = len(list(inp_dir.glob("pose_*.json")))
        if n_poses < 2:
            print(f"  {space_d.name}: {n_poses} pose，跳过 warp")
            n_skipped += 1
            continue

        print(f"\n  [{space_d.name}] ERPT Warp ({n_poses} poses)...")

        cmd = [
            sys.executable, str(script),
            "--stage", "warp_only",
            "--data_dir", str(inp_dir),
            "--output_dir", str(inp_dir),
            "--device", device,
            "--center_frame", "0",
        ]

        proc = subprocess.run(cmd, text=True)
        if proc.returncode != 0:
            print(f"    [Error] {space_d.name} warp 失败 (退出码: {proc.returncode})")
            total_ret = proc.returncode
            continue

        warp_rgb_dir = inp_dir / "warp_rgb"
        warp_depth_dir = inp_dir / "warp_depth"
        keep_suffixes = ("_rgb.png", "_mask.png", "_depth_range.npy")

        n_moved = 0
        for subdir in [warp_rgb_dir, warp_depth_dir]:
            if subdir.exists():
                for f in subdir.iterdir():
                    if f.is_file() and any(f.name.endswith(s) for s in keep_suffixes):
                        shutil.move(str(f), str(inp_dir / f.name))
                        n_moved += 1
                shutil.rmtree(str(subdir), ignore_errors=True)

        n_warped += 1
        print(f"    warp 文件: {n_moved} 个")

    print(f"\n  Warp 完成: {n_warped} 个空间, 跳过 {n_skipped} 个")
    return total_ret


def run_step3_erpt_warp(scene_dir: str, device: str = "cuda") -> int:
    """步骤 3: 调 run_pipeline.py 执行 ERPT warp

    读取 scene_dir/input/ 里的中心帧 + pose → 生成 warp 文件
    warp 文件直接写到 input/ 目录
    """
    script = Path(__file__).parent / "run_pipeline.py"
    if not script.exists():
        raise FileNotFoundError(f"找不到 run_pipeline.py: {script}")

    inp_dir = Path(scene_dir) / "input"

    print(f"\n{'='*60}")
    print("[Step 3] ERPT Warp")
    print(f"{'='*60}")

    cmd = [
        sys.executable, str(script),
        "--stage", "warp_only",
        "--data_dir", str(inp_dir),
        "--output_dir", str(inp_dir),
        "--device", device,
        "--center_frame", "0",
    ]

    proc = subprocess.run(cmd, text=True)
    if proc.returncode != 0:
        print(f"  [Error] run_pipeline 退出码: {proc.returncode}")
        return proc.returncode

    # 把 warp 子目录里需要的文件提到 input/ 根目录
    # 只保留: _rgb.png, _mask.png, _depth_range.npy
    warp_rgb_dir = inp_dir / "warp_rgb"
    warp_depth_dir = inp_dir / "warp_depth"
    keep_suffixes = ("_rgb.png", "_mask.png", "_depth_range.npy")

    n_moved = 0
    for subdir in [warp_rgb_dir, warp_depth_dir]:
        if subdir.exists():
            for f in subdir.iterdir():
                if f.is_file() and any(f.name.endswith(s) for s in keep_suffixes):
                    shutil.move(str(f), str(inp_dir / f.name))
                    n_moved += 1
            # 删除整个子目录（包含不需要的 flow/weight_sum/comparison 等）
            shutil.rmtree(str(subdir), ignore_errors=True)

    print(f"  warp 文件已移到 input/: {n_moved} 个")
    return 0


def is_already_done(output_root, scene_name):
    """检查是否已经跑完"""
    sel_path = os.path.join(
        output_root, scene_name, "input", "frame_selection",
        "selected_frames.json")
    if not os.path.exists(sel_path):
        # 也检查 input/ 下直接放的
        sel_path = os.path.join(output_root, scene_name, "input",
                                "selected_frames.json")
        if not os.path.exists(sel_path):
            return False
    try:
        with open(sel_path) as f:
            data = json.load(f)
        return data.get("total_frames", 0) > 0
    except Exception:
        return False


def find_glb_files(input_dir):
    """递归查找所有 .glb / .gltf 文件，返回 [(glb_path, scene_name), ...]

    scene_name 取文件名（不含扩展名），或第一级子目录名（如有子目录）。
    """
    input_dir = os.path.abspath(input_dir)
    glb_files = []
    for root, dirs, files in os.walk(input_dir):
        for f in files:
            if f.lower().endswith(".glb") or f.lower().endswith(".gltf"):
                glb_path = os.path.join(root, f)
                rel = os.path.relpath(root, input_dir)
                if rel == ".":
                    scene_name = os.path.splitext(f)[0]
                else:
                    scene_name = rel.split(os.sep)[0]
                glb_files.append((glb_path, scene_name))
    glb_files.sort(key=lambda x: x[1])
    return glb_files


def find_blend_files(input_dir):
    """递归查找所有 .blend 文件，返回 [(blend_path, scene_name), ...]
    
    scene_name 取 input_dir 下的第一级子目录名（scene_indoor_XXXX），
    不管 .blend 文件嵌套了几层。
    
    例如:
      input_dir = /path/to/dataset/indoor
      .blend 在 /path/to/dataset/indoor/scene_indoor_0001/1407m1/xxx.blend
      → scene_name = scene_indoor_0001
    """
    input_dir = os.path.abspath(input_dir)
    blend_files = []
    for root, dirs, files in os.walk(input_dir):
        for f in files:
            if f.endswith(".blend"):
                blend_path = os.path.join(root, f)
                rel = os.path.relpath(root, input_dir)
                scene_name = rel.split(os.sep)[0]
                blend_files.append((blend_path, scene_name))
    blend_files.sort(key=lambda x: x[1])
    return blend_files


def find_ply_files(input_dir):
    """递归查找所有 .ply 文件，返回 [(ply_path, scene_name), ...]

    scene_name 取文件名（不含扩展名），或第一级子目录名（如有子目录）。
    """
    input_dir = os.path.abspath(input_dir)
    ply_files = []
    for root, dirs, files in os.walk(input_dir):
        for f in files:
            if f.lower().endswith(".ply"):
                ply_path = os.path.join(root, f)
                rel = os.path.relpath(root, input_dir)
                if rel == ".":
                    scene_name = os.path.splitext(f)[0]
                else:
                    scene_name = rel.split(os.sep)[0]
                ply_files.append((ply_path, scene_name))
    ply_files.sort(key=lambda x: x[1])
    return ply_files


def run_single_scene(args, scene_path, scene_name, scene_type="blend"):
    """跑单个场景，返回 0=成功 / 非0=失败

    scene_type: "blend" | "glb" | "hm3d" | "ply"
    """
    output_root = str(Path(args.output_root).resolve())
    scene_dir = os.path.join(output_root, scene_name)
    temp_dir = os.path.join(scene_dir, "_render_temp")
    os.makedirs(scene_dir, exist_ok=True)

    type_labels = {
        "blend": ".blend (Blender Cycles)",
        "glb":   ".glb/.gltf (Blender Cycles)",
        "hm3d":  ".glb/.gltf (HM3D 渲染)",
        "ply":   ".ply (点云渲染)",
    }
    type_label = type_labels.get(scene_type, scene_type)
    print("=" * 60)
    print(f"全流程 Pipeline: {type_label} → 边渲边选 → ERPT Warp")
    print("=" * 60)
    print(f"  Scene:  {scene_name}")
    print(f"  Input:  {scene_path}")
    print(f"  Output: {scene_dir}/")
    t_start = time.time()

    # Step 1
    if scene_type == "hm3d":
        ret = run_step1_hm3d_pipeline(
            blender_exe=args.blender,
            scene_path=scene_path,
            temp_dir=temp_dir,
            num_frames=args.num_frames,
            resolution=args.resolution,
            samples=args.samples,
            engine=args.engine,
            exposure=args.exposure,
            grid_spacing=args.grid_spacing,
            camera_height=args.camera_height,
            stop_gain=args.stop_gain,
            stop_score=args.stop_score,
            stop_delta=args.stop_delta,
            min_frames=args.min_frames,
            rotation_type=args.rotation_type,
            gain_curve=getattr(args, "gain_curve", True),
        )
    elif scene_type in ("blend", "glb"):
        scene_flag = "--blend" if scene_type == "blend" else "--glb"
        ret = run_step1_blend_pipeline(
            blender_exe=args.blender,
            scene_path=scene_path,
            temp_dir=temp_dir,
            num_frames=args.num_frames,
            resolution=args.resolution,
            samples=args.samples,
            engine=args.engine,
            exposure=args.exposure,
            grid_spacing=args.grid_spacing,
            camera_height=args.camera_height,
            stop_gain=args.stop_gain,
            stop_score=args.stop_score,
            stop_delta=args.stop_delta,
            min_frames=args.min_frames,
            rotation_type=args.rotation_type,
            gain_curve=getattr(args, "gain_curve", True),
            scene_flag=scene_flag,
        )
    else:
        ret = run_step1_ply_pipeline(
            ply_path=scene_path,
            temp_dir=temp_dir,
            num_frames=args.num_frames,
            resolution=args.resolution,
            grid_spacing=args.grid_spacing,
            camera_height=args.camera_height,
            stop_gain=args.stop_gain,
            stop_score=args.stop_score,
            stop_delta=args.stop_delta,
            min_frames=args.min_frames,
            rotation_type=args.rotation_type,
            point_size=getattr(args, "point_size", 2.0),
            z_up=getattr(args, "z_up", True),
        )

    if ret != 0:
        print(f"[Error] Step 1 失败")
        return ret

    # Step 2
    if scene_type == "hm3d":
        ret = run_step2_organize_hm3d(temp_dir, scene_dir)
    else:
        ret = run_step2_organize(temp_dir, scene_dir)
    if ret != 0:
        print(f"[Error] Step 2 失败")
        return ret

    # Step 3
    if not args.skip_warp:
        if scene_type == "hm3d":
            ret = run_step3_erpt_warp_hm3d(scene_dir, device=args.device)
        else:
            ret = run_step3_erpt_warp(scene_dir, device=args.device)
        if ret != 0:
            print(f"[Error] Step 3 失败")

    # 清理
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir, ignore_errors=True)

    dt = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"完成! {scene_name}, {dt:.1f}s ({dt/60:.1f}min)")
    print(f"{'='*60}")
    return 0


def run_single(args):
    """单场景模式（blend、glb、hm3d 或 ply）"""
    if args.ply:
        ply_path = str(Path(args.ply).resolve())
        scene_name = args.scene_name or Path(args.ply).stem
        ret = run_single_scene(args, ply_path, scene_name, scene_type="ply")
    elif args.hm3d:
        glb_path = str(Path(args.hm3d).resolve())
        scene_name = args.scene_name or Path(args.hm3d).stem
        ret = run_single_scene(args, glb_path, scene_name, scene_type="hm3d")
    elif args.glb:
        glb_path = str(Path(args.glb).resolve())
        scene_name = args.scene_name or Path(args.glb).stem
        ret = run_single_scene(args, glb_path, scene_name, scene_type="glb")
    else:
        blend_path = str(Path(args.blend).resolve())
        scene_name = args.scene_name or Path(args.blend).stem
        ret = run_single_scene(args, blend_path, scene_name, scene_type="blend")
    if ret != 0:
        sys.exit(1)


def run_batch(args):
    """批量模式（自动检测 .blend / .glb / .gltf / .ply）"""
    input_dir_abs = os.path.abspath(args.input_dir)
    if not os.path.isdir(input_dir_abs):
        print(f"[Error] --input-dir 目录不存在: {input_dir_abs}")
        print(f"        (原始参数: {args.input_dir})")
        sys.exit(1)

    # 没有 --blender → 只能跑 PLY
    if not getattr(args, "blender", None):
        scene_files = find_ply_files(args.input_dir)
        scene_type = "ply"
        ext_label = ".ply"
    else:
        # 有 blender：优先 .blend，其次 .glb/.gltf (HM3D)，最后 .ply
        scene_files = find_blend_files(args.input_dir)
        scene_type = "blend"
        ext_label = ".blend"
        if not scene_files:
            scene_files = find_glb_files(args.input_dir)
            scene_type = "hm3d"  # 默认使用 HM3D 渲染管线
            ext_label = ".glb/.gltf (HM3D)"
        if not scene_files:
            scene_files = find_ply_files(args.input_dir)
            scene_type = "ply"
            ext_label = ".ply"

    if not scene_files:
        print(f"[Error] 在 {args.input_dir} 下没找到 {ext_label} 文件")
        sys.exit(1)

    output_root = str(Path(args.output_root).resolve())

    print(f"{'='*60}")
    print(f"批量处理模式 ({ext_label})")
    print(f"{'='*60}")
    print(f"  输入目录: {args.input_dir}")
    print(f"  输出目录: {output_root}")
    print(f"  找到 {len(scene_files)} 个 {ext_label} 文件")
    # input(f"  按 Enter 键继续，或 Ctrl+C 取消...")
    
    to_run = []
    skipped = []
    for scene_path, scene_name in scene_files:
        if args.skip_done and is_already_done(output_root, scene_name):
            skipped.append((scene_path, scene_name))
        else:
            to_run.append((scene_path, scene_name))

    if skipped:
        print(f"  跳过 {len(skipped)} 个已完成:")
        for _, sn in skipped:
            print(f"    ✓ {sn}")

    print(f"  待处理 {len(to_run)} 个:")
    for bp, sn in to_run:
        print(f"    → {sn}  ({os.path.basename(bp)})")

    if args.dry_run:
        print(f"\n[Dry run] 不实际运行")
        return

    if not to_run:
        print(f"\n全部已完成!")
        return

    t_all = time.time()
    success = []
    failed = []

    for idx, (scene_path, scene_name) in enumerate(to_run):
        print(f"\n{'='*60}")
        print(f"[{idx+1}/{len(to_run)}] {scene_name}")
        print(f"{'='*60}")

        t_scene = time.time()
        try:
            ret = run_single_scene(args, scene_path, scene_name, scene_type)
            dt = time.time() - t_scene
            if ret == 0:
                success.append((scene_name, dt))
                print(f"\n  ✓ {scene_name} ({dt:.0f}s)")
            else:
                failed.append((scene_name, f"exit code {ret}"))
                print(f"\n  ✗ {scene_name} 失败 ({dt:.0f}s)")
        except Exception as e:
            dt = time.time() - t_scene
            failed.append((scene_name, str(e)))
            print(f"\n  ✗ {scene_name} 异常: {e} ({dt:.0f}s)")

    dt_all = time.time() - t_all
    print(f"\n{'='*60}")
    print(f"批量处理完成")
    print(f"{'='*60}")
    print(f"  总耗时: {dt_all:.0f}s ({dt_all/60:.1f}min = {dt_all/3600:.1f}h)")
    print(f"  成功: {len(success)} 个")
    for sn, dt in success:
        print(f"    ✓ {sn} ({dt:.0f}s)")
    if failed:
        print(f"  失败: {len(failed)} 个")
        for sn, reason in failed:
            print(f"    ✗ {sn}: {reason}")
    if skipped:
        print(f"  跳过: {len(skipped)} 个 (已完成)")


def main():
    parser = argparse.ArgumentParser(
        description="全流程: .blend/.glb/.ply → 边渲边选 → ERPT Warp"
    )

    # 输入（四种模式互斥）
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument("--blend", type=str, default=None,
                             help=".blend 场景文件路径（单 Blend 场景模式）")
    input_group.add_argument("--glb", type=str, default=None,
                             help=".glb / .gltf 场景文件路径（单 GLB 场景模式）")
    input_group.add_argument("--hm3d", type=str, default=None,
                             help=".glb / .gltf 场景文件路径（单 HM3D 场景模式）")
    input_group.add_argument("--ply", type=str, default=None,
                             help=".ply 场景文件路径（单 PLY 场景模式）")

    parser.add_argument("--input-dir", type=str, default=None,
                        help="包含场景文件的根目录（批量模式，自动检测 .blend/.glb/.ply）")
    parser.add_argument("--scene-name", type=str, default=None,
                        help="场景名（默认从文件名提取）")
    parser.add_argument("--output-root", type=str, default="./dataset",
                        help="输出根目录（默认 ./dataset）")

    # Blender 参数（仅 Blend/GLB/HM3D 模式需要）
    parser.add_argument("--blender", type=str, default=None,
                        help="Blender 可执行文件路径（Blend/GLB/HM3D 模式必须）")
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--engine", type=str, default="CYCLES")
    parser.add_argument("--exposure", type=float, default=0.0)
    parser.add_argument("--gain-curve", action="store_true", default=True,
                        help="画增益曲线 (默认开启)")
    parser.add_argument("--no-gain-curve", dest="gain_curve", action="store_false")

    # PLY 参数（仅 PLY 模式）
    parser.add_argument("--point-size", type=float, default=2.0,
                        help="点云渲染点径（像素），PLY 模式有效（默认 2.0）")
    parser.add_argument("--z-up", action="store_true", default=True,
                        help="PLY 坐标系为 Z-up（默认 True）")
    parser.add_argument("--no-z-up", dest="z_up", action="store_false",
                        help="PLY 坐标系为 Y-up（已是 ERPT_native，不转换）")

    # 通用渲染参数
    parser.add_argument("--num-frames", type=int, default=30)
    parser.add_argument("--resolution", type=str, default="2048,1024")

    # 选帧参数
    parser.add_argument("--grid-spacing", type=float, default=0.5)
    parser.add_argument("--camera-height", type=float, default=None)
    parser.add_argument("--stop-gain", type=float, default=0.08)
    parser.add_argument("--stop-score", type=float, default=-0.3)
    parser.add_argument("--stop-delta", type=float, default=0.08)
    parser.add_argument("--min-frames", type=int, default=5)
    parser.add_argument("--rotation-type", type=str, default="random_yaw",
                        choices=["none", "rotate_x_90", "rotate_x_180",
                                 "rotate_z_90", "random_yaw"])

    # ERPT 参数
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--skip-warp", action="store_true",
                        help="只做步骤 1+2，跳过 ERPT warp")

    # 批量模式参数
    parser.add_argument("--skip-done", action="store_true", default=True,
                        help="跳过已跑完的场景（默认开启）")
    parser.add_argument("--no-skip-done", action="store_true",
                        help="强制重跑所有场景")
    parser.add_argument("--dry-run", action="store_true",
                        help="只列出要跑的场景，不实际运行")

    args = parser.parse_args()

    if args.no_skip_done:
        args.skip_done = False

    # 校验 Blend/GLB/HM3D 模式必须提供 --blender
    if (args.blend or args.glb or args.hm3d) and not args.blender:
        parser.error("--blend / --glb / --hm3d 模式必须同时提供 --blender 可执行文件路径")

    # 模式判定
    if args.input_dir:
        run_batch(args)
    elif args.blend or args.glb or args.hm3d or args.ply:
        run_single(args)
    else:
        parser.error("必须指定 --blend / --glb / --hm3d / --ply（单场景）或 --input-dir（批量）")


if __name__ == "__main__":
    main()
