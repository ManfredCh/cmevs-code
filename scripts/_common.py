from __future__ import annotations

import binascii
import csv
import json
import math
import shutil
import struct
import zlib
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write("\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True))
            f.write("\n")


def write_csv(path: str | Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    if fieldnames is None:
        keys: list[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _table_fieldnames(rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> list[str]:
    if fieldnames is not None:
        return fieldnames
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    return keys


def write_markdown_table(path: str | Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    fieldnames = _table_fieldnames(rows, fieldnames)
    with path.open("w", encoding="utf-8") as f:
        f.write("| " + " | ".join(fieldnames) + " |\n")
        f.write("| " + " | ".join(["---"] * len(fieldnames)) + " |\n")
        for row in rows:
            f.write("| " + " | ".join(str(row.get(name, "")) for name in fieldnames) + " |\n")


def _latex_escape(value: Any) -> str:
    text = str(value)
    return (
        text.replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("$", "\\$")
        .replace("#", "\\#")
        .replace("_", "\\_")
        .replace("{", "\\{")
        .replace("}", "\\}")
    )


def write_latex_table(
    path: str | Path,
    rows: list[dict[str, Any]],
    fieldnames: list[str] | None = None,
    caption: str = "Table-ready experiment results.",
    label: str = "tab:cmevs_results",
) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    fieldnames = _table_fieldnames(rows, fieldnames)
    align = "l" * len(fieldnames)
    with path.open("w", encoding="utf-8") as f:
        f.write("\\begin{table}[t]\n")
        f.write("\\centering\n")
        f.write(f"\\caption{{{_latex_escape(caption)}}}\n")
        safe_label = str(label).replace("{", "").replace("}", "")
        f.write(f"\\label{{{safe_label}}}\n")
        f.write(f"\\begin{{tabular}}{{{align}}}\n")
        f.write("\\toprule\n")
        f.write(" & ".join(_latex_escape(name) for name in fieldnames) + " \\\\\n")
        f.write("\\midrule\n")
        for row in rows:
            f.write(" & ".join(_latex_escape(row.get(name, "")) for name in fieldnames) + " \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")


def copy_file(src: str | Path, dst: str | Path) -> None:
    dst = Path(dst)
    ensure_dir(dst.parent)
    shutil.copy2(src, dst)


def candidate_by_id(candidates: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row["candidate_id"]): row for row in candidates}


def valid_candidates(candidates: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in candidates if bool(row.get("valid", True))]


def cell_set(candidate: dict[str, Any]) -> set[str]:
    return {str(cell) for cell in candidate.get("covered_cells", [])}


def universe_cells(candidates: Iterable[dict[str, Any]]) -> set[str]:
    cells: set[str] = set()
    for candidate in candidates:
        if bool(candidate.get("valid", True)):
            cells.update(cell_set(candidate))
    return cells


def selected_ids(selected_doc: dict[str, Any]) -> list[str]:
    return [str(row["candidate_id"]) for row in selected_doc.get("selected_viewpoints", [])]


def safe_div(num: float, den: float) -> float:
    return 0.0 if den == 0 else num / den


def pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return float("nan")
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0.0 or vy <= 0.0:
        return float("nan")
    return num / math.sqrt(vx * vy)


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    return (
        struct.pack(">I", len(payload))
        + kind
        + payload
        + struct.pack(">I", binascii.crc32(kind + payload) & 0xFFFFFFFF)
    )


def write_solid_png(path: str | Path, width: int, height: int, rgb: tuple[int, int, int]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    raw = bytearray()
    row = bytes(rgb) * width
    for _ in range(height):
        raw.append(0)
        raw.extend(row)
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    data = zlib.compress(bytes(raw), level=9)
    with path.open("wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n")
        f.write(_png_chunk(b"IHDR", ihdr))
        f.write(_png_chunk(b"IDAT", data))
        f.write(_png_chunk(b"IEND", b""))


def write_npy_f4(path: str | Path, height: int, width: int, value: float) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    header = "{'descr': '<f4', 'fortran_order': False, 'shape': (%d, %d), }" % (height, width)
    header_bytes = header.encode("latin1")
    prefix_len = 6 + 2 + 2
    padding = 16 - ((prefix_len + len(header_bytes) + 1) % 16)
    header_bytes += b" " * padding + b"\n"
    row = struct.pack("<" + "f" * width, *([float(value)] * width))
    with path.open("wb") as f:
        f.write(b"\x93NUMPY")
        f.write(b"\x01\x00")
        f.write(struct.pack("<H", len(header_bytes)))
        f.write(header_bytes)
        for _ in range(height):
            f.write(row)
