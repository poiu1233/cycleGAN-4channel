"""
split_day_night_dataset.py

把“白天/夜晚混在同一目录”的图片，自动拆分成:
  out_root/{split}/day
  out_root/{split}/night

特点：
- 支持自动检测 split（train/val/test 等一级子目录）；如果没有，则当作一个整体 split=all
- 用每张图的平均亮度（luma）做特征，并用 Otsu 自动阈值做二分（可手动指定阈值）
- 支持 copy（默认）或 move
- 支持 dry-run 预览

注意：
- 这是一个启发式拆分工具。强烈建议你抽样检查结果，并根据需要调整阈值或改用更强的分类器。
"""

from __future__ import annotations

import argparse
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageStat


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def is_image_file(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMG_EXTS


def iter_images(root: Path, recursive: bool = True) -> Iterable[Path]:
    if recursive:
        for p in root.rglob("*"):
            if is_image_file(p):
                yield p
    else:
        for p in root.iterdir():
            if is_image_file(p):
                yield p


def compute_mean_luma(img: Image.Image, max_side: int = 256) -> float:
    """
    计算平均亮度（luma），范围约 [0,255]。
    为了速度，会把图缩到 max_side 以内再算。
    """
    # 更快做法：转灰度 + thumbnail + ImageStat.mean（比 numpy 转换快，内存更省）
    img = img.convert("L")
    if max_side and max_side > 0:
        img.thumbnail((max_side, max_side), Image.BICUBIC)
    stat = ImageStat.Stat(img)
    return float(stat.mean[0])


def otsu_threshold(values: np.ndarray, nbins: int = 256) -> float:
    """
    对一维数值做 Otsu 二值化阈值。
    返回阈值（float）。
    """
    if values.size == 0:
        raise ValueError("No values for Otsu threshold.")
    v = values.astype(np.float32)
    vmin = float(v.min())
    vmax = float(v.max())
    if vmax <= vmin + 1e-6:
        return vmin

    hist, bin_edges = np.histogram(v, bins=nbins, range=(vmin, vmax))
    hist = hist.astype(np.float64)
    prob = hist / (hist.sum() + 1e-12)

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) * 0.5
    omega = np.cumsum(prob)
    mu = np.cumsum(prob * bin_centers)
    mu_t = mu[-1]

    # between-class variance: (mu_t*omega - mu)^2 / (omega*(1-omega))
    denom = omega * (1.0 - omega)
    denom[denom < 1e-12] = np.nan
    sigma_b2 = (mu_t * omega - mu) ** 2 / denom
    idx = int(np.nanargmax(sigma_b2))
    return float(bin_centers[idx])


@dataclass
class ImageInfo:
    path: Path
    mean_luma: float


def detect_splits(in_root: Path, splits: Optional[List[str]]) -> List[Tuple[str, Path]]:
    """
    返回 [(split_name, split_path), ...]
    - 若指定 splits，则只取存在的那些子目录
    - 若未指定 splits 且 in_root 下存在常见 split 目录（train/val/test），则自动使用
    - 否则返回 [("all", in_root)]
    """
    if splits:
        found = []
        for s in splits:
            sp = in_root / s
            if sp.exists() and sp.is_dir():
                found.append((s, sp))
        if not found:
            raise FileNotFoundError(f"None of the specified splits exist under: {in_root}")
        return found

    common = ["train", "val", "test", "valid", "validation"]
    found_common = [(s, in_root / s) for s in common if (in_root / s).exists() and (in_root / s).is_dir()]
    if found_common:
        # 去重：valid/validation 都当作 val 的话容易冲突，这里按目录名原样返回
        return found_common

    return [("all", in_root)]


def copy_or_move(src: Path, dst: Path, move: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if move:
        shutil.move(str(src), str(dst))
    else:
        shutil.copy2(str(src), str(dst))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_root", type=str, required=True, help="输入根目录（可能包含 train/val/test，也可能就是混合图像目录）")
    ap.add_argument("--out_root", type=str, required=True, help="输出根目录（会创建 split/day 和 split/night）")
    ap.add_argument("--out_split", type=str, default=None, help="可选：强制输出到指定 split（例如 train），用于把额外数据并入已有 split")
    ap.add_argument("--splits", type=str, nargs="*", default=None, help="可选：显式指定 split 名称，例如 train val test")
    ap.add_argument("--recursive", action="store_true", help="递归扫描子目录（默认不开启）")
    ap.add_argument("--method", type=str, default="otsu", choices=["otsu", "fixed"], help="阈值方法：otsu 自动或 fixed 固定阈值")
    ap.add_argument("--threshold", type=float, default=None, help="fixed 模式下的阈值（0~255，越大越偏 day）")
    ap.add_argument("--move", action="store_true", help="移动文件（默认是复制）")
    ap.add_argument("--dry_run", action="store_true", help="只打印统计信息，不实际复制/移动")
    ap.add_argument("--max_side", type=int, default=256, help="计算亮度时缩放的最大边，越小越快")
    ap.add_argument("--sample_for_threshold", type=int, default=3000, help="Otsu 阈值计算的抽样数量（0=全量；越小越快）")
    ap.add_argument("--max_images", type=int, default=0, help="每个 split 最多处理多少张图（0=不限制；用于快速 dry-run）")
    args = ap.parse_args()

    in_root = Path(args.in_root)
    out_root = Path(args.out_root)
    if not in_root.exists():
        raise FileNotFoundError(f"in_root not found: {in_root}")

    if args.out_split:
        # 强制把 in_root 当作一个整体 split 处理，并将输出写入 out_root/out_split/...
        splits = [(str(args.out_split), in_root)]
    else:
        splits = detect_splits(in_root, args.splits)
    print(f"[INFO] splits: {[s for s, _ in splits]}")
    print(f"[INFO] mode: {'move' if args.move else 'copy'} | method: {args.method} | recursive: {args.recursive} | dry_run: {args.dry_run}")

    for split_name, split_path in splits:
        imgs = list(iter_images(split_path, recursive=args.recursive))
        if not imgs:
            print(f"[WARN] split={split_name} no images found under: {split_path}")
            continue

        if args.max_images and args.max_images > 0 and len(imgs) > args.max_images:
            imgs = imgs[: args.max_images]
            print(f"[INFO] split={split_name} truncated to max_images={args.max_images} (for speed)")

        # 1) 先为阈值计算抽样（otsu）或全量（fixed）
        if args.method == "fixed":
            if args.threshold is None:
                raise ValueError("fixed method requires --threshold")
            thr = float(args.threshold)
            sample_imgs = imgs
        else:
            n = int(args.sample_for_threshold) if args.sample_for_threshold is not None else 0
            if n <= 0 or n >= len(imgs):
                sample_imgs = imgs
            else:
                # 抽样用随机更稳一些（避免只取到文件排序前的一段）
                rng = np.random.default_rng(0)
                idx = rng.choice(len(imgs), size=n, replace=False)
                sample_imgs = [imgs[i] for i in idx.tolist()]

            sample_vals: List[float] = []
            for k, p in enumerate(sample_imgs, start=1):
                try:
                    with Image.open(p) as im:
                        ml = compute_mean_luma(im, max_side=args.max_side)
                    sample_vals.append(ml)
                except Exception as e:
                    # 个别坏图跳过
                    if k <= 5:
                        print(f"[WARN] failed to read {p}: {e}")
                if k % 1000 == 0:
                    print(f"[INFO] split={split_name} threshold-sample progress: {k}/{len(sample_imgs)}")

            vals = np.array(sample_vals, dtype=np.float32)
            if vals.size == 0:
                raise RuntimeError(f"split={split_name} no readable images for threshold.")
            thr = otsu_threshold(vals)

        # 2) 用阈值对当前 split 的 imgs 做分类（dry_run 也会做统计）
        day_cnt = 0
        night_cnt = 0
        print(f"[INFO] split={split_name} images={len(imgs)} thr≈{thr:.2f} (threshold_sample={len(sample_imgs) if args.method=='otsu' else 'all'})")

        for k, p in enumerate(imgs, start=1):
            try:
                with Image.open(p) as im:
                    ml = compute_mean_luma(im, max_side=args.max_side)
            except Exception as e:
                if k <= 5:
                    print(f"[WARN] failed to read {p}: {e}")
                continue

            label = "day" if ml >= thr else "night"
            if label == "day":
                day_cnt += 1
            else:
                night_cnt += 1

            # 保留相对层级：当 out_split 生效时，rel 相对 split_path（即 in_root）；
            # 否则 rel 相对各自 split 目录（train/val/test）
            rel = p.relative_to(split_path)
            # 输出目录：out_root/split/label/（保留相对层级，避免重名冲突）
            dst = out_root / split_name / label / rel
            if args.dry_run:
                if k % 5000 == 0:
                    print(f"[INFO] split={split_name} classify progress: {k}/{len(imgs)} (dry_run)")
                continue

            copy_or_move(p, dst, move=args.move)
            if k % 5000 == 0:
                print(f"[INFO] split={split_name} copy progress: {k}/{len(imgs)}")

        print(f"[DONE] split={split_name} -> day={day_cnt}, night={night_cnt}")

    print("[ALL DONE]")


if __name__ == "__main__":
    main()


