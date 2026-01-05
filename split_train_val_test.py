"""
split_train_val_test.py

把一个图片数据集按比例划分为 train/val/test。

支持两种常见输入结构：
1) in_root 下直接是图片（flat）
2) in_root 下是“类别/域”子文件夹（如 day/night 或 classA/classB），会按子文件夹分别划分并保留结构：
   out_root/train/day, out_root/val/day, out_root/test/day ...

示例（复制划分 8/1/1）：
  python split_train_val_test.py --in_root "D:\\dataset" --out_root "D:\\dataset_splits" --train_ratio 0.8 --val_ratio 0.1 --seed 42

示例（移动文件而不是复制）：
  python split_train_val_test.py --in_root "D:\\dataset" --out_root "D:\\dataset_splits" --train_ratio 0.8 --val_ratio 0.1 --seed 42 --move
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def is_image(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMG_EXTS


def list_images(root: Path, recursive: bool) -> List[Path]:
    it = root.rglob("*") if recursive else root.iterdir()
    return [p for p in it if is_image(p)]


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def copy_or_move(src: Path, dst: Path, move: bool) -> None:
    ensure_dir(dst.parent)
    if move:
        shutil.move(str(src), str(dst))
    else:
        shutil.copy2(str(src), str(dst))


def split_indices(
    n: int, train_ratio: float, val_ratio: float, test_ratio: float, seed: int
) -> Tuple[List[int], List[int], List[int]]:
    idx = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(idx)
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    n_test = int(round(n * test_ratio))

    # 避免四舍五入导致越界：按顺序截断，确保总和 <= n
    n_train = min(n_train, n)
    n_val = min(n_val, n - n_train)
    n_test = min(n_test, n - n_train - n_val)

    train_idx = idx[:n_train]
    val_idx = idx[n_train : n_train + n_val]
    test_idx = idx[n_train + n_val : n_train + n_val + n_test]
    return train_idx, val_idx, test_idx


def detect_labeled_folders(in_root: Path, recursive: bool) -> Dict[str, List[Path]]:
    """
    判断 in_root 是否为“子目录=类别/域”的结构。
    返回 dict: {label_name: [img_paths...]}
    """
    labeled: Dict[str, List[Path]] = {}
    subdirs = [p for p in in_root.iterdir() if p.is_dir()]
    if not subdirs:
        return labeled
    for sd in subdirs:
        imgs = list_images(sd, recursive=recursive)
        if imgs:
            labeled[sd.name] = imgs
    return labeled


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_root", required=True, help="输入数据集根目录")
    ap.add_argument("--out_root", required=True, help="输出根目录（会生成 train/val/test）")
    ap.add_argument("--train_ratio", type=float, default=0.7, help="train 比例（0~1）")
    ap.add_argument("--val_ratio", type=float, default=0.15, help="val 比例（0~1）")
    ap.add_argument("--test_ratio", type=float, default=0.15, help="test 比例（0~1）")
    ap.add_argument("--seed", type=int, default=0, help="随机种子（保证可复现）")
    ap.add_argument("--recursive", action="store_true", help="递归扫描子目录（默认不递归）")
    ap.add_argument("--move", action="store_true", help="移动文件（默认复制）")
    args = ap.parse_args()

    in_root = Path(args.in_root)
    out_root = Path(args.out_root)
    if not in_root.exists():
        raise FileNotFoundError(f"in_root not found: {in_root}")
    if not (0.0 < args.train_ratio < 1.0):
        raise ValueError("--train_ratio must be in (0,1)")
    if not (0.0 <= args.val_ratio < 1.0):
        raise ValueError("--val_ratio must be in [0,1)")
    if not (0.0 <= args.test_ratio < 1.0):
        raise ValueError("--test_ratio must be in [0,1)")
    s = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(s - 1.0) > 1e-6:
        raise ValueError(f"train_ratio + val_ratio + test_ratio must equal 1.0, got {s}")

    labeled = detect_labeled_folders(in_root, recursive=args.recursive)

    if labeled:
        print(f"[INFO] 检测到子目录标签结构: {list(labeled.keys())}")
        for label, imgs in labeled.items():
            train_idx, val_idx, test_idx = split_indices(
                len(imgs), args.train_ratio, args.val_ratio, args.test_ratio, args.seed
            )
            splits = {"train": train_idx, "val": val_idx, "test": test_idx}
            for split_name, indices in splits.items():
                for i in indices:
                    src = imgs[i]
                    # 保留 label 下的相对层级（如 label 内还有子目录）
                    rel = src.relative_to(in_root)
                    dst = out_root / split_name / rel
                    copy_or_move(src, dst, move=args.move)
            print(f"[DONE] {label}: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")
    else:
        imgs = list_images(in_root, recursive=args.recursive)
        if not imgs:
            print(f"[WARN] 未找到图片: {in_root}")
            return

        train_idx, val_idx, test_idx = split_indices(
            len(imgs), args.train_ratio, args.val_ratio, args.test_ratio, args.seed
        )
        splits = {"train": train_idx, "val": val_idx, "test": test_idx}
        for split_name, indices in splits.items():
            for i in indices:
                src = imgs[i]
                rel = src.relative_to(in_root)
                dst = out_root / split_name / rel
                copy_or_move(src, dst, move=args.move)
        print(f"[DONE] all: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

    print("[ALL DONE]")


if __name__ == "__main__":
    main()


