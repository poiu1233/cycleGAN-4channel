"""
rename_train_val_test_images.py

将 dataset_root 下的 train/val/test 中的图片按“全局递增编号”重新命名。

特性：
- 默认按 split 顺序：train -> val -> test 依次编号（可自定义 splits）
- 递归扫描子目录（保留目录结构，只改文件名）
- 两阶段重命名，避免文件名冲突/覆盖
- 可选：同步重命名 YOLO 标注（同名 .txt），支持两种常见结构：
  1) 与图片同目录：xxx.jpg + xxx.txt
  2) images/ 与 labels/ 平行：.../images/xxx.jpg -> .../labels/xxx.txt
- 可选输出映射表 CSV：old_rel,new_rel

示例：
  python rename_train_val_test_images.py --root "D:\\dataset_splits" --prefix "" --start 1 --digits 6
  python rename_train_val_test_images.py --root "D:\\dataset_splits" --dry_run
  python rename_train_val_test_images.py --root "D:\\dataset_splits" --sync_labels --label_exts ".txt,.xml" --save_map "D:\\map.csv"
"""

from __future__ import annotations

import argparse
import csv
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def is_image(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMG_EXTS


def iter_images(root: Path, recursive: bool) -> Iterable[Path]:
    it = root.rglob("*") if recursive else root.iterdir()
    for p in it:
        # 跳过我们自己的临时文件（万一上次中断）
        if p.is_file() and p.name.startswith(".__tmp__"):
            continue
        if is_image(p):
            yield p


def parse_list_arg(s: str) -> List[str]:
    # 支持 ".txt,.xml" 或 "train,val,test"
    parts = [x.strip() for x in s.replace(";", ",").split(",")]
    return [p for p in parts if p]


def find_label_candidates(
    img_path: Path,
    dataset_root: Path,
    split_name: str,
    label_exts: Sequence[str],
) -> List[Path]:
    """
    返回这张图可能对应的标注文件列表（存在的才返回）。
    """
    candidates: List[Path] = []

    # 1) 同目录同名标注：xxx.jpg + xxx.txt
    for ext in label_exts:
        p = img_path.with_suffix(ext)
        if p.exists() and p.is_file():
            candidates.append(p)

    # 2) images/ -> labels/ 的 YOLO 常见结构
    #    只在 train/val/test 范围下做相对替换，避免误替换其它路径片段
    try:
        rel = img_path.relative_to(dataset_root / split_name)
    except Exception:
        return candidates

    parts = list(rel.parts)
    if "images" in parts:
        idx = parts.index("images")
        parts[idx] = "labels"
        rel_labels = Path(*parts)
        for ext in label_exts:
            p2 = (dataset_root / split_name / rel_labels).with_suffix(ext)
            if p2.exists() and p2.is_file():
                candidates.append(p2)

    # 去重、保持稳定顺序
    uniq: List[Path] = []
    seen: Set[Path] = set()
    for p in candidates:
        if p not in seen:
            uniq.append(p)
            seen.add(p)
    return uniq


@dataclass(frozen=True)
class RenameOp:
    src: Path
    tmp: Path
    dst: Path


def build_rename_ops(
    dataset_root: Path,
    splits: Sequence[str],
    recursive: bool,
    start: int,
    digits: int,
    prefix: str,
    reset_per_split: bool,
) -> Tuple[List[Tuple[Path, Path]], List[Path]]:
    """
    返回：
    - image_pairs: [(old_img, new_img), ...]
    - ordered_imgs: 参与编号的图片（按稳定顺序）
    """
    ordered_imgs: List[Path] = []
    per_split_imgs: List[Tuple[str, List[Path]]] = []
    for s in splits:
        sp = dataset_root / s
        if not sp.exists() or not sp.is_dir():
            raise FileNotFoundError(f"split not found: {sp}")
        imgs = sorted(iter_images(sp, recursive=recursive), key=lambda p: str(p).lower())
        ordered_imgs.extend(imgs)
        per_split_imgs.append((s, imgs))

    if not ordered_imgs:
        return [], []

    image_pairs: List[Tuple[Path, Path]] = []
    if reset_per_split:
        for _, imgs in per_split_imgs:
            idx = int(start)
            for img in imgs:
                new_name = f"{prefix}{idx:0{digits}d}{img.suffix}"
                image_pairs.append((img, img.with_name(new_name)))
                idx += 1
    else:
        idx = int(start)
        for img in ordered_imgs:
            new_name = f"{prefix}{idx:0{digits}d}{img.suffix}"
            image_pairs.append((img, img.with_name(new_name)))
            idx += 1

    return image_pairs, ordered_imgs


def ensure_no_collisions(pairs: Sequence[Tuple[Path, Path]]) -> None:
    dsts = [dst for _, dst in pairs]
    if len(dsts) != len(set(dsts)):
        raise RuntimeError("目标文件名存在重复（dst 冲突），请检查目录结构或命名规则。")

    src_set = {src.resolve() for src, _ in pairs}
    for src, dst in pairs:
        if dst.exists():
            # 若 dst 就是 src 自身（理论上不会，因为我们改了名字），允许
            if dst.resolve() not in src_set:
                raise FileExistsError(f"目标文件已存在，避免覆盖：{dst}")


def to_tmp_path(p: Path, token: str) -> Path:
    # 临时名尽量短，且避免同目录冲突
    return p.with_name(f".__tmp__{token}__{p.name}")


def execute_two_phase_rename(ops: Sequence[RenameOp], dry_run: bool) -> None:
    # Phase 1: src -> tmp
    for op in ops:
        if dry_run:
            continue
        op.src.rename(op.tmp)

    # Phase 2: tmp -> dst
    for op in ops:
        if dry_run:
            continue
        op.tmp.rename(op.dst)


def main() -> None:
    # 尽量让 Windows 终端正常显示中文（若终端本身非 UTF-8，仍可能需要 chcp 65001）
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8")
        except Exception:
            pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True, help="数据集根目录，包含 train/val/test")
    ap.add_argument("--splits", type=str, default="train,val,test", help="split 名称，逗号分隔（默认 train,val,test）")
    ap.add_argument("--recursive", action="store_true", help="递归扫描子目录（默认不递归）")
    ap.add_argument("--start", type=int, default=1, help="编号起始值（默认 1）")
    ap.add_argument("--digits", type=int, default=6, help="编号位数（默认 6），例如 000001")
    # 兼容 PowerShell 下传空字符串时偶发的参数解析问题：允许 --prefix 不带值，等价于空前缀
    ap.add_argument("--prefix", type=str, nargs="?", const="", default="", help="文件名前缀（默认空），例如 img_")
    ap.add_argument("--reset_per_split", action="store_true", help="每个 split 内重新从 --start 开始编号（默认关闭）")
    ap.add_argument("--sync_labels", action="store_true", help="同步重命名标注文件（默认关闭）")
    ap.add_argument("--label_exts", type=str, default=".txt", help="标注后缀列表，逗号分隔（默认 .txt）")
    ap.add_argument("--save_map", type=str, default=None, help="可选：保存 old_rel,new_rel 的 CSV 路径")
    ap.add_argument("--dry_run", action="store_true", help="只打印计划，不实际改名")
    ap.add_argument("--preview", type=int, default=20, help="dry_run/日志时预览前 N 条（默认 20）")
    args = ap.parse_args()

    dataset_root = Path(args.root)
    if not dataset_root.exists():
        raise FileNotFoundError(f"root not found: {dataset_root}")

    splits = parse_list_arg(args.splits)
    if not splits:
        raise ValueError("--splits 不能为空")

    label_exts = parse_list_arg(args.label_exts)
    if not label_exts:
        label_exts = [".txt"]
    label_exts = [e if e.startswith(".") else f".{e}" for e in label_exts]

    image_pairs, ordered_imgs = build_rename_ops(
        dataset_root=dataset_root,
        splits=splits,
        recursive=bool(args.recursive),
        start=int(args.start),
        digits=int(args.digits),
        prefix=str(args.prefix),
        reset_per_split=bool(args.reset_per_split),
    )

    if not image_pairs:
        print(f"[WARN] 未找到图片：root={dataset_root} splits={splits} recursive={args.recursive}")
        return

    # 额外收集 label 的重命名（如果开启）
    label_pairs: List[Tuple[Path, Path]] = []
    if args.sync_labels:
        # 建立 old_img -> new_img 映射，用于从“旧 stem”推导新 stem
        img_map: Dict[Path, Path] = {old: new for old, new in image_pairs}
        for old_img, new_img in image_pairs:
            # split_name 用于 images->labels 规则
            split_name = None
            for s in splits:
                try:
                    old_img.relative_to(dataset_root / s)
                    split_name = s
                    break
                except Exception:
                    continue
            if split_name is None:
                continue

            old_labels = find_label_candidates(
                img_path=old_img,
                dataset_root=dataset_root,
                split_name=split_name,
                label_exts=label_exts,
            )
            if not old_labels:
                continue
            for old_lab in old_labels:
                new_lab = old_lab.with_name(new_img.stem + old_lab.suffix)
                label_pairs.append((old_lab, new_lab))

        # 去重（同一 label 可能同时被两种规则找到）
        dedup: Dict[Path, Path] = {}
        for a, b in label_pairs:
            dedup[a] = b
        label_pairs = list(dedup.items())

    # 碰撞检测：图片 + label 合并检查
    all_pairs: List[Tuple[Path, Path]] = list(image_pairs) + list(label_pairs)
    ensure_no_collisions(all_pairs)

    # 构建两阶段 rename 操作
    token = uuid.uuid4().hex[:8]
    ops: List[RenameOp] = []
    for src, dst in all_pairs:
        ops.append(RenameOp(src=src, tmp=to_tmp_path(src, token), dst=dst))

    print(f"[INFO] root={dataset_root}")
    print(f"[INFO] splits={splits} | recursive={args.recursive}")
    print(f"[INFO] images={len(image_pairs)} | labels={len(label_pairs)} | start={args.start} | digits={args.digits} | prefix='{args.prefix}'")
    print(f"[INFO] mode={'dry_run' if args.dry_run else 'rename'}")

    # 预览
    n_prev = max(0, int(args.preview))
    if n_prev > 0:
        print("[PREVIEW] 前几条改名：")
        shown = 0
        for old_img, new_img in image_pairs[:n_prev]:
            try:
                orel = old_img.relative_to(dataset_root)
                nrel = new_img.relative_to(dataset_root)
            except Exception:
                orel = old_img
                nrel = new_img
            print(f"  IMG  {orel} -> {nrel}")
            shown += 1
        if args.sync_labels and label_pairs:
            for old_lab, new_lab in label_pairs[: max(0, n_prev - shown)]:
                try:
                    orel = old_lab.relative_to(dataset_root)
                    nrel = new_lab.relative_to(dataset_root)
                except Exception:
                    orel = old_lab
                    nrel = new_lab
                print(f"  LAB  {orel} -> {nrel}")

    # 保存映射
    if args.save_map:
        map_path = Path(args.save_map)
        map_path.parent.mkdir(parents=True, exist_ok=True)
        with map_path.open("w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["type", "old_rel", "new_rel"])
            for old_img, new_img in image_pairs:
                w.writerow(["image", str(old_img.relative_to(dataset_root)), str(new_img.relative_to(dataset_root))])
            for old_lab, new_lab in label_pairs:
                w.writerow(["label", str(old_lab.relative_to(dataset_root)), str(new_lab.relative_to(dataset_root))])
        print(f"[DONE] 映射表已写入：{map_path}")

    # 执行改名
    execute_two_phase_rename(ops, dry_run=bool(args.dry_run))
    print("[ALL DONE]")


if __name__ == "__main__":
    main()


