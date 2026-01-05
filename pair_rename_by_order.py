"""
pair_rename_by_order.py

把两个文件夹的图片“按各自排序后的顺序”进行一一配对，并将两边文件统一重命名为相同的连续编号（stem 相同）。
用于满足 `night_2_day_pipeline.py` 中 `ImageFolderPair` 的配对要求（按 sorted(glob) 的索引配对）。

特点：
- 两阶段重命名（src -> tmp -> dst），避免覆盖/冲突
- 支持 dry-run 预览
- 可保存映射表 CSV：day_old,day_new,night_old,night_new
- 保留各自原始扩展名（.jpg/.png...），但两边 stem 统一

用法示例：
  python pair_rename_by_order.py ^
    --day_dir "C:\data\day" ^
    --night_dir "C:\data\night" ^
    --start 1 --digits 6 ^
    --save_map "C:\data\pair_map.csv" ^
    --dry_run

确认无误后去掉 --dry_run 执行真实改名。

注意：
- 两个目录图片数量必须相同，否则无法严格一一配对（你也可以先裁剪到相同数量）。
- 排序规则：按相对路径字符串的 lower() 排序。若目录内还有子目录，请加 --recursive。
"""

from __future__ import annotations

import argparse
import csv
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def is_image(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMG_EXTS


def iter_images(root: Path, recursive: bool) -> Iterable[Path]:
    it = root.rglob("*") if recursive else root.iterdir()
    for p in it:
        if p.is_file() and p.name.startswith(".__tmp__"):
            continue
        if is_image(p):
            yield p


@dataclass(frozen=True)
class RenameOp:
    src: Pat
    tmp: Path
    dst: Path


def to_tmp_path(p: Path, token: str) -> Path:
    return p.with_name(f".__tmp__{token}__{p.name}")


def ensure_no_collisions(pairs: Sequence[Tuple[Path, Path]]) -> None:
    dsts = [dst.resolve() for _, dst in pairs]
    if len(dsts) != len(set(dsts)):
        raise RuntimeError("目标文件名存在重复（dst 冲突），请检查目录结构或命名规则。")

    src_set = {src.resolve() for src, _ in pairs}
    for src, dst in pairs:
        if dst.exists():
            if dst.resolve() not in src_set:
                raise FileExistsError(f"目标文件已存在，避免覆盖：{dst}")


def execute_two_phase_rename(ops: Sequence[RenameOp], dry_run: bool) -> None:
    # phase 1
    for op in ops:
        if dry_run:
            continue
        op.src.rename(op.tmp)
    # phase 2
    for op in ops:
        if dry_run:
            continue
        op.tmp.rename(op.dst)


def main() -> None:
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8")
        except Exception:
            pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--day_dir", required=True, help="day 图片目录")
    ap.add_argument("--night_dir", required=True, help="night 图片目录")
    ap.add_argument("--recursive", action="store_true", help="递归扫描子目录（默认不递归）")
    ap.add_argument("--start", type=int, default=1, help="编号起始值（默认 1）")
    ap.add_argument("--digits", type=int, default=6, help="编号位数（默认 6），例如 000001")
    ap.add_argument("--prefix", type=str, nargs="?", const="", default="", help="文件名前缀（默认空）")
    ap.add_argument("--save_map", type=str, default=None, help="可选：保存映射表 CSV 路径")
    ap.add_argument("--dry_run", action="store_true", help="只预览，不实际改名")
    ap.add_argument("--preview", type=int, default=20, help="预览前 N 对（默认 20）")
    args = ap.parse_args()

    day_dir = Path(args.day_dir)
    night_dir = Path(args.night_dir)
    if not day_dir.exists() or not day_dir.is_dir():
        raise FileNotFoundError(f"day_dir not found: {day_dir}")
    if not night_dir.exists() or not night_dir.is_dir():
        raise FileNotFoundError(f"night_dir not found: {night_dir}")

    day_imgs = sorted(iter_images(day_dir, recursive=bool(args.recursive)), key=lambda p: str(p).lower())
    night_imgs = sorted(iter_images(night_dir, recursive=bool(args.recursive)), key=lambda p: str(p).lower())

    if not day_imgs:
        raise RuntimeError(f"day_dir 未找到图片：{day_dir}")
    if not night_imgs:
        raise RuntimeError(f"night_dir 未找到图片：{night_dir}")

    if len(day_imgs) != len(night_imgs):
        raise RuntimeError(f"数量不一致：day={len(day_imgs)} night={len(night_imgs)}；无法一一配对。")

    idx = int(args.start)
    pairs: List[Tuple[Path, Path, Path, Path]] = []
    rename_pairs: List[Tuple[Path, Path]] = []

    for d, n in zip(day_imgs, night_imgs):
        stem = f"{args.prefix}{idx:0{int(args.digits)}d}"
        d_new = d.with_name(stem + d.suffix.lower())
        n_new = n.with_name(stem + n.suffix.lower())
        pairs.append((d, d_new, n, n_new))
        rename_pairs.append((d, d_new))
        rename_pairs.append((n, n_new))
        idx += 1

    ensure_no_collisions(rename_pairs)

    token = uuid.uuid4().hex[:8]
    ops: List[RenameOp] = [RenameOp(src=a, tmp=to_tmp_path(a, token), dst=b) for a, b in rename_pairs]

    print(f"[INFO] day={len(day_imgs)} night={len(night_imgs)} recursive={args.recursive} start={args.start} digits={args.digits} prefix='{args.prefix}'")
    print(f"[INFO] mode={'dry_run' if args.dry_run else 'rename'}")

    n_prev = max(0, int(args.preview))
    if n_prev > 0:
        print("[PREVIEW] 前几对配对+改名：")
        for i, (d, d_new, n, n_new) in enumerate(pairs[:n_prev], start=1):
            try:
                d_rel = d.relative_to(day_dir)
                d_new_rel = d_new.relative_to(day_dir)
            except Exception:
                d_rel, d_new_rel = d, d_new
            try:
                n_rel = n.relative_to(night_dir)
                n_new_rel = n_new.relative_to(night_dir)
            except Exception:
                n_rel, n_new_rel = n, n_new
            print(f"  [{i:02d}] DAY  {d_rel} -> {d_new_rel}")
            print(f"       NIG  {n_rel} -> {n_new_rel}")

    if args.save_map:
        mp = Path(args.save_map)
        mp.parent.mkdir(parents=True, exist_ok=True)
        with mp.open("w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["day_old", "day_new", "night_old", "night_new"])
            for d, d_new, n, n_new in pairs:
                w.writerow([str(d), str(d_new), str(n), str(n_new)])
        print(f"[DONE] 映射表已写入：{mp}")

    execute_two_phase_rename(ops, dry_run=bool(args.dry_run))
    print("[ALL DONE]")


if __name__ == "__main__":
    main()


