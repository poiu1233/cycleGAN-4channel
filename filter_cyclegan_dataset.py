"""
filter_cyclegan_dataset.py

根据 `qc_generated.py` 输出的 CSV 指标，从 100k_split 数据集中筛选出更适合训练 CycleGAN 的图片子集。

设计目标：
- 夜图过暗/模糊/大面积裁剪异常会让 CycleGAN 更容易产生“内容缺失/混乱/假纹理”
- 因此优先筛掉：过暗、低锐度、边缘弱、裁剪异常（clip0/255 比例过高）、噪声异常（可选）

输出结构：
  out_root/
    night/  (筛选后的 night)
    day/    (筛选后的 day；若未提供 day CSV，则按相同数量随机抽样)

示例（只有 night CSV，day 自动抽样同等数量；set3 推荐阈值）：
  python filter_cyclegan_dataset.py ^
    --night_dir "D:\\image\\100k_split\\train\\night" ^
    --day_dir   "D:\\image\\100k_split\\train\\day" ^
    --csv_night "D:\\image\\noise_night.csv" ^
    --out_root  "D:\\image\\cyclegan_filtered_set3" ^
    --min_brightness 45 --min_sharpness 80 --min_edge 12 --max_clip 0.03 ^
    --seed 42

示例（night/day 都有 CSV，分别按阈值筛选；set3 推荐阈值；并过滤 day 的 noise/clip）：
  python filter_cyclegan_dataset.py ^
    --night_dir "D:\\image\\100k_split\\train\\night" ^
    --day_dir   "D:\\image\\100k_split\\train\\day" ^
    --csv_night "D:\\image\\qc_night.csv" ^
    --csv_day   "D:\\image\\qc_day.csv" ^
    --out_root  "D:\\image\\cyclegan_filtered_set3" ^
    --min_brightness 45 --min_sharpness 80 --min_edge 12 --max_clip 0.03 ^
    --max_brightness_day 210 --max_noise_std_day 20 --max_clip_day 0.03
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path
from typing import Iterable, List, Optional, Set


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def iter_images(root: Path) -> Iterable[Path]:
    for p in sorted(root.glob("*")):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            yield p


def load_csv_files(csv_path: Path) -> List[dict]:
    # 用内置 csv，避免强依赖 pandas
    import csv

    rows: List[dict] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def to_float(v: Optional[str], default: float = 0.0) -> float:
    if v is None:
        return default
    try:
        return float(v)
    except Exception:
        return default


def select_from_metrics(
    rows: List[dict],
    *,
    min_brightness: float,
    min_sharpness: float,
    min_edge: float,
    max_clip: float,
    max_noise_std: float,
    max_images: int,
) -> List[str]:
    keep: List[str] = []
    for r in rows:
        fn = (r.get("file") or "").strip()
        if not fn:
            continue

        b = to_float(r.get("brightness_mean"), default=0.0)
        lap = to_float(r.get("sharpness_laplacian_var"), default=0.0)
        edge = to_float(r.get("edge_strength_mean"), default=0.0)
        clip0 = to_float(r.get("artifact_clip0_ratio"), default=0.0)
        clip255 = to_float(r.get("artifact_clip255_ratio"), default=0.0)
        noise = to_float(r.get("noise_std"), default=0.0)

        if b < min_brightness:
            continue
        if lap < min_sharpness:
            continue
        if edge < min_edge:
            continue
        if (clip0 + clip255) > max_clip:
            continue
        if noise > max_noise_std:
            continue

        keep.append(fn)
        if max_images > 0 and len(keep) >= max_images:
            break
    return keep


def copy_files_by_name(src_dir: Path, dst_dir: Path, names: List[str]) -> int:
    dst_dir.mkdir(parents=True, exist_ok=True)
    ok = 0
    for name in names:
        src = src_dir / name
        if not src.exists():
            # 容错：有些 CSV 可能只记录 file 名，但目录下后缀大小写不同
            base = src_dir / Path(name).stem
            found = None
            for ext in IMG_EXTS:
                cand = base.with_suffix(ext)
                if cand.exists():
                    found = cand
                    break
                cand2 = base.with_suffix(ext.upper())
                if cand2.exists():
                    found = cand2
                    break
            if found is None:
                continue
            src = found

        shutil.copy2(str(src), str(dst_dir / src.name))
        ok += 1
    return ok


def sample_day_by_count(day_dir: Path, n: int, seed: int) -> List[str]:
    all_files = [p.name for p in iter_images(day_dir)]
    rng = random.Random(int(seed))
    rng.shuffle(all_files)
    if n <= 0:
        return []
    return all_files[: min(n, len(all_files))]


def main():
    ap = argparse.ArgumentParser(description="Filter 100k_split into better CycleGAN training subset.")
    ap.add_argument("--night_dir", required=True)
    ap.add_argument("--day_dir", required=True)
    ap.add_argument("--csv_night", required=True, help="qc_generated.py 输出的 night CSV")
    ap.add_argument("--csv_day", default=None, help="可选：day CSV（不提供则 day 随机抽样）")
    ap.add_argument("--out_root", required=True, help="输出根目录，会创建 night/day 子目录")
    # set3 推荐默认阈值（更贴合当前 night 数据分布）
    ap.add_argument("--min_brightness", type=float, default=45.0)
    ap.add_argument("--min_sharpness", type=float, default=80.0)
    ap.add_argument("--min_edge", type=float, default=12.0)
    ap.add_argument("--max_clip", type=float, default=0.03, help="artifact_clip0_ratio+artifact_clip255_ratio 上限")
    ap.add_argument("--max_noise_std", type=float, default=999.0, help="噪声 std 上限（默认不过滤噪声）")
    ap.add_argument("--max_images", type=int, default=0, help="最多保留多少张（0=不限制）")
    ap.add_argument("--seed", type=int, default=42, help="day 抽样随机种子（csv_day 为空时生效）")
    ap.add_argument("--max_brightness_day", type=float, default=255.0, help="仅对 day CSV 生效：过滤过曝 day（brightness_mean 上限）")
    ap.add_argument("--max_noise_std_day", type=float, default=999.0, help="仅对 day CSV 生效：过滤高噪声 day（noise_std 上限，例如 20）")
    ap.add_argument("--max_clip_day", type=float, default=1.0, help="仅对 day CSV 生效：过滤裁剪异常 day（clip0+clip255 上限，例如 0.03）")
    args = ap.parse_args()

    night_dir = Path(args.night_dir)
    day_dir = Path(args.day_dir)
    csv_night = Path(args.csv_night)
    csv_day = Path(args.csv_day) if args.csv_day else None
    out_root = Path(args.out_root)

    out_night = out_root / "night"
    out_day = out_root / "day"
    out_root.mkdir(parents=True, exist_ok=True)

    # night selection
    rows_n = load_csv_files(csv_night)
    keep_night = select_from_metrics(
        rows_n,
        min_brightness=float(args.min_brightness),
        min_sharpness=float(args.min_sharpness),
        min_edge=float(args.min_edge),
        max_clip=float(args.max_clip),
        max_noise_std=float(args.max_noise_std),
        max_images=int(args.max_images),
    )

    if not keep_night:
        raise RuntimeError("按当前阈值筛不到 night 图片，请放宽阈值（例如降低 min_brightness/min_sharpness/min_edge）。")

    # day selection
    keep_day: List[str]
    if csv_day is None:
        keep_day = sample_day_by_count(day_dir, len(keep_night), seed=int(args.seed))
    else:
        rows_d = load_csv_files(csv_day)
        # day 的锐度/边缘一般更强，不必太严格；重点过滤过曝
        keep_day = []
        for r in rows_d:
            fn = (r.get("file") or "").strip()
            if not fn:
                continue
            b = to_float(r.get("brightness_mean"), default=0.0)
            if b > float(args.max_brightness_day):
                continue
            # 可选：过滤高噪声 & 裁剪异常（适合 YOLO/训练稳定性）
            noise = to_float(r.get("noise_std"), default=0.0)
            if noise > float(args.max_noise_std_day):
                continue
            clip0 = to_float(r.get("artifact_clip0_ratio"), default=0.0)
            clip255 = to_float(r.get("artifact_clip255_ratio"), default=0.0)
            if (clip0 + clip255) > float(args.max_clip_day):
                continue
            keep_day.append(fn)
            if int(args.max_images) > 0 and len(keep_day) >= int(args.max_images):
                break
        # 若 day 太多，随机抽样到与 night 同数量
        if len(keep_day) > len(keep_night):
            rng = random.Random(int(args.seed))
            rng.shuffle(keep_day)
            keep_day = keep_day[: len(keep_night)]
        # 若 day 太少，给出明确提示（避免 silently 缺配对）
        if len(keep_day) < len(keep_night):
            print(
                f"[WARN] day 通过筛选的数量 ({len(keep_day)}) 小于 night 数量 ({len(keep_night)})，"
                f"将导致 day copied 少于 night；可放宽 --max_noise_std_day/--max_clip_day 或提供更多 day 数据。"
            )

    # copy
    n_ok = copy_files_by_name(night_dir, out_night, keep_night)
    d_ok = copy_files_by_name(day_dir, out_day, keep_day)

    print("==== Filter done ====")
    print(
        f"thresholds: min_brightness={args.min_brightness} | min_sharpness={args.min_sharpness} | "
        f"min_edge={args.min_edge} | max_clip={args.max_clip} | max_noise_std={args.max_noise_std} | "
        f"max_images={args.max_images}"
    )
    print(f"night selected: {len(keep_night)} | copied: {n_ok} -> {out_night}")
    print(f"day   selected: {len(keep_day)} | copied: {d_ok} -> {out_day}")
    print("tips: 若 copied 明显小于 selected，通常是 CSV 的 file 名与目录实际文件名不一致（可检查后缀/大小写）。")


if __name__ == "__main__":
    main()


