"""
unpaired_generate.py

使用 unpaired CycleGAN 的生成器权重批量生成图片（仅 G，不需要四通道模型）。

典型用法（PowerShell/CMD）：
  # 夜->昼（推荐用最终权重）
  python unpaired_generate.py ^
    --in_path "C:\\Users\\vipuser\\Documents\\train\\night" ^
    --out_dir "C:\\Users\\vipuser\\Documents\\gen_day" ^
    --ckpt "C:\\Users\\vipuser\\Documents\\ckpt\\unpaired\\G_night2day.pth" ^
    --device cuda

  # 单张图片
  python unpaired_generate.py --in_path "C:\\path\\night.jpg" --out_dir "C:\\out" --ckpt "...\\G_night2day.pth"

说明：
  - 默认会做 16:9 中心裁剪并 resize 到 1280x720（与 pipeline 推理保持一致）
  - 仅用于生成 CycleGAN 输出（day-like），不做融合
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.cuda.amp import autocast
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image

from night_2_day_pipeline import ResnetGenerator, center_crop_to_aspect


def iter_images(p: Path):
    if p.is_file():
        yield p
        return
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    for f in sorted(p.glob("*")):
        if f.is_file() and f.suffix.lower() in exts:
            yield f


def main():
    ap = argparse.ArgumentParser(description="Batch generate images using unpaired CycleGAN generator.")
    ap.add_argument("--in_path", required=True, help="输入：单张图片路径或图片文件夹")
    ap.add_argument("--out_dir", required=True, help="输出文件夹")
    ap.add_argument("--ckpt", required=True, help="生成器权重路径（例如 G_night2day.pth）")
    ap.add_argument("--device", default="cuda", help="cuda / cpu")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--no_amp", action="store_true", help="禁用 AMP")
    ap.add_argument("--suffix", default="", help="输出文件名后缀，例如 _day")
    args = ap.parse_args()

    in_path = Path(args.in_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    is_cuda = device.type == "cuda"

    # load model
    G = ResnetGenerator(in_channels=3, out_channels=3).to(device)
    state = torch.load(args.ckpt, map_location=device)
    G.load_state_dict(state)
    G.eval()

    tf = transforms.ToTensor()
    out_size = (int(args.width), int(args.height))

    images = list(iter_images(in_path))
    if not images:
        print(f"[WARN] 未找到可处理的图片：{in_path}")
        return

    with torch.no_grad():
        for p in images:
            pil = Image.open(p).convert("RGB")
            pil = center_crop_to_aspect(pil, 16.0 / 9.0)
            pil = pil.resize(out_size, Image.BICUBIC)

            t = tf(pil).unsqueeze(0).to(device, non_blocking=is_cuda)  # [0,1]
            t = t * 2.0 - 1.0  # [-1,1]

            with autocast(enabled=(not args.no_amp) and is_cuda):
                y = G(t)
            y = (y.squeeze(0).float().cpu().clamp(-1, 1) + 1.0) / 2.0  # [0,1]

            out_name = f"{p.stem}{args.suffix}{p.suffix}"
            save_image(y, str(out_dir / out_name))

    print(f"Done. Saved outputs to: {out_dir} (processed {len(images)} images)")


if __name__ == "__main__":
    main()


