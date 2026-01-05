"""
paired4c_generate.py

只使用“四通道模型”(RGB + inverse RLA -> day) 做推理生成，不使用 CycleGAN、不做融合。

用法（PowerShell/CMD）：
  # 单张
  python paired4c_generate.py ^
    --in_path "D:\\image\\one_night.jpg" ^
    --out_dir "D:\\image\\gen_4c" ^
    --ckpt "D:\\image\\ckpt\\paired4c\\_30" ^
    --device cuda

  # 整目录
  python paired4c_generate.py ^
    --in_path "D:\\image\\100k_split\\train\\night" ^
    --out_dir "D:\\image\\gen_4c" ^
    --ckpt "D:\\image\\ckpt\\paired4c\\_30" ^
    --device cuda

说明：
  - --ckpt 可以是权重文件（G4c_last.pth / G4c_epochXX.pth），也可以是包含这些文件的目录。
  - 默认会做 16:9 中心裁剪并 resize 到 1280x720（可用 --width/--height 改）。
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.cuda.amp import autocast
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image

from night_2_day_pipeline import FourChannelGenerator, center_crop_to_aspect, compute_inverse_rla


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def iter_images(p: Path):
    if p.is_file():
        yield p
        return
    for f in sorted(p.glob("*")):
        if f.is_file() and f.suffix.lower() in IMG_EXTS:
            yield f


def _torch_load_state(path: Path, device: torch.device):
    # PyTorch 2.1+ 支持 weights_only=True（更安全）；老版本 fallback
    try:
        return torch.load(str(path), map_location=device, weights_only=True)
    except TypeError:
        return torch.load(str(path), map_location=device)


def resolve_ckpt(ckpt: Path) -> Path:
    """ckpt 可以是文件或目录：优先 G4c_last.pth，其次最新的 G4c_epoch*.pth"""
    if ckpt.is_file():
        return ckpt
    if not ckpt.exists() or not ckpt.is_dir():
        raise FileNotFoundError(f"ckpt not found: {ckpt}")

    last = ckpt / "G4c_last.pth"
    if last.exists():
        return last

    # 找 epoch 最大的
    cand = list(ckpt.glob("G4c_epoch*.pth"))
    if not cand:
        raise FileNotFoundError(f"no G4c_last.pth or G4c_epoch*.pth found in: {ckpt}")

    def epoch_num(p: Path) -> int:
        m = re.search(r"epoch(\\d+)", p.stem)
        return int(m.group(1)) if m else -1

    cand.sort(key=lambda p: (epoch_num(p), p.stat().st_mtime), reverse=True)
    return cand[0]


def main():
    ap = argparse.ArgumentParser(description="Generate images using paired 4-channel generator only.")
    ap.add_argument("--in_path", required=True, help="输入：单张图片路径或图片文件夹")
    ap.add_argument("--out_dir", required=True, help="输出文件夹")
    ap.add_argument("--ckpt", required=True, help="四通道生成器权重路径或目录（包含 G4c_last.pth / G4c_epoch*.pth）")
    ap.add_argument("--device", default="cuda", help="cuda / cpu")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--no_amp", action="store_true", help="禁用 AMP")
    ap.add_argument("--suffix", default="_4c", help="输出文件名后缀，例如 _4c")
    args = ap.parse_args()

    in_path = Path(args.in_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    is_cuda = device.type == "cuda"

    ckpt_path = resolve_ckpt(Path(args.ckpt))
    print(f"[INFO] Using ckpt: {ckpt_path}")

    G = FourChannelGenerator(in_channels=4, out_channels=3, n_blocks=6).to(device)
    state = _torch_load_state(ckpt_path, device)
    G.load_state_dict(state)
    G.eval()

    images = list(iter_images(in_path))
    if not images:
        print(f"[WARN] 未找到可处理的图片：{in_path}")
        return

    tf = transforms.ToTensor()
    out_size = (int(args.width), int(args.height))

    with torch.no_grad():
        for p in images:
            pil = Image.open(p).convert("RGB")
            pil = center_crop_to_aspect(pil, 16.0 / 9.0)
            pil = pil.resize(out_size, Image.BICUBIC)

            np_rgb = np.array(pil)  # H,W,3 RGB uint8
            np_bgr = cv2.cvtColor(np_rgb, cv2.COLOR_RGB2BGR)
            inv_rla = compute_inverse_rla(np_bgr)  # H,W in [0,1]

            t_rgb = tf(pil).unsqueeze(0).to(device, non_blocking=is_cuda)  # [0,1]
            t_rgb = t_rgb * 2.0 - 1.0  # [-1,1]
            t_inv = torch.from_numpy(inv_rla).unsqueeze(0).unsqueeze(0).to(device, non_blocking=is_cuda).float()
            t_inv = t_inv * 2.0 - 1.0
            x4 = torch.cat([t_rgb, t_inv], dim=1)  # 1,4,H,W

            with autocast(enabled=(not args.no_amp) and is_cuda):
                y = G(x4)
            y = (y.squeeze(0).float().cpu().clamp(-1, 1) + 1.0) / 2.0  # [0,1]

            out_name = f"{p.stem}{args.suffix}{p.suffix}"
            save_image(y, str(out_dir / out_name))

    print(f"Done. Saved outputs to: {out_dir} (processed {len(images)} images)")


if __name__ == "__main__":
    main()


