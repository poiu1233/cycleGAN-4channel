"""
validate_cyclegan.py

CycleGAN 生成质量验证脚本
关键指标解读（用于 CycleGAN 调参/选权重）
PSNR：>30dB 优秀，25–30 良好，<20 较差
SSIM：>0.9 优秀，0.8–0.9 良好
亮度均值：60–180 合理（你原夜图 34，如果生成图到 100+ 说明增强有效）
锐度/边缘：越高越清晰，但要防止过锐化造假边缘
噪声/伪影：越低越好
支持两种验证模式：
1. 无参考验证（盲评）：评估生成图的亮度、锐度、边缘、噪声、伪影等客观指标
2. 有参考验证（配对）：如果提供 GT 目录，额外计算 PSNR/SSIM/LPIPS

Usage:
    # 无参考验证（只评估生成图质量）
    python validate_cyclegan.py --gen_dir "path/to/generated" --out_dir "results"
    
    # 有参考验证（与 GT 对比）
    python validate_cyclegan.py --gen_dir "path/to/generated" --gt_dir "path/to/ground_truth" --out_dir "results"
    
    # 同时对比多个生成目录
    python validate_cyclegan.py --gen_dir "gen1" "gen2" "gen3" --gt_dir "path/to/gt" --out_dir "results"

    # 一键：用 CycleGAN 生成器权重批量生成 + 评测（适合你这种只有 .pth 的情况）
    # - 先把 input_dir 里每张图过一遍生成器，输出到 out_dir/generated_xxx/
    # - 再对该输出目录跑同样的评测（可选 gt_dir 做有参考指标）
    python validate_cyclegan.py --G_path "D:/image/ckpt/unpaired_e40_c384_b8/G_day2night.pth" --input_dir "D:/image/day_images" --out_dir "results"
"""

import argparse
import warnings
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import numpy as np
import pandas as pd
from tqdm import tqdm
import cv2
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings('ignore')

# =============================================================================
# （可选）生成：用 CycleGAN 生成器权重批量生成图片
# =============================================================================

def _try_import_torch():
    try:
        import torch  # noqa
        from torchvision import transforms  # noqa
        return True
    except Exception:
        return False


def generate_images_with_generator(
    G_path: Path,
    input_dir: Path,
    out_gen_dir: Path,
    *,
    device: str = 'cuda',
    width: int = 1280,
    height: int = 720,
    exts: Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.bmp'),
    n_blocks: int = 9,
    ngf: int = 64,
    use_amp: bool = True,
    overwrite: bool = False,
) -> Path:
    """
    用 CycleGAN 的 ResNet Generator（in=3, out=3）对 input_dir 批量推理，保存到 out_gen_dir。
    约定输入输出文件名一致，方便和 GT 对齐（如果你有 GT）。
    """
    if not _try_import_torch():
        raise RuntimeError(
            "你启用了 --G_path/--input_dir 的“一键生成+评测”模式，但当前环境缺少 PyTorch 依赖。\n"
            "请先安装：pip install torch torchvision\n"
            "（以及本脚本评测需要：pip install opencv-python numpy pandas tqdm pillow）"
        )

    import torch
    import torch.nn as nn
    from torch.cuda.amp import autocast
    from torchvision import transforms
    from PIL import Image

    # 复用本项目里定义的同款网络结构/预处理（避免你训练/推理结构不一致）
    from night_2_day_pipeline import ResnetGenerator, center_crop_to_aspect

    input_dir = Path(input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"input_dir not found: {input_dir}")
    G_path = Path(G_path)
    if not G_path.exists():
        raise FileNotFoundError(f"G_path not found: {G_path}")

    out_gen_dir = Path(out_gen_dir)
    out_gen_dir.mkdir(parents=True, exist_ok=True)

    device_t = torch.device(device if torch.cuda.is_available() else 'cpu')
    is_cuda = (device_t.type == 'cuda')

    G = ResnetGenerator(in_channels=3, out_channels=3, ngf=int(ngf), n_blocks=int(n_blocks)).to(device_t)
    state = torch.load(str(G_path), map_location=device_t)
    # 兼容：有些训练脚本会把权重包在 {'state_dict': ...} 里
    if isinstance(state, dict) and 'state_dict' in state and isinstance(state['state_dict'], dict):
        state = state['state_dict']
    G.load_state_dict(state)
    G.eval()

    allowed_exts = {e.lower() for e in exts}
    in_paths = sorted([p for p in input_dir.glob('*') if p.suffix.lower() in allowed_exts])
    if not in_paths:
        raise RuntimeError(f"No images found in input_dir: {input_dir}")

    to_tensor = transforms.ToTensor()

    for p in tqdm(in_paths, desc=f'Generating ({G_path.name})'):
        out_p = out_gen_dir / p.name
        if out_p.exists() and not overwrite:
            continue

        pil = Image.open(p).convert('RGB')
        pil_16x9 = center_crop_to_aspect(pil, 16.0 / 9.0)
        pil_resized = pil_16x9.resize((int(width), int(height)), Image.BICUBIC)
        t = to_tensor(pil_resized).unsqueeze(0).to(device_t, non_blocking=is_cuda)  # [0,1]
        t = t * 2.0 - 1.0  # [-1,1]

        with torch.no_grad():
            with autocast(enabled=bool(use_amp) and is_cuda):
                fake = G(t)
        fake = fake.squeeze(0).detach().float().cpu().clamp(-1, 1)
        fake = (fake + 1.0) / 2.0  # [0,1]
        fake_np = (fake.numpy().transpose(1, 2, 0) * 255.0).round().astype(np.uint8)  # HWC RGB
        Image.fromarray(fake_np, mode='RGB').save(str(out_p))

    return out_gen_dir


# =============================================================================
# 有参考指标（PSNR / SSIM）
# =============================================================================

def calculate_psnr(img1: np.ndarray, img2: np.ndarray, max_val: float = 255.0) -> float:
    """计算 PSNR（峰值信噪比）"""
    mse = np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)
    if mse == 0:
        return 100.0
    return float(20 * np.log10(max_val / np.sqrt(mse)))


def calculate_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """计算 SSIM（结构相似性）- 简化版"""
    # 转为灰度
    if len(img1.shape) == 3:
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    if len(img2.shape) == 3:
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    
    # 常数
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    
    # 均值
    mu1 = cv2.GaussianBlur(img1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(img2, (11, 11), 1.5)
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    # 方差和协方差
    sigma1_sq = cv2.GaussianBlur(img1 ** 2, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(img2 ** 2, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(img1 * img2, (11, 11), 1.5) - mu1_mu2
    
    # SSIM
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return float(ssim_map.mean())


def calculate_mae(img1: np.ndarray, img2: np.ndarray) -> float:
    """计算 MAE（平均绝对误差）"""
    return float(np.mean(np.abs(img1.astype(np.float32) - img2.astype(np.float32))))


# =============================================================================
# 无参考指标（调用 qc_generated 模块）
# =============================================================================

def get_blind_metrics(img_bgr: np.ndarray) -> Dict[str, float]:
    """获取无参考质量指标（简化版）"""
    from qc_generated import (
        calc_brightness_metrics,
        calc_sharpness_metrics,
        calc_edge_metrics,
        calc_noise_metrics,
        calc_artifact_metrics,
    )
    
    metrics = {}
    metrics.update(calc_brightness_metrics(img_bgr))
    metrics.update(calc_sharpness_metrics(img_bgr))
    metrics.update(calc_edge_metrics(img_bgr))
    metrics.update(calc_noise_metrics(img_bgr))
    metrics.update(calc_artifact_metrics(img_bgr))
    
    return metrics


# =============================================================================
# 验证单张图
# =============================================================================

def validate_single_image(
    gen_path: Path,
    gt_path: Optional[Path],
    include_blind: bool = True
) -> Optional[Dict]:
    """验证单张生成图"""
    try:
        gen_img = cv2.imread(str(gen_path), cv2.IMREAD_COLOR)
        if gen_img is None:
            return None
        
        result = {'file': gen_path.name}
        
        # 有参考指标
        if gt_path and gt_path.exists():
            gt_img = cv2.imread(str(gt_path), cv2.IMREAD_COLOR)
            if gt_img is not None:
                # 确保尺寸一致
                if gen_img.shape != gt_img.shape:
                    gt_img = cv2.resize(gt_img, (gen_img.shape[1], gen_img.shape[0]), 
                                       interpolation=cv2.INTER_CUBIC)
                
                result['psnr'] = calculate_psnr(gen_img, gt_img)
                result['ssim'] = calculate_ssim(gen_img, gt_img)
                result['mae'] = calculate_mae(gen_img, gt_img)
        
        # 无参考指标
        if include_blind:
            blind = get_blind_metrics(gen_img)
            result.update(blind)
        
        return result
    
    except Exception as e:
        print(f"Error processing {gen_path}: {e}")
        return None


# =============================================================================
# 主函数
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description='CycleGAN 生成质量验证')
    ap.add_argument('--gen_dir', nargs='+', default=None, help='生成图目录（可多个）')
    ap.add_argument('--gt_dir', default=None, help='Ground Truth 目录（可选，用于有参考验证）')
    ap.add_argument('--out_dir', required=True, help='输出结果目录')
    ap.add_argument('--exts', default='.jpg,.jpeg,.png,.bmp', help='允许的图片后缀')
    ap.add_argument('--workers', type=int, default=4, help='并行 worker 数')
    ap.add_argument('--no_blind', action='store_true', help='不计算无参考指标（只做 PSNR/SSIM）')

    # 一键生成 + 评测（可选）
    ap.add_argument('--G_path', default=None, help='CycleGAN 生成器权重路径（.pth），指定则会先生成再评测')
    ap.add_argument('--input_dir', default=None, help='输入图片目录（与 --G_path 配对使用）')
    ap.add_argument('--device', type=str, default='cuda', help='推理设备：cuda/cpu（仅一键模式使用）')
    ap.add_argument('--width', type=int, default=1280, help='推理输出宽（仅一键模式使用）')
    ap.add_argument('--height', type=int, default=720, help='推理输出高（仅一键模式使用）')
    ap.add_argument('--n_blocks', type=int, default=9, help='ResnetGenerator 的残差块数（默认 9；需与你训练时一致）')
    ap.add_argument('--ngf', type=int, default=64, help='生成器基准通道数（默认 64；需与你训练时一致）')
    ap.add_argument('--no_amp_infer', action='store_true', help='禁用 AMP（仅一键模式使用）')
    ap.add_argument('--overwrite_gen', action='store_true', help='生成目录存在同名文件时覆盖（仅一键模式使用）')
    args = ap.parse_args()
    
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    exts = {e.lower().strip() for e in args.exts.split(',')}
    gt_dir = Path(args.gt_dir) if args.gt_dir else None

    # --- 一键模式：先生成 ---
    gen_dir_list: List[str] = []
    if args.G_path or args.input_dir:
        if not (args.G_path and args.input_dir):
            raise SystemExit("使用一键模式时必须同时提供：--G_path 和 --input_dir")
        model_path = Path(args.G_path)
        # 生成目录命名：out_dir/generated_<ckpt_name>/
        safe_name = model_path.stem
        gen_out = out_dir / f"generated_{safe_name}"
        gen_out = generate_images_with_generator(
            model_path,
            Path(args.input_dir),
            gen_out,
            device=args.device,
            width=args.width,
            height=args.height,
            exts=tuple(exts),
            n_blocks=args.n_blocks,
            ngf=args.ngf,
            use_amp=(not args.no_amp_infer),
            overwrite=bool(args.overwrite_gen),
        )
        gen_dir_list = [str(gen_out)]
    elif args.gen_dir:
        gen_dir_list = list(args.gen_dir)
    else:
        raise SystemExit("必须提供 --gen_dir 或者（--G_path + --input_dir）其中一种方式")
    
    # 对每个生成目录分别验证
    for gen_dir_str in gen_dir_list:
        gen_dir = Path(gen_dir_str)
        if not gen_dir.exists():
            print(f"[WARN] {gen_dir} not found, skip")
            continue
        
        print(f"\n{'='*60}")
        print(f"Validating: {gen_dir}")
        print(f"{'='*60}")
        
        # 收集图片路径
        gen_paths = sorted([p for p in gen_dir.glob('*') if p.suffix.lower() in exts])
        if not gen_paths:
            print(f"[WARN] No images found in {gen_dir}")
            continue
        
        # 构建 GT 映射（如果有）
        gt_map = {}
        if gt_dir and gt_dir.exists():
            gt_files = {p.name: p for p in gt_dir.glob('*') if p.suffix.lower() in exts}
            for gp in gen_paths:
                if gp.name in gt_files:
                    gt_map[gp] = gt_files[gp.name]
        
        # 并行验证
        rows = []
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    validate_single_image,
                    gp,
                    gt_map.get(gp),
                    not args.no_blind
                ): gp for gp in gen_paths
            }
            
            for future in tqdm(as_completed(futures), total=len(futures), desc='Validating'):
                result = future.result()
                if result:
                    rows.append(result)
        
        if not rows:
            print(f"[WARN] No valid results for {gen_dir}")
            continue
        
        # 生成 DataFrame
        df = pd.DataFrame(rows)
        
        # 保存 CSV
        csv_name = f"val_{gen_dir.name}.csv"
        csv_path = out_dir / csv_name
        df.to_csv(csv_path, index=False, float_format='%.4f')
        print(f"Saved: {csv_path}")
        
        # 生成摘要
        summary_lines = [
            "=" * 60,
            f"CycleGAN 生成质量验证报告：{gen_dir.name}",
            "=" * 60,
            f"图片数量: {len(df)}",
            "",
        ]
        
        # 有参考指标
        if 'psnr' in df.columns:
            summary_lines.extend([
                "【有参考指标（与 GT 对比）】",
                f"  PSNR: {df['psnr'].mean():.2f} ± {df['psnr'].std():.2f} dB",
                f"  SSIM: {df['ssim'].mean():.4f} ± {df['ssim'].std():.4f}",
                f"  MAE: {df['mae'].mean():.2f} ± {df['mae'].std():.2f}",
                "",
                "  PSNR 分布：",
                f"    优秀 (>30dB): {(df['psnr'] > 30).sum()} 张",
                f"    良好 (25-30dB): {((df['psnr'] >= 25) & (df['psnr'] <= 30)).sum()} 张",
                f"    一般 (20-25dB): {((df['psnr'] >= 20) & (df['psnr'] < 25)).sum()} 张",
                f"    较差 (<20dB): {(df['psnr'] < 20).sum()} 张",
                "",
            ])
        
        # 无参考指标
        if 'brightness_mean' in df.columns:
            summary_lines.extend([
                "【无参考指标（生成图质量）】",
                f"  亮度均值: {df['brightness_mean'].mean():.1f} (提示：day→night 通常会更低；night→day 通常会更高)",
                f"  锐度 (Laplacian): {df['sharpness_laplacian_var'].mean():.1f}",
                f"  边缘强度: {df['edge_strength_mean'].mean():.2f}",
                f"  噪声水平: {df['noise_std'].mean():.2f}",
                f"  伪影 (clip0+clip255): {(df['artifact_clip0_ratio'] + df['artifact_clip255_ratio']).mean()*100:.2f}%",
                "",
                "【潜在问题】",
                f"  过暗 (亮度<60): {(df['brightness_mean'] < 60).sum()} 张",
                f"  过曝 (亮度>200): {(df['brightness_mean'] > 200).sum()} 张",
                f"  模糊 (Lap<100): {(df['sharpness_laplacian_var'] < 100).sum()} 张",
                f"  高噪声 (noise>20): {(df['noise_std'] > 20).sum()} 张",
                "",
            ])
        
        summary_lines.append("=" * 60)
        
        summary_text = "\n".join(summary_lines)
        print(summary_text)
        
        # 保存摘要
        summary_path = out_dir / f"val_{gen_dir.name}_summary.txt"
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(summary_text)
        print(f"Saved: {summary_path}")
    
    print("\n[ALL DONE]")


if __name__ == '__main__':
    main()

