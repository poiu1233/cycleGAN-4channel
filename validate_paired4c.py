"""
validate_paired4c.py

四通道模型生成质量验证脚本

专门用于验证四通道（RGB + inverse RLA -> day）模型的输出质量，支持：
1. 三向对比：输入（night/low）、输出（generated）、GT（day/high）
2. 有参考指标：PSNR / SSIM / MAE（与 GT 对比）
3. 无参考指标：亮度、锐度、边缘、噪声、伪影
4. 颗粒感/伪纹理专项检测

关键指标解读：
- PSNR：>30dB 优秀，25-30 良好，<20 较差
- SSIM：>0.9 优秀，0.8-0.9 良好
- 亮度均值：60-180 合理
- 锐度/边缘：越高越清晰，但过高可能是过锐化/伪纹理
- 噪声/颗粒感：越低越好（你之前四通道输出颗粒感重，重点关注此指标）

Usage:
    # 基本用法：验证生成图质量（与 GT 对比）
    python validate_paired4c.py --gen_dir "path/to/generated" --gt_dir "path/to/high" --out_dir "results"
    
    # 完整三向对比：输入 vs 输出 vs GT
    python validate_paired4c.py --gen_dir "path/to/generated" --input_dir "path/to/low" --gt_dir "path/to/high" --out_dir "results"
    
    # 同时验证多个生成目录（例如不同 epoch 的输出）
    python validate_paired4c.py --gen_dir "gen_e10" "gen_e20" "gen_e30" --gt_dir "path/to/high" --out_dir "results"
"""

import argparse
import warnings
from pathlib import Path
from typing import Optional, Dict, List
import numpy as np
import pandas as pd
from tqdm import tqdm
import cv2
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings('ignore')


# =============================================================================
# 有参考指标（PSNR / SSIM / MAE）
# =============================================================================

def calculate_psnr(img1: np.ndarray, img2: np.ndarray, max_val: float = 255.0) -> float:
    """计算 PSNR（峰值信噪比）"""
    mse = np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)
    if mse == 0:
        return 100.0
    return float(20 * np.log10(max_val / np.sqrt(mse)))


def calculate_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """计算 SSIM（结构相似性）"""
    if len(img1.shape) == 3:
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    if len(img2.shape) == 3:
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    
    mu1 = cv2.GaussianBlur(img1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(img2, (11, 11), 1.5)
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = cv2.GaussianBlur(img1 ** 2, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(img2 ** 2, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(img1 * img2, (11, 11), 1.5) - mu1_mu2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return float(ssim_map.mean())


def calculate_mae(img1: np.ndarray, img2: np.ndarray) -> float:
    """计算 MAE（平均绝对误差）"""
    return float(np.mean(np.abs(img1.astype(np.float32) - img2.astype(np.float32))))


# =============================================================================
# 无参考指标
# =============================================================================

def calc_brightness(img_bgr: np.ndarray) -> Dict[str, float]:
    """亮度指标"""
    y = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(np.float32)
    return {
        'brightness_mean': float(y.mean()),
        'brightness_std': float(y.std()),
    }


def calc_sharpness(img_bgr: np.ndarray) -> Dict[str, float]:
    """锐度指标"""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    return {
        'sharpness_laplacian_var': float(lap.var()),
        'sharpness_laplacian_mean': float(np.abs(lap).mean()),
    }


def calc_edge(img_bgr: np.ndarray) -> Dict[str, float]:
    """边缘指标"""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    edge_mag = np.sqrt(gx**2 + gy**2)
    return {
        'edge_strength_mean': float(edge_mag.mean()),
        'edge_strength_max': float(edge_mag.max()),
    }


def calc_noise(img_bgr: np.ndarray) -> Dict[str, float]:
    """噪声指标"""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    diff = gray - blur
    noise_std = float(diff.std())
    
    # Laplacian 噪声估计（Immerkaer 方法）
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    h, w = gray.shape
    sigma_immerkaer = float(np.sqrt(np.pi / 2) * np.abs(lap).sum() / (6 * (w - 2) * (h - 2)))
    
    return {
        'noise_std': noise_std,
        'noise_sigma_immerkaer': sigma_immerkaer,
    }


def calc_artifact(img_bgr: np.ndarray) -> Dict[str, float]:
    """伪影指标"""
    h, w, _ = img_bgr.shape
    zeros = float((img_bgr == 0).sum()) / (h * w * 3)
    ones = float((img_bgr == 255).sum()) / (h * w * 3)
    
    # 块状伪影检测
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=1)
    
    block_edge_x = np.abs(gx[:, 7::8]).mean() if w > 8 else 0
    block_edge_y = np.abs(gy[7::8, :]).mean() if h > 8 else 0
    overall_gradient = (np.abs(gx).mean() + np.abs(gy).mean()) / 2 + 1e-8
    blockiness = float((block_edge_x + block_edge_y) / (2 * overall_gradient))
    
    return {
        'artifact_clip0_ratio': zeros,
        'artifact_clip255_ratio': ones,
        'artifact_blockiness': blockiness,
    }


def calc_grain_texture(img_bgr: np.ndarray) -> Dict[str, float]:
    """
    颗粒感/伪纹理专项检测（针对你之前四通道输出的问题）
    
    检测方法：
    1. 高频噪声能量比：在频域计算高频部分占比
    2. 局部方差一致性：真实纹理局部方差变化大，颗粒噪声较均匀
    3. Laplacian 绝对值的局部标准差：颗粒感图像在平坦区域也有高响应
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h, w = gray.shape
    
    # 1. 高频能量比（FFT）
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)
    cy, cx = h // 2, w // 2
    r = min(h, w) // 8
    mask_lf = np.zeros((h, w), np.float32)
    cv2.circle(mask_lf, (cx, cy), r, 1, -1)
    lf_energy = (magnitude * mask_lf).sum()
    total_energy = magnitude.sum() + 1e-8
    hf_ratio = float(1.0 - lf_energy / total_energy)
    
    # 2. 局部方差一致性（颗粒噪声在平坦区也有高方差）
    block_size = 32
    local_vars = []
    for i in range(0, h - block_size, block_size):
        for j in range(0, w - block_size, block_size):
            block = gray[i:i+block_size, j:j+block_size]
            local_vars.append(block.var())
    
    if local_vars:
        var_mean = np.mean(local_vars)
        var_std = np.std(local_vars)
        # 颗粒感图像：var_std/var_mean 较低（方差分布均匀）
        grain_uniformity = float(var_std / (var_mean + 1e-8))
    else:
        grain_uniformity = 0.0
    
    # 3. Laplacian 在"应该平坦"区域的响应（检测伪纹理）
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    blur = cv2.GaussianBlur(gray, (15, 15), 0)
    flat_mask = np.abs(gray - blur) < 10  # "平坦区域"
    if flat_mask.sum() > 100:
        lap_in_flat = float(np.abs(lap[flat_mask]).mean())
    else:
        lap_in_flat = 0.0
    
    return {
        'grain_hf_ratio': hf_ratio,
        'grain_var_uniformity': grain_uniformity,
        'grain_lap_in_flat': lap_in_flat,
    }


# =============================================================================
# 验证单张图
# =============================================================================

def validate_single_image(
    gen_path: Path,
    input_path: Optional[Path],
    gt_path: Optional[Path],
) -> Optional[Dict]:
    """验证单张图"""
    try:
        gen_img = cv2.imread(str(gen_path), cv2.IMREAD_COLOR)
        if gen_img is None:
            return None
        
        result = {'file': gen_path.name}
        
        # 生成图的无参考指标
        result.update(calc_brightness(gen_img))
        result.update(calc_sharpness(gen_img))
        result.update(calc_edge(gen_img))
        result.update(calc_noise(gen_img))
        result.update(calc_artifact(gen_img))
        result.update(calc_grain_texture(gen_img))
        
        # 与 GT 对比（有参考指标）
        if gt_path and gt_path.exists():
            gt_img = cv2.imread(str(gt_path), cv2.IMREAD_COLOR)
            if gt_img is not None:
                if gen_img.shape != gt_img.shape:
                    gt_img = cv2.resize(gt_img, (gen_img.shape[1], gen_img.shape[0]),
                                       interpolation=cv2.INTER_CUBIC)
                
                result['psnr_vs_gt'] = calculate_psnr(gen_img, gt_img)
                result['ssim_vs_gt'] = calculate_ssim(gen_img, gt_img)
                result['mae_vs_gt'] = calculate_mae(gen_img, gt_img)
        
        # 与输入对比（看增强幅度）
        if input_path and input_path.exists():
            input_img = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
            if input_img is not None:
                if gen_img.shape != input_img.shape:
                    input_img = cv2.resize(input_img, (gen_img.shape[1], gen_img.shape[0]),
                                          interpolation=cv2.INTER_CUBIC)
                
                # 输入图亮度（用于计算增强幅度）
                input_y = cv2.cvtColor(input_img, cv2.COLOR_BGR2YCrCb)[:, :, 0]
                result['input_brightness'] = float(input_y.mean())
                result['brightness_gain'] = result['brightness_mean'] - result['input_brightness']
                
                # 与输入的 PSNR（用于判断是否"改动过大"）
                result['psnr_vs_input'] = calculate_psnr(gen_img, input_img)
        
        return result
    
    except Exception as e:
        print(f"Error processing {gen_path}: {e}")
        return None


# =============================================================================
# 主函数
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description='四通道模型生成质量验证')
    ap.add_argument('--gen_dir', nargs='+', required=True, help='生成图目录（可多个）')
    ap.add_argument('--input_dir', default=None, help='输入图目录（night/low，可选）')
    ap.add_argument('--gt_dir', default=None, help='Ground Truth 目录（day/high，可选）')
    ap.add_argument('--out_dir', required=True, help='输出结果目录')
    ap.add_argument('--exts', default='.jpg,.jpeg,.png,.bmp', help='允许的图片后缀')
    ap.add_argument('--workers', type=int, default=4, help='并行 worker 数')
    args = ap.parse_args()
    
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    exts = {e.lower().strip() for e in args.exts.split(',')}
    input_dir = Path(args.input_dir) if args.input_dir else None
    gt_dir = Path(args.gt_dir) if args.gt_dir else None
    
    for gen_dir_str in args.gen_dir:
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
        
        # 构建映射
        def build_map(ref_dir):
            if ref_dir and ref_dir.exists():
                return {p.stem: p for p in ref_dir.glob('*') if p.suffix.lower() in exts}
            return {}
        
        input_map = build_map(input_dir)
        gt_map = build_map(gt_dir)
        
        # 并行验证
        rows = []
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {}
            for gp in gen_paths:
                # 去掉可能的后缀（如 _4c）匹配原文件名
                stem = gp.stem
                for suffix in ['_4c', '_day', '_gen', '_out']:
                    if stem.endswith(suffix):
                        stem = stem[:-len(suffix)]
                        break
                
                inp = input_map.get(stem) or input_map.get(gp.stem)
                gt = gt_map.get(stem) or gt_map.get(gp.stem)
                
                futures[executor.submit(validate_single_image, gp, inp, gt)] = gp
            
            for future in tqdm(as_completed(futures), total=len(futures), desc='Validating'):
                result = future.result()
                if result:
                    rows.append(result)
        
        if not rows:
            print(f"[WARN] No valid results for {gen_dir}")
            continue
        
        df = pd.DataFrame(rows)
        
        # 保存 CSV
        csv_path = out_dir / f"val4c_{gen_dir.name}.csv"
        df.to_csv(csv_path, index=False, float_format='%.4f')
        print(f"Saved: {csv_path}")
        
        # 生成摘要
        summary_lines = [
            "=" * 60,
            f"四通道模型生成质量验证报告：{gen_dir.name}",
            "=" * 60,
            f"图片数量: {len(df)}",
            "",
        ]
        
        # 有参考指标
        if 'psnr_vs_gt' in df.columns:
            summary_lines.extend([
                "【有参考指标（与 GT 对比）】",
                f"  PSNR: {df['psnr_vs_gt'].mean():.2f} ± {df['psnr_vs_gt'].std():.2f} dB",
                f"  SSIM: {df['ssim_vs_gt'].mean():.4f} ± {df['ssim_vs_gt'].std():.4f}",
                f"  MAE: {df['mae_vs_gt'].mean():.2f} ± {df['mae_vs_gt'].std():.2f}",
                "",
                "  PSNR 分布：",
                f"    优秀 (>30dB): {(df['psnr_vs_gt'] > 30).sum()} 张",
                f"    良好 (25-30dB): {((df['psnr_vs_gt'] >= 25) & (df['psnr_vs_gt'] <= 30)).sum()} 张",
                f"    一般 (20-25dB): {((df['psnr_vs_gt'] >= 20) & (df['psnr_vs_gt'] < 25)).sum()} 张",
                f"    较差 (<20dB): {(df['psnr_vs_gt'] < 20).sum()} 张",
                "",
            ])
        
        # 增强幅度
        if 'brightness_gain' in df.columns:
            summary_lines.extend([
                "【增强幅度（输入 vs 输出）】",
                f"  输入亮度均值: {df['input_brightness'].mean():.1f}",
                f"  输出亮度均值: {df['brightness_mean'].mean():.1f}",
                f"  亮度增益: {df['brightness_gain'].mean():.1f}",
                "",
            ])
        
        # 无参考指标
        summary_lines.extend([
            "【无参考指标（生成图质量）】",
            f"  亮度均值: {df['brightness_mean'].mean():.1f} (理想: 60-180)",
            f"  锐度 (Laplacian): {df['sharpness_laplacian_var'].mean():.1f}",
            f"  边缘强度: {df['edge_strength_mean'].mean():.2f}",
            f"  噪声水平: {df['noise_std'].mean():.2f}",
            "",
        ])
        
        # 颗粒感专项
        summary_lines.extend([
            "【颗粒感/伪纹理检测（重点关注）】",
            f"  高频能量比: {df['grain_hf_ratio'].mean():.4f} (越高可能颗粒感越重)",
            f"  方差均匀度: {df['grain_var_uniformity'].mean():.4f} (越低可能颗粒感越重)",
            f"  平坦区Lap响应: {df['grain_lap_in_flat'].mean():.2f} (越高伪纹理越重)",
            "",
        ])
        
        # 潜在问题
        summary_lines.extend([
            "【潜在问题】",
            f"  过暗 (亮度<60): {(df['brightness_mean'] < 60).sum()} 张",
            f"  过曝 (亮度>200): {(df['brightness_mean'] > 200).sum()} 张",
            f"  模糊 (Lap<100): {(df['sharpness_laplacian_var'] < 100).sum()} 张",
            f"  高噪声 (noise>20): {(df['noise_std'] > 20).sum()} 张",
            f"  高颗粒感 (lap_in_flat>5): {(df['grain_lap_in_flat'] > 5).sum()} 张",
            "",
            "=" * 60,
        ])
        
        summary_text = "\n".join(summary_lines)
        print(summary_text)
        
        summary_path = out_dir / f"val4c_{gen_dir.name}_summary.txt"
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(summary_text)
        print(f"Saved: {summary_path}")
    
    print("\n[ALL DONE]")


if __name__ == '__main__':
    main()

