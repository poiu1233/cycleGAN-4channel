"""
qc_generated.py

CycleGAN 生成图像质量评估脚本
用于判断生成的图像是否适合后续融合及 YOLO 目标检测（车牌/标线）

指标分类：
1. 亮度/曝光指标
2. 对比度/动态范围
3. 锐度/清晰度
4. 色彩质量
5. 噪声估计
6. 边缘质量（对车牌/标线检测至关重要）
7. 频域分析
8. 局部统计
9. 伪影/异常检测

Usage:
    python qc_generated.py --gen_dir "path/to/generated" --out_csv qc_report.csv
    python qc_generated.py --gen_dir "path/to/generated" --ref_dir "path/to/original_night" --out_csv qc_report.csv
"""

import argparse
import warnings
from pathlib import Path
from typing import Optional, Dict, Any
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

warnings.filterwarnings('ignore')


# =============================================================================
# 1. 亮度 / 曝光指标
# =============================================================================
def calc_brightness_metrics(img_bgr: np.ndarray) -> Dict[str, float]:
    """亮度相关指标"""
    y = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(np.float32)
    
    mean_y = float(y.mean())
    std_y = float(y.std())
    median_y = float(np.median(y))
    
    # 分位数（检测极端值）
    p5, p95 = np.percentile(y, [5, 95])
    dynamic_range = float(p95 - p5)
    
    # 曝光异常比例
    underexposed = float((y < 30).sum() / y.size)      # 过暗
    overexposed = float((y > 225).sum() / y.size)      # 过曝
    well_exposed = float(((y >= 30) & (y <= 225)).sum() / y.size)
    
    # 亮度直方图均匀度（熵）
    hist, _ = np.histogram(y.flatten(), bins=256, range=(0, 256))
    hist = hist / (hist.sum() + 1e-8)
    hist = hist[hist > 0]
    brightness_entropy = float(-np.sum(hist * np.log2(hist + 1e-10)))
    
    return {
        'brightness_mean': mean_y,
        'brightness_std': std_y,
        'brightness_median': median_y,
        'brightness_dynamic_range': dynamic_range,
        'underexposed_ratio': underexposed,
        'overexposed_ratio': overexposed,
        'well_exposed_ratio': well_exposed,
        'brightness_entropy': brightness_entropy,
    }


# =============================================================================
# 2. 对比度指标
# =============================================================================
def calc_contrast_metrics(img_bgr: np.ndarray) -> Dict[str, float]:
    """对比度相关指标"""
    # NOTE: 一些 Windows OpenCV 轮子对 (src=CV_32F, dst=CV_64F) 的线性滤波组合支持不完整，
    # 会报 “Unsupported combination of source format (=5), and destination format (=6)”。
    # 这里统一用 CV_32F 计算梯度/滤波，避免触发该路径。
    gray_u8 = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = gray_u8.astype(np.float32)
    
    # 全局对比度（Michelson）
    i_max, i_min = gray.max(), gray.min()
    michelson = float((i_max - i_min) / (i_max + i_min + 1e-8))
    
    # RMS 对比度
    rms_contrast = float(gray.std())
    
    # 局部对比度（基于梯度）
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient_mag = np.sqrt(gx**2 + gy**2)
    local_contrast = float(gradient_mag.mean())
    
    # Weber 对比度（局部均值）
    kernel = np.ones((11, 11), np.float32) / 121
    local_mean = cv2.filter2D(gray, cv2.CV_32F, kernel)
    weber = np.abs(gray - local_mean) / (local_mean + 1e-8)
    weber_contrast = float(weber.mean())
    
    return {
        'contrast_michelson': michelson,
        'contrast_rms': rms_contrast,
        'contrast_local_gradient': local_contrast,
        'contrast_weber': weber_contrast,
    }


# =============================================================================
# 3. 锐度 / 清晰度指标
# =============================================================================
def calc_sharpness_metrics(img_bgr: np.ndarray) -> Dict[str, float]:
    """锐度/清晰度相关指标"""
    gray_u8 = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = gray_u8.astype(np.float32)
    
    # Laplacian 方差（最常用锐度指标）
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    laplacian_var = float(lap.var())
    laplacian_mean = float(np.abs(lap).mean())
    
    # Brenner 梯度（聚焦度量）
    brenner = np.sum((gray[:, 2:].astype(float) - gray[:, :-2].astype(float))**2)
    brenner = float(brenner / gray.size)
    
    # Tenengrad（Sobel 梯度能量）
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    tenengrad = float(np.sqrt(gx**2 + gy**2).mean())
    
    # SMD（Sum of Modified Laplacian）
    sml = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    smd = float(np.abs(sml).sum() / gray.size)
    
    # 高频能量比（FFT）
    f = np.fft.fft2(gray.astype(np.float32))
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)
    h, w = gray.shape
    cy, cx = h // 2, w // 2
    r = min(h, w) // 8  # 低频半径
    mask_lf = np.zeros((h, w), np.float32)
    cv2.circle(mask_lf, (cx, cy), r, 1, -1)
    lf_energy = (magnitude * mask_lf).sum()
    total_energy = magnitude.sum() + 1e-8
    hf_ratio = float(1.0 - lf_energy / total_energy)
    
    return {
        'sharpness_laplacian_var': laplacian_var,
        'sharpness_laplacian_mean': laplacian_mean,
        'sharpness_brenner': brenner,
        'sharpness_tenengrad': tenengrad,
        'sharpness_smd': smd,
        'sharpness_hf_ratio': hf_ratio,
    }


# =============================================================================
# 4. 色彩质量指标
# =============================================================================
def calc_color_metrics(img_bgr: np.ndarray) -> Dict[str, float]:
    """色彩相关指标"""
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(img_hsv)
    
    # 饱和度
    sat_mean = float(s.mean())
    sat_std = float(s.std())
    
    # 色彩丰富度（色相分布熵）
    hist_h, _ = np.histogram(h.flatten(), bins=180, range=(0, 180))
    hist_h = hist_h / (hist_h.sum() + 1e-8)
    hist_h = hist_h[hist_h > 0]
    hue_entropy = float(-np.sum(hist_h * np.log2(hist_h + 1e-10)))
    
    # 色彩均衡（RGB 通道差异）
    b, g, r = cv2.split(img_bgr)
    channel_diff = float(np.std([b.mean(), g.mean(), r.mean()]))
    
    # 色偏检测（相对于灰度的偏移）
    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    l, a, b_ch = cv2.split(img_lab)
    a_mean = float(a.mean() - 128)  # 红-绿偏移
    b_mean = float(b_ch.mean() - 128)  # 黄-蓝偏移
    color_cast = float(np.sqrt(a_mean**2 + b_mean**2))
    
    # 自然度指标（基于统计学习的简化版）
    # 过高饱和度或过低饱和度都不自然
    naturalness = float(1.0 - abs(sat_mean - 80) / 128)
    naturalness = max(0.0, min(1.0, naturalness))
    
    return {
        'color_saturation_mean': sat_mean,
        'color_saturation_std': sat_std,
        'color_hue_entropy': hue_entropy,
        'color_channel_diff': channel_diff,
        'color_cast_a': a_mean,
        'color_cast_b': b_mean,
        'color_cast_magnitude': color_cast,
        'color_naturalness': naturalness,
    }


# =============================================================================
# 5. 噪声估计
# =============================================================================
def calc_noise_metrics(img_bgr: np.ndarray) -> Dict[str, float]:
    """噪声相关指标"""
    gray_u8 = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = gray_u8.astype(np.float32)
    
    # 高斯差分噪声估计
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    diff = gray - blur
    noise_std = float(diff.std())
    
    # Laplacian 噪声估计（Immerkaer 方法）
    # σ = sqrt(π/2) * (1/(6*(W-2)*(H-2))) * Σ|L|
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    h, w = gray.shape
    sigma_immerkaer = float(np.sqrt(np.pi / 2) * np.abs(lap).sum() / (6 * (w - 2) * (h - 2)))
    
    # 信噪比估计 (简化版)
    signal = gray.std()
    snr = float(signal / (noise_std + 1e-8))
    
    # 局部噪声一致性（噪声应均匀分布）
    block_size = 32
    noise_vars = []
    for i in range(0, h - block_size, block_size):
        for j in range(0, w - block_size, block_size):
            block = diff[i:i+block_size, j:j+block_size]
            noise_vars.append(block.var())
    noise_uniformity = float(1.0 - np.std(noise_vars) / (np.mean(noise_vars) + 1e-8))
    
    return {
        'noise_std': noise_std,
        'noise_sigma_immerkaer': sigma_immerkaer,
        'noise_snr': snr,
        'noise_uniformity': noise_uniformity,
    }


# =============================================================================
# 6. 边缘质量（对车牌/标线检测至关重要）
# =============================================================================
def calc_edge_metrics(img_bgr: np.ndarray) -> Dict[str, float]:
    """边缘质量指标（关键：车牌/标线检测依赖清晰边缘）"""
    gray_u8 = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = gray_u8
    
    # Canny 边缘密度
    edges_low = cv2.Canny(gray, 50, 150)
    edges_high = cv2.Canny(gray, 100, 200)
    edge_density_low = float(edges_low.sum() / 255 / edges_low.size)
    edge_density_high = float(edges_high.sum() / 255 / edges_high.size)
    
    # 边缘强度（Sobel 梯度均值）
    gray_f = gray.astype(np.float32)
    gx = cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=3)
    edge_magnitude = np.sqrt(gx**2 + gy**2)
    edge_strength_mean = float(edge_magnitude.mean())
    edge_strength_max = float(edge_magnitude.max())
    
    # 边缘连续性（检测断裂边缘，对标线检测重要）
    # 使用形态学操作检测连通区域
    kernel = np.ones((3, 3), np.uint8)
    edges_dilated = cv2.dilate(edges_high, kernel, iterations=1)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(edges_dilated, connectivity=8)
    if num_labels > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]  # 排除背景
        mean_area = float(areas.mean()) if len(areas) > 0 else 0
        max_area = float(areas.max()) if len(areas) > 0 else 0
        # 连续性：大连通区域占比越高越好
        continuity = float(max_area / (areas.sum() + 1e-8))
    else:
        mean_area, max_area, continuity = 0.0, 0.0, 0.0
    
    # 水平/垂直边缘比例（标线通常有明确方向）
    horizontal_edges = float(np.abs(gy).sum())
    vertical_edges = float(np.abs(gx).sum())
    total_edges = horizontal_edges + vertical_edges + 1e-8
    h_ratio = horizontal_edges / total_edges
    v_ratio = vertical_edges / total_edges
    
    return {
        'edge_density_low': edge_density_low,
        'edge_density_high': edge_density_high,
        'edge_strength_mean': edge_strength_mean,
        'edge_strength_max': edge_strength_max,
        'edge_component_mean_area': mean_area,
        'edge_component_max_area': max_area,
        'edge_continuity': continuity,
        'edge_horizontal_ratio': h_ratio,
        'edge_vertical_ratio': v_ratio,
    }


# =============================================================================
# 7. 频域分析
# =============================================================================
def calc_frequency_metrics(img_bgr: np.ndarray) -> Dict[str, float]:
    """频域分析指标"""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h, w = gray.shape
    
    # FFT
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude = np.log1p(np.abs(fshift))
    
    # 频谱质心（越靠外说明高频越丰富）
    y_coords, x_coords = np.mgrid[0:h, 0:w]
    cy, cx = h // 2, w // 2
    dist = np.sqrt((y_coords - cy)**2 + (x_coords - cx)**2)
    total_mag = magnitude.sum() + 1e-8
    centroid_dist = float((magnitude * dist).sum() / total_mag)
    
    # 频谱均匀度
    # 分成低/中/高频带
    r_low = min(h, w) // 8
    r_mid = min(h, w) // 4
    
    mask_low = dist <= r_low
    mask_mid = (dist > r_low) & (dist <= r_mid)
    mask_high = dist > r_mid
    
    energy_low = float(magnitude[mask_low].sum())
    energy_mid = float(magnitude[mask_mid].sum())
    energy_high = float(magnitude[mask_high].sum())
    total = energy_low + energy_mid + energy_high + 1e-8
    
    return {
        'freq_centroid_dist': centroid_dist,
        'freq_energy_low_ratio': energy_low / total,
        'freq_energy_mid_ratio': energy_mid / total,
        'freq_energy_high_ratio': energy_high / total,
    }


# =============================================================================
# 8. 局部统计 / 纹理
# =============================================================================
def calc_texture_metrics(img_bgr: np.ndarray) -> Dict[str, float]:
    """纹理/局部统计指标"""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h, w = gray.shape
    
    # GLCM 简化版（计算局部方差和均匀度）
    block_size = 64
    local_vars = []
    local_means = []
    for i in range(0, h - block_size, block_size):
        for j in range(0, w - block_size, block_size):
            block = gray[i:i+block_size, j:j+block_size]
            local_vars.append(block.var())
            local_means.append(block.mean())
    
    # 局部方差的统计
    texture_var_mean = float(np.mean(local_vars))
    texture_var_std = float(np.std(local_vars))
    texture_uniformity = float(1.0 - texture_var_std / (texture_var_mean + 1e-8))
    
    # 局部均值的统计（检测大块平坦区域）
    mean_of_means = float(np.mean(local_means))
    std_of_means = float(np.std(local_means))
    
    return {
        'texture_local_var_mean': texture_var_mean,
        'texture_local_var_std': texture_var_std,
        'texture_uniformity': texture_uniformity,
        'texture_local_mean_std': std_of_means,
    }


# =============================================================================
# 9. 伪影 / 异常检测
# =============================================================================
def calc_artifact_metrics(img_bgr: np.ndarray) -> Dict[str, float]:
    """伪影/异常检测指标"""
    h, w, _ = img_bgr.shape
    gray_u8 = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = gray_u8.astype(np.float32)
    
    # 裁剪比例（全黑/全白像素）
    zeros = float((img_bgr == 0).sum()) / (h * w * 3)
    ones = float((img_bgr == 255).sum()) / (h * w * 3)
    
    # 块状伪影检测（棋盘格/块效应）
    # 检测 8x8 块边界的梯度异常
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=1)
    
    # 在 8 的倍数位置的梯度
    block_edge_x = np.abs(gx[:, 7::8]).mean() if w > 8 else 0
    block_edge_y = np.abs(gy[7::8, :]).mean() if h > 8 else 0
    overall_gradient = (np.abs(gx).mean() + np.abs(gy).mean()) / 2 + 1e-8
    blockiness = float((block_edge_x + block_edge_y) / (2 * overall_gradient))
    
    # 色彩带状伪影（banding）
    # 检测相邻行/列的突变
    row_diff = np.abs(np.diff(gray, axis=0)).mean()
    col_diff = np.abs(np.diff(gray, axis=1)).mean()
    banding = float(abs(row_diff - col_diff) / (row_diff + col_diff + 1e-8))
    
    # 重复纹理检测（GAN 常见问题）
    # 使用自相关检测周期性
    autocorr = cv2.matchTemplate(gray.astype(np.uint8), 
                                  gray[h//4:3*h//4, w//4:3*w//4].astype(np.uint8),
                                  cv2.TM_CCORR_NORMED)
    # 排除中心峰值后的最大相关
    cy, cx = autocorr.shape[0] // 2, autocorr.shape[1] // 2
    autocorr[max(0,cy-10):cy+10, max(0,cx-10):cx+10] = 0
    repetition_score = float(autocorr.max())
    
    return {
        'artifact_clip0_ratio': zeros,
        'artifact_clip255_ratio': ones,
        'artifact_blockiness': blockiness,
        'artifact_banding': banding,
        'artifact_repetition': repetition_score,
    }


# =============================================================================
# 10. 综合评分
# =============================================================================
def calc_overall_score(metrics: Dict[str, float]) -> Dict[str, float]:
    """计算综合质量评分（0-100）"""
    scores = {}
    
    # 亮度评分（60-180 为理想范围）
    brightness = metrics.get('brightness_mean', 128)
    if 60 <= brightness <= 180:
        scores['score_brightness'] = 100
    elif brightness < 60:
        scores['score_brightness'] = max(0, 100 - (60 - brightness) * 2)
    else:
        scores['score_brightness'] = max(0, 100 - (brightness - 180) * 2)
    
    # 曝光评分
    well_exposed = metrics.get('well_exposed_ratio', 0.8)
    scores['score_exposure'] = min(100, well_exposed * 100)
    
    # 锐度评分（基于 Laplacian，经验阈值）
    lap_var = metrics.get('sharpness_laplacian_var', 0)
    scores['score_sharpness'] = min(100, lap_var / 10)  # 1000+ 为清晰
    
    # 边缘评分（对 YOLO 重要）
    edge_strength = metrics.get('edge_strength_mean', 0)
    scores['score_edge'] = min(100, edge_strength / 0.5)  # 50+ 为良好
    
    # 噪声评分（噪声越低越好）
    noise = metrics.get('noise_std', 0)
    scores['score_noise'] = max(0, 100 - noise * 5)  # <20 为良好
    
    # 伪影评分
    clip_total = metrics.get('artifact_clip0_ratio', 0) + metrics.get('artifact_clip255_ratio', 0)
    scores['score_artifact'] = max(0, 100 - clip_total * 500)
    
    # 自然度
    naturalness = metrics.get('color_naturalness', 0.5)
    scores['score_naturalness'] = naturalness * 100
    
    # 综合评分（加权平均）
    weights = {
        'score_brightness': 0.10,
        'score_exposure': 0.15,
        'score_sharpness': 0.25,
        'score_edge': 0.20,
        'score_noise': 0.10,
        'score_artifact': 0.10,
        'score_naturalness': 0.10,
    }
    overall = sum(scores[k] * weights[k] for k in weights)
    scores['score_overall'] = overall
    
    # YOLO 适用性评分（侧重锐度和边缘）
    yolo_score = (
        scores['score_sharpness'] * 0.35 +
        scores['score_edge'] * 0.35 +
        scores['score_exposure'] * 0.15 +
        scores['score_artifact'] * 0.15
    )
    scores['score_yolo_ready'] = yolo_score
    
    return scores


# =============================================================================
# 主函数
# =============================================================================
def analyze_image(img_path: Path) -> Optional[Dict[str, Any]]:
    """分析单张图像"""
    try:
        img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            return None
        
        h, w, _ = img_bgr.shape
        
        metrics = {'file': img_path.name, 'width': w, 'height': h}
        
        # 收集所有指标
        metrics.update(calc_brightness_metrics(img_bgr))
        metrics.update(calc_contrast_metrics(img_bgr))
        metrics.update(calc_sharpness_metrics(img_bgr))
        metrics.update(calc_color_metrics(img_bgr))
        metrics.update(calc_noise_metrics(img_bgr))
        metrics.update(calc_edge_metrics(img_bgr))
        metrics.update(calc_frequency_metrics(img_bgr))
        metrics.update(calc_texture_metrics(img_bgr))
        metrics.update(calc_artifact_metrics(img_bgr))
        
        # 计算综合评分
        metrics.update(calc_overall_score(metrics))
        
        return metrics
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return None


def main():
    ap = argparse.ArgumentParser(description='CycleGAN 生成图像质量评估')
    ap.add_argument('--gen_dir', required=True, help='生成图像目录')
    ap.add_argument('--out_csv', default='qc_report.csv', help='输出 CSV 文件')
    ap.add_argument('--out_summary', default='qc_summary.txt', help='输出摘要文件')
    ap.add_argument('--exts', default='.jpg,.jpeg,.png,.bmp', help='允许的图片后缀')
    ap.add_argument('--max_images', type=int, default=0, help='最大处理图片数（0=全部）')
    ap.add_argument('--workers', type=int, default=4, help='并行 worker 数')
    args = ap.parse_args()
    
    exts = {e.lower().strip() for e in args.exts.split(',')}
    gen_dir = Path(args.gen_dir)
    
    # 收集图片路径
    img_paths = [p for p in sorted(gen_dir.glob('*')) if p.suffix.lower() in exts]
    if args.max_images > 0:
        img_paths = img_paths[:args.max_images]
    
    print(f"Found {len(img_paths)} images in {gen_dir}")
    
    # 并行处理
    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(analyze_image, p): p for p in img_paths}
        for future in tqdm(as_completed(futures), total=len(futures), desc='Analyzing'):
            result = future.result()
            if result:
                rows.append(result)
    
    if not rows:
        print("No valid images found.")
        return
    
    # 创建 DataFrame
    df = pd.DataFrame(rows)
    
    # 按综合评分排序
    df.sort_values(by='score_overall', ascending=False, inplace=True)
    
    # 保存 CSV
    df.to_csv(args.out_csv, index=False, float_format='%.4f')
    print(f"Saved detailed metrics to: {args.out_csv}")
    
    # 生成摘要
    summary_lines = [
        "=" * 60,
        "CycleGAN 生成图像质量评估报告",
        "=" * 60,
        f"分析图片数量: {len(df)}",
        f"图像目录: {gen_dir}",
        "",
        "【综合评分统计】",
        f"  整体评分 (score_overall): {df['score_overall'].mean():.1f} ± {df['score_overall'].std():.1f}",
        f"  YOLO适用性 (score_yolo_ready): {df['score_yolo_ready'].mean():.1f} ± {df['score_yolo_ready'].std():.1f}",
        "",
        "【关键指标均值】",
        f"  亮度均值: {df['brightness_mean'].mean():.1f} (理想: 60-180)",
        f"  锐度 (Laplacian): {df['sharpness_laplacian_var'].mean():.1f} (越高越清晰)",
        f"  边缘强度: {df['edge_strength_mean'].mean():.2f} (越高边缘越清晰)",
        f"  噪声水平: {df['noise_std'].mean():.2f} (越低越好)",
        f"  饱和度: {df['color_saturation_mean'].mean():.1f} (40-120 较自然)",
        f"  曝光良好比例: {df['well_exposed_ratio'].mean()*100:.1f}%",
        "",
        "【质量分布】",
        f"  优秀 (>80): {(df['score_overall'] > 80).sum()} 张 ({(df['score_overall'] > 80).mean()*100:.1f}%)",
        f"  良好 (60-80): {((df['score_overall'] >= 60) & (df['score_overall'] <= 80)).sum()} 张",
        f"  一般 (40-60): {((df['score_overall'] >= 40) & (df['score_overall'] < 60)).sum()} 张",
        f"  较差 (<40): {(df['score_overall'] < 40).sum()} 张",
        "",
        "【YOLO 检测适用性】",
        f"  高度适用 (>80): {(df['score_yolo_ready'] > 80).sum()} 张",
        f"  适用 (60-80): {((df['score_yolo_ready'] >= 60) & (df['score_yolo_ready'] <= 80)).sum()} 张",
        f"  需谨慎 (<60): {(df['score_yolo_ready'] < 60).sum()} 张",
        "",
        "【潜在问题检测】",
        f"  过暗图像 (亮度<60): {(df['brightness_mean'] < 60).sum()} 张",
        f"  过曝图像 (亮度>200): {(df['brightness_mean'] > 200).sum()} 张",
        f"  模糊图像 (Lap<100): {(df['sharpness_laplacian_var'] < 100).sum()} 张",
        f"  高噪声 (noise>20): {(df['noise_std'] > 20).sum()} 张",
        f"  裁剪异常 (clip>1%): {((df['artifact_clip0_ratio'] + df['artifact_clip255_ratio']) > 0.01).sum()} 张",
        "",
        "【最佳/最差样本】",
        f"  最佳: {df.iloc[0]['file']} (评分: {df.iloc[0]['score_overall']:.1f})",
        f"  最差: {df.iloc[-1]['file']} (评分: {df.iloc[-1]['score_overall']:.1f})",
        "",
        "【建议】",
    ]
    
    # 根据指标给出建议
    avg_brightness = df['brightness_mean'].mean()
    avg_sharpness = df['sharpness_laplacian_var'].mean()
    avg_noise = df['noise_std'].mean()
    avg_yolo = df['score_yolo_ready'].mean()
    
    suggestions = []
    if avg_brightness < 80:
        suggestions.append("- 生成图像整体偏暗，考虑增加训练轮次或调整 cycle_lambda")
    if avg_brightness > 180:
        suggestions.append("- 生成图像整体偏亮，可能过度增强，建议减少训练轮次")
    if avg_sharpness < 200:
        suggestions.append("- 图像锐度不足，可能影响车牌/标线识别，建议检查是否过度训练导致平滑")
    if avg_noise > 15:
        suggestions.append("- 噪声水平较高，可能需要后处理去噪")
    if avg_yolo < 60:
        suggestions.append("- YOLO 适用性评分较低，建议减少训练轮次或调整模型参数")
    if avg_yolo >= 70:
        suggestions.append("- YOLO 适用性评分良好，可以尝试用于目标检测")
    
    if not suggestions:
        suggestions.append("- 整体质量良好，可用于后续处理")
    
    summary_lines.extend(suggestions)
    summary_lines.append("")
    summary_lines.append("=" * 60)
    
    summary_text = "\n".join(summary_lines)
    print(summary_text)
    
    with open(args.out_summary, 'w', encoding='utf-8') as f:
        f.write(summary_text)
    print(f"\nSaved summary to: {args.out_summary}")


if __name__ == '__main__':
    main()

