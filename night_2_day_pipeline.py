"""
night2day_pipeline.py

PyTorch implementation to reproduce the method described in Lee et al. (2025):
- Unpaired CycleGAN module (night <-> day)
- Four-channel paired module (RGB + inverse RLA -> day)
- RLA / inverse-RLA generation (Algorithm 2)
- Cross-blending fusion (Algorithm 3)

This file contains:
1. Image processing helpers (RLA generation using OpenCV)
2. Model definitions (ResNet-based Generator, PatchGAN Discriminator)
3. Training skeleton for unpaired CycleGAN
4. Synthetic paired data generation using trained unpaired generator
5. Training skeleton for four-channel paired generator (4-channel input)
6. Cross-blending fusion and end-to-end inference pipeline

Hyperparameters and variable defaults are set according to values referenced in the paper and assistant summary:
- cycle_lambda = 10.0
- identity_lambda = 0.5
- lr = 2e-4
- betas = (0.5, 0.999)
- unpaired dataset size example: 2800 (paper used similar scale)
- paired dataset size example: 3000
- fusion gamma = 2

Notes:
- The training loops are provided in a reproducible, well-documented skeleton form. Running full training requires GPU(s) and properly prepared datasets.
- This code uses OpenCV (cv2), PyTorch (torch, torchvision).

Dependencies (example):
pip install torch torchvision opencv-python tqdm pillow numpy

"""

import os
import math
import random
import time
from pathlib import Path
import argparse

import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode
from torchvision.utils import save_image
from tqdm import tqdm

# -------------------------
# Utilities: Image & RLA
# -------------------------

def pil_to_cv2(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def cv2_to_pil(img_cv2: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB))


def compute_rla_map(rgb_bgr: np.ndarray, bilateral_d=9, bilateral_sigma_color=75, bilateral_sigma_space=75, exp_coef=2.0):
    """
    Compute the RLA (Road Light Attention) map as described in the paper (Algorithm 2 summary):
    - Input: BGR uint8 image (H, W, 3)
    - Convert to LAB, extract L channel, apply bilateral filter to preserve edges
    - Compute an exponential vertical weighting emphasizing lower image rows (near-field)
    - Combine and normalize -> RLA in [0,1]

    Returns: rla: float32 array shape (H, W) in [0,1]
    """
    # Convert to LAB and extract L channel
    lab = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[:, :, 0]  # range 0..255

    # Bilateral filter to smooth illumination while preserving edges
    L_blur = cv2.bilateralFilter(L.astype(np.uint8), d=bilateral_d, sigmaColor=bilateral_sigma_color, sigmaSpace=bilateral_sigma_space).astype(np.float32)

    # Normalize L_blur
    Ln = (L_blur - L_blur.min()) / (1e-8 + (L_blur.max() - L_blur.min()))

    # Exponential vertical weighting (bottom rows emphasized)
    H, W = Ln.shape
    ys = np.arange(H, dtype=np.float32)
    # normalize 0..1 top->bottom
    ys_norm = ys / (H - 1.0)
    vertical_weight = np.exp(exp_coef * ys_norm) - 1.0
    vertical_weight = vertical_weight / (vertical_weight.max() + 1e-8)
    vertical_map = np.tile(vertical_weight[:, None], (1, W))

    # Combine
    rla = Ln * vertical_map
    # Normalize to [0,1]
    rla = (rla - rla.min()) / (1e-8 + (rla.max() - rla.min()))
    return rla.astype(np.float32)


def compute_inverse_rla(rgb_bgr: np.ndarray, **kwargs):
    rla = compute_rla_map(rgb_bgr, **kwargs)
    inv = 1.0 - rla
    # optional smoothing (paper mentions smoothing / normalization)
    inv = cv2.GaussianBlur(inv, (5,5), 0)
    inv = (inv - inv.min()) / (1e-8 + (inv.max()-inv.min()))
    return inv.astype(np.float32)


# Helper: convert numpy arrays to torch tensors (C,H,W)
def npimg_to_tensor(img_rgb: np.ndarray):
    # img_rgb: H,W,3 uint8 or float in [0,255]
    if img_rgb.dtype == np.uint8:
        img_rgb = img_rgb.astype(np.float32) / 255.0
    img_rgb = np.transpose(img_rgb, (2,0,1))
    return torch.from_numpy(img_rgb).float()

def center_crop_to_aspect(img: Image.Image, aspect_w_over_h: float = 16.0 / 9.0) -> Image.Image:
    """
    将任意输入图像做“中心裁剪到固定宽高比”，避免直接 Resize 到正方形导致几何拉伸。
    - aspect_w_over_h: 例如 16/9
    """
    w, h = img.size
    target_aspect = float(aspect_w_over_h)
    current_aspect = w / float(h)

    if abs(current_aspect - target_aspect) < 1e-6:
        return img

    if current_aspect > target_aspect:
        # too wide -> crop width
        new_w = int(round(h * target_aspect))
        left = (w - new_w) // 2
        return img.crop((left, 0, left + new_w, h))
    else:
        # too tall -> crop height
        new_h = int(round(w / target_aspect))
        top = (h - new_h) // 2
        return img.crop((0, top, w, top + new_h))


class Paired16x9RandomCropTransform:
    """
    给 paired 数据（day, night）应用一致的随机变换：
    1) 中心裁剪到 16:9（避免拉伸）
    2) Resize 到 load_size（保证后续 random crop 足够大）
    3) 同一组参数的 RandomCrop 到 crop_size（默认 256）
    4) 可选：同一随机水平翻转
    5) ToTensor + Normalize 到 [-1,1]
    """
    def __init__(self, crop_size=256, load_size=286, hflip=True):
        self.crop_size = int(crop_size)
        self.load_size = int(load_size)
        self.hflip = bool(hflip)

    def __call__(self, day_img: Image.Image, night_img: Image.Image):
        day_img = center_crop_to_aspect(day_img, 16.0 / 9.0)
        night_img = center_crop_to_aspect(night_img, 16.0 / 9.0)

        # Resize: torchvision Resize(int) keeps aspect ratio by matching shorter side
        day_img = TF.resize(day_img, self.load_size, interpolation=InterpolationMode.BICUBIC)
        night_img = TF.resize(night_img, self.load_size, interpolation=InterpolationMode.BICUBIC)

        i, j, h, w = transforms.RandomCrop.get_params(day_img, output_size=(self.crop_size, self.crop_size))
        day_img = TF.crop(day_img, i, j, h, w)
        night_img = TF.crop(night_img, i, j, h, w)

        if self.hflip and random.random() < 0.5:
            day_img = TF.hflip(day_img)
            night_img = TF.hflip(night_img)

        day_t = TF.to_tensor(day_img)
        night_t = TF.to_tensor(night_img)
        day_t = TF.normalize(day_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        night_t = TF.normalize(night_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        return day_t, night_t


# -------------------------
# Datasets
# -------------------------
class ImageFolderPair(Dataset):
    """Dataset for paired training (day_image, night_image_or_synthetic)
    Expects two directories or a list of pairs.
    Returns: day_tensor (3,H,W), night_tensor (3,H,W)
    """
    def __init__(self, day_dir, night_dir, transform=None, size=(512,512), crop_size=256, load_size=286):
        self.day_paths = sorted(list(Path(day_dir).glob('*.*')))
        self.night_paths = sorted(list(Path(night_dir).glob('*.*')))
        assert len(self.day_paths) == len(self.night_paths), "Paired dataset requires same number of day/night files"
        self.transform = transform
        self.size = size
        self.crop_size = int(crop_size)
        self.load_size = int(load_size)
        self._default_paired_tf = Paired16x9RandomCropTransform(
            crop_size=self.crop_size,
            load_size=self.load_size,
            hflip=True,
        )

    def __len__(self):
        return len(self.day_paths)

    def __getitem__(self, idx):
        day = Image.open(self.day_paths[idx]).convert('RGB')
        night = Image.open(self.night_paths[idx]).convert('RGB')
        if self.transform is not None:
            # 约定：paired transform 必须是 callable(day_img, night_img) -> (day_t, night_t)
            return self.transform(day, night)

        # default paired transform：16:9 中心裁剪 + resize + random crop（对 day/night 同步）
        return self._default_paired_tf(day, night)


class UnpairedFolderDataset(Dataset):
    """Simple wrapper for two folders (A and B) returned unpaired (random pairing in __getitem__)
    """
    def __init__(self, folder_A, folder_B, transform=None, size=(512,512), crop_size=256, load_size=286):
        self.A_paths = sorted(list(Path(folder_A).glob('*.*')))
        self.B_paths = sorted(list(Path(folder_B).glob('*.*')))
        self.transform = transform
        self.size = size
        self.crop_size = int(crop_size)
        self.load_size = int(load_size)

    def __len__(self):
        return max(len(self.A_paths), len(self.B_paths))

    def __getitem__(self, idx):
        a_path = self.A_paths[idx % len(self.A_paths)]
        b_path = random.choice(self.B_paths)
        A = Image.open(a_path).convert('RGB')
        B = Image.open(b_path).convert('RGB')
        if self.transform is None:
            # default unpaired transform：各自独立做 16:9 中心裁剪 + resize + random crop 256
            preprocess = transforms.Compose([
                transforms.Lambda(lambda im: center_crop_to_aspect(im, 16.0 / 9.0)),
                transforms.Resize(self.load_size, interpolation=InterpolationMode.BICUBIC),
                transforms.RandomCrop(self.crop_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
            ])
            A = preprocess(A)
            B = preprocess(B)
        else:
            A = self.transform(A)
            B = self.transform(B)
        return A, B


# -------------------------
# Models: ResNet Generator & PatchGAN Discriminator (CycleGAN standard)
# -------------------------

# Helper conv blocks
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, norm=True, activation='relu'):
        super().__init__()
        layers = [nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding)]
        if norm:
            layers.append(nn.InstanceNorm2d(out_ch))
        if activation == 'relu':
            layers.append(nn.ReLU(True))
        elif activation == 'leaky':
            layers.append(nn.LeakyReLU(0.2, True))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ResnetBlock(nn.Module):
    def __init__(self, dim, norm_layer=nn.InstanceNorm2d):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=0),
            norm_layer(dim),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=0),
            norm_layer(dim),
        )

    def forward(self, x):
        return x + self.conv_block(x)


class ResnetGenerator(nn.Module):
    """Standard ResNet-based generator that accepts `in_channels` input channels (3 or 4)
    """
    def __init__(self, in_channels=3, out_channels=3, ngf=64, n_blocks=9):
        super().__init__()
        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(in_channels, ngf, kernel_size=7, padding=0),
                 nn.InstanceNorm2d(ngf),
                 nn.ReLU(True)]
        # downsampling
        n_down = 2
        mult = 1
        for i in range(n_down):
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                      nn.InstanceNorm2d(ngf * mult * 2),
                      nn.ReLU(True)]
            mult *= 2
        # resnet blocks
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult)]
        # upsampling
        for i in range(n_down):
            # NOTE:
            # - 不使用 ConvTranspose2d（反卷积）以避免常见的 checkerboard/颗粒伪影
            # - 改用 resize-conv：Upsample + Conv2d（更稳、更利于下游检测任务）
            out_ch = int(ngf * mult / 2)
            model += [
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.ReflectionPad2d(1),
                nn.Conv2d(ngf * mult, out_ch, kernel_size=3, stride=1, padding=0),
                nn.InstanceNorm2d(out_ch),
                nn.ReLU(True),
            ]
            mult = int(mult / 2)
        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, out_channels, kernel_size=7, padding=0), nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc=3, ndf=64, n_layers=3):
        super().__init__()
        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
                    nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw),
                         nn.InstanceNorm2d(ndf * nf_mult),
                         nn.LeakyReLU(0.2, True)]
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw),
                     nn.InstanceNorm2d(ndf * nf_mult),
                     nn.LeakyReLU(0.2, True)]
        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]
        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        return self.model(x)


class MultiScaleDiscriminator(nn.Module):
    """
    多尺度判别器（Multi-Scale D）：
    - 对同一张图在多个分辨率上做 PatchGAN 判别
    - 对“大块结构/天空区域/远景纹理”的一致性更敏感，通常可减少贴片/块状崩坏
    """
    def __init__(self, input_nc=3, ndf=64, n_layers=3, num_scales: int = 2):
        super().__init__()
        self.num_scales = max(int(num_scales), 1)
        self.discriminators = nn.ModuleList([
            NLayerDiscriminator(input_nc=input_nc, ndf=ndf, n_layers=n_layers)
            for _ in range(self.num_scales)
        ])
        # 经典 downsample：3x3 avgpool stride=2（更平滑，减少 aliasing）
        self.downsample = nn.AvgPool2d(kernel_size=3, stride=2, padding=1, count_include_pad=False)

    def forward(self, x):
        outs = []
        cur = x
        for i, D in enumerate(self.discriminators):
            outs.append(D(cur))
            if i != self.num_scales - 1:
                cur = self.downsample(cur)
        return outs


# -------------------------
# Losses & training helpers
# -------------------------

def init_weights(net, init_type='normal', init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1 or classname.find('InstanceNorm2d') != -1:
            if hasattr(m, 'weight') and m.weight is not None:
                nn.init.normal_(m.weight.data, 1.0, init_gain)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
    net.apply(init_func)


class ImagePool:
    """
    CycleGAN 经典的“历史假样本缓冲池”（image buffer / replay buffer）。
    作用：让判别器看到“过去生成过的 fake”，抑制 D 过拟合当前 batch 的 fake，
    从而减少训练振荡、块状/贴片伪影与模式崩坏风险。
    """
    def __init__(self, pool_size: int = 50):
        self.pool_size = int(pool_size)
        self.num_imgs = 0
        self.images = []

    def query(self, images):
        """
        Args:
            images: Tensor [B,C,H,W]，一般传入 fake.detach()
        Returns:
            Tensor [B,C,H,W]，部分来自历史池，部分来自当前 fake
        """
        if self.pool_size <= 0:
            return images

        # 保证是 batch 维度
        if images.dim() == 3:
            images = images.unsqueeze(0)

        return_images = []
        for image in images:
            image = image.unsqueeze(0)
            if self.num_imgs < self.pool_size:
                self.images.append(image.detach())
                self.num_imgs += 1
                return_images.append(image)
            else:
                if random.random() > 0.5:
                    idx = random.randint(0, self.pool_size - 1)
                    old = self.images[idx].clone()
                    self.images[idx] = image.detach()
                    return_images.append(old)
                else:
                    return_images.append(image)
        return torch.cat(return_images, dim=0)


def _gan_loss(preds, target_is_real: bool, criterionGAN: nn.Module) -> torch.Tensor:
    """
    支持单尺度判别器输出（Tensor）或多尺度输出（List[Tensor]）。
    返回各尺度损失的平均值。
    """
    if isinstance(preds, (list, tuple)):
        losses = []
        for p in preds:
            tgt = torch.ones_like(p) if target_is_real else torch.zeros_like(p)
            losses.append(criterionGAN(p, tgt))
        return sum(losses) / float(len(losses))
    tgt = torch.ones_like(preds) if target_is_real else torch.zeros_like(preds)
    return criterionGAN(preds, tgt)


def _rgb_to_luma01(x: torch.Tensor) -> torch.Tensor:
    """
    x: [B,3,H,W] in [-1,1] -> Y: [B,1,H,W] in [0,1]
    """
    x01 = (x + 1.0) * 0.5
    r = x01[:, 0:1]
    g = x01[:, 1:2]
    b = x01[:, 2:3]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    return y.clamp(0.0, 1.0)


def _sobel_grad_mag(y: torch.Tensor) -> torch.Tensor:
    """
    y: [B,1,H,W] -> grad magnitude [B,1,H,W]
    使用 Sobel 计算梯度幅值，强调边缘/轮廓一致性（车道线/建筑边/树轮廓）。
    """
    # Sobel kernels (float32); 会自动按输入 dtype 做计算
    kx = torch.tensor([[-1., 0., 1.],
                       [-2., 0., 2.],
                       [-1., 0., 1.]], device=y.device, dtype=y.dtype).view(1, 1, 3, 3)
    ky = torch.tensor([[-1., -2., -1.],
                       [ 0.,  0.,  0.],
                       [ 1.,  2.,  1.]], device=y.device, dtype=y.dtype).view(1, 1, 3, 3)
    gx = F.conv2d(y, kx, padding=1)
    gy = F.conv2d(y, ky, padding=1)
    return torch.sqrt(gx * gx + gy * gy + 1e-12)


def edge_consistency_loss(fake_rgb: torch.Tensor, real_rgb: torch.Tensor, criterionL1: nn.Module) -> torch.Tensor:
    """
    边缘一致性损失：L1(grad(fake_Y), grad(real_Y))
    - 抑制天空乱贴纹理/大块拼贴
    - 保持建筑/绿植轮廓与路面线条
    """
    y_fake = _rgb_to_luma01(fake_rgb)
    y_real = _rgb_to_luma01(real_rgb)
    g_fake = _sobel_grad_mag(y_fake)
    g_real = _sobel_grad_mag(y_real)
    return criterionL1(g_fake, g_real)


# -------------------------
# Training: Unpaired CycleGAN
# -------------------------

def train_unpaired_cyclegan(
    dataset_A_dir,
    dataset_B_dir,
    out_dir,
    device='cuda',
    epochs=200,
    batch_size=1,
    image_size=(256, 256),
    num_workers=4,
    use_amp=True,
    grad_accum_steps=1,
    start_epoch=1,
    resume_G_A2B_path=None,
    resume_G_B2A_path=None,
    resume_D_A_path=None,
    resume_D_B_path=None,
    lr: float = 2e-4,
    cycle_lambda: float = 10.0,
    identity_lambda: float = 0.5,
    pool_size: int = 50,
    d_scales: int = 2,
    lambda_edge: float = 0.0,
):
    """
    Train an unpaired CycleGAN (A <-> B), where A=night, B=day or vice versa.
    This function contains a standard CycleGAN training loop with hyperparameters referenced from paper summary.
    """
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    is_cuda = (device.type == 'cuda')

    # 训练使用“16:9 中心裁剪 + resize(load_size) + random crop(crop_size)”以避免拉伸。
    crop_size = int(image_size[0])
    load_size = max(crop_size + 30, crop_size)
    dataset = UnpairedFolderDataset(
        dataset_A_dir,
        dataset_B_dir,
        transform=None,
        size=image_size,
        crop_size=crop_size,
        load_size=load_size,
    )
    # NOTE:
    # - CycleGAN 通常 batch_size=1 最稳（InstanceNorm + 风格迁移任务常见设置）
    # - 如需更大的“有效 batch”，建议启用 grad_accum_steps（梯度累积）
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=is_cuda,
        persistent_workers=(num_workers > 0),
        drop_last=True,
    )

    # Models
    G_A2B = ResnetGenerator(in_channels=3, out_channels=3).to(device)
    G_B2A = ResnetGenerator(in_channels=3, out_channels=3).to(device)
    if int(d_scales) > 1:
        D_A = MultiScaleDiscriminator(input_nc=3, ndf=64, n_layers=3, num_scales=int(d_scales)).to(device)
        D_B = MultiScaleDiscriminator(input_nc=3, ndf=64, n_layers=3, num_scales=int(d_scales)).to(device)
    else:
        D_A = NLayerDiscriminator(input_nc=3).to(device)
        D_B = NLayerDiscriminator(input_nc=3).to(device)

    # Image pool（replay buffer）：稳定判别器训练，减少贴片/块状伪影
    fake_A_pool = ImagePool(pool_size=pool_size)
    fake_B_pool = ImagePool(pool_size=pool_size)

    # Init or resume
    if resume_G_A2B_path or resume_G_B2A_path or resume_D_A_path or resume_D_B_path:
        # 只恢复网络权重（本脚本不保存 optimizer/scaler 状态，因此严格意义不是“无缝续训”）
        if resume_G_A2B_path:
            G_A2B.load_state_dict(torch.load(resume_G_A2B_path, map_location=device))
        if resume_G_B2A_path:
            G_B2A.load_state_dict(torch.load(resume_G_B2A_path, map_location=device))
        if resume_D_A_path:
            D_A.load_state_dict(torch.load(resume_D_A_path, map_location=device))
        if resume_D_B_path:
            D_B.load_state_dict(torch.load(resume_D_B_path, map_location=device))
    else:
        init_weights(G_A2B); init_weights(G_B2A); init_weights(D_A); init_weights(D_B)

    # Losses
    criterionGAN = nn.MSELoss().to(device)  # LSGAN
    criterionCycle = nn.L1Loss().to(device)
    criterionId = nn.L1Loss().to(device)
    criterionEdge = nn.L1Loss().to(device)

    # Optimizers
    lr = float(lr)
    optimizer_G = optim.Adam(list(G_A2B.parameters()) + list(G_B2A.parameters()), lr=lr, betas=(0.5,0.999))
    optimizer_D_A = optim.Adam(D_A.parameters(), lr=lr, betas=(0.5,0.999))
    optimizer_D_B = optim.Adam(D_B.parameters(), lr=lr, betas=(0.5,0.999))

    # Learning rate schedulers (linear decay after some epochs)
    lambda_lr = lambda epoch: 1.0 - max(0, epoch - epochs//2) / float(epochs//2 + 1)
    scheduler_G = optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=lambda_lr)
    scheduler_D_A = optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=lambda_lr)
    scheduler_D_B = optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=lambda_lr)

    # Labels
    real_label = 1.0
    fake_label = 0.0

    cycle_lambda = float(cycle_lambda)
    id_lambda = float(identity_lambda)
    lambda_edge = float(lambda_edge)

    grad_accum_steps = max(int(grad_accum_steps), 1)
    scaler = GradScaler(enabled=bool(use_amp) and is_cuda)

    start_epoch = max(int(start_epoch), 1)
    for epoch in range(start_epoch, epochs+1):
        loop = tqdm(loader, desc=f'Unpaired Epoch {epoch}/{epochs}', leave=False)
        optimizer_G.zero_grad(set_to_none=True)
        optimizer_D_A.zero_grad(set_to_none=True)
        optimizer_D_B.zero_grad(set_to_none=True)

        for i, (dataA, dataB) in enumerate(loop):
            real_A = dataA.to(device, non_blocking=True)
            real_B = dataB.to(device, non_blocking=True)
            bs = real_A.size(0)
            # ------------------
            #  Train Generators
            # ------------------
            with autocast(enabled=scaler.is_enabled()):
                # Identity loss
                idt_B = G_A2B(real_B)
                loss_idt_B = id_lambda * criterionId(idt_B, real_B) * cycle_lambda  # scaled
                idt_A = G_B2A(real_A)
                loss_idt_A = id_lambda * criterionId(idt_A, real_A) * cycle_lambda

                # GAN loss
                fake_B = G_A2B(real_A)
                pred_fake_B = D_B(fake_B)
                loss_GAN_A2B = _gan_loss(pred_fake_B, True, criterionGAN)

                fake_A = G_B2A(real_B)
                pred_fake_A = D_A(fake_A)
                loss_GAN_B2A = _gan_loss(pred_fake_A, True, criterionGAN)

                # Cycle loss
                rec_A = G_B2A(fake_B)
                rec_B = G_A2B(fake_A)
                loss_cycle_A = criterionCycle(rec_A, real_A)
                loss_cycle_B = criterionCycle(rec_B, real_B)

                loss_G_raw = (
                    loss_GAN_A2B
                    + loss_GAN_B2A
                    + cycle_lambda * (loss_cycle_A + loss_cycle_B)
                    + (loss_idt_A + loss_idt_B)
                )

                # Edge / gradient consistency（可选，默认 0）
                # 让 fake 的边缘响应更贴近对应域的 real，有助于抑制天空贴片、保结构轮廓
                if lambda_edge > 0:
                    loss_edge = edge_consistency_loss(fake_B, real_B, criterionEdge) + edge_consistency_loss(fake_A, real_A, criterionEdge)
                    loss_G_raw = loss_G_raw + (lambda_edge * loss_edge)
                loss_G = loss_G_raw / float(grad_accum_steps)

            scaler.scale(loss_G).backward()

            do_step = ((i + 1) % grad_accum_steps == 0)

            # ------------------
            #  Train Discriminators A and B
            # ------------------
            # D_A
            with autocast(enabled=scaler.is_enabled()):
                pred_real = D_A(real_A)
                loss_D_real = _gan_loss(pred_real, True, criterionGAN)
                fake_A_for_D = fake_A_pool.query(fake_A.detach())
                pred_fake = D_A(fake_A_for_D)
                loss_D_fake = _gan_loss(pred_fake, False, criterionGAN)
                loss_D_A_raw = (loss_D_real + loss_D_fake) * 0.5
                loss_D_A = loss_D_A_raw / float(grad_accum_steps)

            scaler.scale(loss_D_A).backward()

            # D_B
            with autocast(enabled=scaler.is_enabled()):
                pred_real = D_B(real_B)
                loss_D_real = _gan_loss(pred_real, True, criterionGAN)
                fake_B_for_D = fake_B_pool.query(fake_B.detach())
                pred_fake = D_B(fake_B_for_D)
                loss_D_fake = _gan_loss(pred_fake, False, criterionGAN)
                loss_D_B_raw = (loss_D_real + loss_D_fake) * 0.5
                loss_D_B = loss_D_B_raw / float(grad_accum_steps)

            scaler.scale(loss_D_B).backward()
            if do_step:
                scaler.step(optimizer_G)
                scaler.step(optimizer_D_A)
                scaler.step(optimizer_D_B)
                scaler.update()
                optimizer_G.zero_grad(set_to_none=True)
                optimizer_D_A.zero_grad(set_to_none=True)
                optimizer_D_B.zero_grad(set_to_none=True)

            # logging
            loop.set_postfix({'loss_G': loss_G_raw.item(), 'loss_D_A': loss_D_A_raw.item(), 'loss_D_B': loss_D_B_raw.item()})

        # step schedulers
        scheduler_G.step()
        scheduler_D_A.step()
        scheduler_D_B.step()

        # Save checkpoint periodically
        if epoch % 10 == 0:
            torch.save(G_A2B.state_dict(), os.path.join(out_dir, f'G_A2B_epoch{epoch}.pth'))
            torch.save(G_B2A.state_dict(), os.path.join(out_dir, f'G_B2A_epoch{epoch}.pth'))
            torch.save(D_A.state_dict(), os.path.join(out_dir, f'D_A_epoch{epoch}.pth'))
            torch.save(D_B.state_dict(), os.path.join(out_dir, f'D_B_epoch{epoch}.pth'))

    return G_A2B, G_B2A


# -------------------------
# Generate synthetic paired data using trained G (unpaired generator)
# -------------------------

def generate_synthetic_nights(generator_G_A2B, day_dir, out_synth_dir, device='cuda', image_size=(512,512)):
    """
    Use the unpaired generator in *inverse* direction as needed depending on training mapping.
    Here we assume G_A2B maps night->day; to synthesize night from day we need the opposite generator
    (paper used unpaired model to synthesize night images from day images). Adjust accordingly.
    For simplicity: generator_G_day2night maps day -> synthetic_night (if naming follows paper, rename accordingly).
    """
    os.makedirs(out_synth_dir, exist_ok=True)
    transform = transforms.Compose([
        transforms.Resize(image_size),
    ])
    for p in sorted(Path(day_dir).glob('*.*')):
        img = Image.open(p).convert('RGB')
        img_t = transform(img)
        img_np = np.array(img_t)
        # To tensor normalized [-1,1]
        t = transforms.ToTensor()(img_t).unsqueeze(0).to(device)
        t = t * 2.0 - 1.0
        with torch.no_grad():
            # If your generator maps day->night, feed accordingly; adapt name
            fake = generator_G_A2B(t)
            # de-normalize
            fake = (fake.squeeze(0).cpu().clamp(-1,1) + 1.0) / 2.0
            save_path = Path(out_synth_dir) / p.name
            save_image(fake, str(save_path))
    print(f"Saved synthetic nights to: {out_synth_dir}")


def generate_synthetic_nights_from_checkpoint(
    G_day2night_path: str,
    day_dir: str,
    out_synth_dir: str,
    device: str = 'cuda',
    output_size=(1280, 720),
    use_amp: bool = True,
):
    """
    使用训练好的 unpaired 生成器（day->night）把“白天图”合成为“伪夜图”，用于后续 paired(4-channel) 训练。
    - G_day2night_path: unpaired 训练得到的 day->night 权重（建议命名为 G_day2night.pth）
    - day_dir: 白天图文件夹
    - out_synth_dir: 输出伪夜图文件夹
    - output_size: 统一输出分辨率（默认 1280x720，与推理 pipeline 一致）
    """
    os.makedirs(out_synth_dir, exist_ok=True)
    device_t = torch.device(device if torch.cuda.is_available() else 'cpu')
    is_cuda = (device_t.type == 'cuda')

    G = ResnetGenerator(in_channels=3, out_channels=3).to(device_t)
    G.load_state_dict(torch.load(G_day2night_path, map_location=device_t))
    G.eval()

    day_paths = sorted(list(Path(day_dir).glob('*.*')))
    for p in tqdm(day_paths, desc='Synthesizing night from day'):
        pil = Image.open(p).convert('RGB')
        pil_16x9 = center_crop_to_aspect(pil, 16.0 / 9.0)
        pil_resized = pil_16x9.resize(tuple(map(int, output_size)), Image.BICUBIC)
        t = transforms.ToTensor()(pil_resized).unsqueeze(0).to(device_t, non_blocking=is_cuda)
        t = t * 2.0 - 1.0

        with torch.no_grad():
            with autocast(enabled=bool(use_amp) and is_cuda):
                fake = G(t)
                fake = (fake.squeeze(0).detach().float().cpu().clamp(-1, 1) + 1.0) / 2.0

        save_path = Path(out_synth_dir) / p.name
        save_image(fake, str(save_path))

    print(f"Saved synthetic nights to: {out_synth_dir}")


# -------------------------
# Paired Four-channel training
# -------------------------
class FourChannelGenerator(ResnetGenerator):
    def __init__(self, in_channels=4, out_channels=3, ngf=64, n_blocks=9):
        super().__init__(in_channels=in_channels, out_channels=out_channels, ngf=ngf, n_blocks=n_blocks)


def train_paired_four_channel(
    paired_day_dir,
    paired_night_dir,
    out_dir,
    device='cuda',
    epochs=100,
    batch_size=4,
    image_size=(256, 256),
    num_workers=4,
    use_amp=True,
    grad_accum_steps=1,
    lambda_l1=100.0,
    lambda_gan=1.0,
    n_blocks=6,
):
    """
    Train a supervised paired model where input is 4-channel (RGB + inverse RLA) and target is day RGB.
    """
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    is_cuda = (device.type == 'cuda')
    crop_size = int(image_size[0])
    load_size = max(crop_size + 30, crop_size)
    paired_transform = Paired16x9RandomCropTransform(crop_size=crop_size, load_size=load_size, hflip=True)
    dataset = ImageFolderPair(
        paired_day_dir,
        paired_night_dir,
        transform=paired_transform,
        size=image_size,
        crop_size=crop_size,
        load_size=load_size,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=is_cuda,
        persistent_workers=(num_workers > 0),
        drop_last=True,
    )

    G_4c = FourChannelGenerator(in_channels=4, out_channels=3, ngf=64, n_blocks=int(n_blocks)).to(device)
    D = NLayerDiscriminator(input_nc=3).to(device)
    init_weights(G_4c); init_weights(D)

    criterionGAN = nn.MSELoss().to(device)
    criterionL1 = nn.L1Loss().to(device)

    optimizer_G = optim.Adam(G_4c.parameters(), lr=2e-4, betas=(0.5,0.999))
    optimizer_D = optim.Adam(D.parameters(), lr=2e-4, betas=(0.5,0.999))

    lambda_l1 = float(lambda_l1)  # pix2pix 常用 100，数值越大越“保结构/保边缘”，越不易幻觉
    lambda_gan = float(lambda_gan)  # 越小越不追求“逼真风格”，更适合检测任务（减少伪纹理）

    grad_accum_steps = max(int(grad_accum_steps), 1)
    scaler = GradScaler(enabled=bool(use_amp) and is_cuda)

    for epoch in range(1, epochs+1):
        loop = tqdm(loader, desc=f'Paired Epoch {epoch}/{epochs}', leave=False)
        optimizer_G.zero_grad(set_to_none=True)
        optimizer_D.zero_grad(set_to_none=True)
        for i, (day, night) in enumerate(loop):
            day = day.to(device, non_blocking=True)
            night = night.to(device, non_blocking=True)
            bs = day.size(0)
            # Build inverse RLA maps per image and append as 4th channel
            inv_rla_batch = []
            for b in range(bs):
                # convert night[b] tensor to numpy uint8
                np_img = ((night[b].cpu().numpy().transpose(1,2,0) * 0.5 + 0.5) * 255.0).astype(np.uint8)
                bgr = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
                inv_rla = compute_inverse_rla(bgr, bilateral_d=9, bilateral_sigma_color=75, bilateral_sigma_space=75, exp_coef=2.0)
                inv_rla_t = torch.from_numpy(inv_rla).unsqueeze(0).float()  # 1,H,W
                inv_rla_batch.append(inv_rla_t)
            inv_rla_batch = torch.stack(inv_rla_batch, dim=0).to(device, non_blocking=True)
            # 关键：与推理阶段保持一致，把 inv_rla 从 [0,1] 归一化到 [-1,1]
            # 否则第4通道尺度与 night RGB（已在 [-1,1]）不一致，容易导致训练不稳定/颗粒感伪纹理
            inv_rla_batch = inv_rla_batch * 2.0 - 1.0

            # construct 4-channel input: night_rgb + inv_rla
            input_4c = torch.cat([night, inv_rla_batch], dim=1)

            # ------------------
            #  Train G
            # ------------------
            with autocast(enabled=scaler.is_enabled()):
                fake_day = G_4c(input_4c)
                pred_fake = D(fake_day)
                loss_G_GAN = criterionGAN(pred_fake, torch.ones_like(pred_fake))
                loss_G_L1 = criterionL1(fake_day, day)
                loss_G_raw = (lambda_gan * loss_G_GAN) + (lambda_l1 * loss_G_L1)
                loss_G = loss_G_raw / float(grad_accum_steps)
            scaler.scale(loss_G).backward()

            # ------------------
            #  Train D
            # ------------------
            with autocast(enabled=scaler.is_enabled()):
                pred_real = D(day)
                loss_D_real = criterionGAN(pred_real, torch.ones_like(pred_real))
                pred_fake = D(fake_day.detach())
                loss_D_fake = criterionGAN(pred_fake, torch.zeros_like(pred_fake))
                loss_D_raw = (loss_D_real + loss_D_fake) * 0.5
                loss_D = loss_D_raw / float(grad_accum_steps)
            scaler.scale(loss_D).backward()

            do_step = ((i + 1) % grad_accum_steps == 0)
            if do_step:
                scaler.step(optimizer_G)
                scaler.step(optimizer_D)
                scaler.update()
                optimizer_G.zero_grad(set_to_none=True)
                optimizer_D.zero_grad(set_to_none=True)

            loop.set_postfix({'loss_G': loss_G_raw.item(), 'loss_D': loss_D_raw.item()})

        # Save checkpoints
        if epoch % 10 == 0:
            torch.save(G_4c.state_dict(), os.path.join(out_dir, f'G4c_epoch{epoch}.pth'))
            torch.save(D.state_dict(), os.path.join(out_dir, f'D_epoch{epoch}.pth'))

    return G_4c, D


# -------------------------
# Cross-blending fusion (Algorithm 3 summary)
# -------------------------

def cross_blend_results(unpaired_out_rgb: np.ndarray, paired_out_rgb: np.ndarray, night_bgr: np.ndarray, gamma=2.0):
    """
    unpaired_out_rgb, paired_out_rgb: H,W,3 arrays in [0,255] or [0,1]
    night_bgr: original input night image uint8 BGR
    gamma: fusion sharpness (paper used gamma=2)

    returns final_rgb float32 H,W,3 in [0,1]
    """
    # ensure float [0,1]
    def to01(x):
        if x.dtype == np.uint8:
            return x.astype(np.float32) / 255.0
        return x.astype(np.float32)

    up = to01(unpaired_out_rgb)
    pp = to01(paired_out_rgb)

    rla = compute_rla_map(night_bgr)
    mask_p = np.power(rla, gamma)
    mask_p = (mask_p - mask_p.min()) / (1e-8 + (mask_p.max() - mask_p.min()))
    mask_p = np.expand_dims(mask_p, axis=2)
    final = mask_p * pp + (1.0 - mask_p) * up
    final = np.clip(final, 0.0, 1.0)
    return final


# -------------------------
# End-to-end inference pipeline
# -------------------------

def inference_pipeline(
    night_img_path,
    G_unpaired_path,
    G_paired_path,
    device='cuda',
    output_size=(1280, 720),
    gamma=2.0,
    use_amp=True,
):
    """
    Full inference:
    1. Load original night image
    2. Run unpaired generator (night->day) to get global style output
    3. Compute inverse RLA from night image and run paired 4-channel generator to get local-detail output
    4. Cross-blend using gamma
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    is_cuda = (device.type == 'cuda')

    # load image -> center crop to 16:9 -> resize to fixed output_size (default 1280x720)
    pil = Image.open(night_img_path).convert('RGB')
    pil_16x9 = center_crop_to_aspect(pil, 16.0 / 9.0)
    pil_resized = pil_16x9.resize(tuple(map(int, output_size)), Image.BICUBIC)

    np_rgb = np.array(pil_resized)
    np_bgr = cv2.cvtColor(np_rgb, cv2.COLOR_RGB2BGR)

    # Prepare unpaired input (normalized [-1,1])
    t = transforms.ToTensor()(pil_resized).unsqueeze(0).to(device, non_blocking=is_cuda)
    t = t * 2.0 - 1.0

    # Load models
    G_un = ResnetGenerator(in_channels=3, out_channels=3).to(device)
    G_un.load_state_dict(torch.load(G_unpaired_path, map_location=device))
    G_un.eval()

    G_p = FourChannelGenerator(in_channels=4, out_channels=3).to(device)
    G_p.load_state_dict(torch.load(G_paired_path, map_location=device))
    G_p.eval()

    with torch.no_grad():
        with autocast(enabled=bool(use_amp) and is_cuda):
            fake_un = G_un(t)
            fake_un = (fake_un.squeeze(0).detach().float().cpu().clamp(-1, 1) + 1.0) / 2.0  # [0,1]
            fake_un_np = np.transpose((fake_un.numpy() * 255.0).astype(np.uint8), (1, 2, 0))  # H,W,3 RGB

            # Paired input: compute inverse RLA from resized image
            inv_rla = compute_inverse_rla(np_bgr)
            inv_rla_tensor = torch.from_numpy(inv_rla).unsqueeze(0).unsqueeze(0).to(device, non_blocking=is_cuda).float()  # 1,1,H,W
            t_rgb = transforms.ToTensor()(pil_resized).unsqueeze(0).to(device, non_blocking=is_cuda)  # 1,3,H,W in [0,1]
            t_rgb_norm = t_rgb * 2.0 - 1.0
            input4 = torch.cat([t_rgb_norm, inv_rla_tensor * 2.0 - 1.0], dim=1)  # normalize inv rla similarly

            fake_p = G_p(input4)
            fake_p = (fake_p.squeeze(0).detach().float().cpu().clamp(-1, 1) + 1.0) / 2.0
            fake_p_np = np.transpose((fake_p.numpy() * 255.0).astype(np.uint8), (1, 2, 0))  # H,W,3 RGB

    # cross blend
    final = cross_blend_results(fake_un_np, fake_p_np, np_bgr, gamma=gamma)
    final_uint8 = (final * 255.0).astype(np.uint8)
    # final_uint8 已是 RGB
    return Image.fromarray(final_uint8, mode='RGB')


# -------------------------
# Example usage & saving helpers
# -------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Night->Day pipeline (unpaired CycleGAN + 4-channel paired + cross-blend).'
    )
    sub = parser.add_subparsers(dest='cmd')

    # ---- train_unpaired ----
    p_un = sub.add_parser('train_unpaired', help='训练 unpaired CycleGAN（night <-> day）')
    p_un.add_argument('--night_dir', required=True, help='夜间图片文件夹（domain A）')
    p_un.add_argument('--day_dir', required=True, help='白天图片文件夹（domain B）')
    p_un.add_argument('--out_dir', required=True, help='输出 checkpoint 文件夹')
    p_un.add_argument('--epochs', type=int, default=200)
    p_un.add_argument('--batch_size', type=int, default=1)
    p_un.add_argument('--crop_size', type=int, default=256, help='训练 random crop 尺寸（正方形）')
    p_un.add_argument('--num_workers', type=int, default=0, help='Windows 建议先用 0，稳定后再加')
    p_un.add_argument('--device', type=str, default='cuda')
    p_un.add_argument('--lr', type=float, default=1e-4, help='学习率（更稳建议 1e-4；默认 1e-4）')
    p_un.add_argument('--cycle_lambda', type=float, default=10.0, help='Cycle consistency 权重（常用 10）')
    p_un.add_argument('--identity_lambda', type=float, default=1.0, help='Identity 权重（抑制乱编纹理；建议 0.5~1.0）')
    p_un.add_argument('--pool_size', type=int, default=50, help='ImagePool 大小（replay buffer；常用 50~100）')
    p_un.add_argument('--d_scales', type=int, default=2, help='多尺度判别器尺度数（1=关闭；建议 2~3）')
    p_un.add_argument('--lambda_edge', type=float, default=2.0, help='边缘/梯度一致性损失权重（0=关闭；建议 1~5）')
    p_un.add_argument('--no_amp', action='store_true', help='禁用 AMP（混合精度）')
    p_un.add_argument('--grad_accum_steps', type=int, default=1)
    p_un.add_argument('--resume_dir', type=str, default=None, help='可选：从某个 checkpoint 目录继续训练（只恢复权重）')
    p_un.add_argument('--resume_epoch', type=int, default=0, help='配合 --resume_dir，指定要从哪个 epoch 的 checkpoint 继续（例如 20）')

    # ---- synthesize_night ----
    p_syn = sub.add_parser('synthesize_night', help='用 day->night 权重把白天图合成伪夜图')
    p_syn.add_argument('--day_dir', required=True, help='白天图片文件夹')
    p_syn.add_argument('--G_day2night_path', required=True, help='day->night 权重路径（G_day2night.pth）')
    p_syn.add_argument('--out_dir', required=True, help='输出伪夜图文件夹')
    p_syn.add_argument('--device', type=str, default='cuda')
    p_syn.add_argument('--width', type=int, default=1280)
    p_syn.add_argument('--height', type=int, default=720)
    p_syn.add_argument('--no_amp', action='store_true')

    # ---- train_paired ----
    p_paired = sub.add_parser('train_paired', help='训练 paired 4-channel 模型（RGB + inverse RLA -> day）')
    p_paired.add_argument('--paired_day_dir', required=True, help='paired 的 day 目录（目标）')
    p_paired.add_argument('--paired_night_dir', required=True, help='paired 的 night 目录（输入 RGB；可用 synthesize_night 生成）')
    p_paired.add_argument('--out_dir', required=True, help='输出 checkpoint 文件夹')
    p_paired.add_argument('--epochs', type=int, default=100)
    p_paired.add_argument('--batch_size', type=int, default=4)
    p_paired.add_argument('--crop_size', type=int, default=256)
    p_paired.add_argument('--num_workers', type=int, default=0)
    p_paired.add_argument('--device', type=str, default='cuda')
    p_paired.add_argument('--no_amp', action='store_true')
    p_paired.add_argument('--grad_accum_steps', type=int, default=1)
    p_paired.add_argument('--lambda_l1', type=float, default=100.0, help='L1 重建权重（建议 50~200；越大越保结构）')
    p_paired.add_argument('--lambda_gan', type=float, default=0.1, help='GAN 权重（检测任务建议更小，如 0~0.3）')
    p_paired.add_argument('--n_blocks', type=int, default=6, help='生成器 ResNet blocks（默认 6；更大更强但更慢）')

    # ---- infer ----
    p_inf = sub.add_parser('infer', help='单张图片推理（night->day）')
    p_inf.add_argument('--night_img', required=True, help='夜间输入图片路径')
    p_inf.add_argument('--G_night2day_path', required=True, help='unpaired 的 night->day 权重路径（G_night2day.pth）')
    p_inf.add_argument('--G_paired_path', required=True, help='paired 4-channel 权重路径（G4c_last.pth 或 G4c_epochXX.pth）')
    p_inf.add_argument('--out_img', required=True, help='输出图片路径（png/jpg）')
    p_inf.add_argument('--device', type=str, default='cuda')
    p_inf.add_argument('--width', type=int, default=1280)
    p_inf.add_argument('--height', type=int, default=720)
    p_inf.add_argument('--gamma', type=float, default=2.0)
    p_inf.add_argument('--no_amp', action='store_true')

    args = parser.parse_args()

    if args.cmd is None:
        print('This module provides functions to train and run the pipeline.\n')
        print('Typical steps:')
        print('1) python night_2_day_pipeline.py train_unpaired --night_dir ... --day_dir ... --out_dir ...')
        print('2) python night_2_day_pipeline.py synthesize_night --day_dir ... --G_day2night_path ... --out_dir ...')
        print('3) python night_2_day_pipeline.py train_paired --paired_day_dir ... --paired_night_dir ... --out_dir ...')
        print('4) python night_2_day_pipeline.py infer --night_img ... --G_night2day_path ... --G_paired_path ... --out_img ...')
        raise SystemExit(0)

    if args.cmd == 'train_unpaired':
        # A=night, B=day  =>  G_A2B: night->day, G_B2A: day->night
        resume_kwargs = {}
        if args.resume_dir and int(args.resume_epoch) > 0:
            rdir = Path(args.resume_dir)
            rep = int(args.resume_epoch)
            resume_kwargs = dict(
                start_epoch=rep + 1,
                resume_G_A2B_path=str(rdir / f'G_A2B_epoch{rep}.pth'),
                resume_G_B2A_path=str(rdir / f'G_B2A_epoch{rep}.pth'),
                resume_D_A_path=str(rdir / f'D_A_epoch{rep}.pth'),
                resume_D_B_path=str(rdir / f'D_B_epoch{rep}.pth'),
            )
            # 简单检查文件存在性（友好报错）
            for k, pth in resume_kwargs.items():
                if k.startswith("resume_") and (not Path(pth).exists()):
                    raise FileNotFoundError(f"resume checkpoint not found: {pth}")

        G_night2day, G_day2night = train_unpaired_cyclegan(
            dataset_A_dir=args.night_dir,
            dataset_B_dir=args.day_dir,
            out_dir=args.out_dir,
            device=args.device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            image_size=(args.crop_size, args.crop_size),
            num_workers=args.num_workers,
            use_amp=(not args.no_amp),
            grad_accum_steps=args.grad_accum_steps,
            lr=args.lr,
            cycle_lambda=args.cycle_lambda,
            identity_lambda=args.identity_lambda,
            pool_size=args.pool_size,
            d_scales=args.d_scales,
            lambda_edge=args.lambda_edge,
            **resume_kwargs,
        )
        # 额外保存“更语义化”的最终权重名，方便后续步骤引用
        os.makedirs(args.out_dir, exist_ok=True)
        torch.save(G_night2day.state_dict(), os.path.join(args.out_dir, 'G_night2day.pth'))
        torch.save(G_day2night.state_dict(), os.path.join(args.out_dir, 'G_day2night.pth'))
        print(f"Saved final unpaired generators to: {args.out_dir}")

    elif args.cmd == 'synthesize_night':
        generate_synthetic_nights_from_checkpoint(
            G_day2night_path=args.G_day2night_path,
            day_dir=args.day_dir,
            out_synth_dir=args.out_dir,
            device=args.device,
            output_size=(args.width, args.height),
            use_amp=(not args.no_amp),
        )

    elif args.cmd == 'train_paired':
        G4c, D = train_paired_four_channel(
            paired_day_dir=args.paired_day_dir,
            paired_night_dir=args.paired_night_dir,
            out_dir=args.out_dir,
            device=args.device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            image_size=(args.crop_size, args.crop_size),
            num_workers=args.num_workers,
            use_amp=(not args.no_amp),
            grad_accum_steps=args.grad_accum_steps,
            lambda_l1=args.lambda_l1,
            lambda_gan=args.lambda_gan,
            n_blocks=args.n_blocks,
        )
        os.makedirs(args.out_dir, exist_ok=True)
        torch.save(G4c.state_dict(), os.path.join(args.out_dir, 'G4c_last.pth'))
        torch.save(D.state_dict(), os.path.join(args.out_dir, 'D_last.pth'))
        print(f"Saved final paired models to: {args.out_dir}")

    elif args.cmd == 'infer':
        out_img = inference_pipeline(
            night_img_path=args.night_img,
            G_unpaired_path=args.G_night2day_path,
            G_paired_path=args.G_paired_path,
            device=args.device,
            output_size=(args.width, args.height),
            gamma=args.gamma,
            use_amp=(not args.no_amp),
        )
        out_path = Path(args.out_img)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_img.save(str(out_path))
        print(f"Saved output to: {args.out_img}")


# End of file
