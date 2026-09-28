#!/usr/bin/env python3
"""
Wan2.2 VAE encode（python / torch 实现）
用途: run.sh 四段式改造的「VAE encode」段 —— 输入图 -> 48通道 latent (SDLT 格式)
避开 sd-cpp 内部 Wan2.2 encode 的马赛克 bug。

用法:
  python wan22_vae_encode.py --image <png/jpg> --vae models/vae/Wan2.2_VAE.safetensors \
      --size 320x448 --out /path/init_latent.bin [--frames 49]

输出: SDLT 格式 [W,H,1,C,1] f32 (ver=24, dim=5, shape=[W,H,1,C,1])
      T 帧复制由 sd-cpp 侧完成 (与 encode_first_stage 路径一致)。

结构依据 (与 sd-cpp wan_vae.hpp / 权重 key 对齐):
  conv1(96->96,1x1x1) conv2(48->48,1x1x1)
  encoder: conv1(12->160) + downsamples(4段:2xResidual+Resample, 空间16x, 时间2次3d) + middle(Residual+Attn+Residual) + head(->96)
  decoder: conv1(48->1024) + middle + upsamples(4段) + head(->12)
"""
import argparse
import struct
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lingbot-world-v2'))
from wan.modules.vae2_1 import CausalConv3d, RMS_norm, ResidualBlock, AttentionBlock, Resample, Upsample

CACHE_T = 4


class DownResidualBlock(nn.Module):
    """段容器: num_res_blocks 个 ResidualBlock + 可选 Resample (key: downsamples.N.downsamples.M.*)"""

    def __init__(self, in_dim, out_dim, num_res_blocks, down_flag, temperal_downsample, dropout=0.0):
        super().__init__()
        blocks = []
        for _ in range(num_res_blocks):
            blocks.append(ResidualBlock(in_dim, out_dim, dropout))
            in_dim = out_dim
        if down_flag:
            mode = 'downsample3d' if temperal_downsample else 'downsample2d'
            blocks.append(Resample(out_dim, mode=mode))
        self.downsamples = nn.Sequential(*blocks)


class Wan22UpsampleResample(nn.Module):
    """Wan2.2 upsample Resample: 空间2x + Conv(dim->dim) + time_conv (dim->dim*2) [key: resample.1.* / time_conv.*]"""

    def __init__(self, dim, mode):
        super().__init__()
        assert mode in ('upsample2d', 'upsample3d')
        self.dim = dim
        self.mode = mode
        if mode == 'upsample2d':
            self.resample = nn.Sequential(
                Upsample(scale_factor=(2., 2.), mode='nearest-exact'),
                nn.Conv2d(dim, dim, 3, padding=1))
        else:
            self.resample = nn.Sequential(
                Upsample(scale_factor=(2., 2.), mode='nearest-exact'),
                nn.Conv2d(dim, dim, 3, padding=1))
            self.time_conv = CausalConv3d(dim, dim * 2, (3, 1, 1), padding=(1, 0, 0))

    def forward(self, x, feat_cache=None, feat_idx=None):
        if self.mode == 'upsample2d':
            # 2D 空间上采样 (5D): 空间 2x, 时间不变; Conv2d 需合并 T 到 batch
            b, c, t, h, w = x.shape
            x = F.interpolate(x, scale_factor=(1, 2, 2), mode='nearest')
            x = x.transpose(1, 2).reshape(b * t, c, h * 2, w * 2)
            x = self.resample[1](x)  # Conv2d
            x = x.reshape(b, t, c, h * 2, w * 2).transpose(1, 2)
        else:
            # upsample3d: 先空间 2x (2D Conv), 再时间 2x
            b, c, t, h, w = x.shape
            x = F.interpolate(x, scale_factor=(1, 2, 2), mode='nearest')
            x = x.transpose(1, 2).reshape(b * t, c, h * 2, w * 2)
            x = self.resample[1](x)  # Conv2d
            x = x.reshape(b, t, c, h * 2, w * 2).transpose(1, 2)
            x = F.interpolate(x, scale_factor=(2, 1, 1), mode='trilinear', align_corners=False)
            x = self.time_conv(x)
        return x


class UpResidualBlock(nn.Module):
    """decoder 段容器: num_res_blocks+1 个 ResidualBlock + 可选 Resample (key: upsamples.N.upsamples.M.*)"""

    def __init__(self, in_dim, out_dim, num_res_blocks, up_flag, temperal_upsample, dropout=0.0):
        super().__init__()
        blocks = []
        for _ in range(num_res_blocks + 1):
            blocks.append(ResidualBlock(in_dim, out_dim, dropout))
            in_dim = out_dim
        if up_flag:
            mode = 'upsample3d' if temperal_upsample else 'upsample2d'
            blocks.append(Wan22UpsampleResample(in_dim, mode=mode))
        self.upsamples = nn.Sequential(*blocks)


class Encoder3d2(nn.Module):
    def __init__(self, dim=160, z_dim=48, in_channels=12, dim_mult=None,
                 num_res_blocks=2, temperal_downsample=None, dropout=0.0):
        super().__init__()
        if dim_mult is None:
            dim_mult = [1, 2, 4, 4]
        if temperal_downsample is None:
            temperal_downsample = [False, True, True, False]
        self.dim_mult = dim_mult
        self.temperal_downsample = temperal_downsample

        dims = [dim * u for u in [1] + dim_mult]
        self.conv1 = CausalConv3d(in_channels, dims[0], 3, padding=1)

        downsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            # 空间 down 3 次 (8x), 时间 3d down 2 次 (4x)
            down_flag = (i != len(dim_mult) - 1)
            t_down = temperal_downsample[i] if i < len(temperal_downsample) else False
            downsamples.append(DownResidualBlock(
                in_dim, out_dim, num_res_blocks,
                down_flag=down_flag, temperal_downsample=t_down, dropout=dropout))
        self.downsamples = nn.Sequential(*downsamples)

        out_dim = dims[-1]
        self.middle = nn.Sequential(
            ResidualBlock(out_dim, out_dim, dropout),
            AttentionBlock(out_dim),
            ResidualBlock(out_dim, out_dim, dropout))
        self.head = nn.Sequential(
            RMS_norm(out_dim, images=False), nn.SiLU(),
            CausalConv3d(out_dim, z_dim * 2, 3, padding=1))

    def forward(self, x, feat_cache=None, feat_idx=None):
        x = self.conv1(x)
        for m in self.downsamples:
            for b in m.downsamples:
                x = b(x)
        for b in self.middle:
            x = b(x)
        for b in self.head:
            if isinstance(b, CausalConv3d):
                x = b(x)
            else:
                x = b(x)
        return x


class Decoder3d2(nn.Module):
    def __init__(self, dim=256, z_dim=48, out_channels=12, dim_mult=None,
                 num_res_blocks=2, temperal_upsample=None, dropout=0.0):
        super().__init__()
        if dim_mult is None:
            dim_mult = [1, 2, 4, 4]
        if temperal_upsample is None:
            temperal_upsample = [True, True, False, False]
        self.dim_mult = dim_mult
        self.temperal_upsample = temperal_upsample

        dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]
        self.conv1 = CausalConv3d(z_dim, dims[0], 3, padding=1)

        self.middle = nn.Sequential(
            ResidualBlock(dims[0], dims[0], dropout),
            AttentionBlock(dims[0]),
            ResidualBlock(dims[0], dims[0], dropout))

        upsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            t_up = temperal_upsample[i] if i < len(temperal_upsample) else False
            upsamples.append(UpResidualBlock(
                in_dim, out_dim, num_res_blocks,
                up_flag=(i != len(dim_mult) - 1), temperal_upsample=t_up, dropout=dropout))
        self.upsamples = nn.Sequential(*upsamples)

        out_dim = dims[-1]
        self.head = nn.Sequential(
            RMS_norm(out_dim, images=False), nn.SiLU(),
            CausalConv3d(out_dim, out_channels, 3, padding=1))

    def forward(self, x, feat_cache=None, feat_idx=None):
        x = self.conv1(x)
        for b in self.middle:
            x = b(x)
        for m in self.upsamples:
            for b in m.upsamples:
                x = b(x)
        for b in self.head:
            if isinstance(b, CausalConv3d):
                x = b(x)
            else:
                x = b(x)
        return x


# 官方 Wan2.2 VAE per-channel latent 归一化 (ComfyUI-Ovi ovi/modules/vae2_2.py L995-1103)
WAN22_LATENT_MEAN = [
    -0.2289, -0.0052, -0.1323, -0.2339, -0.2799, 0.0174, 0.1838, 0.1557,
    -0.1382, 0.0542, 0.2813, 0.0891, 0.1570, -0.0098, 0.0375, -0.1825,
    -0.2246, -0.1207, -0.0698, 0.5109, 0.2665, -0.2108, -0.2158, 0.2502,
    -0.2055, -0.0322, 0.1109, 0.1567, -0.0729, 0.0899, -0.2799, -0.1230,
    -0.0313, -0.1649, 0.0117, 0.0723, -0.2839, -0.2083, -0.0520, 0.3748,
    0.0152, 0.1957, 0.1433, -0.2944, 0.3573, -0.0548, -0.1681, -0.0667]
WAN22_LATENT_STD = [
    0.4765, 1.0364, 0.4514, 1.1677, 0.5313, 0.4990, 0.4818, 0.5013,
    0.8158, 1.0344, 0.5894, 1.0901, 0.6885, 0.6165, 0.8454, 0.4978,
    0.5759, 0.3523, 0.7135, 0.6804, 0.5833, 1.4146, 0.8986, 0.5659,
    0.7069, 0.5338, 0.4889, 0.4917, 0.4069, 0.4999, 0.6866, 0.4093,
    0.5709, 0.6065, 0.6415, 0.4944, 0.5726, 1.2042, 0.5458, 1.6887,
    0.3971, 1.0600, 0.3943, 0.5537, 0.5444, 0.4089, 0.7468, 0.7744]


class Wan22VAE(nn.Module):
    def __init__(self, dim=160, dec_dim=256, z_dim=48, in_channels=12, out_channels=12):
        super().__init__()
        self.z_dim = z_dim
        self.encoder = Encoder3d2(dim, z_dim, in_channels)
        self.conv1 = CausalConv3d(z_dim * 2, z_dim * 2, 1)
        self.conv2 = CausalConv3d(z_dim, z_dim, 1)
        self.decoder = Decoder3d2(dec_dim, z_dim, out_channels)
        self.register_buffer('_mean', torch.tensor(WAN22_LATENT_MEAN).view(1, z_dim, 1, 1, 1), persistent=False)
        self.register_buffer('_inv_std', torch.tensor([1.0 / s for s in WAN22_LATENT_STD]).view(1, z_dim, 1, 1, 1), persistent=False)

    def encode(self, x, scale=1.0):
        """x: [B,C,T,H,W] float32, 0-1 -> latent [B,48,T,H/16,W/16] (确定性, 与官方一致: 直接返回归一化 mu)"""
        x = x * scale
        h = self.encoder(x)
        h = self.conv1(h)
        mu, _ = h.chunk(2, dim=1)
        mu = (mu - self._mean) * self._inv_std
        return mu

    def decode(self, z, scale=1.0):
        z = z * self._std + self._mean
        h = self.conv2(z)
        h = self.decoder(h)
        return h / scale


def load_vae(path, device='cpu'):
    from safetensors import safe_open
    model = Wan22VAE()
    state = {}
    with safe_open(path, framework='pt', device=device) as f:
        for k in f.keys():
            state[k] = f.get_tensor(k)
    missing, unexpected = model.load_state_dict(state, strict=True)
    if missing:
        print('MISSING:', missing[:10], '...' if len(missing) > 10 else '')
        raise SystemExit('state_dict missing keys')
    if unexpected:
        print('UNEXPECTED:', unexpected[:10], '...' if len(unexpected) > 10 else '')
        raise SystemExit('state_dict unexpected keys')
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--image', required=True)
    ap.add_argument('--vae', default='models/vae/Wan2.2_VAE.safetensors')
    ap.add_argument('--size', default='320x448')
    ap.add_argument('--out', required=True)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--scale', type=float, default=1.0)
    ap.add_argument('--downsample', type=int, default=1,
                    help='deprecated: kept for compatibility; structure now produces native 16x latent')
    ap.add_argument('--no-sample', action='store_true', help='确定性: 不采样, 直接用 mu')
    args = ap.parse_args()

    W, H = [int(x) for x in args.size.lower().split('x')]
    img = Image.open(args.image).convert('RGB').resize((W, H), Image.LANCZOS)
    arr = np.asarray(img, dtype=np.float32) / 255.0  # [H,W,3]
    x = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(1)  # [3,1,H,W] (官方类内部 patchify, 无 batch 维)

    device = args.device
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from wan22_vae2_2_official import Wan2_2_VAE
    vae = Wan2_2_VAE(z_dim=48, c_dim=160, vae_pth=args.vae, dtype=torch.float32, device=device)
    with torch.no_grad():
        z = vae.encode([x.to(device)])[0]  # [C=48,T,Hh,Ww] 归一化 mu (官方确定性 encode)
    z = z.cpu().numpy()  # [C=48, T=1, Hh, Ww]

    # latent 布局 [W,H,T,C] -> [Ww, Hh, 1, 48]
    C, T, Hh, Ww = z.shape
    zt = z.transpose(3, 2, 1, 0)  # [Ww, Hh, T, C]
    print(f'latent shape [W,H,T,C] = {zt.shape}, range [{zt.min():.3f}, {zt.max():.3f}], '
          f'nan={np.isnan(zt).sum()}, inf={np.isinf(zt).sum()}')

    # SDLT 5D: ver=24, dim=5, shape=[W,H,T,C,1]
    sdlt = struct.pack('<4sii', b'SDLT', 24, 5) + struct.pack('<5i', *zt.shape, 1)
    sdlt += np.ascontiguousarray(zt, dtype=np.float32).tobytes()
    with open(args.out, 'wb') as f:
        f.write(sdlt)
    print(f'saved {args.out} ({os.path.getsize(args.out)} bytes)')


if __name__ == '__main__':
    main()
