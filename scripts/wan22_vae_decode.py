#!/usr/bin/env python3
"""
Wan2.2 VAE decode（python / torch，官方 ComfyUI-Ovi Wan2_2_VAE 类）
用途: runwan.sh 四段式的「VAE decode」段 —— latent -> 视频帧
注意: 完整 Wan2.2 VAE latent 与 TAE latent 空间不兼容 (TAE decode 官方 VAE latent = 马赛克)。
     本脚本与 wan22_vae_encode.py (官方类) 配套使用, 构成完整官方 VAE 管线。

用法:
  python wan22_vae_decode.py --latent <latent.bin> --vae models/vae/Wan2.2_VAE.safetensors \
      --out /path/frames_dir [--width 320 --height 448 --frames 49]

输出: out 目录下 frame_%04d.png (0-255 RGB)
"""
import argparse
import os
import struct
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wan22_vae2_2_official import Wan2_2_VAE


def read_sdlt(path):
    d = open(path, 'rb').read()
    dim = struct.unpack('<i', d[8:12])[0]
    shape = struct.unpack('<' + 'i' * dim, d[12:12 + 4 * dim])
    n = int(np.prod(shape))
    arr = np.frombuffer(d, dtype=np.float32, offset=12 + 4 * dim, count=n).copy()
    return shape, arr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--latent', required=True)
    ap.add_argument('--vae', default='models/vae/Wan2.2_VAE.safetensors')
    ap.add_argument('--out', required=True)
    ap.add_argument('--device', default='cpu')
    args = ap.parse_args()

    shape, arr = read_sdlt(args.latent)
    # SDLT [W,H,T,C,1] -> [C,T,H,W]
    z = torch.from_numpy(arr).reshape(shape[0], shape[1], shape[2], shape[3])
    z = z.permute(3, 2, 0, 1)  # [C,T,H,W]
    print('latent [W,H,T,C] =', shape, '-> [C,T,H,W]', z.shape)

    vae = Wan2_2_VAE(z_dim=48, c_dim=160, vae_pth=args.vae,
                     dtype=torch.float32, device=args.device)
    with torch.no_grad():
        vids = vae.decode([z])  # [C,T',H',W'] (T' 时间上采样, 官方 4x)
    x = vids[0].float().clamp_(-1, 1)  # [C,T,H,W]
    print('decoded', x.shape, 'range %.2f %.2f' % (x.min().item(), x.max().item()))

    os.makedirs(args.out, exist_ok=True)
    c, t, h, w = x.shape
    x = (x + 1) / 2
    for i in range(t):
        img = x[:, i].permute(1, 2, 0).numpy()
        Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).save(
            os.path.join(args.out, 'frame_%04d.png' % i))
    print('saved %d frames to %s' % (t, args.out))


if __name__ == '__main__':
    main()
