"""Decode-only stage (段 2): load Wan2.1 VAE + SDLT latent bin -> mp4.

Usage:
    FORCE_CPU=1 python3 decode_latent.py --latent /tmp/mlx_lat.bin \
        --vae ../models/lingbot-world-v2-assets/Wan2.1_VAE.pth \
        --fps 16 -o /tmp/out.mp4
"""
import argparse
import os
import struct

import numpy as np
import torch

from wan.modules.vae2_1 import Wan2_1_VAE
from wan.utils.utils import save_video


def load_sdlt(path):
    with open(path, 'rb') as f:
        magic = f.read(4)
        assert magic == b'SDLT', f'bad magic {magic}'
        ver = struct.unpack('<i', f.read(4))[0]
        dim = struct.unpack('<i', f.read(4))[0]
        shape = struct.unpack('<' + 'i' * dim, f.read(4 * dim))
        data = np.frombuffer(f.read(), dtype=np.float32)
    return ver, shape, data.reshape(shape)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--latent', required=True)
    ap.add_argument('--vae', required=True)
    ap.add_argument('--fps', type=int, default=16)
    ap.add_argument('-o', '--output', required=True)
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    vae = Wan2_1_VAE(vae_pth=args.vae, device=device)

    ver, shape, lat = load_sdlt(args.latent)
    print(f'[decode] latent ver={ver} shape={shape}', flush=True)
    # [W,H,T,C] or [W,H,T,C,1] -> [C,T,H,W]
    if lat.ndim == 5:
        lat = lat[..., 0]
    lat = np.transpose(lat, (3, 2, 1, 0))  # [C,T,H,W]
    lat_t = torch.from_numpy(np.ascontiguousarray(lat)).to(device).float()
    print(f'[decode] VAE input {tuple(lat_t.shape)}', flush=True)
    with torch.no_grad():
        videos = vae.decode([lat_t])
    v = videos[0]
    print(f'[decode] decoded {tuple(v.shape)}', flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or '.', exist_ok=True)
    save_video(tensor=v[None], save_file=args.output, fps=args.fps,
               nrow=1, normalize=True, value_range=(-1, 1))
    print(f'[decode] saved {args.output}', flush=True)


if __name__ == '__main__':
    main()
