# genvideopipeline — Local Video Generation on Mac M2 16G (4-Stage Split)

Personal experiment: running 5B / 1.3B video diffusion models locally on an Apple Silicon Mac with **16GB unified memory** by splitting the pipeline into **4 sequential stages**. Method notes only, for reference.

## Why "4-stage"

A monolithic video generation pipeline (text encoding → VAE encode → DiT sampling → VAE decode) peaks above **30GB** — fatal on a 16GB machine. The 4-stage split runs 4 **independent processes strictly serially**, each loading only what it needs and releasing memory on exit, so memory never stacks: peak drops to **8–12GB**.

## Pipeline 1: Wan2.2-TI2V-5B (sd-cli + official python VAE)

```
Stage 1  T5 encode     sd-cli --save-text-embedding (T5 is image-independent)
Stage 2  VAE encode    python scripts/wan22_vae_encode.py (official Wan2.2 VAE, 16x normalized latent)
Stage 3  DiT sampling  sd-cli --load-init-latent + --save-latent
Stage 4  decode        python scripts/wan22_vae_decode.py (full VAE, final quality) | sd-cli TAE (quick check)
```

- Script: `scripts/runwan.sh <workdir> <prompt> <start_image>`
- Peak memory ~12GB; verified at 320×448 / 49 frames / 30 steps.
- Details: `docs/wan22-4stage.md` (Chinese).

**Two verified rules**:
1. **Latent spaces must pair**: full-VAE↔full-VAE, TAE↔TAE. Mixing produces mosaic artifacts.
2. **sd-cpp's DiT only accepts TAE latents** (official VAE latents mismatch → corrupted output). For full-VAE quality, the DiT must also run on the official implementation (python torch or MLX).

## Pipeline 2: LingBot-World-V2 1.3B (MLX 30-layer DiT)

```
Stage 1  VAE encode    input image → input latent (.pt)
Stage 2  DiT sampling  MLX 30-layer forward (bf16 weights 2.6GB, no VAE/T5 loaded) → latent.bin
Stage 3  VAE decode    lingbot-mlx/decode_latent.py (Wan2.1 VAE only) → mp4
Stage 4  mux           ffmpeg
```

- Script: `scripts/runlingbot.sh <workdir> <prompt> [start_image]`
- Peak memory **30GB → 8GB**; ~3min total (9 frames 320×432); output **bit-identical** to the monolithic run.
- Details: `docs/lingbot-mlx.md` (Chinese).

## Memory & parameter experience

| Rule | Finding |
|---|---|
| What drives memory | **Total latent elements (frames × resolution)** — independent of sampling steps |
| Resolution ceiling | M2 Metal silently corrupts latents at ≥480×832; 384×640 is the verified safe max |
| Frame floor | <9 frames glitches (tearing); start at 9 |
| Out-of-memory | Triggers swap → machine freezes. Watch Activity Monitor while probing |
| Probe order | frames (9→17→25→33) → resolution → steps (10/20/30, quality/speed only) |

## Models & tools (official links)

| Component | Source |
|---|---|
| Wan2.2 code | https://github.com/Wan-Video/Wan2.2 |
| Wan2.2-TI2V-5B weights | https://modelscope.cn/models/Wan-AI/Wan2.2-TI2V-5B |
| stable-diffusion.cpp (sd-cli) | https://github.com/leejet/stable-diffusion.cpp |
| TAEHV (Wan2.1 TAE) | https://github.com/madebyollin/taehv |
| LingBot-World-V2 code | https://github.com/robbyant/lingbot-world-v2 |
| LingBot 1.3B weights | https://modelscope.cn/models/Robbyant/lingbot-world-v2-1.3b-causal-fast |
| Apple MLX | https://github.com/ml-explore/mlx |
| ComfyUI-Ovi (Wan2.2 VAE reference) | https://github.com/snicolast/ComfyUI-Ovi |

Model weights (GGUF/safetensors) are large — download them yourself from the sources above. **This repo contains no weights, sample images, credentials, or personal data**; provide your own start image.

## Layout

```
genvideopipeline/
├── README.md / README.en.md
├── LICENSE
├── docs/
│   ├── wan22-4stage.md      # Wan2.2-TI2V-5B 4-stage conclusions & fixes
│   └── lingbot-mlx.md       # LingBot MLX 4-stage record
├── scripts/
│   ├── runwan.sh            # Wan2.2-TI2V-5B 4-stage
│   ├── runlingbot.sh        # LingBot MLX 4-stage
│   ├── wan22_vae_encode.py
│   ├── wan22_vae_decode.py
│   └── wan22_vae2_2_official.py  # Wan2.2 VAE implementation (Alibaba Wan Team copyright)
└── lingbot-mlx/
    ├── lingbot_mlx.py       # 30-layer DiT MLX implementation
    └── decode_latent.py     # VAE-only decode
```
