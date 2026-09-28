# LingBot-World-V2 1.3B 四段式（MLX 路线）记录

> 目标：在 Mac M2 16G 上本地跑通 LingBot-World-V2 1.3B causal-fast 的 i2v 视频生成。
> 官方 torch `generate.py` 在 MPS 上反复崩溃/死机后，改走自研 **MLX 并行路线**：
> 用 Apple MLX（Metal 原生、bf16）重写 30 层 DiT 主体，其余输入资产复用官方已生成数据。
> 硬约束：每层网络与官方 Equal 对齐，禁止近似替代、禁止改网络结构。

## 1. 背景：为什么放弃 sd-cpp 路线

- 早期 sd-cpp 移植（fork + causal rope + c2ws）出现 NaN，根因指向 ggml gallocr 分配器的中间 buffer 复用（同一代码跨时间：一次干净、一次全 NaN，不可稳定复现）。
- 修复方案（每层输出独立 buffer）不具可复现性；用户拍板开 **MLX 并行路线**止损。
- MLX 无 gallocr，bf16 原生支持，是 Metal 上的原生框架。

## 2. 四段式架构（scripts/runlingbot.sh）

```
段1  VAE encode  官方 encode 后存输入 latent（y=msk+latent）到 .pt，exit
段2  DiT 采样    MLX 30 层 forward（不加载 VAE/T5，从文件读输入 latent）→ SDLT bin
段3  VAE decode  lingbot-mlx/decode_latent.py（只加载 Wan2.1 VAE）→ mp4
段4  封装         ffmpeg
```

- 每段独立进程、串行执行、内存不叠加。
- **实测：峰值内存 30G → 8G；总时长 ~3min（39s encode + 71s DiT + ~70s decode，9 帧 320×432）；输出与完整流程 bit 一致（266983B）。**
- T5 文本嵌入使用预计算的 `output/embeds_17f`（如需重新生成，先单独跑 T5 段）。

## 3. 复用/重写边界

- **不重写（官方 forward 内 dump 成 npy，直接喂 MLX）**：T5 文本嵌入（inp_ctx）、c2ws 预处理后嵌入（inp_c2ws）、VAE encode+patch（inp_x）、time embedding/projection（inp_e/inp_e2）。
- **重写（lingbot-mlx/lingbot_mlx.py，~350 行）**：30 层 CausalWanAttentionBlock + CausalHead + unpatchify，对齐官方 golden（逐层 relL1 ≈ 6e-6，f32）。
- **解码**：官方 CPU Wan2.1 VAE（lingbot-mlx/decode_latent.py）。

## 4. 已确认的网络事实（实现时必须遵守）

- `num_heads=12, head_dim=128`（wan/configs/wan_i2v_1_3B.py：dim=1536, ffn_dim=8960, num_layers=30, cross_attn_norm=True）。
- **RMSNorm 应用在 view 成 head 之前**（qkv Linear → RMSNorm 全 dim → reshape [B,L,12,128]）。
- **norm1/norm2 是无 affine 的 LayerNorm**（权重里没有 norm1/norm2.weight）；norm3 有 affine（cross_attn_norm=True）。
- **cam 注入激活是 SiLU**（不是 GELU）——cam_injector_layer1 → SiLU → cam_injector_layer2 → +c2ws 残差 → cam_scale/shift。
- **官方 CPU_F32 路径 self/cross-attn 都是无 mask 全注意力**（SDPA is_causal=False）——非 CUDA 路径。
- **rope**：freqs = cat([rope_params(1024,44), rope_params(1024,42), rope_params(1024,42)])，causal_rope_apply split [22,21,21] 复数段，grid=(F,H',W')（patch 后）=(3,17,31)，token 序 w' 快。
- patch_size=(1,2,2)（T 不 patch）。

## 5. 优化项（2026-09-17 落地）

1. **bf16 权重**（MLX_BF16=1，5.2GB → 2.6GB）：每步 4-9s，全流程 ~2min14s，误差复测 relL1 ≈ 0.6%（可接受）。
2. **加载时过滤 blocks/head 权重**（官方 `_load_safetensors_state_dict` 在 MLX_BACKEND 下丢弃 `blocks.*`/`head.head.*` key）：省 ~6GB，且从加载源头就不进内存。
3. **四段式生成**（见上）。
4. **sd-cpp VAE Metal decode 弃用**：WanVAE decode 350s 且只输出单帧；TAE 是 Wan2.2 的（latent 12 通道）≠ lingbot Wan2.1（16 通道），通道不匹配。decode 保留官方 CPU（~70s/9帧）。

## 6. 运行命令（官方 generate.py + MLX_BACKEND 开关）

> 推荐直接使用 `scripts/runlingbot.sh`（会自动设置 `LINGBOT_INDEX` 指向 MLX 权重索引）。以下为等价的手动命令：

```bash
# 段 1：VAE encode
(cd lingbot-world-v2 && env FORCE_CPU=1 FORCE_CPU_F32=1 MLX_BACKEND=1 \
    MLX_SAVE_INPUT_LATENT=<workdir>/input_latent.pt PYTHONPATH=. python3 generate.py \
    --task i2v-1.3B --size 320*448 --ckpt_dir ../models/lingbot-world-v2-1.3b-causal-fast \
    --assets_dir ../models/lingbot-world-v2-assets --embeds_dir output/embeds_17f \
    --image <start_image> --action_path examples/03 --frame_num 9 --base_seed 42 \
    --chunk_size 3 --prompt "<prompt>")

# 段 2：DiT 采样（MLX, bf16）
(cd lingbot-world-v2 && env FORCE_CPU=1 FORCE_CPU_F32=1 MLX_BACKEND=1 MLX_BF16=1 \
    MLX_INPUT_LATENT=<workdir>/input_latent.pt MLX_LATENT_ONLY=<workdir>/latent.bin PYTHONPATH=. \
    python3 generate.py --task i2v-1.3B --size 320*448 --ckpt_dir ../models/... \
    --assets_dir ../models/... --embeds_dir output/embeds_17f --image <start_image> \
    --action_path examples/03 --frame_num 9 --base_seed 42 --chunk_size 3 --prompt "<prompt>")

# 段 3：VAE decode
(cd lingbot-world-v2 && env FORCE_CPU=1 FORCE_CPU_F32=1 PYTHONPATH=. \
    python3 ../lingbot-mlx/decode_latent.py --latent <workdir>/latent.bin \
    --vae ../models/lingbot-world-v2-assets/Wan2.1_VAE.pth --fps 16 -o <workdir>/output.mp4)
```

推荐直接使用 `scripts/runlingbot.sh <workdir> <prompt> [start_image]`。

## 7. 前置依赖

- 官方源码 `lingbot-world-v2`（已按 FORCE_CPU/FORCE_CPU_F32/MLX_BACKEND 打补丁：generate.py、image2video.py、model_fast.py、attention.py，见仓库内脚本注释）。
- 权重：`models/lingbot-world-v2-1.3b-causal-fast`（ModelScope，6.84GB，6 分片 safetensors）+ `models/lingbot-world-v2-assets`（含 Wan2.1 VAE）。
- 环境：Python venv 含 torch + safetensors + mlx。

## 8. 已知坑

| 坑 | 结论 |
|---|---|
| 官方 MPS 推理 | 反复崩溃/死机，M2 不可用 → 强制 CPU（FORCE_CPU=1）或 MLX |
| sd-cpp 路线 NaN | gallocr buffer 复用，修复不具可复现性 → 弃用，走 MLX |
| TAE 通道不匹配 | Wan2.2 TAE（12ch）≠ lingbot Wan2.1 latent（16ch），decode 必须官方 Wan2.1 VAE |
| chunk_size | causal_fast 9 帧需 `--chunk_size 3`（默认 6 时 lat_f=3 不合法会崩） |

## 9. 官方资料

- 项目页：https://technology.robbyant.com/lingbot-world-v2
- 代码：https://github.com/robbyant/lingbot-world-v2
- 权重：https://modelscope.cn/models/Robbyant/lingbot-world-v2-1.3b-causal-fast
- MLX：https://github.com/ml-explore/mlx
- Wan2.1（VAE 来源）：https://github.com/Wan-Video/Wan2.1
