# genvideopipeline — 在 Mac M2 16G 上本地跑视频生成（四段式）

> English: [README.md](README.md)

个人实验项目：在 Apple Silicon **16GB 统一内存**的 Mac 上，把 5B / 1.3B 视频扩散模型用**四段式**拆分跑通本地视频生成。仅作方法记录，仅供参考。

## 为什么是「四段式」

视频生成全流程（文本编码 → VAE encode → DiT 采样 → VAE decode）单进程跑，峰值内存 **30G+**，16G 机器必死机。四段式把管线拆成 4 个**独立进程、严格串行**执行，每段只加载自己需要的模型、跑完即释放，内存不叠加，峰值降到 **8–12GB**。

## 管线一：Wan2.2-TI2V-5B（sd-cli + python 官方 VAE）

```
段1  T5 编码      sd-cli --save-text-embedding（T5 与图无关，不加载 DiT/VAE）
段2  VAE encode   python scripts/wan22_vae_encode.py（官方 Wan2.2 VAE，16x 归一化 latent）
段3  DiT 采样     sd-cli --load-init-latent + --save-latent
段4  decode       python scripts/wan22_vae_decode.py（完整 VAE，最终画质）| sd-cli TAE（快速确认）
```

- 脚本：`scripts/runwan.sh <workdir> <prompt> <start_image>`
- 峰值内存 ~12GB；验证配置 320×448 / 49 帧 / 30 步。
- 详细结论见 `docs/wan22-4stage.md`。

**两条铁律（已实证）**：
1. **latent 空间必须成对**：完整 VAE↔完整 VAE、TAE↔TAE，混用 = 马赛克/花屏。
2. **sd-cpp 的 DiT 只吃 TAE latent**：喂官方 VAE latent 分布不匹配（花屏）。要完整 VAE 高画质，DiT 也必须走官方实现（python torch 或 MLX）。

## 管线二：LingBot-World-V2 1.3B（MLX 30 层 DiT）

```
段1  VAE encode   输入图 → 输入 latent（.pt）
段2  DiT 采样     MLX 30 层 forward（bf16 权重 2.6GB，不加载 VAE/T5）→ latent.bin
段3  VAE decode   lingbot-mlx/decode_latent.py（只加载 Wan2.1 VAE）→ mp4
段4  封装          ffmpeg
```

- 脚本：`scripts/runlingbot.sh <workdir> <prompt> [start_image]`
- 峰值内存 **30G → 8G**；总时长 ~3min（9 帧 320×432）；输出与完整流程 **bit 一致**。
- 详细记录见 `docs/lingbot-mlx.md`。

## 内存与参数经验

| 规律 | 结论 |
|---|---|
| 内存由什么决定 | **latent 总元素数（帧数 × 分辨率）**，与采样步数无关 |
| 分辨率上限 | M2 Metal ≥480×832 会**静默损坏 latent**（模糊色块），384×640 是验证过的安全上限 |
| 帧数下限 | <9 帧会 glitch（画面撕裂），9 帧起步 |
| 超内存表现 | 触发 swap，机器卡死——探测时盯着活动监视器，稳定不涨才安全 |
| 探测顺序 | 帧数（9→17→25→33）→ 分辨率 → 步数（10/20/30，只影响质量与速度） |

## 模型与工具（官方链接）

| 组件 | 来源 |
|---|---|
| Wan2.2 代码 | https://github.com/Wan-Video/Wan2.2 |
| Wan2.2-TI2V-5B 权重 | https://modelscope.cn/models/Wan-AI/Wan2.2-TI2V-5B |
| stable-diffusion.cpp（sd-cli） | https://github.com/leejet/stable-diffusion.cpp |
| TAEHV（Wan2.1 TAE） | https://github.com/madebyollin/taehv |
| LingBot-World-V2 代码 | https://github.com/robbyant/lingbot-world-v2 |
| LingBot 1.3B 权重 | https://modelscope.cn/models/Robbyant/lingbot-world-v2-1.3b-causal-fast |
| Apple MLX | https://github.com/ml-explore/mlx |
| ComfyUI-Ovi（Wan2.2 VAE 参考） | https://github.com/snicolast/ComfyUI-Ovi |

本地模型权重（GGUF/safetensors）体积大，需按上述来源自行下载；**仓库不含任何权重、样例图片、密钥或个人数据**，起始图由用户自行提供。

## 目录结构

```
genvideopipeline/
├── README.md / README.zh.md
├── LICENSE
├── docs/
│   ├── wan22-4stage.md      # Wan2.2-TI2V-5B 四段式结论与修复记录
│   └── lingbot-mlx.md       # LingBot MLX 四段式记录
├── scripts/
│   ├── runwan.sh            # Wan2.2-TI2V-5B 四段式
│   ├── runlingbot.sh        # LingBot MLX 四段式
│   ├── wan22_vae_encode.py
│   ├── wan22_vae_decode.py
│   └── wan22_vae2_2_official.py  # Wan2.2 VAE 实现（Alibaba Wan Team 版权）
└── lingbot-mlx/
    ├── lingbot_mlx.py       # 30 层 DiT MLX 实现
    └── decode_latent.py     # VAE-only 解码
```
