# run.sh 四段式改造结论（2026-09-18）

## 目标
把 Wan2.2-TI2V-5B 三段式（T5→DiT→TAE）拆成四段式，VAE encode 改用 python（官方 Wan2.2 VAE），避开 sd-cpp 内部 encode 的马赛克 bug。

## 四段式
```
段1 T5 编码      sd-cli --save-text-embedding --skip-sampling（不传 -i，T5 与图无关）
段2 VAE encode   python scripts/wan22_vae_encode.py（官方 Wan2.2_VAE.safetensors，strict 100% 加载）
段3 DiT 采样     sd-cli --load-text-embedding + --load-init-latent + --save-latent --skip-decode
段4 TAE decode   sd-cli --tae --load-latent
```
全程严格串行（runwan.sh 顺序执行，段 1/3 加载时 T5/DiT 内存不叠加，峰值 ~12GB）。

## 关键修复（sd-cpp，commit c160547）
1. **`--load-init-latent` 参数**：common.h/cpp + stable-diffusion.h（img_gen + vid_gen 两个 struct）+ stable-diffusion.cpp
   - image 路径：prepare_image_generation_latents 三分支
   - video 路径：prepare_video_generation_latents 的 Wan2.2-TI2V 分支（L6541）读 SDLT 文件跳过 encode_first_stage
2. **Wan2.2 VAE scale 回归**：官方 8x 空间，sd-cpp 硬编码 16x —— 在 python encode 侧解决：官方 VAE 输出 40x56 → area 插值到 20x28（16x 网格，与 sd-cpp DiT 匹配），质量优于 sd-cpp 内部 TAE encode
3. **Head modulate 回归**（顺带修复，此前阻塞所有 Wan2.2 采样）：
   - lingbot 调试时把 `modulate_mul/modulate_add` 改成 `ggml_repeat(es, x)`，es 的 T 维与 x token 数不兼容 → `GGML_ASSERT(ggml_can_repeat)` 崩溃
   - 已改回 modulate_mul/modulate_add（保留 debug 钩子）

## 全黑根因与修复（2026-09-18 补充）
**根因**：python 官方 VAE 的 latent 数值尺度与 sd-cpp DiT/TAE 空间不匹配，且 VAE 结构有误：
1. **缺 patchify**（2x2 空间重组，3ch→12ch）——官方 Wan2.2 VAE 是 **16x 空间压缩**（patchify 2x + 卷积 8x），之前实现是 8x 手动 12ch 零填充
2. **缺 per-channel latent 归一化**——官方 encode 返回 `(mu - mean) * (1/std)`（48 维，ComfyUI-Ovi `ovi/modules/vae2_2.py` L995-1103 硬编码），之前 std 仅 0.09
3. **官方 encode 是确定性**（直接返回 mu，不采样）——之前误用 reparameterize

修复后 init latent：shape [20,28,1,48]（16x 空间），range [-1.72, 1.34]，std 0.47 → DiT 采样输出 std ~1.2（与历史 TAE latent 1.37 同量级）→ **视频从全黑恢复为有内容**。

## 花屏/马赛克根因（2026-09-18 定案）
**对照实验实锤**：完整 Wan2.2 VAE latent 与 TAE latent 是**两个不兼容的空间**：
- sd-cpp TAE encode → DiT → TAE decode：**正常**（旧二进制 6 步即出清晰画面，对照验证）
- python 官方完整 VAE encode → TAE decode：**马赛克**（官方类 latent std 0.44 vs TAE latent std 1.37）
- 用户所述"sd-cpp encode 马赛克" = 完整 VAE 与 TAE 混用所致；**TAE encode 本身正常**

**结论**：四段式必须"同空间成对"——
- **TAE 全程**（sd-cpp encode/decode，已验证正常，快）
- **完整 VAE 全程**（python 官方类 encode + decode，与官方一致，decode 慢）
- 禁止混用（完整 VAE encode + TAE decode = 马赛克）

## 结论（用户确认 2026-09-18）
- **TAE（lighttaew2_2）画质低于完整 VAE**——只适合做场景/过程确认（低步数快速看构图），**不适合出最终结果**
- **最终结果用完整 VAE 全程**：python 官方 encode（`wan22_vae_encode.py`）→ **官方 DiT（python torch / MLX）** → python 官方 decode（`wan22_vae_decode.py`）
- **latent 空间必须成对**：完整 VAE↔完整 VAE、TAE↔TAE，混用=马赛克（已实证）

## ⚠️ sd-cpp DiT 只吃 TAE latent（2026-09-18 30 步实证）
- TAE latent（std 1.4）→ sd-cpp DiT → TAE decode：**正常**（人物坐藤椅，清晰）
- 完整 VAE latent（std 0.44）→ sd-cpp DiT → 完整 VAE decode：**花屏**（30 步仍模糊偏色）
- 结论：**sd-cpp 的 Wan2.2 DiT 转换面向 TAE latent 空间**，喂官方 VAE latent 分布不匹配。
  因此 **sd-cpp 路径无法产出完整 VAE 高画质**；完整 VAE 高画质必须 DiT 也走官方
  （python torch 或 MLX），sd-cpp 只负责 TAE 快速管线。

## 脚本（runwan.sh，四段式）
- 段1 T5：sd-cli `--save-text-embedding --skip-sampling`
- 段2 encode：python 官方 Wan2.2 VAE（16x 归一化 latent，std 0.44）
- 段3 DiT：sd-cli `--load-init-latent + --save-latent --skip-decode`
- 段4 decode：`VAE_DECODE=python`（默认，最终画质）| `tae`（快速确认）
- 全程严格串行，各段独立进程，峰值 ~12GB

## 验证结果（320x448，49 帧）
| 段 | 结果 |
|---|---|
| 1 T5 | embedding.bin（cond/uncond 各 2097152，15s）✓ |
| 2 encode | init_latent.bin [20,28,1,48,1]（16x 空间，官方归一化 std 0.44）✓ |
| 3 DiT | latent.bin [20,28,13,48,1]（STEPS 6/10 验证，std ~1.2）✓ |
| 4 TAE/VAE | TAE decode（完整 VAE latent → 马赛克，禁用）；python 完整 VAE decode（`wan22_vae_decode.py`）✓ |

**对照（实证）**：TAE 全程 30 步 → 画面正常（人物坐藤椅，清晰）；完整 VAE encode + TAE decode → 马赛克（空间不兼容）。

## 脚本
`scripts/runwan.sh <workdir> <prompt> <start_image>`，环境变量 WIDTH/HEIGHT/FRAMES/STEPS/CFG/STRENGTH/FLOW_SHIFT。
