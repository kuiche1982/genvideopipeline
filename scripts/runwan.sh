#!/bin/bash
# Wan2.2-TI2V-5B 四段式视频生成脚本
# Wan2.2-TI2V-5B 四段式视频生成脚本
# 段1: T5 编码 (sd-cli, umt5)
# 段2: VAE encode (python 官方 Wan2.2 VAE, 16x 归一化 latent)
# 段3: DiT 采样 (sd-cli, --load-init-latent + --save-latent)
# 段4: decode (默认 python 完整 VAE —— 最终画质; 或 TAE 快速确认)
#
# ⚠️ latent 空间必须成对: 完整 VAE encode 只能配完整 VAE decode (python),
#    TAE encode 只能配 TAE decode (sd-cli)。混用 = 马赛克/花屏 (已实证)。
#    TAE (lighttaew2_2) 画质低, 只适合过程确认; 最终结果用完整 VAE。
#
# 用法: ./scripts/runwan.sh <workdir> <prompt> <start_image>
#
# 环境变量: WIDTH=320 HEIGHT=448 FRAMES=49 STEPS=30 CFG=6.0 FLOW_SHIFT=3.0
#           VAE_DECODE=python(默认, 完整VAE最终画质) | tae(快速确认)
set -euo pipefail

WORKDIR="${1:?用法: runwan.sh <workdir> <prompt> <start_image>}"
PROMPT="${2:?缺少 prompt}"
START_IMG="${3:?缺少 start_image}"

WIDTH="${WIDTH:-320}"
HEIGHT="${HEIGHT:-448}"
FRAMES="${FRAMES:-49}"
STEPS="${STEPS:-30}"
CFG="${CFG:-6.0}"
STRENGTH="${STRENGTH:-0.3}"
FLOW_SHIFT="${FLOW_SHIFT:-3.0}"

cd "$(dirname "$0")/.."
ROOT="$(pwd)"

SD_CLI="${SD_CLI:-$ROOT/sd-cpp/build/bin/sd-cli}"
DIFFUSION="$ROOT/models/diffusion/Wan2.2-TI2V-5B-Q8_0.gguf"
T5XXL="$ROOT/models/text_encoder/umt5-xxl-encoder-Q4_K_S.gguf"
TAE="$ROOT/models/tae/lighttaew2_2.safetensors"
WAN22_VAE="$ROOT/models/vae/Wan2.2_VAE.safetensors"
ENCODE_PY="$ROOT/scripts/wan22_vae_encode.py"
PY="$ROOT/.venv/bin/python3"

mkdir -p "$WORKDIR"
LOG="$WORKDIR/run.log"

log() { local ts; ts="$(date '+%Y-%m-%d %H:%M:%S')"; echo "[$ts] $*" | tee -a "$LOG"; }
die() { log "ERROR: $*"; exit 1; }

[ -x "$SD_CLI" ] || die "sd-cli 不存在: $SD_CLI"
[ -f "$DIFFUSION" ] || die "diffusion 模型不存在: $DIFFUSION"
[ -f "$T5XXL" ] || die "t5xxl 模型不存在: $T5XXL"
[ -f "$TAE" ] || die "tae 模型不存在: $TAE"
[ -f "$WAN22_VAE" ] || die "Wan2.2 VAE 不存在: $WAN22_VAE"
[ -f "$START_IMG" ] || die "起始图不存在: $START_IMG"
START_IMG_ABS="$(cd "$(dirname "$START_IMG")" && pwd)/$(basename "$START_IMG")"

EMB="$WORKDIR/embedding.bin"
INIT_LAT="$WORKDIR/init_latent.bin"
LAT="$WORKDIR/latent.bin"
VIDEO="$WORKDIR/output.mp4"

# ── 段 1/4: T5 编码 ──────────────────────────────────
log "── 段 1/4: T5 编码 ──"
"$SD_CLI" -M vid_gen \
    --backend "diffusion=mtl0,te=mtl0" \
    --diffusion-model "$DIFFUSION" \
    --tae "$TAE" \
    --t5xxl "$T5XXL" \
    -p "$PROMPT" \
    -W "$WIDTH" -H "$HEIGHT" \
    --video-frames "$FRAMES" \
    --steps "$STEPS" \
    --cfg-scale "$CFG" \
    --flow-shift "$FLOW_SHIFT" \
    --save-text-embedding "$EMB" \
    --skip-sampling \
    -o "$WORKDIR/t5.bin" 2>&1 | tee -a "$LOG"
[ -f "$EMB" ] || die "T5 输出不存在"
log "段 1 完成: $(du -h "$EMB" | cut -f1), 耗时见上"

# ── 段 2/4: VAE encode (python) ──────────────────────
log "── 段 2/4: VAE encode (python, Wan2.2_VAE) ──"
"$PY" "$ENCODE_PY" --image "$START_IMG_ABS" --vae "$WAN22_VAE" \
    --size "${WIDTH}x${HEIGHT}" --out "$INIT_LAT" 2>&1 | tee -a "$LOG"
[ -f "$INIT_LAT" ] || die "init latent 不存在"
log "段 2 完成: $(du -h "$INIT_LAT" | cut -f1)"

# ── 段 3/4: DiT 采样 (读预计算 init latent) ───────────
log "── 段 3/4: DiT 采样 (--load-init-latent) ──"
"$SD_CLI" -M vid_gen \
    --backend "diffusion=mtl0,te=mtl0" \
    --diffusion-model "$DIFFUSION" \
    --tae "$TAE" \
    -p "$PROMPT" \
    -W "$WIDTH" -H "$HEIGHT" \
    --video-frames "$FRAMES" \
    --steps "$STEPS" \
    --cfg-scale "$CFG" \
    --flow-shift "$FLOW_SHIFT" \
    --load-text-embedding "$EMB" \
    --load-init-latent "$INIT_LAT" \
    --save-latent "$LAT" \
    --skip-decode \
    -o "$WORKDIR/latent_out.avi" 2>&1 | tee -a "$LOG"
[ -f "$LAT" ] || die "latent 输出不存在"
log "段 3 完成: $(du -h "$LAT" | cut -f1)"

# ── 段 4/4: decode (python 完整 VAE 最终画质 / TAE 快速确认) ──
VAE_DECODE="${VAE_DECODE:-python}"
if [ "$VAE_DECODE" = "python" ]; then
    log "── 段 4/4: python 完整 VAE decode (最终画质) ──"
    "$PY" "$ROOT/scripts/wan22_vae_decode.py" \
        --latent "$LAT" --vae "$ROOT/models/vae/Wan2.2_VAE.safetensors" \
        --out "$WORKDIR/frames" 2>&1 | tee -a "$LOG"
    [ -f "$WORKDIR/frames/frame_0000.png" ] || die "decode 帧不存在"
    log "段 4 完成 (python VAE): $(ls "$WORKDIR/frames" | wc -l | tr -d ' ') 帧"
else
    log "── 段 4/4: TAE decode (快速确认, 低画质) ──"
    "$SD_CLI" -M vid_gen \
        --backend "vae=mtl0" \
        --tae "$TAE" \
        --model-version "Wan 2.2 TI2V" \
        -W "$WIDTH" -H "$HEIGHT" \
        --video-frames "$FRAMES" \
        --load-latent "$LAT" \
        -o "$WORKDIR/output.avi" 2>&1 | tee -a "$LOG"
    [ -f "$WORKDIR/output.avi" ] || die "视频输出不存在"
    log "段 4 完成 (TAE)"""
fi

# ── 封装 mp4 ─────────────────────────────────────────
log "── 封装 mp4 ──"
if [ "$VAE_DECODE" = "python" ]; then
    # python decode 输出帧序列 -> mp4
    "$PY" - "$WORKDIR" "$VIDEO" <<'PYEOF'
import subprocess, sys
workdir, video = sys.argv[1], sys.argv[2]
subprocess.run(['ffmpeg', '-y', '-framerate', '16',
                '-i', workdir + '/frames/frame_%04d.png',
                '-c:v', 'libx264', '-pix_fmt', 'yuv420p', video], check=True)
PYEOF
else
    if command -v ffmpeg >/dev/null 2>&1; then
        ffmpeg -y -i "$WORKDIR/output.avi" -c:v libx264 -pix_fmt yuv420p "$VIDEO" 2>&1 | tail -2 | tee -a "$LOG"
    else
        cp "$WORKDIR/output.avi" "$VIDEO"
    fi
fi
[ -f "$VIDEO" ] || die "mp4 封装失败"
log "完成: $VIDEO ($(du -h "$VIDEO" | cut -f1))"
