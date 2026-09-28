#!/bin/bash
# LingBot-World-V2 (MLX) 四段式视频生成脚本
#   段 1/4: VAE encode —— 输入图 → 输入 latent（只加载 VAE + 非 blocks DiT 权重）
#   段 2/4: DiT 采样  —— MLX 30 层 forward（不加载 VAE/T5, 从文件读输入 latent）
#   段 3/4: VAE decode —— 输出 latent → 视频（只加载 VAE）
#   段 4/4: MP4 封装（ffmpeg）
# T5 文本嵌入使用预计算 output/embeds_17f（如需重新生成, 先跑 T5 段）
# 用法: ./scripts/runlingbot.sh <workdir> <prompt> [start_image]
# 环境变量: SIZE=320*448 FRAMES=9 CHUNK=3 SEED=42 FPS=16
set -euo pipefail

WORKDIR="${1:?用法: runlingbot.sh <workdir> <prompt> [start_image]}"
PROMPT="${2:?缺少 prompt}"
START_IMG="${3:-input.jpg}"   # 起始图由用户提供（如仓库根目录 input.jpg），本仓库不附带样例图片

cd "$(dirname "$0")/.."
ROOT="$(pwd)"

PY="$ROOT/.venv/bin/python3"
LINGBOT="$ROOT/lingbot-world-v2"
CKPT="$ROOT/models/lingbot-world-v2-1.3b-causal-fast"
ASSETS="$ROOT/models/lingbot-world-v2-assets"
EMBEDS="$LINGBOT/output/embeds_17f"
VAE_PTH="$ASSETS/Wan2.1_VAE.pth"
ACTION="examples/03"
# MLX 侧自行加载 blocks/head 权重（官方加载器会跳过 blocks.* key）
export LINGBOT_INDEX="$CKPT/model.safetensors.index.json"

SIZE="${SIZE:-320*448}"
FRAMES="${FRAMES:-9}"
CHUNK="${CHUNK:-3}"
SEED="${SEED:-42}"
FPS="${FPS:-16}"

mkdir -p "$WORKDIR"
LOG="$WORKDIR/run.log"

log() {
    local ts
    ts="$(date '+%Y-%m-%d %H:%M:%S')"
    echo "[$ts] $*" | tee -a "$LOG"
}
die() {
    log "ERROR: $*"
    exit 1
}

[ -f "$VAE_PTH" ] || die "VAE 不存在: $VAE_PTH"
[ -d "$EMBEDS" ] || die "T5 embeds 不存在: $EMBEDS"
[ -f "$ROOT/$START_IMG" ] || die "输入图不存在: $ROOT/$START_IMG"

log "========================================"
log "LingBot-World-V2 (MLX) 四段式生成"
log "========================================"
log "workdir:    $WORKDIR"
log "prompt:     $PROMPT"
log "start_img:  $ROOT/$START_IMG"
log "size:       $SIZE  frames: $FRAMES  chunk: $CHUNK  seed: $SEED"

# ── 段 1/4: VAE encode ──────────────────────────────
IN_Y="$WORKDIR/input_latent.pt"
log "── 段 1/4: VAE encode ──"
T0=$(date +%s)
(cd "$LINGBOT" && env FORCE_CPU=1 FORCE_CPU_F32=1 MLX_BACKEND=1 \
    MLX_SAVE_INPUT_LATENT="$IN_Y" PYTHONPATH=. "$PY" generate.py \
    --task i2v-1.3B --size "$SIZE" --ckpt_dir "$CKPT" --assets_dir "$ASSETS" \
    --embeds_dir "$EMBEDS" --image "$ROOT/$START_IMG" --action_path "$ACTION" \
    --frame_num "$FRAMES" --base_seed "$SEED" --chunk_size "$CHUNK" \
    --prompt "$PROMPT") 2>&1 | tee -a "$LOG"
T1=$(date +%s)
[ -f "$IN_Y" ] || die "段 1 失败: 输入 latent 未生成"
log "段 1 完成: $(du -h "$IN_Y" | cut -f1), 耗时 $((T1-T0))s"

# ── 段 2/4: DiT 采样 (MLX, 不加载 VAE/T5) ───────────
OUT_LAT="$WORKDIR/latent.bin"
log "── 段 2/4: DiT 采样 (MLX, 不加载 VAE) ──"
T0=$(date +%s)
(cd "$LINGBOT" && env FORCE_CPU=1 FORCE_CPU_F32=1 MLX_BACKEND=1 MLX_BF16=1 \
    MLX_INPUT_LATENT="$IN_Y" MLX_LATENT_ONLY="$OUT_LAT" PYTHONPATH=. "$PY" generate.py \
    --task i2v-1.3B --size "$SIZE" --ckpt_dir "$CKPT" --assets_dir "$ASSETS" \
    --embeds_dir "$EMBEDS" --image "$ROOT/$START_IMG" --action_path "$ACTION" \
    --frame_num "$FRAMES" --base_seed "$SEED" --chunk_size "$CHUNK" \
    --prompt "$PROMPT") 2>&1 | tee -a "$LOG"
T1=$(date +%s)
[ -f "$OUT_LAT" ] || die "段 2 失败: latent 未生成"
log "段 2 完成: $(du -h "$OUT_LAT" | cut -f1), 耗时 $((T1-T0))s"

# ── 段 3/4: VAE decode ─────────────────────────────
MP4="$WORKDIR/output.mp4"
log "── 段 3/4: VAE decode ──"
T0=$(date +%s)
(cd "$LINGBOT" && env FORCE_CPU=1 FORCE_CPU_F32=1 PYTHONPATH=. \
    "$PY" "$ROOT/lingbot-mlx/decode_latent.py" --latent "$OUT_LAT" \
    --vae "$VAE_PTH" --fps "$FPS" -o "$MP4") 2>&1 | tee -a "$LOG"
T1=$(date +%s)
[ -f "$MP4" ] || die "段 3 失败: 视频未生成"
log "段 3 完成: $(du -h "$MP4" | cut -f1), 耗时 $((T1-T0))s"

# ── 段 4/4: 清理/汇总 ───────────────────────────────
log "========================================"
log "完成! 产物:"
log "  输入 latent: $IN_Y"
log "  输出 latent: $OUT_LAT"
log "  视频:        $MP4"
log "  日志:        $LOG"
log "========================================"
echo "$MP4"
