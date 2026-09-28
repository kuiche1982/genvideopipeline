#!/usr/bin/env python3
"""LingBot-World-V2 1.3B DiT — MLX reimplementation (f32, alignment-first).

Alignment target: official FORCE_CPU_F32 run (golden dumps in /tmp/golden1).
Inputs are the official pre-computed patch/token tensors (inp_*.npy):
  x      [1, 1581, 1536]  token sequence (post patch_embedding, post concat)
  e0     [1, 1581, 6, 1536] time_projection output (per-block modulation)
  e2     [1, 1581, 1536] time_embedding output (head modulation)
  ctx    [1, 512, 1536]  text_embedding output
  c2ws   [1, 1581, 1536] c2ws plucker embedding (post pre-processing)
Only the 30 transformer blocks + head + unpatchify are reimplemented here.
"""
import json
import math
import os
import sys

import mlx.core as mx

# ----------------------------------------------------------------------------
# Weights
# ----------------------------------------------------------------------------

def load_weights(index_path, dtype=mx.float32):
    """Load all safetensors shards -> {key: mx.array}."""
    idx = json.load(open(index_path))
    wmap = idx["weight_map"]
    dirn = os.path.dirname(os.path.abspath(index_path))
    out = {}
    for key, shard in wmap.items():
        from safetensors import safe_open
        p = os.path.join(dirn, shard)
        with safe_open(p, framework="numpy") as f:
            out[key] = mx.array(f.get_tensor(key)).astype(dtype)
    return out


# ----------------------------------------------------------------------------
# Rope
# ----------------------------------------------------------------------------

def rope_params(max_seq_len, dim, theta=10000.0):
    """Returns (re, im) of polar(1, outer(pos, 1/theta^(arange(0,dim,2)/dim))).
    dim is the *real* dimension -> dim//2 complex components."""
    half = dim // 2
    pos = mx.arange(max_seq_len, dtype=mx.float32)
    freq = 1.0 / mx.power(theta, mx.arange(0, dim, 2, dtype=mx.float32) / dim)
    ang = mx.outer(pos, freq)  # [seq, half]
    return mx.cos(ang), mx.sin(ang)


def build_freqs(d):
    """d = head_dim = dim // num_heads. Matches WanModelFast.freqs."""
    a = d - 4 * (d // 6)
    b = 2 * (d // 6)
    re0, im0 = rope_params(1024, a)
    re1, im1 = rope_params(1024, b)
    re2, im2 = rope_params(1024, b)
    re = mx.concatenate([re0, re1, re2], axis=-1)  # [1024, 22+21+21]
    im = mx.concatenate([im0, im1, im2], axis=-1)
    return re, im


def _grid_freqs(re, im, grid, start_frame=0):
    """Per-token complex multipliers for grid (f,h,w). re/im are [1024, 64]
    complex components (22 t-freqs, 21 h-freqs, 21 w-freqs)."""
    f, h, w = grid
    L = f * h * w
    c0, c1 = 22, 21  # complex counts per segment (d=128: 64-2*21=22)
    re_t = mx.broadcast_to(mx.reshape(re[start_frame:start_frame + f, :c0], (f, 1, 1, c0)), (f, h, w, c0))
    re_h = mx.broadcast_to(mx.reshape(re[:h, c0:c0 + c1], (1, h, 1, c1)), (f, h, w, c1))
    re_w = mx.broadcast_to(mx.reshape(re[:w, c0 + c1:], (1, 1, w, c1)), (f, h, w, c1))
    im_t = mx.broadcast_to(mx.reshape(im[start_frame:start_frame + f, :c0], (f, 1, 1, c0)), (f, h, w, c0))
    im_h = mx.broadcast_to(mx.reshape(im[:h, c0:c0 + c1], (1, h, 1, c1)), (f, h, w, c1))
    im_w = mx.broadcast_to(mx.reshape(im[:w, c0 + c1:], (1, 1, w, c1)), (f, h, w, c1))
    re_g = mx.reshape(mx.concatenate([re_t, re_h, re_w], axis=-1), (L, -1))
    im_g = mx.reshape(mx.concatenate([im_t, im_h, im_w], axis=-1), (L, -1))
    return re_g, im_g


def causal_rope_apply(x, re, im, grid, start_frame=0):
    """x: [1, L, n, d] -> roped (same shape). Complex multiplication in real form."""
    f, h, w = grid
    L = f * h * w
    b, Lx, n, d = x.shape
    assert Lx == L
    c = d // 2
    xr = mx.reshape(x[0, :L], (L, n, c, 2))  # (re, im) pairs
    x_re, x_im = xr[..., 0], xr[..., 1]
    re_g, im_g = _grid_freqs(re, im, grid, start_frame)
    re_g = mx.reshape(re_g, (L, 1, c))
    im_g = mx.reshape(im_g, (L, 1, c))
    o_re = x_re * re_g - x_im * im_g
    o_im = x_re * im_g + x_im * re_g
    out = mx.reshape(mx.stack([o_re, o_im], axis=-1), (L, n, d))
    return mx.expand_dims(out, 0)


# ----------------------------------------------------------------------------
# Ops
# ----------------------------------------------------------------------------

def rms_norm(x, w, eps=1e-6):
    return x * mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + eps) * w


def layer_norm(x, w, b, eps=1e-6):
    mean = mx.mean(x, axis=-1, keepdims=True)
    var = mx.mean((x - mean) ** 2, axis=-1, keepdims=True)
    return (x - mean) * mx.rsqrt(var + eps) * w + b


def layer_norm_noaffine(x, eps=1e-6):
    mean = mx.mean(x, axis=-1, keepdims=True)
    var = mx.mean((x - mean) ** 2, axis=-1, keepdims=True)
    return (x - mean) * mx.rsqrt(var + eps)


def gelu_tanh(x):
    return 0.5 * x * (1.0 + mx.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)))


def linear(x, w, b):
    """x [.., in] -> [.., out] = x @ W^T + b"""
    return mx.addmm(b, x, mx.transpose(w))


def sdpa(q, k, v):
    """q,k,v: [1, L, n, d] f32. Exact softmax attention, no mask (matches
    official FORCE_CPU_F32 SDPA is_causal=False)."""
    scale = 1.0 / math.sqrt(q.shape[-1])
    qt = mx.transpose(q, (0, 2, 1, 3))  # [1, n, Lq, d]
    kt = mx.transpose(k, (0, 2, 3, 1))  # [1, n, d, Lk]
    vt = mx.transpose(v, (0, 2, 1, 3))
    s = mx.matmul(qt, kt) * scale      # [1, n, Lq, Lk]
    p = mx.softmax(s, axis=-1)
    o = mx.matmul(p, vt)               # [1, n, Lq, d]
    return mx.transpose(o, (0, 2, 1, 3))


# ----------------------------------------------------------------------------
# Layers
# ----------------------------------------------------------------------------

def self_attn(x, freqs, grid, W, p, num_heads, dbg=None):
    """CausalWanSelfAttention: q/k/v linear -> RMSNorm(full dim) -> view heads
    -> rope -> SDPA -> o."""
    q = linear(x, W[p + "q.weight"], W[p + "q.bias"])
    k = linear(x, W[p + "k.weight"], W[p + "k.bias"])
    v = linear(x, W[p + "v.weight"], W[p + "v.bias"])
    q = rms_norm(q, W[p + "norm_q.weight"], eps=1e-6)
    k = rms_norm(k, W[p + "norm_k.weight"], eps=1e-6)
    n, d = num_heads, q.shape[-1] // num_heads
    b, s, _ = q.shape
    q = mx.reshape(q, (b, s, n, d))
    k = mx.reshape(k, (b, s, n, d))
    v = mx.reshape(v, (b, s, n, d))
    if dbg is not None:
        dbg["q_pre_rope"] = q
        dbg["k_pre_rope"] = k
    q = causal_rope_apply(q, freqs[0], freqs[1], grid, 0)
    k = causal_rope_apply(k, freqs[0], freqs[1], grid, 0)
    if dbg is not None:
        dbg["q_roped"] = q
        dbg["v"] = v
    o = sdpa(q, k, v)
    o = mx.reshape(o, (b, s, -1))
    return linear(o, W[p + "o.weight"], W[p + "o.bias"])


def cross_attn(x, ctx, W, p, num_heads):
    """WanCrossAttention: q from x, k/v from ctx, exact SDPA, no mask."""
    q = linear(x, W[p + "q.weight"], W[p + "q.bias"])
    k = linear(ctx, W[p + "k.weight"], W[p + "k.bias"])
    v = linear(ctx, W[p + "v.weight"], W[p + "v.bias"])
    q = rms_norm(q, W[p + "norm_q.weight"], eps=1e-6)
    k = rms_norm(k, W[p + "norm_k.weight"], eps=1e-6)
    n, d = num_heads, q.shape[-1] // num_heads
    b, s, _ = q.shape
    q = mx.reshape(q, (b, s, n, d))
    k = mx.reshape(k, (b, -1, n, d))
    v = mx.reshape(v, (b, -1, n, d))
    o = sdpa(q, k, v)
    o = mx.reshape(o, (b, s, -1))
    return linear(o, W[p + "o.weight"], W[p + "o.bias"])


def block(x, e, ctx, c2ws, freqs, grid, W, i, num_heads=12, dbg=None):
    """CausalWanAttentionBlock. e: [1, L, 6, d] (e0 + block.modulation)."""
    p = f"blocks.{i}."
    d = x.shape[-1]
    # modulation chunks
    mod = mx.reshape(W[p + "modulation"], (1, 1, 6, d))
    e = e + mod
    e0 = e[..., 0, :]  # [1, L, d]
    e1 = e[..., 1, :]
    e2 = e[..., 2, :]
    e3 = e[..., 3, :]
    e4 = e[..., 4, :]
    e5 = e[..., 5, :]

    # self-attention path (norm1 is no-affine LayerNorm)
    attn_in = layer_norm_noaffine(x, eps=1e-6) * (1.0 + e1) + e0
    if dbg is not None:
        dbg["attn_in"] = attn_in
    y = self_attn(attn_in, freqs, grid, W, p + "self_attn.", num_heads, dbg=dbg)
    if dbg is not None:
        dbg["sa_out"] = y
    x = x + y * e2
    if dbg is not None:
        dbg["after_sa_add"] = x

    # cam injection (silu activation)
    h = linear(c2ws, W[p + "cam_injector_layer1.weight"], W[p + "cam_injector_layer1.bias"])
    h = h * mx.sigmoid(h)
    h = linear(h, W[p + "cam_injector_layer2.weight"], W[p + "cam_injector_layer2.bias"])
    h = h + c2ws
    scale = linear(h, W[p + "cam_scale_layer.weight"], W[p + "cam_scale_layer.bias"])
    shift = linear(h, W[p + "cam_shift_layer.weight"], W[p + "cam_shift_layer.bias"])
    x = (1.0 + scale) * x + shift
    if dbg is not None:
        dbg["after_cam"] = x

    # cross attention + ffn (norm3 is affine LayerNorm; norm2 is no-affine)
    xn = layer_norm(x, W[p + "norm3.weight"], W[p + "norm3.bias"], eps=1e-6)
    x = x + cross_attn(xn, ctx, W, p + "cross_attn.", num_heads)
    if dbg is not None:
        dbg["after_cross"] = x
    ffn_in = layer_norm_noaffine(x, eps=1e-6) * (1.0 + e4) + e3
    y = linear(ffn_in, W[p + "ffn.0.weight"], W[p + "ffn.0.bias"])
    y = gelu_tanh(y)
    y = linear(y, W[p + "ffn.2.weight"], W[p + "ffn.2.bias"])
    x = x + y * e5
    if dbg is not None:
        dbg["after_ffn"] = x
    return x


def head_fn(x, e, W):
    """CausalHead: layer_norm (no affine) + modulation(2) + linear."""
    d = x.shape[-1]
    mod = mx.reshape(W["head.modulation"], (1, 1, 2, d))
    e = e[..., None, :] + mod  # [1, L, 2, d]
    e0 = e[..., 0, :]
    e1 = e[..., 1, :]
    h = layer_norm_noaffine(x, eps=1e-6) * (1.0 + e1) + e0
    return linear(h, W["head.head.weight"], W["head.head.bias"])


def unpatchify(x, grid, patch=(1, 2, 2), out_dim=16):
    """x [1, L, out_dim*prod(patch)] -> [out_dim, F, H, W]."""
    f, h, w = grid
    L = f * h * w
    u = x[0, :L]
    u = mx.reshape(u, (f, h, w, *patch, out_dim))
    # einsum 'fhwpqrc->cfphqwr'
    u = mx.transpose(u, (6, 0, 3, 1, 4, 2, 5))
    u = mx.reshape(u, (out_dim, f * patch[0], h * patch[1], w * patch[2]))
    return u


# ----------------------------------------------------------------------------
# Full model
# ----------------------------------------------------------------------------

def forward(x, e0, e2, ctx, c2ws, W, grid=(3, 17, 31), nlayers=30, num_heads=12):
    """x/e0/e2/ctx/c2ws: f32. Returns block outputs list, head output, latent."""
    d = x.shape[-1]
    freqs = build_freqs(d // num_heads)
    outs = []
    dbg = {}
    for i in range(nlayers):
        d0 = dbg if i == 0 else None
        x = block(x, e0, ctx, c2ws, freqs, grid, W, i, num_heads, dbg=d0)
        outs.append(x)
    if dbg:
        import numpy as np
        for k, v in dbg.items():
            mx.eval(v)
            np.save(f"/tmp/mlx_b0_{k}.npy", np.asarray(v))
    x = head_fn(x, e2, W)
    lat = unpatchify(x, grid)
    return outs, x, lat


# ----------------------------------------------------------------------------
# Torch drop-in interface (used by official pipeline with MLX_BACKEND=1)
# ----------------------------------------------------------------------------

_W_GLOBAL = None
# 权重索引路径：由环境变量 LINGBOT_INDEX 指定（scripts/runlingbot.sh 会自动设置），
# 默认相对仓库根目录，便于单独调试。
_INDEX_PATH = os.environ.get(
    "LINGBOT_INDEX",
    "models/lingbot-world-v2-1.3b-causal-fast/model.safetensors.index.json",
)


def _get_W():
    global _W_GLOBAL
    if _W_GLOBAL is None:
        import time
        dtype = mx.bfloat16 if os.environ.get("MLX_BF16") else mx.float32
        t0 = time.time()
        _W_GLOBAL = load_weights(_INDEX_PATH, dtype=dtype)
        print(f"[MLX] weights loaded in {time.time()-t0:.1f}s dtype={dtype}", flush=True)
    return _W_GLOBAL


def forward_torch(x, e0, e2, ctx, c2ws, grid=(3, 17, 31)):
    """Drop-in replacement for blocks + head. All inputs torch f32 tensors.
    grid: (F, H_patch, W_patch) after patch embedding. Returns head output
    [1, L, 64] as torch tensor (official does unpatchify)."""
    import numpy as np
    import torch
    import time
    W = _get_W()
    t0 = time.time()
    xa = mx.array(x.detach().float().numpy())
    e0a = mx.array(e0.detach().float().numpy())
    e2a = mx.array(e2.detach().float().numpy())
    cta = mx.array(ctx.detach().float().numpy())
    c2a = mx.array(c2ws.detach().float().numpy())
    _, hx, _ = forward(xa, e0a, e2a, cta, c2a, W, grid=tuple(int(v) for v in grid), nlayers=30, num_heads=12)
    mx.eval(hx)
    print(f"[MLX] fwd {(time.time()-t0)*1000:.0f}ms dtype={hx.dtype}", flush=True)
    return torch.from_numpy(np.asarray(hx)).to(x.device)


# ----------------------------------------------------------------------------
# CLI: run and compare against golden
# ----------------------------------------------------------------------------

def load_npy(p):
    import numpy as np
    return mx.array(np.load(p)).astype(mx.float32)


def load_pt(p):
    import numpy as np
    import torch
    t = torch.load(p, map_location="cpu", weights_only=True)
    return mx.array(t.detach().float().numpy())


def main():
    # 对齐 CLI：仓库根目录可用 LINGBOT_BASE 覆盖（默认当前目录）
    base = os.environ.get("LINGBOT_BASE", ".")
    idx = f"{base}/models/lingbot-world-v2-1.3b-causal-fast/model.safetensors.index.json"
    gd = os.environ.get("GOLDEN_DIR", "/tmp/golden1")
    t0 = int(os.environ.get("T_STEP", "999"))

    print("loading weights...", flush=True)
    W = load_weights(idx, dtype=mx.bfloat16 if os.environ.get("MLX_BF16") else mx.float32)

    x = load_npy(f"{gd}/inp_x_t{t0}.npy")
    e0 = load_npy(f"{gd}/inp_e_t{t0}.npy")
    e2 = load_npy(f"{gd}/inp_e2_t{t0}.npy")
    ctx = load_npy(f"{gd}/inp_ctx_t{t0}.npy")
    c2ws = load_npy(f"{gd}/inp_c2ws_t{t0}.npy")
    print(f"inputs x{x.shape} e0{e0.shape} e2{e2.shape} ctx{ctx.shape} c2ws{c2ws.shape}", flush=True)

    outs, hx, lat = forward(x, e0, e2, ctx, c2ws, W)
    print("forward done", flush=True)

    # compare per block
    worst = 0.0
    for i, o in enumerate(outs):
        g = load_pt(f"{gd}/b{i:02d}_t{t0}.pt")
        err = mx.abs(o - g)
        rel = mx.sum(err) / (mx.sum(mx.abs(g)) + 1e-9)
        rel = float(rel)
        worst = max(worst, rel)
        if i % 5 == 0 or rel > 0.01:
            print(f"  block{i:02d} relL1={rel:.6f}", flush=True)
    g = load_pt(f"{gd}/head_t{t0}.pt")
    rel = float(mx.sum(mx.abs(hx - g)) / (mx.sum(mx.abs(g)) + 1e-9))
    print(f"  head   relL1={rel:.6f}", flush=True)
    g = load_pt(f"{gd}/lat_t{t0}.pt")
    rel = float(mx.sum(mx.abs(lat - g)) / (mx.sum(mx.abs(g)) + 1e-9))
    print(f"  latent relL1={rel:.6f}", flush=True)
    print(f"  WORST block relL1 = {worst:.6f}", flush=True)


if __name__ == "__main__":
    main()
