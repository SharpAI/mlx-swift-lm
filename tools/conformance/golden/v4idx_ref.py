"""Float64 torch-CPU oracle for the DeepSeek-V4 sparse indexer and the sparse-attention merge, transcribed from vLLM.

Every function cites vLLM file:line. Paths are relative to the vLLM source root (the `src` directory of the shallow
clone of vllm-project/vllm at commit 2eae56c). `cpu_sparse.py`, `cpu_mla.py` are under vllm/models/deepseek_v4/cpu/,
`attention.py` under vllm/models/deepseek_v4/, the tests under tests/.

WHAT vLLM DOES (all CPU/CUDA kernels are C++/Triton; the torch code that EXISTS is the CPU path's eager glue and the
eager references in tests/kernels/test_deepseek_v4_cpu_kernels.py. This file transcribes both):

  index scores   score[t, b] = sum_h w[t,h] * relu(q[t,h] . k[b])                 tests/kernels/test_deepseek_v4_cpu_kernels.py:1122-1123
                 q[t] = rope_last_rope_dim(wq_b(q_norm(wq_a x))[t], position t)    attention.py:1147, :1095-1101 ; test :1253-1290
                 w[t,h] = weights_proj(x)[t,h] * head_dim^-0.5 * n_head^-0.5       attention.py:1004, :1100-1101 ; test :1287
  visibility     token at position p sees the first (p + 1) // ratio compressed    vllm/v1/attention/backends/mla/indexer.py:549-551
                 entries (entry b = tokens [b*ratio, b*ratio + ratio - 1])
  top-k          torch.topk semantics over the visible columns; fewer than k       tests/kernels/test_top_k_per_row.py:213-243, :772-781
                 visible -> all of them, the rest -1; ties: set of VALUES only     :157-210 ; cpu_sparse.py:246-261
  SWA window     [max(p - window + 1, 0), p]                                       cpu_sparse.py:587-590 ; tests/v1/attention/test_deepseek_v4_swa_visible.py:63-70
  merge          one softmax over [window keys ++ selected compressed keys ++ sink] cpu_sparse.py:630-641 (flash_mla_with_kvcache_cpu) ;
                 k = v (one shared kv row per slot), value of the sink is 0         test :697-736 (_flash_mla_eager)
  compressed-key RoPE position of entry b is (position // ratio) * ratio = b*ratio  compressor.py:453 ; test :310 and :560 (compressed_pos)

INFERRED / NOT MODELLED (kept short, each is also listed in MANIFEST.json "note"):
  I1  fp8 quantisation. vLLM quantises q (per-head power-of-two scale folded into w) and k (per-token scale) to fp8; the
      Swift path has no quantisation. The oracle takes the dequantised value as exact (k_scale = 1, q_scale folded back).
      v4idx_crosscheck.py reports the size of what this omits (informational, not asserted at 1e-10).
  I2  vLLM's V4 tree has no Hadamard rotation of q/k (grep -ri hadamard vllm/models/deepseek_v4 -> nothing). An orthonormal
      rotation of both operands would leave q.k unchanged anyway.
  I3  Tie-breaking among equal scores is not specified by vLLM (its tests compare the multiset of selected VALUES). The golden
      therefore encodes {strict: indices that must be selected, tie: indices tied at the threshold, need: how many of them}.
  I4  RoPE: fixtures use plain GPT-J RoPE (theta given). That is exactly vLLM's `deepseek_yarn` with factor 1.0 (the SWA layers,
      common/rope.py:46-52, deepseek_scaling_rope.py:73-101 with scaling_factor == 1). The YaRN variant of compressed layers is a
      separate unit and is NOT covered by these fixtures.
  I5  The compressed keys are INPUTS (the compressor is another lane); this oracle only RoPEs raw entries (position b*ratio).
  I6  The attention sink is an extra logit (NOT multiplied by the softmax scale) whose value row is 0; -inf == no sink
      (weight_utils.py:1564-1572 make_attention_sink initialises to -inf "no sink effect").
"""
import math

import torch

F64 = torch.float64
NEG_INF = float("-inf")


# --------------------------------------------------------------------------------------------------------------------
# RoPE
# --------------------------------------------------------------------------------------------------------------------
def inv_freq(theta, rot_dim):
    """vllm/model_executor/layers/rotary_embedding/base.py:86-91: 1 / (base ** (arange(0, rotary_dim, 2) / rotary_dim));
    equal to deepseek_scaling_rope.py:73-101 for scaling_factor == 1 (interpolation == extrapolation)."""
    return 1.0 / (theta ** (torch.arange(0, rot_dim, 2, dtype=F64) / rot_dim))


def gptj_rope(x, pos, theta):
    """x [..., rot_dim] rotated at integer positions `pos` (broadcast against x's leading dims, last dim removed).
    tests/kernels/test_deepseek_v4_cpu_kernels.py:71-76 _apply_gptj_rope: even/odd interleaved pairs,
    out_even = x_even*cos - x_odd*sin ; out_odd = x_even*sin + x_odd*cos. cos/sin = cos/sin(pos * inv_freq)
    (rotary_embedding/base.py:96-102)."""
    rot_dim = x.shape[-1]
    ang = pos.to(F64).unsqueeze(-1) * inv_freq(theta, rot_dim)          # [..., rot_dim/2]
    cos, sin = ang.cos(), ang.sin()
    x_even, x_odd = x[..., 0::2], x[..., 1::2]
    out_even = x_even * cos - x_odd * sin
    out_odd = x_even * sin + x_odd * cos
    return torch.stack((out_even, out_odd), dim=-1).flatten(-2)


def rope_tail(x, pos, theta, rope_dim):
    """RoPE on the LAST rope_dim channels, the rest untouched.
    compressor.py:452 ("applied to LAST rope_head_dim elements of head_dim"); test :1263-1273 (nope_dim = head_dim - rot_dim)."""
    nope = x.shape[-1] - rope_dim
    return torch.cat([x[..., :nope], gptj_rope(x[..., nope:], pos, theta)], dim=-1)


def rope_compressed_keys(raw, theta, rope_dim, ratio, first_block=0):
    """Entry b is rotated at position (position // ratio) * ratio where `position` is the LAST token of block b
    (compressor.py:453 ; test :310/:560 compressed_pos = (position // compress_ratio) * compress_ratio, written when
    (position + 1) % ratio == 0 so position = b*ratio + ratio - 1) -> b * ratio."""
    nb = raw.shape[0]
    pos = (torch.arange(nb) + first_block) * ratio
    return rope_tail(raw, pos, theta, rope_dim)


# --------------------------------------------------------------------------------------------------------------------
# visibility
# --------------------------------------------------------------------------------------------------------------------
def visible_blocks(pos, ratio):
    """vllm/v1/attention/backends/mla/indexer.py:549-551: global_ctx = start_pos + 1 + offset (= position + 1);
    len_per_token = global_ctx // COMPRESS_RATIO. Also cpu_sparse.py:252-254 (num_compressed = (positions + 1) // ratio)."""
    return (pos + 1) // ratio


def visibility_matrix(length, blocks, ratio, start=0):
    """[length][blocks] bool: block b is readable by the query at position start + t iff b < (start + t + 1) // ratio."""
    return [[b < visible_blocks(start + t, ratio) for b in range(blocks)] for t in range(length)]


def swa_window_bounds(pos, window):
    """[start, end) of the keys a query at `pos` reads from the raw (per-token) cache.
    cpu_sparse.py:587-590: start_pos = (positions - window + 1).clamp(min=0); local_pos = start_pos + arange(window);
    valid = local_pos <= positions. Same as tests/v1/attention/test_deepseek_v4_swa_visible.py:63-70 ref_swa_bounds with
    left = right = 0 (no image span)."""
    return max(pos - window + 1, 0), pos + 1


def swa_visibility_matrix(length, window, start=0, total=None):
    """[length][total] bool, total = start + length by default: query t (abs position start + t) reads raw key j iff
    j in swa_window_bounds."""
    total = start + length if total is None else total
    out = []
    for t in range(length):
        lo, hi = swa_window_bounds(start + t, window)
        out.append([lo <= j < hi for j in range(total)])
    return out


# --------------------------------------------------------------------------------------------------------------------
# index scores
# --------------------------------------------------------------------------------------------------------------------
def index_scores(x, qr, wq_b, w_proj, keys, start, n_head, head_dim, rope_dim, theta, ratio=4):
    """x [L, hidden], qr [L, q_lora] (= q_norm(wq_a x), attention.py + cpu_sparse.py:112-113 _split_qkv_and_norm), wq_b
    [n_head*head_dim, q_lora], w_proj [n_head, hidden], keys [nb, head_dim] (RoPE already applied, the indexer K cache).
    Returns [L, nb] with -inf where the block is not complete at the query position."""
    L, nb = x.shape[0], keys.shape[0]
    q = (qr @ wq_b.T).view(L, n_head, head_dim)                                  # attention.py:1146-1147 (_wq_b_proj, view)
    pos = start + torch.arange(L)                                                # positions of the L query tokens
    q = rope_tail(q, pos.unsqueeze(-1), theta, rope_dim)                         # attention.py:1095-1101 ; test :1253-1290
    w = (x @ w_proj.T) * head_dim ** -0.5 * n_head ** -0.5                       # attention.py:712 weights_proj ; :1004 softmax_scale ; :1100-1101 head scale ; test :1287 (I1: q_scale folds back to identity)
    dot = torch.einsum("jd,thd->tjh", keys, q).clamp(min=0.0)                    # test :1122 dot = einsum("jd,hd->jh", k, q).clamp(min=0.0)
    score = (dot * w.unsqueeze(1)).sum(dim=2)                                    # test :1123 score = (dot * w.unsqueeze(0)).sum(dim=1) ; k_scale = 1 (I1)
    n_vis = visible_blocks(pos, ratio)                                           # indexer.py:549-551
    visible = torch.arange(nb).unsqueeze(0) < n_vis.unsqueeze(1)
    return torch.where(visible, score, torch.full_like(score, NEG_INF))


# --------------------------------------------------------------------------------------------------------------------
# top-k
# --------------------------------------------------------------------------------------------------------------------
def topk_select(row, k):
    """One row of scores (-inf = not visible). Returns dict(strict=[...], tie=[...], need=int) such that ANY valid
    selection (vLLM's contract: same multiset of values as torch.topk over the visible columns, tests/kernels/
    test_top_k_per_row.py:157-210, :213-243) contains every `strict` index and exactly `need` of the `tie` indices and
    nothing else. Fewer than k visible: all visible columns, need = 0 (cpu_sparse.py:246-261, test :772-781)."""
    vis = [i for i, v in enumerate(row) if v != NEG_INF]
    n_sel = min(k, len(vis))
    if n_sel == 0:
        return dict(strict=[], tie=[], need=0)
    vals = sorted((row[i] for i in vis), reverse=True)                           # torch.topk values (descending)
    thr = vals[n_sel - 1]
    strict = [i for i in vis if row[i] > thr]
    tie = [i for i in vis if row[i] == thr]
    return dict(strict=strict, tie=tie, need=n_sel - len(strict))


# --------------------------------------------------------------------------------------------------------------------
# merge of the SWA window and the compressed branch
# --------------------------------------------------------------------------------------------------------------------
def merge_attention(q, kv, comp, win_vis, comp_vis, sink, scale):
    """q [H, L, D]; kv [S, D] raw per-token rows (k = v); comp [nb, D] compressed rows (k = v);
    win_vis [L][S] bool (which raw rows the query reads); comp_vis [L][nb] bool (which compressed rows);
    sink [H] or None (== -inf); scale float. Returns [H, L, D].
    tests/kernels/test_deepseek_v4_cpu_kernels.py:697-735 _flash_mla_eager: gather window ++ compressed rows, scores =
    q.kv * scale (:728), invalid -> -inf (:729), append the sink logit (:730-731; NOT scaled), softmax over all (:732), out =
    probs[:-1] . kv (:733). cpu_sparse.py:630-641 passes window_slots, compressed_slots, attn_sink, scale."""
    H, L, D = q.shape
    out = torch.zeros(H, L, D, dtype=F64)
    for t in range(L):
        rows = [kv[j] for j in range(kv.shape[0]) if win_vis[t][j]] + [comp[b] for b in range(comp.shape[0]) if comp_vis[t][b]]
        for h in range(H):
            s_h = torch.tensor(NEG_INF if sink is None else float(sink[h]), dtype=F64)
            if not rows:
                continue                                                         # nothing readable and no sink mass: output 0
            keys = torch.stack(rows)                                             # [N, D]
            logits = (keys @ q[h, t]) * scale                                    # test :728
            full = torch.cat([logits, s_h.view(1)])                              # test :730-731
            p = torch.softmax(full, dim=0)                                       # test :732
            out[h, t] = p[:-1] @ keys                                            # test :733
    return out


def layer_layout(compress_ratios, has_index_cfg=True):
    """Which layers own a compressor / an indexer. attention.py:240-242 compress_ratio = max(1, rate); :302 `if
    self.compress_ratio == 4:` -> indexer; :369 `if self.compress_ratio > 1:` -> compressor. (Ratio 0 / 1 = SWA only.)"""
    out = []
    for i, r in enumerate(compress_ratios):
        cr = max(1, r)
        out.append(dict(layer=i, ratio=r, compressor=cr > 1, indexer=cr == 4 and has_index_cfg))
    return out
