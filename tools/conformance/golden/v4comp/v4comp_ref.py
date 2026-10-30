"""float64 torch-CPU oracle for the DeepSeek-V4 KV compressor (lane V4COMP, sl36/v4-compressor-golden).

Transcribed from the vLLM source at commit 2eae56c (main, shallow clone). Paths are relative to the
vLLM repo root; "cpp:N" = csrc/cpu/sgl-kernels/compressor.cpp:N, "cpucomp:N" =
vllm/models/deepseek_v4/cpu/cpu_compressor.py:N, "comp:N" = vllm/models/deepseek_v4/compressor.py:N,
"cpuk:N" = tests/kernels/test_deepseek_v4_cpu_kernels.py:N, "kvc:N" = tests/kernels/test_compressor_kv_cache.py:N,
"rope:N" = tests/kernels/test_fused_deepseek_v4_qnorm_rope_kv_insert.py:N, "attn:N" = vllm/models/deepseek_v4/attention.py:N,
"smla:N" = vllm/models/deepseek_v4/sparse_mla.py:N, "cpusp:N" = vllm/models/deepseek_v4/cpu/cpu_sparse.py:N.

Everything the C++ kernel does in fp32 is done here in float64. NOT transcribed (out of scope, no Swift counterpart): the fp8/UE8M0
quantisation of the stored entry (cpp:347-377). INFERRED by us (not in the vLLM source): head_dim / rope_dim are parameters here,
the C++ hard-codes 512/64 (cpp:49-51) and 128/64 (cpp:391-393); the algebra is dimension independent and vLLM's own
torch reference (kvc:2432) is generic in head_dim / rope_dim, which crosscheck() exercises.
"""
import torch

F64 = torch.float64


def make_cos_sin_cache(max_pos, rope_dim, base):
    # rope:50-60 (float64 here, float32 in vLLM): cache[pos, :rope_dim/2] = cos, [pos, rope_dim/2:] = sin
    inv_freq = 1.0 / (base ** (torch.arange(0, rope_dim, 2, dtype=F64) / rope_dim))
    t = torch.arange(max_pos, dtype=F64)
    freqs = torch.einsum("i,j -> ij", t, inv_freq)
    return torch.cat((freqs.cos(), freqs.sin()), dim=-1)


def gptj_rope(x, cos, sin):
    # cpp:74-92 (forward GPT-J, interleaved pairs; sign mask negates sin on the even lane) == cpuk:71-76
    xe, xo = x[0::2], x[1::2]
    oe = xe * cos - xo * sin   # cpuk:74
    oo = xe * sin + xo * cos   # cpuk:75
    return torch.stack((oe, oo), dim=-1).flatten()


def save_partial_states(kv, score, ape, ratio):
    """Per-token state rows. cpp:245-255 == common/ops/save_partial_states.py:77-89 (ape_row = position % COMPRESS_RATIO, :82): row[:W] = kv[t]; row[W:] = score[t] + ape[position % ratio] (position == t here,
    sequences start at 0). Returned as two [T, W] tensors indexed by absolute position (block table = identity)."""
    T = kv.shape[0]
    pos = torch.arange(T)
    return kv.clone(), score + ape[pos % ratio]


def compress_at(p, st_kv, st_score, ratio, overlap, head_dim, rope_dim, norm_w, eps, cs, variant=None):
    """One compressed entry for the token at absolute position p (only called when (p + 1) % ratio == 0, cpp:289).
    variant: None = vLLM; others are deliberately WRONG readings used to prove the fixtures discriminate."""
    coff = 1 + int(overlap)
    window = coff * ratio                                   # cpucomp:78
    mx = torch.full((head_dim,), float("-inf"), dtype=F64)  # cpp:293
    rows = []
    for w in range(window):
        q = p - window + 1 + w                              # cpucomp:80 (win_local)
        if q < 0:                                           # cpu_utils.py:28-37 -> -1; cpp:301 `continue`
            continue
        off = head_dim if w >= ratio else 0                 # cpp:304 (previous block: first channel half, current: second half)
        if variant == "swap_halves":
            off = 0 if w >= ratio else (head_dim if overlap else 0)
        k = st_kv[q, off:off + head_dim]
        s = st_score[q, off:off + head_dim]
        rows.append((k, s))
        mx = torch.maximum(mx, s)                           # cpp:307-311
    denom = torch.zeros(head_dim, dtype=F64)
    acc = torch.zeros(head_dim, dtype=F64)
    for k, s in rows:                                       # cpp:313-326: per-channel softmax over the window
        e = torch.exp(s - mx)
        denom = denom + e
        acc = acc + e * k
    v = acc / denom                                         # cpp:333
    inv_rms = 1.0 / torch.sqrt((v * v).sum() / head_dim + eps)  # cpp:339
    normed = v * inv_rms * norm_w                           # cpp:343 (RMSNorm over the FULL head_dim, before RoPE)
    pre_rope = normed.clone()
    cpos = (p // ratio) * ratio                             # cpp:353
    if variant == "last_token_pos":
        cpos = p
    half = rope_dim // 2
    cos, sin = cs[cpos, :half], cs[cpos, half:]             # cpp:354-355
    nope = head_dim - rope_dim                              # cpp:356: RoPE on the LAST rope_dim channels
    out = normed.clone()
    out[nope:] = gptj_rope(normed[nope:], cos, sin)
    return pre_rope, out


def compress_sequence(x, wkv, wgate, ape, norm_w, eps, ratio, head_dim, rope_dim, rope_theta, variant=None):
    """x [T, hidden] -> (pre_rope [nb, head_dim], entries [nb, head_dim]); nb = T // ratio (partial trailing block carried, no entry)."""
    overlap = ratio == 4                                    # comp:239-240 `self.overlap = compress_ratio == 4; self.coff = 1 + self.overlap` (comp:158 asserts ratio in [4, 128])
    T = x.shape[0]
    kv = x @ wkv.T                                          # attn:698-703 torch.mm(hidden_states, fused_wkv_wgate.weight.T, fp32)
    sc = x @ wgate.T                                        # comp:364 kv, score = kv_score.split([coff*hd, coff*hd]) (kv first, score second)
    if variant == "no_overlap_ratio4":
        overlap = False
    st_kv, st_score = save_partial_states(kv, sc, ape, ratio)
    cs = make_cos_sin_cache(max(T, 2), rope_dim, rope_theta)
    pre, ent = [], []
    for p in range(T):
        if (p + 1) % ratio != 0:                            # cpp:289
            continue
        a, b = compress_at(p, st_kv, st_score, ratio, overlap, head_dim, rope_dim, norm_w, eps, cs, variant)
        pre.append(a); ent.append(b)
    z = torch.zeros(0, head_dim, dtype=F64)
    return (torch.stack(pre) if pre else z), (torch.stack(ent) if ent else z)


def visible_count(t, ratio):
    # smla:405 / attn:86 / cpusp:252-255: num_compressed = (position + 1) // compress_ratio; entry b visible iff b < num_compressed
    return (t + 1) // ratio
