"""Float64 torch-CPU transcription of vLLM's Kimi Delta Attention (KDA) maths: the ground-truth oracle
for the SharpAI KDA Metal kernel (Libraries/MLXLMCommon/SharpAIOverlay/GatedDeltaKDA.swift).

Source: vLLM commit 739e5945d, paths relative to vllm/model_executor/layers/fla/ops/ :
    FR = fused_recurrent.py   kernel fused_recurrent_gated_delta_rule_fwd_kernel (IS_KDA=True branch)
    KD = kda.py               fused_recurrent_kda_fwd, kda_gate_fwd_kernel / fused_kda_gate, chunk_kda
    L2 = l2norm.py
Every line of maths below carries the vLLM file:line it transcribes.

Layouts (all float64 on CPU):
    q, k   [B, T, Hk, Dk]     v [B, T, Hv, Dv]     Hv a multiple of Hk (GQA)
    gk     [B, T, Hv, Dk]     LOG decay (<= 0); the kernel under test takes exp(gk)
    beta   [B, T, Hv]         per-head scalar (already sigmoid-ed by the caller, see kda.py model layer)
    state  [B, Hv, Dv, Dk]    FR:119 (p_h0 offset  i_hv*V*K + o_v[:,None]*K + o_k[None,:])

INFERRED (NOT in the vLLM source; every item here is a decision of this file, not a fact of vLLM):
    I1  `mask` [B, T] bool. The vLLM kernel has no per-token mask (it has cu_seqlens varlen, FR:65-74).
        Semantics used: a masked token is a complete no-op, state unchanged and y = 0. This is what
        the SharpAI kernel documents (GatedDeltaKDA.swift, "masked step: state unchanged, y = 0").
    I2  `kda_gate(lower_bound=...)`: the "safe gate" lower_bound * sigmoid(exp(A_log) * (g + bias)) is
        NOT in the vLLM tree at 739e5945d (grep lower_bound under vllm/model_executor finds nothing).
        It is the formula cited by Tests/.../GapReference/GdnReference.swift (kdaLogDecay) from a
        different vLLM snapshot that is not available here. UNVERIFIED against vLLM.
    I3  log(1 + exp(x)) is evaluated as log1p(exp(x)) (identical function, exact in float64 for very
        negative x where 1 + exp(x) would round to 1). KD:1330 writes log(1.0 + exp(g_scaled)).
    I4  Output shape. fused_recurrent_kda_fwd allocates o = torch.empty_like(k) (KD:55), which is only
        right for Hk == Hv and Dk == Dv; the kernel indexes o as [B, T, Hv, Dv] (FR:96, FR:169) and
        that is the shape returned here.
    I5  Hv % Hk == 0 and head mapping i_h = i_hv // (Hv // Hk) (FR:64) is the repeat_interleave
        (contiguous-group) convention; asserted, not derived.
    I6  Final state is the state after the last un-masked token. vLLM's per-token ht writes
        (FR:152-165) are not modelled.
"""
import math

import torch

F64 = torch.float64


def _f64(x):
    return None if x is None else torch.as_tensor(x).to(device="cpu", dtype=F64)


# ----------------------------------------------------------------------------------------------
# L2 norm of one head vector
# ----------------------------------------------------------------------------------------------
def l2norm_vec(x, eps=1e-6):
    """FR:128-129  b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)   (same eps as L2:96 default eps=1e-6)."""
    return x / torch.sqrt(torch.sum(x * x) + eps)


# ----------------------------------------------------------------------------------------------
# recurrence
# ----------------------------------------------------------------------------------------------
def recurrent(q, k, v, gk, beta, state=None, scale=None, use_qk_l2norm=False, mask=None):
    """Token-by-token KDA recurrence, one (batch, value-head) tile [Dv, Dk] at a time.

    Returns (y [B,T,Hv,Dv], state [B,Hv,Dv,Dk]) float64.
    scale=None -> Dk ** -0.5 (FR:599-600 fused_recurrent_gated_delta_rule, same default in chunk_kda KD:1243-1244).
    use_qk_l2norm -> FR:127-129 (applied to q and k BEFORE the scale, per head vector).
    """
    q, k, v, gk, beta = (_f64(t) for t in (q, k, v, gk, beta))
    B, T, Hk, Dk = k.shape
    Hv, Dv = v.shape[2], v.shape[3]
    assert q.shape == k.shape
    assert Hv % Hk == 0, "I5"
    assert gk.shape == (B, T, Hv, Dk), "KDA gate is per key dimension (FR:92, p_gk +K)"
    assert beta.shape == (B, T, Hv)  # FR:89  p_beta = beta + bos*HV + i_hv  (not headwise)
    if scale is None:
        scale = Dk ** -0.5  # FR:599-600
    if mask is not None:
        mask = torch.as_tensor(mask).to(dtype=torch.bool)
        assert mask.shape == (B, T)

    h_all = torch.zeros(B, Hv, Dv, Dk, dtype=F64) if state is None else _f64(state).clone()  # FR:102, FR:119-120
    assert h_all.shape == (B, Hv, Dv, Dk)
    y = torch.zeros(B, T, Hv, Dv, dtype=F64)

    for b in range(B):
        for i_hv in range(Hv):
            i_h = i_hv // (Hv // Hk)  # FR:64
            b_h = h_all[b, i_hv].clone()  # [BV, BK] = [Dv, Dk], FR:102 (zeros) / FR:120 (initial state)
            for t in range(T):
                if mask is not None and not bool(mask[b, t]):
                    continue  # I1: no-op, y stays 0
                b_q = q[b, t, i_h].clone()  # FR:123
                b_k = k[b, t, i_h].clone()  # FR:124
                b_v = v[b, t, i_hv].clone()  # FR:125
                if use_qk_l2norm:  # FR:127
                    b_q = l2norm_vec(b_q)  # FR:128
                    b_k = l2norm_vec(b_k)  # FR:129
                b_q = b_q * scale  # FR:130
                b_gk = gk[b, t, i_hv]  # FR:136
                b_h = b_h * torch.exp(b_gk)[None, :]  # FR:137  b_h *= exp(b_gk[None, :])
                b_v = b_v - torch.sum(b_h * b_k[None, :], dim=1)  # FR:139  b_v -= tl.sum(b_h * b_k[None, :], 1)
                b_beta = beta[b, t, i_hv]  # FR:143
                b_v = b_v * b_beta  # FR:144  b_v *= b_beta
                b_h = b_h + b_v[:, None] * b_k[None, :]  # FR:146  b_h += b_v[:, None] * b_k[None, :]
                y[b, t, i_hv] = torch.sum(b_h * b_q[None, :], dim=1)  # FR:148  b_o = tl.sum(b_h * b_q[None, :], 1)
            h_all[b, i_hv] = b_h  # I6
    return y, h_all


# ----------------------------------------------------------------------------------------------
# gate
# ----------------------------------------------------------------------------------------------
def kda_gate(g, A_log, dt_bias=None, beta=1.0, threshold=20.0, lower_bound=None):
    """KD:1274-1331 kda_gate_fwd_kernel / KD:1336-1379 fused_kda_gate.

    g [..., H*D], A_log [H], dt_bias [H*D] or None (flat, KD:1320 g_bias + i_h*D + n_d).
    Returns the LOG decay [..., H, D] float64 (KD:1379 y.view(*orig_shape, H, head_k_dim)).
    `beta` is the SOFTPLUS beta (KD:1279), not the delta-rule beta.
    lower_bound: I2, INFERRED, see module header.
    """
    g, A_log, dt_bias = _f64(g), _f64(A_log).reshape(-1), _f64(dt_bias)
    H = A_log.numel()  # KD:1357
    assert g.shape[-1] % H == 0
    D = g.shape[-1] // H  # KD:1358 assert H * head_k_dim == HD
    lead = g.shape[:-1]
    g2 = g.reshape(-1, H, D)
    out = torch.zeros_like(g2)
    for t in range(g2.shape[0]):
        for h in range(H):
            b_a = -math.exp(float(A_log[h]))  # KD:1291-1292  b_a = -tl.exp(b_a)
            for d in range(D):
                b_g = float(g2[t, h, d])  # KD:1315
                if dt_bias is not None:
                    b_g = b_g + float(dt_bias[h * D + d])  # KD:1320-1323  b_g = b_g + b_bias
                if lower_bound is None:
                    g_scaled = b_g * beta  # KD:1328
                    use_linear = g_scaled > threshold  # KD:1329
                    sp = b_g if use_linear else (1.0 / beta) * math.log1p(math.exp(g_scaled))  # KD:1330 (+I3)
                    out[t, h, d] = b_a * sp  # KD:1331
                else:
                    # I2 (inferred): lower_bound * sigmoid(exp(A_log) * (g + bias))
                    out[t, h, d] = lower_bound / (1.0 + math.exp(-math.exp(float(A_log[h])) * b_g))
    return out.reshape(*lead, H, D)
