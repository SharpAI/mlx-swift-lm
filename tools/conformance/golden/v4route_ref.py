"""Float64 torch-CPU oracle for the DeepSeek-V4 MoE routing, clamped SwiGLU and mHC residual stream.

Everything here is transcribed from vLLM (commit 2eae56c; provenance statement in vllm_provenance.py). Every line cites the vLLM file:line it comes from. File names are
relative to the vLLM tree root:

  [TK]  tests/kernels/moe/test_topk_softplus_sqrt.py     (torch reference _torch_topk_softplus_sqrt, lines 20-81)
  [CU]  csrc/moe/topk_softplus_sqrt_kernels.cu           (CUDA kernels the reference is tested against)
  [CM]  vllm/models/deepseek_v4/cpu/model.py             (the model: gate, hash table, shared expert, mHC wiring)
  [MT]  vllm/model_executor/kernels/mhc/torch.py         (mhc_pre_torch, mhc_post_torch)
  [MK]  tests/kernels/mhc/test_mhc_kernels.py            (sinkhorn_normalize_ref, mhc_pre_ref, mhc_post_ref, hc_head_ref)
  [AC]  vllm/model_executor/layers/activation.py         (SiluAndMulWithClamp.forward_native)
  [FA]  vllm/model_executor/layers/fused_moe/activation.py / utils.py (the routed-expert clamp, same maths)

INFERRED (not stated by any vLLM source in the tree; the Swift suite lists these as such):
  I1  The gate logits are the plain linear map `x @ W^T` (no bias): [CM:136-142] builds GateLinear(bias=False,
      out_dtype=float32) and hands `router_logits=hidden_states` to FusedMoE, whose GateLinear is not in the tree.
  I2  A routed expert is gate_proj/up_proj/down_proj with silu(clamp(gate)) * clamp(up) between them, summed with
      the routing weights; the shared expert is added unscaled. [CM:176-186] shows the shared expert is a
      DeepseekV4MLP with swiglu_limit; the combine step of FusedMoE (routed sum + shared) is not in the tree.
  I3  The hash layers are the first `num_hash_layers` layers: [CM:146] uses `layer_index < mlp_layer_types.count(
      "hash_moe")`, and `mlp_layer_types` comes from the HF config, which is not in the tree.
  I4  The fixture hidden states reach the gate unchanged (the post-ffn_norm hidden state); the norm itself is not
      part of this lane.
  I5  Sinkhorn with `repeat` = 0 is undefined (every vLLM call site has repeat >= 1: the first column
      normalisation always runs); fixtures use repeat >= 1.
"""
import torch

F64 = torch.float64


# --------------------------------------------------------------------------------------------------------------
# scoring and routing
# --------------------------------------------------------------------------------------------------------------
def sqrtsoftplus(x):
    """sqrt(softplus(x)), the numerically stable form: sqrt(max(x,0) + log1p(exp(-|x|))).

    [CU:116 and CU:173] `sqrtf(fmaxf(x, 0.f) + __logf(1.f + __expf(-fabsf(x))))`. The torch reference [TK:32] is
    `F.softplus(x).sqrt()`; torch's softplus switches to the identity for x > 20 (threshold), which differs from
    the exact value by at most e^-20 relative; v4route_crosscheck.py measures the two against each other.
    """
    x = x.to(F64)
    return torch.sqrt(torch.clamp(x, min=0) + torch.log1p(torch.exp(-torch.abs(x))))


def route(logits, topk, renormalize, scale, bias=None, input_ids=None, table=None):
    """Top-k routing of `logits` [T, E]. Returns (weights [T, k] float64, ids [T, k] int64).

    Order of operations (the one the matrix called unverified): score -> choose ids -> gather the UNBIASED score at
    those ids -> divide by the sum (only if renormalize) -> multiply by routed_scaling_factor. [TK:76-81]
    """
    scores = sqrtsoftplus(logits)  # [TK:32]
    if table is not None:  # hash layer: ids come from the table, bias is not used [TK:35-38, CM:146-160, CU:171]
        ids = table[input_ids.long()].long()
    else:
        choice = scores if bias is None else scores + bias.to(F64).unsqueeze(0)  # [TK:63-68]
        # Ties go to the lower expert id [TK:69-74] (stable descending argsort), as the kernel does [CU:140].
        ids = torch.argsort(choice, dim=-1, descending=True, stable=True)[:, :topk]
    w = scores.gather(1, ids)  # [TK:76] weights are the ORIGINAL (unbiased) scores
    if renormalize:  # [TK:77-78]; the kernels divide by (sum > 0 ? sum : 1) [CU:191, CU:469]
        s = w.sum(dim=-1, keepdim=True)
        w = w / torch.where(s > 0, s, torch.ones_like(s))
    if scale != 1.0:  # [TK:79-80]
        w = w * scale
    return w, ids


# --------------------------------------------------------------------------------------------------------------
# clamped SwiGLU and the MoE
# --------------------------------------------------------------------------------------------------------------
def swiglu_clamp(gate, up, limit):
    """[AC:239-243] gate = clamp(gate, max=limit); up = clamp(up, -limit, limit); gate * sigmoid(gate) * up
    (alpha = 1, beta = 0). The routed experts use the same maths [FA: utils.py `_swiglu_limit_torch`]. The clamp is
    ONE-SIDED on the gate (max only) and two-sided on the up half."""
    gate = torch.clamp(gate.to(F64), max=limit)
    up = torch.clamp(up.to(F64), min=-limit, max=limit)
    return gate * torch.sigmoid(gate) * up


def expert(x, wg, wu, wd, limit):
    """One SwiGLU MLP: down(swiglu_clamp(x Wg^T, x Wu^T)). [CM:176-186] (gate_up_proj is gate then up)."""
    return swiglu_clamp(x @ wg.T, x @ wu.T, limit) @ wd.T


def moe(x, gate_w, bias, table, input_ids, topk, renormalize, scale, limit, wg, wu, wd, sg, su, sd):
    """x [T, H]; routed experts wg/wu [E, I, H], wd [E, H, I]; shared expert sg/su [Is, H], sd [H, Is].
    routed = sum_k w_k * expert_{id_k}(x); out = routed + shared(x) (I2)."""
    logits = x @ gate_w.T  # I1
    w, ids = route(logits, topk, renormalize, scale, bias, input_ids, table)
    out = torch.zeros_like(x)
    for t in range(x.shape[0]):
        for j in range(topk):
            e = int(ids[t, j])
            out[t] += w[t, j] * expert(x[t : t + 1], wg[e], wu[e], wd[e], limit)[0]
    out = out + expert(x, sg, su, sd, limit)
    return out, w, ids


# --------------------------------------------------------------------------------------------------------------
# mHC (hyper-connections): pre, post, head
# --------------------------------------------------------------------------------------------------------------
HC_POST_ALPHA = 2.0  # [CM:263] `self.hc_post_alpha = 2.0`


def sinkhorn(logits, repeat, eps):
    """logits [..., n, n] -> doubly-normalised matrix. [MK:866-872] / [MT:119-123]:
    softmax over the LAST dim, + eps, divide by (column sums + eps); then (repeat - 1) times: divide by (row sums
    + eps), divide by (column sums + eps). eps is added INSIDE the denominators."""
    x = torch.softmax(logits, dim=-1) + eps
    x = x / (x.sum(dim=-2, keepdim=True) + eps)
    for _ in range(repeat - 1):
        x = x / (x.sum(dim=-1, keepdim=True) + eps)
        x = x / (x.sum(dim=-2, keepdim=True) + eps)
    return x


def mhc_pre(residual, fn, scale, base, rms_eps, hc_eps, repeat):
    """residual [T, hc, D]; fn [(2+hc)*hc, hc*D]; scale [3]; base [(2+hc)*hc].
    Returns (layer_input [T, D], post [T, hc], comb [T, hc, hc]). [MT:86-132]; the same maths is [MK:875-916].
    rms_eps is the MODEL's rms_norm_eps [CM:327]; hc_pre_eps = hc_sinkhorn_eps = hc_eps [CM:328-329];
    hc_post_mult_value = 2.0 [CM:330]."""
    residual = residual.to(F64)
    T, hc, D = residual.shape
    x = residual.reshape(T, hc * D)  # [MT:103]
    mixes = x @ fn.to(F64).T  # [MT:104]
    mixes = mixes * torch.rsqrt(x.square().sum(-1, keepdim=True) / (hc * D) + rms_eps)  # [MT:105-106]
    scale, base = scale.to(F64), base.to(F64)
    pre = torch.sigmoid(mixes[:, :hc] * scale[0] + base[:hc]) + hc_eps  # [MT:108-109] NOT normalised to sum 1
    post = torch.sigmoid(mixes[:, hc : 2 * hc] * scale[1] + base[hc : 2 * hc]) * HC_POST_ALPHA  # [MT:111-114]
    comb_logits = mixes[:, 2 * hc :].reshape(T, hc, hc) * scale[2] + base[2 * hc :].reshape(1, hc, hc)  # [MT:116-118]
    comb = sinkhorn(comb_logits, repeat, hc_eps)  # [MT:119-123]
    y = (pre.unsqueeze(-1) * residual).sum(dim=1)  # [MT:125-127] (the kernel then casts to bf16)
    return y, post, comb


def mhc_post(x, residual, post, comb):
    """out[j] = post[j] * x + sum_i comb[i, j] * residual[i]. [MT:141-147] / [MK:919-927].
    x [T, D]; residual [T, hc, D]; post [T, hc]; comb [T, hc, hc]."""
    mixed = torch.einsum("tij,tih->tjh", comb.to(F64), residual.to(F64))  # [MT:141-145]
    return mixed + post.to(F64).unsqueeze(-1) * x.to(F64).unsqueeze(-2)  # [MT:146-147]


def hc_head(residual, fn, scale, base, rms_eps, hc_eps):
    """Final HC head: residual [T, hc, D]; fn [hc, hc*D]; scale [1]; base [hc]. [MK:930-944]:
    the RMS norm uses rms_eps, the gate is sigmoid(. * scale + base) + hc_eps, NO normalisation. The model passes
    (rms_norm_eps, hc_eps) [CM:619-626]."""
    residual = residual.to(F64)
    T, hc, D = residual.shape
    flat = residual.reshape(T, hc * D)
    normed = flat * torch.rsqrt(flat.square().mean(dim=-1, keepdim=True) + rms_eps)  # [MK:936-938]
    pre = normed @ fn.to(F64).T  # [MK:939]
    pre = torch.sigmoid(pre * scale.to(F64) + base.to(F64)) + hc_eps  # [MK:940]
    return (pre.unsqueeze(-1) * residual).sum(dim=-2)  # [MK:941]
