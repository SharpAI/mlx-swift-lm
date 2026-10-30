"""Generate the V4 routing / clamped SwiGLU / mHC golden fixtures for ConformanceV4RouteGoldenTests.

PROVENANCE
  Maths: vLLM commit 2eae56c (main, shallow clone; the clone has no .git, the commit is the one the lane was given),
    transcribed in tools/conformance/golden/v4route_ref.py with file:line citations; INFERRED points are I1..I5 in
    that header. v4route_crosscheck.py checks the oracle against vLLM's own torch reference code (extracted with ast
    and run in float64) on >= 200 random cases per component; it must print OK before this generator is run.
  Python: float64 torch CPU. The torch version used is recorded in MANIFEST.json ("torch").

HOW TO RUN (once, then commit the outputs; tests never run python, CI is air-gapped)
    uv venv .venv-golden && uv pip install --python .venv-golden/bin/python torch numpy \\
        --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple
    .venv-golden/bin/python -I tools/conformance/golden/v4route_crosscheck.py     # must print OK
    .venv-golden/bin/python -I tools/conformance/golden/gen_v4route_golden.py     # rewrites Tests/MLXLMTests/Conformance/Golden/v4route/
  Network is needed ONLY for the one-time torch install. Output is deterministic (per-case seed = crc32(name)).

FIXTURE CONTRACT
  1. Every input tensor is ALREADY rounded to the case dtype (float32 / bfloat16) before the oracle sees it, so a
     tolerance only has to cover compute error. Inputs are first cut to 4 decimals (short text), then rounded to
     the dtype; they are stored with 9 significant digits, which round-trips float32 (and so bfloat16) exactly.
     Outputs are the oracle's float64 values stored with 12 significant digits.
  2. kind "gate"/"hash": DeepseekV4Gate on x [B,S,H] (hash: with tokenIds [B,S]). Expected output: per token the
     chosen expert ids SORTED ASCENDING and the routing weights in that order (expert order inside the top-k is not
     part of the contract, [TK] sorts too). Selection margins are asserted by this generator (a rejection loop), so
     float32 / bfloat16 rounding of the logits cannot flip a selection; the exact-tie case is the one deliberate
     exception.
  3. kind "expert": DeepseekV4Expert (the shared expert); kind "moe": DeepseekV4MoE (gate + routed experts via
     SwitchGLU + shared expert). Expected: y [B,S,H].
  4. kinds "hcpre", "hcpost", "hchead", "hcchain": the hyper-connection residual stream [B,S,hc,D] (pre = reduce
     hc copies to one, post = expand back, head = final reduction, chain = pre -> sublayer(y) = 0.5 y + 0.1 -> post).
"""
import hashlib
import json
import math
import os
import sys
import zlib

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v4route_ref as R  # noqa: E402

F64 = torch.float64
VLLM_COMMIT = "2eae56c"
DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16}
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.normpath(os.path.join(HERE, "..", "..", "..", "Tests", "MLXLMTests", "Conformance", "Golden", "v4route"))


def seeded(name):
    g = torch.Generator()
    g.manual_seed(zlib.crc32(name.encode()))
    return g


class Case:
    """Draws rounded inputs for one named case."""

    def __init__(self, name, dtype):
        self.name, self.dtype, self.g = name, dtype, seeded(name)

    def randn(self, *shape, scale=1.0, dtype=None):
        x = torch.randn(*shape, generator=self.g, dtype=F64) * scale
        return self.rnd(x, dtype)

    def rnd(self, x, dtype=None):
        return torch.round(x * 1e4).div(1e4).to(DTYPES[dtype or self.dtype]).to(F64)

    def ints(self, lo, hi, *shape):
        return torch.randint(lo, hi, shape, generator=self.g)

    def table(self, vocab, E, k):
        return torch.stack([torch.randperm(E, generator=self.g)[:k] for _ in range(vocab)]).to(torch.int32)


def T(x, ints=False, digits=9):
    flat = x.reshape(-1).tolist()
    data = [int(v) for v in flat] if ints else [float(f"{v:.{digits}g}") for v in flat]
    return {"shape": list(x.shape), "data": data}


def O(x):
    return T(x, digits=12)


def sorted_by_id(w, ids):
    ids_s, order = torch.sort(ids, dim=-1)
    return w.gather(-1, order), ids_s


# ---------------------------------------------------------------------------------------------------------------
# tolerances (reasoning in the manifest)
# ---------------------------------------------------------------------------------------------------------------
TOL = {
    "route_w_fp32": {
        "mode": "elementwise", "rtol": 5e-6, "atol": 0.0,
        "why": "float32 compute vs float64 truth, per routing weight (all weights are positive, so a per-element "
               "relative bound is meaningful). Logit = sum over H<=32 terms: float32 matmul error ~ sqrt(H) u |term| "
               "<= 1e-6 absolute for |logit| <= 8 (u = 2^-24 = 6e-8). score = sqrt(softplus(l)): d ln(score)/dl = "
               "sigmoid(l) / (2 softplus(l)) <= 1/2 for every l, so the score's relative error is <= 5e-7 from the "
               "logit plus ~4 ulp (2.4e-7) from exp/log1p/add/sqrt. Normalisation (sum of k<=6 values, one divide) "
               "and the routed scale add ~3 ulp (1.8e-7). Total <= ~1.0e-6; 5e-6 = that + 5x margin."},
    "route_w_bf16": {
        "mode": "elementwise", "rtol": 4.5e-3, "atol": 0.0,
        "why": "bfloat16 inputs, float32 compute (vLLM: GateLinear out_dtype=float32, CM:136-142; the Swift gate "
               "widens x and the weight to float32 before the matmul), then ONE rounding of the finished routing "
               "weight to bfloat16 (the gate hands bf16 weights to the expert combine). Round to nearest gives "
               "|round(v) - v| <= 2^-8 |v| = 3.906e-3 |v| (bf16 has 8 significant bits). The float32 part is the same "
               "as route_w_fp32, <= ~1e-6 relative. 4.5e-3 = 3.906e-3 + 1e-6 + ~15% margin. A 1-2% error in the bf16 "
               "path (a wrong eps or scale) now fails; the earlier 2.5e-2 bound existed only to absorb bf16 logits."},
    "out_fp32": {
        "mode": "normwise", "rtol": 5e-6, "atol": 1e-9,
        "why": "float32 compute vs float64 truth, max|got-want| <= atol + rtol * max|want| over the tensor (normwise "
               "because these outputs are sums with cancellation). Two or three chained matmuls with <= 32 terms "
               "(sqrt(n) u |term| ~ 3.4e-7 each), silu/sigmoid/softmax/exp (<= 4 ulp = 2.4e-7), up to 20 Sinkhorn "
               "normalisations (contractive: random-walk sqrt(40) u = 3.8e-7) and a cancellation factor "
               "sum|terms|/|sum| <= ~5: ~2e-6 worst case; 5e-6 = that + 2.5x margin. atol 1e-9 only covers an "
               "all-zero tensor and the 1e-12 text rounding of the golden."},
    "hcmix_fp32": {
        "mode": "elementwise", "rtol": 1e-5, "atol": 0.0,
        "why": "float32 vs float64, per element of post [B,S,hc] and comb [B,S,hc,hc]; both are strictly positive "
               "(sigmoid * 2 and a softmax + eps) so no cancellation. mixes = x . fn (hc*D <= 64 terms: sqrt(64) u "
               "|term| ~ 5e-7 relative to the logit scale), sigmoid / softmax (<= 4 ulp), and up to 2 x 19 further "
               "Sinkhorn divisions after the first (each <= 1 ulp, contractive so random-walk: sqrt(40) u = 3.8e-7). "
               "Total <= ~1.5e-6; 1e-5 = 6x margin."},
    "hcpost_bf16": {
        "mode": "normwise", "rtol": 4e-3, "atol": 1e-9,
        "why": "x and residual are bfloat16-exact and post/comb are float32, so the Swift result is computed in "
               "float32 (promotion) and rounded to bfloat16 once: |round(v) - v| <= 2^-8 |v| (round to nearest, 8-bit "
               "significand, u = 3.9e-3) <= 3.9e-3 max|want| in the normwise sense; 4e-3 = u + float32 compute "
               "(<= 1e-6) + 2% margin."},
}


# ---------------------------------------------------------------------------------------------------------------
# gate / hash / expert / moe
# ---------------------------------------------------------------------------------------------------------------
def gate_like(name, group, dtype, *, B=1, S=4, H=16, E=16, k=4, renorm=True, scale=1.5, hash_=False, vocab=24,
              wscale=1.0, xscale=1.0, bias_scale=0.0, tie=False, max_abs_logit=8.0, min_gap=2e-3, force_flip=False,
              ids_override=None):
    c = Case(name, dtype)
    for _ in range(2000):
        x = c.randn(B * S, H, scale=xscale)
        W = c.randn(E, H, scale=wscale / math.sqrt(H))
        bias = None if hash_ else c.randn(E, scale=bias_scale, dtype="float32")
        table = ids = None
        if hash_:
            table = c.table(vocab, E, k)
            ids = c.ints(0, vocab, B * S) if ids_override is None else torch.tensor(ids_override)
        if tie:
            # experts 2 and 5 are the k-th / (k+1)-th choice and EXACTLY tied (same weight row, same bias)
            assert E == 8 and k == 3 and not hash_
            W[5] = W[2]
            bias = torch.tensor([10.0, -10.0, 0.0, -10.0, -10.0, 0.0, -10.0, 10.0], dtype=F64)
        logits = x @ W.T
        if tie:
            logits[:, 5] = logits[:, 2]
        if logits.abs().max() > max_abs_logit:
            continue
        scores = R.sqrtsoftplus(logits)
        if not hash_:
            choice = scores + (bias if bias is not None else 0)
            srt = torch.sort(choice, dim=-1, descending=True).values
            if tie:
                assert bool((srt[:, k - 1] == srt[:, k]).all()), "tie lost"
                if not bool(((srt[:, k - 2] - srt[:, k - 1]) > min_gap).all() and ((srt[:, k] - srt[:, k + 1]) > min_gap).all()):
                    continue
            elif not bool(((srt[:, k - 1] - srt[:, k]) > min_gap).all()):
                continue
            if force_flip:
                unb = torch.argsort(scores, dim=-1, descending=True, stable=True)[:, :k].sort(-1).values
                bia = torch.argsort(choice, dim=-1, descending=True, stable=True)[:, :k].sort(-1).values
                if not bool((unb != bia).any()):
                    continue
        break
    else:
        raise RuntimeError(f"{name}: no input satisfied the constraints")
    w, topi = R.route(logits, k, renorm, scale, bias, ids, table)
    w, topi = sorted_by_id(w, topi)
    inputs = {"x": T(x.reshape(B, S, H)), "weight": T(W)}
    if hash_:
        inputs["table"] = T(table, ints=True)
        inputs["tokenIds"] = T(ids.reshape(B, S), ints=True)
    else:
        inputs["bias"] = T(bias)
    return {
        "name": name, "kind": "hash" if hash_ else "gate", "group": group, "dtype": dtype,
        "tolerance": {"weights": "route_w_bf16" if dtype == "bfloat16" else "route_w_fp32"},
        "params": {"B": B, "S": S, "H": H, "E": E, "topk": k, "vocab": vocab, "normTopkProb": renorm,
                   "routedScalingFactor": scale},
        "inputs": inputs,
        "outputs": {"ids": T(topi.reshape(B, S, k), ints=True), "weights": O(w.reshape(B, S, k))},
    }


def expert_inputs(c, H, I, wscale):
    wg = c.randn(I, H, scale=wscale / math.sqrt(H))
    wu = c.randn(I, H, scale=wscale / math.sqrt(H))
    wd = c.randn(H, I, scale=1.0 / math.sqrt(I))
    return wg, wu, wd


def expert_case(name, group, *, B=1, S=4, H=16, I=12, limit=10.0, wscale=1.0, xscale=1.0, need_gate_lo=False,
                need_gate_hi=False, need_up_out=False):
    c = Case(name, "float32")
    for _ in range(2000):
        x = c.randn(B * S, H, scale=xscale)
        wg, wu, wd = expert_inputs(c, H, I, wscale)
        g, u = x @ wg.T, x @ wu.T
        if need_gate_lo and not bool((g < -limit - 0.05).any()):
            continue
        if need_gate_hi and not bool((g > limit + 0.05).any()):
            continue
        if need_up_out and not bool((u.abs() > limit + 0.05).any()):
            continue
        break
    else:
        raise RuntimeError(f"{name}: constraints not met")
    y = R.expert(x, wg, wu, wd, limit)
    return {
        "name": name, "kind": "expert", "group": group, "dtype": "float32", "tolerance": {"y": "out_fp32"},
        "params": {"B": B, "S": S, "H": H, "I": I, "swigluLimit": limit},
        "inputs": {"x": T(x.reshape(B, S, H)), "wg": T(wg), "wu": T(wu), "wd": T(wd)},
        "outputs": {"y": O(y.reshape(B, S, H))},
        "stats": {"gateBelowNegLimit": int((g < -limit).sum()), "gateAboveLimit": int((g > limit).sum()),
                  "upOutsideLimit": int((u.abs() > limit).sum())},
    }


def moe_case(name, group, *, B=1, S=4, H=12, I=8, E=8, k=3, hash_=False, limit=10.0, renorm=True, scale=1.5,
             vocab=24, wscale=1.0, xscale=1.0):
    c = Case(name, "float32")
    for _ in range(2000):
        x = c.randn(B * S, H, scale=xscale)
        gw = c.randn(E, H, scale=1.0 / math.sqrt(H))
        bias = None if hash_ else c.randn(E, scale=0.5)
        table = ids = None
        if hash_:
            table = c.table(vocab, E, k)
            ids = c.ints(0, vocab, B * S)
        wg = c.randn(E, I, H, scale=wscale / math.sqrt(H))
        wu = c.randn(E, I, H, scale=wscale / math.sqrt(H))
        wd = c.randn(E, H, I, scale=1.0 / math.sqrt(I))
        sg, su, sd = expert_inputs(c, H, I, wscale)
        logits = x @ gw.T
        scores = R.sqrtsoftplus(logits)
        if not hash_:
            srt = torch.sort(scores + bias, dim=-1, descending=True).values
            if not bool(((srt[:, k - 1] - srt[:, k]) > 2e-3).all()):
                continue
        break
    else:
        raise RuntimeError(f"{name}: constraints not met")
    y, w, ids_k = R.moe(x, gw, bias, table, ids, k, renorm, scale, limit, wg, wu, wd, sg, su, sd)
    inputs = {"x": T(x.reshape(B, S, H)), "gateWeight": T(gw), "wg": T(wg), "wu": T(wu), "wd": T(wd),
              "sg": T(sg), "su": T(su), "sd": T(sd)}
    if hash_:
        inputs["table"] = T(table, ints=True)
        inputs["tokenIds"] = T(ids.reshape(B, S), ints=True)
    else:
        inputs["bias"] = T(bias)
    return {
        "name": name, "kind": "moe", "group": group, "dtype": "float32", "tolerance": {"y": "out_fp32"},
        "params": {"B": B, "S": S, "H": H, "I": I, "E": E, "topk": k, "hash": hash_, "vocab": vocab,
                   "swigluLimit": limit, "normTopkProb": renorm, "routedScalingFactor": scale},
        "inputs": inputs, "outputs": {"y": O(y.reshape(B, S, H))},
    }


# ---------------------------------------------------------------------------------------------------------------
# mHC
# ---------------------------------------------------------------------------------------------------------------
def hc_inputs(c, B, S, hc, D, *, fn_scale=0.3, scale_scale=0.8, base_scale=0.5, x_scale=1.5, dtype="float32"):
    mix = (2 + hc) * hc
    res = c.randn(B * S, hc, D, scale=x_scale, dtype=dtype)
    fn = c.randn(mix, hc * D, scale=fn_scale, dtype="float32")
    scale = c.randn(3, scale=scale_scale, dtype="float32")
    base = c.randn(mix, scale=base_scale, dtype="float32")
    return res, fn, scale, base


def hcpre_case(name, group, *, B=1, S=3, hc=4, D=8, repeat=20, rms_eps=1e-6, hc_eps=1e-6, fn_scale=0.3,
               scale_scale=0.8, base_scale=0.5, zero_fn=False):
    c = Case(name, "float32")
    res, fn, scale, base = hc_inputs(c, B, S, hc, D, fn_scale=fn_scale, scale_scale=scale_scale, base_scale=base_scale)
    if zero_fn:
        fn = torch.zeros_like(fn)
    y, post, comb = R.mhc_pre(res, fn, scale, base, rms_eps, hc_eps, repeat)
    return {
        "name": name, "kind": "hcpre", "group": group, "dtype": "float32",
        "tolerance": {"y": "out_fp32", "post": "hcmix_fp32", "comb": "hcmix_fp32"},
        "params": {"B": B, "S": S, "hc": hc, "D": D, "sinkhornIters": repeat, "rmsEps": rms_eps, "hcEps": hc_eps},
        "inputs": {"x": T(res.reshape(B, S, hc, D)), "fn": T(fn), "scale": T(scale), "base": T(base)},
        "outputs": {"y": O(y.reshape(B, S, D)), "post": O(post.reshape(B, S, hc)),
                    "comb": O(comb.reshape(B, S, hc, hc))},
    }


def hcpost_case(name, group, *, dtype="float32", B=1, S=3, hc=4, D=8):
    c = Case(name, dtype)
    res = c.randn(B * S, hc, D, scale=1.5)
    x = c.randn(B * S, D, scale=1.0)
    # post and comb are arbitrary positive float32 tensors; comb is NOT symmetric and NOT doubly stochastic, so a
    # transposed (i,j) mix cannot pass by accident.
    post = c.randn(B * S, hc, scale=0.4, dtype="float32").abs() + 0.2
    comb = c.randn(B * S, hc, hc, scale=0.5, dtype="float32").abs() + 0.05
    out = R.mhc_post(x, res, post, comb)
    return {
        "name": name, "kind": "hcpost", "group": group, "dtype": dtype,
        "tolerance": {"y": "hcpost_bf16" if dtype == "bfloat16" else "out_fp32"},
        "params": {"B": B, "S": S, "hc": hc, "D": D},
        "inputs": {"x": T(x.reshape(B, S, D)), "residual": T(res.reshape(B, S, hc, D)), "post": T(post.reshape(B, S, hc)),
                   "comb": T(comb.reshape(B, S, hc, hc))},
        "outputs": {"y": O(out.reshape(B, S, hc, D))},
    }


def hchead_case(name, group, *, B=1, S=3, hc=4, D=8, rms_eps=1e-6, hc_eps=1e-6):
    c = Case(name, "float32")
    res = c.randn(B * S, hc, D, scale=1.5)
    fn = c.randn(hc, hc * D, scale=0.3)
    scale = c.randn(1, scale=0.8)
    base = c.randn(hc, scale=0.5)
    y = R.hc_head(res, fn, scale, base, rms_eps, hc_eps)
    return {
        "name": name, "kind": "hchead", "group": group, "dtype": "float32", "tolerance": {"y": "out_fp32"},
        "params": {"B": B, "S": S, "hc": hc, "D": D, "rmsEps": rms_eps, "hcEps": hc_eps},
        "inputs": {"x": T(res.reshape(B, S, hc, D)), "fn": T(fn), "scale": T(scale), "base": T(base)},
        "outputs": {"y": O(y.reshape(B, S, D))},
    }


def hcchain_case(name, group, *, B=1, S=3, hc=4, D=8, repeat=20):
    c = Case(name, "float32")
    res, fn, scale, base = hc_inputs(c, B, S, hc, D)
    y, post, comb = R.mhc_pre(res, fn, scale, base, 1e-6, 1e-6, repeat)
    sub = 0.5 * y + 0.1  # the stand-in sublayer (a fixed affine map, applied by the test, not by Swift model code)
    out = R.mhc_post(sub, res, post, comb)
    return {
        "name": name, "kind": "hcchain", "group": group, "dtype": "float32", "tolerance": {"y": "out_fp32"},
        "params": {"B": B, "S": S, "hc": hc, "D": D, "sinkhornIters": repeat, "rmsEps": 1e-6, "hcEps": 1e-6,
                   "sublayerScale": 0.5, "sublayerShift": 0.1},
        "inputs": {"x": T(res.reshape(B, S, hc, D)), "fn": T(fn), "scale": T(scale), "base": T(base)},
        "outputs": {"y": O(out.reshape(B, S, hc, D))},
    }


# ---------------------------------------------------------------------------------------------------------------
def build_cases():
    cs = []
    # learned (bias) routing --------------------------------------------------------------------------------
    cs.append(gate_like("gate_fp32_renorm_scale", "gate", "float32", bias_scale=0.4))
    cs.append(gate_like("gate_fp32_no_renorm", "gate", "float32", renorm=False, scale=1.5, bias_scale=0.4))
    cs.append(gate_like("gate_fp32_renorm_unit_scale", "gate", "float32", scale=1.0, bias_scale=0.4))
    cs.append(gate_like("gate_fp32_no_renorm_unit_scale", "gate", "float32", renorm=False, scale=1.0, bias_scale=0.4))
    cs.append(gate_like("gate_fp32_bias_flips_selection", "gate", "float32", S=6, bias_scale=1.2, force_flip=True))
    cs.append(gate_like("gate_fp32_k6_e32", "gate", "float32", S=5, H=32, E=32, k=6, bias_scale=0.4))
    cs.append(gate_like("gate_fp32_wide_logits", "gate", "float32", S=5, wscale=7.0, xscale=1.0, bias_scale=0.4,
                        max_abs_logit=30.0, min_gap=1e-2))
    cs.append(gate_like("gate_fp32_negative_tail", "gate", "float32", S=5, wscale=3.0, bias_scale=0.2, max_abs_logit=14.0))
    cs.append(gate_like("gate_fp32_batch2", "gate", "float32", B=2, S=3, bias_scale=0.4))
    cs.append(gate_like("gate_fp32_k1_renorm", "gate_edge", "float32", k=1, renorm=True, scale=1.5, bias_scale=0.4))
    cs.append(gate_like("gate_fp32_tie_at_boundary", "gate_edge", "float32", S=3, E=8, k=3, tie=True, min_gap=1e-3))
    cs.append(gate_like("gate_bf16_basic", "gate_bf16", "bfloat16", S=4, wscale=1.2, bias_scale=0.4,
                        max_abs_logit=4.0, min_gap=8e-2))
    cs.append(gate_like("gate_bf16_k6_e32", "gate_bf16", "bfloat16", S=4, H=32, E=32, k=6, wscale=1.0, bias_scale=0.4,
                        max_abs_logit=4.0, min_gap=8e-2))
    # hash routing ------------------------------------------------------------------------------------------
    cs.append(gate_like("hash_fp32_renorm_scale", "hash", "float32", hash_=True))
    cs.append(gate_like("hash_fp32_no_renorm", "hash", "float32", hash_=True, renorm=False))
    cs.append(gate_like("hash_fp32_unit_scale", "hash", "float32", hash_=True, scale=1.0))
    cs.append(gate_like("hash_fp32_repeated_token", "hash", "float32", hash_=True, S=4, ids_override=[3, 7, 3, 7]))
    cs.append(gate_like("hash_fp32_edge_ids", "hash", "float32", hash_=True, S=4, vocab=24, ids_override=[0, 23, 0, 23]))
    cs.append(gate_like("hash_fp32_batch2_k6", "hash", "float32", hash_=True, B=2, S=3, H=32, E=32, k=6))
    cs.append(gate_like("hash_bf16_basic", "hash_bf16", "bfloat16", hash_=True, wscale=1.2, max_abs_logit=4.0))
    cs.append(gate_like("hash_fp32_k1_renorm", "gate_edge", "float32", hash_=True, k=1))
    # clamped SwiGLU: the shared expert ---------------------------------------------------------------------
    cs.append(expert_case("expert_limit_inactive", "swiglu", limit=100.0))
    cs.append(expert_case("expert_up_outside", "swiglu", limit=1.0, wscale=2.5, need_up_out=True))
    cs.append(expert_case("expert_gate_above", "swiglu", limit=1.0, wscale=2.5, need_gate_hi=True))
    cs.append(expert_case("expert_gate_below", "swiglu", limit=0.5, wscale=2.5, need_gate_lo=True))
    cs.append(expert_case("expert_all_clamps", "swiglu", S=6, limit=0.5, wscale=2.5, need_gate_lo=True,
                          need_gate_hi=True, need_up_out=True))
    # the MoE as a whole ------------------------------------------------------------------------------------
    cs.append(moe_case("moe_learned_limit_inactive", "moe", limit=100.0))
    cs.append(moe_case("moe_hash_limit_inactive", "moe", hash_=True, limit=100.0))
    cs.append(moe_case("moe_learned_limit_active", "moe_clamp", limit=0.5, wscale=2.5))
    cs.append(moe_case("moe_hash_limit_active", "moe_clamp", hash_=True, limit=0.5, wscale=2.5))
    # mHC -----------------------------------------------------------------------------------------------------
    cs.append(hcpre_case("hcpre_hc4_rep20", "hcpre"))
    cs.append(hcpre_case("hcpre_hc4_rep1", "hcpre", repeat=1))
    cs.append(hcpre_case("hcpre_hc4_rep2", "hcpre", repeat=2))
    cs.append(hcpre_case("hcpre_hc2_rep5", "hcpre", hc=2, repeat=5))
    cs.append(hcpre_case("hcpre_batch2", "hcpre", B=2, S=2))
    cs.append(hcpre_case("hcpre_peaky_logits", "hcpre", fn_scale=1.2, scale_scale=2.0, base_scale=1.5))
    cs.append(hcpre_case("hcpre_zero_fn_base_only", "hcpre", zero_fn=True, base_scale=1.0))
    cs.append(hcpre_case("hcpre_large_hc_eps", "hcpre_eps", hc_eps=1e-2, rms_eps=1e-2))
    cs.append(hcpre_case("hcpre_rms_eps_differs", "hcpre_eps", rms_eps=1e-2, hc_eps=1e-6))
    cs.append(hcpre_case("hcpre_hc_eps_differs", "hcpre_eps", rms_eps=1e-6, hc_eps=1e-2))
    cs.append(hcpost_case("hcpost_fp32_hc4", "hcpost"))
    cs.append(hcpost_case("hcpost_fp32_hc2", "hcpost", hc=2))
    cs.append(hcpost_case("hcpost_fp32_batch2", "hcpost", B=2, S=2))
    cs.append(hcpost_case("hcpost_bf16_hc4", "hcpost_bf16", dtype="bfloat16"))
    cs.append(hchead_case("hchead_fp32_hc4", "hchead"))
    cs.append(hchead_case("hchead_fp32_hc2", "hchead", hc=2))
    cs.append(hchead_case("hchead_rms_eps_differs", "hchead_eps", rms_eps=1e-2, hc_eps=1e-6))
    cs.append(hchead_case("hchead_hc_eps_differs", "hchead_eps", rms_eps=1e-6, hc_eps=1e-2))
    cs.append(hcchain_case("hcchain_hc4_rep20", "hcchain"))
    cs.append(hcchain_case("hcchain_hc2_rep3", "hcchain", hc=2, repeat=3))
    return cs


def sha256_file(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def main():
    os.makedirs(OUT, exist_ok=True)
    for f in os.listdir(OUT):
        if f.endswith(".json"):
            os.remove(os.path.join(OUT, f))
    cases = build_cases()
    names = [c["name"] for c in cases]
    assert len(set(names)) == len(names)
    files = {}
    used_tol = set()
    for c in cases:
        for t in c["tolerance"].values():
            used_tol.add(t)
        path = os.path.join(OUT, c["name"] + ".json")
        with open(path, "w") as fh:
            json.dump(c, fh, sort_keys=True, separators=(",", ":"))
        files[c["name"] + ".json"] = {"sha256": sha256_file(path), "kind": c["kind"], "group": c["group"],
                                      "dtype": c["dtype"], "tolerance": c["tolerance"]}
    manifest = {
        "schema": 1,
        "count": len(files),
        "files": files,
        "vllmCommit": VLLM_COMMIT,
        "torch": torch.__version__,
        "generator": "tools/conformance/golden/gen_v4route_golden.py",
        "generatorSha256": sha256_file(os.path.join(HERE, "gen_v4route_golden.py")),
        "oracle": "tools/conformance/golden/v4route_ref.py",
        "oracleSha256": sha256_file(os.path.join(HERE, "v4route_ref.py")),
        "oracleDtype": "float64",
        "criterion": "per tolerance class `mode`: normwise = max|got-want| <= atol + rtol*max|want| over a tensor; "
                     "elementwise = |got_i-want_i| <= atol + rtol*|want_i| for all i; expert ids are compared exactly "
                     "after sorting each token's ids ascending",
        "note": "Generated once and committed; tests never run python. Edit nothing by hand: rerun the generator. "
                "Items inferred rather than taken from vLLM: see v4route_ref.py header I1..I5.",
        "tolerances": {k: TOL[k] for k in sorted(used_tol)},
    }
    with open(os.path.join(OUT, "MANIFEST.json"), "w") as fh:
        json.dump(manifest, fh, sort_keys=True, separators=(",", ":"))
    kinds = {}
    for c in cases:
        kinds[c["kind"]] = kinds.get(c["kind"], 0) + 1
    print(f"wrote {len(cases)} fixtures to {OUT}: {kinds}")


if __name__ == "__main__":
    main()
