"""Cross-check of kda_ref.py against independent formulations of the same vLLM maths. Run:
    uv run --no-project --python <venv>/bin/python -I tools/conformance/golden/kda_crosscheck.py
(or just `python -I kda_crosscheck.py` inside the venv). Exit code 0 only if every case agrees to 1e-10
relative (normwise: max|a-b| / max(max|b|, 1e-300)).

Independent formulations used (none shares code with kda_ref.recurrent):
  V  vectorised einsum recurrence over all heads and the batch at once (no per-tile loops).
  C  the chunk_kda formulation (vllm/model_executor/layers/fla/ops/kda.py:1170-1227 chunk_kda_fwd):
       chunk-local cumsum of log decay                       kda.py:1182
       A  (strict lower) = (beta k) . (k exp(G_i - G_j))     kda.py:698-708  (kkt kernel)
       Aqk (lower+diag)  = scale * q . (k exp(G_i - G_j))     kda.py:709-710
       T = (I + A)^-1                                         solve_tril.py:514 docstring ("inverse of the matrix I + A")
       u = T (beta v)                                         kda.py:882-883 ; w = T (beta k exp(G))   kda.py:904-915, 956
       v_new = u - w h^T                                      chunk_delta_h.py:175, 197
       o = (q scale exp(G)) h^T + Aqk v_new                   kda.py:1088-1098, 1123
       h <- h exp(G_last) + v_new^T (k exp(G_last - G))       chunk_delta_h.py:230 ; kda.py:944 (kg)
     chunk_kda has no GQA and no mask (q, k, v, g all have H heads), so C is checked with Hk == Hv and
     mask=None, and (for GQA) with k/q repeat_interleave'd up front.
  G  gate vs the torch code in vLLM's tests:
       tests/kernels/test_fused_sigmoid_gating_delta_rule.py:68  -exp(A_log) * F.softplus(a + dt_bias)
       tests/kernels/test_fused_recurrent_packed_decode.py:61-64 where(x <= 20, log1p(exp(clamp(x, max=20))), x)
     and F.softplus(x, beta, threshold) for softplus beta != 1.
vLLM has NO torch implementation of the recurrence in its tests (they compare Triton against Triton),
so V and C are the only independent references for it.
"""
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
import kda_ref  # noqa: E402

F64 = torch.float64
TOL = 1e-10


def rel(a, b):
    return float((a - b).abs().max() / max(float(b.abs().max()), 1e-300)) if b.numel() else 0.0


def l2_all(x, eps=1e-6):
    return x / torch.sqrt((x * x).sum(-1, keepdim=True) + eps)  # FR:128-129 / L2:91


def vectorised(q, k, v, gk, beta, state, scale, l2, mask):
    B, T, Hk, Dk = k.shape
    Hv, Dv = v.shape[2], v.shape[3]
    if l2:
        q, k = l2_all(q), l2_all(k)
    q = q * scale
    q = q.repeat_interleave(Hv // Hk, dim=2)
    k = k.repeat_interleave(Hv // Hk, dim=2)
    S = torch.zeros(B, Hv, Dv, Dk, dtype=F64) if state is None else state.clone()
    y = torch.zeros(B, T, Hv, Dv, dtype=F64)
    for t in range(T):
        m = torch.ones(B, dtype=torch.bool) if mask is None else mask[:, t]
        Sn = S * torch.exp(gk[:, t])[:, :, None, :]
        u = (v[:, t] - torch.einsum("bhvk,bhk->bhv", Sn, k[:, t])) * beta[:, t][..., None]
        Sn = Sn + u[..., None] * k[:, t][:, :, None, :]
        yt = torch.einsum("bhvk,bhk->bhv", Sn, q[:, t])
        S = torch.where(m[:, None, None, None], Sn, S)
        y[:, t] = torch.where(m[:, None, None], yt, torch.zeros_like(yt))
    return y, S


def chunked(q, k, v, gk, beta, state, scale, l2, C):
    B, T, Hk, Dk = k.shape
    Hv, Dv = v.shape[2], v.shape[3]
    if l2:  # kda.py:1246-1248
        q, k = l2_all(q), l2_all(k)
    q = q.repeat_interleave(Hv // Hk, dim=2)
    k = k.repeat_interleave(Hv // Hk, dim=2)
    S = torch.zeros(B, Hv, Dv, Dk, dtype=F64) if state is None else state.clone()
    y = torch.zeros(B, T, Hv, Dv, dtype=F64)
    for s in range(0, T, C):
        e = min(T, s + C)
        c = e - s
        for b in range(B):
            for h in range(Hv):
                Gc = torch.cumsum(gk[b, s:e, h], dim=0)  # kda.py:1182  [c, Dk]
                qc, kc, vc, bc = q[b, s:e, h], k[b, s:e, h], v[b, s:e, h], beta[b, s:e, h]
                kb = kc * bc[:, None]
                dec = torch.exp(Gc[:, None, :] - Gc[None, :, :])  # [i, j, Dk] = exp(G_i - G_j)
                A = torch.einsum("id,jd,ijd->ij", kb, kc, dec).tril(-1)  # kda.py:707-708 (o_i > j)
                Aqk = (scale * torch.einsum("id,jd,ijd->ij", qc, kc, dec)).tril(0)  # kda.py:709-710 (o_i >= j)
                Tm = torch.linalg.inv(torch.eye(c, dtype=F64) + A)  # solve_tril.py:514
                u = Tm @ (vc * bc[:, None])  # kda.py:882-883
                w = Tm @ (kb * torch.exp(Gc))  # kda.py:904, 915, 956
                v_new = u - w @ S[b, h].T  # chunk_delta_h.py:175, 197   (h is [Dk, Dv] there; S is [Dv, Dk] here)
                o = (qc * scale * torch.exp(Gc)) @ S[b, h].T + Aqk @ v_new  # kda.py:1088-1098, 1123
                kg = kc * torch.exp(Gc[-1][None, :] - Gc)  # kda.py:944
                S[b, h] = S[b, h] * torch.exp(Gc[-1])[None, :] + v_new.T @ kg  # chunk_delta_h.py:230
                y[b, s:e, h] = o
    return y, S


def rand_case(rng, force_equal_heads=False, allow_mask=True):
    B = int(rng.integers(1, 3))
    T = int(rng.integers(1, 40))
    Hk = int(rng.choice([1, 2, 3]))
    Hv = Hk if force_equal_heads else Hk * int(rng.choice([1, 2, 3]))
    Dk = int(rng.choice([1, 2, 5, 8, 16]))
    Dv = int(rng.choice([1, 3, 4, 8]))
    g = torch.Generator().manual_seed(int(rng.integers(0, 2**31)))
    r = lambda *s: torch.randn(*s, generator=g, dtype=F64)
    q, k, v = r(B, T, Hk, Dk), r(B, T, Hk, Dk), r(B, T, Hv, Dv)
    l2 = bool(rng.integers(0, 2))
    if not l2:  # keep the delta rule stable without the l2 norm: unit-norm k
        k = k / k.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    gk = -torch.rand(B, T, Hv, Dk, generator=g, dtype=F64) * float(rng.choice([0.01, 0.3, 2.0, 30.0]))
    beta = torch.rand(B, T, Hv, generator=g, dtype=F64)
    state = r(B, Hv, Dv, Dk) if rng.integers(0, 2) else None
    scale = float(rng.choice([1.0, 0.5, Dk ** -0.5]))
    mask = (torch.rand(B, T, generator=g) > 0.4) if (allow_mask and rng.integers(0, 2)) else None
    return q, k, v, gk, beta, state, scale, l2, mask


def check_fixtures(directory):
    """Re-read the COMMITTED JSON (not the in-memory oracle result) and recompute every case with the independent
    vectorised formulation / torch softplus; proves the serialised text (layout, constants, 10-digit outputs) is right."""
    import glob, json
    bad, n = [], 0
    for path in sorted(glob.glob(os.path.join(directory, "*.json"))):
        if path.endswith("MANIFEST.json"):
            continue
        fx = json.load(open(path))
        i, o = fx["inputs"], fx["outputs"]
        # inputs are stored as shortest float32 text: parse them as float32 (as Swift's Float(...) does), outputs as float64
        t = lambda a, *shape: torch.tensor(a, dtype=torch.float32).to(F64).reshape(*shape)
        tout = lambda a, *shape: torch.tensor(a, dtype=F64).reshape(*shape)
        if fx["kind"] == "recurrent":
            B, T, Hk, Hv, Dk, Dv = (fx[x] for x in ("B", "T", "Hk", "Hv", "Dk", "Dv"))
            q, k, v = t(i["q"], B, T, Hk, Dk), t(i["k"], B, T, Hk, Dk), t(i["v"], B, T, Hv, Dv)
            gk = torch.full((B, T, Hv, Dk), i["logDecayConst"], dtype=F64) if i["logDecayConst"] is not None else t(i["logDecay"], B, T, Hv, Dk)
            beta = torch.full((B, T, Hv), i["betaConst"], dtype=F64) if i["betaConst"] is not None else t(i["beta"], B, T, Hv)
            st = None if i["state0"] is None else t(i["state0"], B, Hv, Dv, Dk)
            mask = None if i["mask"] is None else torch.tensor(i["mask"], dtype=torch.bool)
            y, s = vectorised(q, k, v, gk, beta, st, fx["scale"], fx["l2norm"], mask)
            e = max(rel(y, tout(o["y"], B, T, Hv, Dv)), rel(s, tout(o["state1"], B, Hv, Dv, Dk)))
        else:
            T, H, D = fx["T"], fx["H"], fx["D"]
            x = t(i["g"], T, H, D) + (t(i["dtBias"], 1, H, D) if i["dtBias"] is not None else 0.0)
            a = t(i["aLog"], 1, H, 1).exp()
            if fx["lowerBound"] is None:
                want = -a * F.softplus(x, beta=fx["softplusBeta"], threshold=fx["threshold"])
            else:
                want = fx["lowerBound"] * torch.sigmoid(a * x)
            e = rel(want, tout(o["logDecay"], T, H, D))
        n += 1
        if e > 1e-9:
            bad.append((os.path.basename(path), e))
    print(f"fixtures re-derived from committed JSON: {n}, disagreements: {bad}")
    return bad


def main():
    import numpy as np

    rng = np.random.default_rng(20261029)
    worst = {"V": 0.0, "C": 0.0, "G": 0.0}
    n = {"V": 0, "C": 0, "G": 0}
    bad = []
    for i in range(240):  # recurrent vs vectorised (GQA, state, mask, l2, scale)
        q, k, v, gk, beta, st, scale, l2, mask = rand_case(rng)
        y, s = kda_ref.recurrent(q, k, v, gk, beta, st, scale, l2, mask)
        y2, s2 = vectorised(q, k, v, gk, beta, st, scale, l2, mask)
        e = max(rel(y, y2), rel(s, s2))
        worst["V"] = max(worst["V"], e); n["V"] += 1
        if e > TOL: bad.append(("V", i, e))
    for i in range(240):  # recurrent vs chunk_kda formulation (no mask; GQA via repeat)
        q, k, v, gk, beta, st, scale, l2, _ = rand_case(rng, allow_mask=False)
        C = int(rng.choice([1, 2, 4, 7, 16, 64]))
        y, s = kda_ref.recurrent(q, k, v, gk, beta, st, scale, l2, None)
        y2, s2 = chunked(q, k, v, gk, beta, st, scale, l2, C)
        # chunk form exponentiates differences of cumulative sums; strongly negative gates make
        # exp(G_i - G_j) fine (<=1) but exp(G) underflow to 0 harmlessly, so no extra tolerance needed.
        e = max(rel(y, y2), rel(s, s2))
        worst["C"] = max(worst["C"], e); n["C"] += 1
        if e > TOL: bad.append(("C", i, e))
    for i in range(240):  # gate
        H = int(rng.integers(1, 5)); D = int(rng.choice([1, 4, 16]))
        T = int(rng.integers(1, 6))
        g = torch.Generator().manual_seed(int(rng.integers(0, 2**31)))
        x = (torch.randn(T, H * D, generator=g, dtype=F64) * float(rng.choice([1.0, 10.0, 30.0])))
        A = torch.randn(H, generator=g, dtype=F64)
        bias = torch.randn(H * D, generator=g, dtype=F64) if rng.integers(0, 2) else None
        sp_beta = float(rng.choice([1.0, 1.0, 0.5, 2.0]))
        got = kda_ref.kda_gate(x, A, bias, sp_beta, 20.0)
        xx = x + (bias if bias is not None else 0.0)
        xx = xx.reshape(T, H, D)
        a = A.exp()[None, :, None]
        want1 = -a * F.softplus(xx, beta=sp_beta, threshold=20.0)  # torch; test_fused_sigmoid_gating...:68 for beta=1
        e = rel(got, want1)
        if sp_beta == 1.0:  # packed-decode test formula, test_fused_recurrent_packed_decode.py:61-64
            sx = torch.where(xx <= 20.0, torch.log1p(torch.exp(torch.clamp(xx, max=20.0))), xx)
            e = max(e, rel(got, -a * sx))
        worst["G"] = max(worst["G"], e); n["G"] += 1
        if e > TOL: bad.append(("G", i, e))
    golden_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "Tests", "MLXLMTests", "Conformance", "Golden", "kda")
    if os.path.isdir(golden_dir):
        bad += check_fixtures(golden_dir)
    print("cases", n, "worst normwise relative error", worst)
    if bad:
        print("DISAGREEMENTS", bad[:10])
        return 1
    print("OK: all cases agree to", TOL)
    return 0


if __name__ == "__main__":
    sys.exit(main())
