"""Cross-check of v4idx_ref.py against vLLM's OWN torch reference code. Run (once, before generating):

    <venv>/bin/python -I tools/conformance/golden/v4idx_crosscheck.py --vllm <vLLM root> (or VLLM_ROOT=<vLLM root>)

Exit code 0 only when every comparison agrees to 1e-10 relative (normwise, max|a-b| / max|b|) or, for the raw float32
vLLM code, to the float32 bound stated next to it. The vLLM functions are NOT retyped: they are extracted with `ast` from the
vLLM files (so they cannot drift from what is in the tree) and executed. Two runs of each:
  raw    exactly as written in vLLM (float32 `.float()` casts) -> must agree with the float64 oracle to <= 2e-5 relative
         (float32 rounding: 24 bits -> a few 1e-7 per op, a handful of ops, the softmax and the dot products)
  f64    the same source text with the mechanical substitutions `.float()` -> `.double()` and `torch.float32` -> `torch.float64`
         (nothing else) -> must agree to <= 1e-10 relative. THIS is the check of the formulae; "raw" shows that the only
         disagreement of the raw code is its own float32 precision.

References used:
  R1 tests/kernels/test_deepseek_v4_cpu_kernels.py::_apply_gptj_rope           vs v4idx_ref.gptj_rope
  R2 tests/kernels/test_deepseek_v4_cpu_kernels.py::_reference_paged_mqa_logits vs v4idx_ref.index_scores (the dot/relu/weights part)
  R3 tests/kernels/test_deepseek_v4_cpu_kernels.py::_flash_mla_eager            vs v4idx_ref.merge_attention
  R4 tests/kernels/test_deepseek_v4_cpu_kernels.py::_indexer_q_rope_quant_reference : the weight scaling and the dequantised q
     (INFORMATIONAL for the quantisation part: fp8 e4m3 has 3 mantissa bits and the RoPE half is rounded to bfloat16, so the
     dequantised q differs from the exact q by up to ~2^-4 relative per element. The formula for the weights is exact.)
  R5 vllm/models/deepseek_v4/cpu/cpu_sparse.py::DeepseekV4CPUIndexer._fill_short_context_topk (the short-context "select every
     candidate" path) vs v4idx_ref.visible_blocks / topk_select
  R6 tests/v1/attention/test_deepseek_v4_swa_visible.py::ref_swa_bounds (no image span) and the inline window code of
     cpu_sparse.py (lines 587-590, retyped op for op because it is inside forward_mqa) vs v4idx_ref.swa_window_bounds
  R7 torch.topk (the reference of tests/kernels/test_top_k_per_row.py:213-243) vs v4idx_ref.topk_select, incl. ties and k > row length
  R8 vllm/v1/attention/backends/mla/indexer.py:549-551 (the Triton kernel's formula, retyped) vs v4idx_ref.visible_blocks
"""
import argparse
import ast
import os
import sys
import textwrap
import types

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v4idx_ref as ref  # noqa: E402

F64 = torch.float64
N_CASES = 220
failures = []


def rel(a, b):
    a, b = torch.as_tensor(a, dtype=F64), torch.as_tensor(b, dtype=F64)
    if b.numel() == 0:
        return 0.0
    return float((a - b).abs().max() / max(float(b.abs().max()), 1e-300))


def check(name, err, tol, note=""):
    ok = err <= tol
    print(f"  {'ok  ' if ok else 'FAIL'} {name}: rel err {err:.3e} (limit {tol:.0e}) {note}")
    if not ok:
        failures.append(name)


def extract(path, names, f64):
    """Source text of the named top-level functions (or Class.method) of a python file, exec'ed in a fresh namespace."""
    src = open(path).read()
    tree = ast.parse(src)
    segs = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in names:
            segs.append(textwrap.dedent(ast.get_source_segment(src, node, padded=True)))
    ns = {"torch": torch, "F": torch.nn.functional, "FP8_MAX": 448.0, "HEAD_DIM": 512, "NOPE_DIM": 448, "ROPE_DIM": 64,
          "QUANT_BLOCK": 64, "INDEXER_HEAD_DIM": 128}
    for seg in segs:
        if f64:
            seg = seg.replace(".float()", ".double()").replace("torch.float32", "torch.float64")
        # strip decorators / annotations that need vllm imports
        exec(compile(ast.parse(seg), "<vllm>", "exec"), ns)
    missing = [n for n in names if n not in ns]
    assert not missing, f"not found in {path}: {missing}"
    return ns


def gen(seed):
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vllm", default=None, help="vLLM checkout root (commit 2eae56c; the directory that contains vllm/ and "
                                                 "tests/); or set VLLM_ROOT")
    a = ap.parse_args()
    root = a.vllm or os.environ.get("VLLM_ROOT")
    if not root or not os.path.isdir(os.path.join(root, "vllm")):
        ap.error("need a vLLM checkout: pass --vllm <root> or set VLLM_ROOT (the directory that contains vllm/)")
    kern = os.path.join(root, "tests/kernels/test_deepseek_v4_cpu_kernels.py")
    swa = os.path.join(root, "tests/v1/attention/test_deepseek_v4_swa_visible.py")
    cpu = os.path.join(root, "vllm/models/deepseek_v4/cpu/cpu_sparse.py")

    # ---- R1 rope ---------------------------------------------------------------------------------------------------
    print("R1 rope")
    for f64 in (False, True):
        ns = extract(kern, ["_apply_gptj_rope"], f64)
        worst = 0.0
        for i in range(N_CASES):
            g = gen(1000 + i)
            rd = int(torch.randint(1, 5, (1,), generator=g)) * 2
            theta = float(10 ** (3 + 3 * torch.rand(1, generator=g)))
            n = int(torch.randint(1, 6, (1,), generator=g))
            x = torch.randn(n, 3, rd, generator=g, dtype=F64)
            pos = torch.randint(0, 200, (n,), generator=g)
            ang = pos.to(F64).unsqueeze(-1) * ref.inv_freq(theta, rd)         # the cos/sin cache row of base.py:96-102
            cos, sin = ang.cos().unsqueeze(1), ang.sin().unsqueeze(1)
            if not f64:
                x, cos, sin = x.float(), cos.float(), sin.float()
            want = ns["_apply_gptj_rope"](x, cos, sin)
            got = ref.gptj_rope(x.to(F64), pos.unsqueeze(-1).expand(n, 3), theta)
            worst = max(worst, rel(got, want))
        check("gptj_rope " + ("f64" if f64 else "raw(f32)"), worst, 1e-10 if f64 else 2e-5)

    # ---- R2 scores -------------------------------------------------------------------------------------------------
    print("R2 index scores")
    for f64 in (False, True):
        ns = extract(kern, ["_reference_paged_mqa_logits", "_apply_gptj_rope"], f64)
        worst = 0.0
        for i in range(N_CASES):
            g = gen(2000 + i)
            nh = int(torch.randint(1, 6, (1,), generator=g))
            hd = int(torch.randint(2, 6, (1,), generator=g)) * 4
            rd = int(torch.randint(1, hd // 2 + 1, (1,), generator=g)) * 2
            L = int(torch.randint(1, 14, (1,), generator=g))
            start = int(torch.randint(0, 30, (1,), generator=g))
            nb = (start + L) // 4
            hidden, ql = 6, 5
            theta = float(10 ** (3 + 2 * torch.rand(1, generator=g)))
            x = torch.randn(L, hidden, generator=g, dtype=F64)
            qr = torch.randn(L, ql, generator=g, dtype=F64)
            wq_b = torch.randn(nh * hd, ql, generator=g, dtype=F64)
            wp = torch.randn(nh, hidden, generator=g, dtype=F64)
            keys = torch.randn(nb, hd, generator=g, dtype=F64) if nb else torch.zeros(0, hd, dtype=F64)
            want_full = ref.index_scores(x, qr, wq_b, wp, keys, start, nh, hd, rd, theta)
            # vLLM's reference, one query token at a time (its batch dim is the token): q after RoPE, folded weights,
            # K of that token's visible entries, scale 1.
            q = (qr @ wq_b.T).view(L, nh, hd)
            pos = start + torch.arange(L)
            ang = pos.to(F64).unsqueeze(-1) * ref.inv_freq(theta, rd)
            cos, sin = ang.cos().unsqueeze(1), ang.sin().unsqueeze(1)
            q = torch.cat([q[..., : hd - rd], _rope_with(ns, q[..., hd - rd:], cos, sin, f64)], dim=-1)
            w = (x @ wp.T) * hd ** -0.5 * nh ** -0.5
            nvis = (pos + 1) // 4
            ref_k = [keys[: int(n)] for n in nvis]
            ref_scale = [torch.ones(int(n), dtype=F64 if f64 else torch.float32) for n in nvis]
            max_len = max(1, int(nvis.max()))
            if not f64:
                q, w = q.float(), w.float()
                ref_k = [k.float() for k in ref_k]
            out = ns["_reference_paged_mqa_logits"](q, w, ref_k, ref_scale, max_len)
            mask = torch.isfinite(want_full)
            got = torch.zeros(L, max_len, dtype=F64)
            wf = torch.where(mask, want_full, torch.zeros_like(want_full))[:, :max_len] if nb else torch.zeros(L, max_len, dtype=F64)
            worst = max(worst, rel(wf, out.to(F64)))
            # the -inf pattern must be exactly "column >= visible count"
            assert all(bool(mask[t, b]) == (b < int(nvis[t])) for t in range(L) for b in range(nb)), "visibility pattern"
        check("index_scores " + ("f64" if f64 else "raw(f32)"), worst, 1e-10 if f64 else 2e-5)

    # ---- R3 merge --------------------------------------------------------------------------------------------------
    print("R3 merge attention")
    for f64 in (False, True):
        ns = extract(kern, ["_flash_mla_eager"], f64)
        worst = 0.0
        for i in range(N_CASES):
            g = gen(3000 + i)
            H = int(torch.randint(1, 4, (1,), generator=g))
            D = int(torch.randint(2, 6, (1,), generator=g)) * 2
            T = int(torch.randint(1, 7, (1,), generator=g))
            S = int(torch.randint(3, 12, (1,), generator=g))
            nb = int(torch.randint(0, 6, (1,), generator=g))
            kv = torch.randn(S, D, generator=g, dtype=F64)
            comp = torch.randn(nb, D, generator=g, dtype=F64) if nb else torch.zeros(0, D, dtype=F64)
            q = torch.randn(T, H, D, generator=g, dtype=F64)
            scale = D ** -0.5
            sink = torch.randn(H, generator=g, dtype=F64) * 2 if i % 3 else torch.full((H,), float("-inf"), dtype=F64)
            win_slots = torch.where(torch.rand(T, S, generator=g) > 0.35, torch.arange(S).expand(T, S), torch.full((T, S), -1))
            win_slots[:, 0] = torch.where(win_slots.max(dim=1).values < 0, 0, win_slots[:, 0])   # >= 1 valid row per token
            comp_slots = torch.where(torch.rand(T, nb, generator=g) > 0.4, torch.arange(nb).expand(T, nb), torch.full((T, nb), -1))
            args = (q, kv, win_slots, comp if nb else kv.new_zeros(0, D), comp_slots if nb else win_slots.new_full((T, 0), -1), sink, scale)
            if not f64:
                args = (q.float(), kv.float(), win_slots, comp.float() if nb else kv.new_zeros(0, D).float(),
                        comp_slots if nb else win_slots.new_full((T, 0), -1), sink.float(), scale)
            want = ns["_flash_mla_eager"](*args).to(F64)                       # [T, H, D]
            win_vis = (win_slots >= 0).tolist()
            comp_vis = (comp_slots >= 0).tolist() if nb else [[] for _ in range(T)]
            got = ref.merge_attention(q.permute(1, 0, 2), kv, comp, win_vis, comp_vis, sink, scale).permute(1, 0, 2)
            worst = max(worst, rel(got, want))
        check("merge_attention " + ("f64" if f64 else "raw(f32)"), worst, 1e-10 if f64 else 2e-5)

    # ---- R4 q rope + weights scaling (informational on the fp8 part) --------------------------------------------------
    print("R4 indexer q: weights scaling exact, dequantised q within fp8/bf16 quantisation")
    ns = extract(kern, ["_indexer_q_rope_quant_reference", "_apply_gptj_rope"], False)
    worst_w, worst_q = 0.0, 0.0
    for i in range(40):
        g = gen(4000 + i)
        T, H, D, RD = 7, 4, 32, 8
        theta = 10000.0
        q = torch.randn(T, H, D, generator=g)
        pos = torch.randint(0, 120, (T,), generator=g)
        cs = torch.zeros(128, RD)
        ang = torch.arange(128, dtype=F64).unsqueeze(-1) * ref.inv_freq(theta, RD)
        cs[:, : RD // 2], cs[:, RD // 2:] = ang.cos().float(), ang.sin().float()
        w = torch.randn(T, H, generator=g)
        ss, hs = D ** -0.5, H ** -0.5
        q8, wout = ns["_indexer_q_rope_quant_reference"](pos, q, cs, w, ss, hs)
        scale = wout / (w * ss * hs)                                           # = the per-(token, head) fp8 scale
        p2 = torch.log2(scale.abs())
        assert torch.allclose(p2, p2.round(), atol=1e-4), "fp8 scale is not a power of two"
        want_q = ref.rope_tail(q.to(F64), pos.unsqueeze(-1).expand(T, H), theta, RD)
        deq = q8.float() * scale.unsqueeze(-1)
        worst_q = max(worst_q, rel(deq, want_q))
        worst_w = max(worst_w, rel(wout / scale, (w * ss * hs)))
    check("weights_out/scale == w * softmax_scale * head_scale", worst_w, 1e-6)
    print(f"  info dequantised fp8 q vs exact q: max normwise rel err {worst_q:.3e} (fp8 e4m3 + bf16 RoPE half; NOT modelled, I1)")
    assert worst_q < 0.1

    # ---- R5 short-context fill, R8 kernel formula -------------------------------------------------------------------
    print("R5/R8 visibility and the short-context top-k fill")
    ns = extract(cpu, ["_fill_short_context_topk"], False)
    fill = ns["_fill_short_context_topk"]
    worst = 0
    for i in range(N_CASES):
        g = gen(5000 + i)
        ratio = 4
        topk = int(torch.randint(1, 20, (1,), generator=g))
        n = int(torch.randint(1, 12, (1,), generator=g))
        positions = torch.randint(0, 120, (n,), generator=g)
        self_ = types.SimpleNamespace(topk_tokens=topk, compress_ratio=ratio,
                                      topk_indices_buffer=torch.full((n + 2, topk), -7, dtype=torch.int32))
        fill(self_, positions, n)
        for t in range(n):
            row = self_.topk_indices_buffer[t].tolist()
            nvis = ref.visible_blocks(int(positions[t]), ratio)
            # my oracle: scores that are finite exactly for the visible columns, k = topk
            sc = [float(b) if b < nvis else float("-inf") for b in range(max(nvis, 1) + 3)]
            sel = ref.topk_select(sc, topk)
            chosen = sorted(sel["strict"] + sel["tie"][: sel["need"]])
            if nvis <= topk:                                                    # every candidate selected, in order, rest -1
                want_row = list(range(nvis)) + [-1] * (topk - nvis)
                worst += int(row != want_row) + int(chosen != list(range(nvis)))
            # R8: the Triton kernel formula len_per_token = (start_pos + 1 + offset) // COMPRESS_RATIO (indexer.py:549-551)
            start_pos, offset = int(positions[t]) - 3, 3
            worst += int((start_pos + 1 + offset) // ratio != nvis)
    check("fill_short_context_topk / kernel formula (mismatching rows)", float(worst), 0.0)

    # ---- R6 SWA window --------------------------------------------------------------------------------------------------
    print("R6 SWA window")
    ns = extract(swa, ["ref_swa_bounds"], False)
    bad = 0
    for i in range(N_CASES):
        g = gen(6000 + i)
        window = int(torch.randint(1, 16, (1,), generator=g))
        n = int(torch.randint(1, 40, (1,), generator=g))
        positions = torch.arange(n)
        # cpu_sparse.py:587-590, op for op
        start_pos = (positions - window + 1).clamp(min=0)
        local_pos = start_pos.unsqueeze(-1) + torch.arange(window)
        valid_win = local_pos <= positions.unsqueeze(-1)
        for p in range(n):
            cpu_rows = local_pos[p][valid_win[p]].tolist()
            lo, hi = ns["ref_swa_bounds"](p, window, 0, 0)
            mine = list(range(*ref.swa_window_bounds(p, window)))
            bad += int(cpu_rows != mine) + int(list(range(lo, hi)) != mine)
    check("swa window (mismatching rows)", float(bad), 0.0)

    # ---- R7 top-k vs torch.topk -------------------------------------------------------------------------------------
    print("R7 top-k vs torch.topk")
    bad = 0
    for i in range(N_CASES):
        g = gen(7000 + i)
        n = int(torch.randint(1, 20, (1,), generator=g))
        nvis = int(torch.randint(0, n + 1, (1,), generator=g))
        k = int(torch.randint(1, 24, (1,), generator=g))
        vals = torch.randint(-3, 4, (n,), generator=g).to(F64) * 0.5            # many ties
        row = [float(v) if j < nvis else float("-inf") for j, v in enumerate(vals)]
        sel = ref.topk_select(row, k)
        k_i = min(k, nvis)
        if k_i == 0:
            bad += int(sel != dict(strict=[], tie=[], need=0))
            continue
        tv = vals[:nvis].topk(k_i).values.sort().values                          # test_top_k_per_row.py:238-241
        chosen = sel["strict"] + sel["tie"][: sel["need"]]
        bad += int(sorted(row[j] for j in chosen) != tv.tolist()) + int(len(chosen) != k_i)
        # any other choice of `need` tied indices gives the same values
        if sel["need"]:
            alt = sel["strict"] + sel["tie"][-sel["need"]:]
            bad += int(sorted(row[j] for j in alt) != tv.tolist())
    check("topk_select (mismatching rows)", float(bad), 0.0)

    if failures:
        print("FAILED:", failures)
        sys.exit(1)
    print("OK")


def _rope_with(ns, x, cos, sin, f64):
    fn = ns["_apply_gptj_rope"]
    if f64:
        return fn(x, cos, sin)
    return fn(x.float(), cos.float(), sin.float()).to(F64)


if __name__ == "__main__":
    main()
