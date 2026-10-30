"""Cross-check dequant_ref.py against vLLM's OWN torch/numpy reference code (739e5945d) on random cases.

vLLM is not importable here (compiled ops, scalar_types), so the needed functions are lifted from the source files with
`ast` and exec'd unchanged in a namespace that provides only torch/numpy plus tiny stubs (listed in STUBS). Nothing is
re-typed: a vLLM edit changes what is checked.

Acceptance: |ours - vLLM| <= tol * max(|vLLM|, tiny); any bound above 1e-10 is explained at the check.
Run:  python gen's venv:  python dequant_crosscheck.py [--vllm DIR]
"""

import argparse
import ast
import os
import sys
import types

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dequant_ref as R  # noqa: E402

VLLM_DEFAULT = "/Users/simba/SwiftLM/3rd_party/vllm"
Q = "vllm/model_executor/layers/quantization/utils/"

# STUBS (the only non-vLLM code that runs): scalar_types.float4_e2m1f.max() (6.0, the E2M1 maximum);
# a ScalarType-like object for uint4b8 / uint4 (size_bits, min, max, bias, flags) -- values from vllm/scalar_type.py
# semantics: uint4b8 has stored range [0,15] and bias 8 (value range [-8, 7]); uint4 is [0, 15] without bias.


def lift(vllm_dir, relpath, names, ns):
    """exec the top-level defs/assigns called `names` from vllm_dir/relpath into ns, in source order."""
    src = open(os.path.join(vllm_dir, relpath)).read()
    tree = ast.parse(src)
    want = set(names)
    got = set()
    for node in tree.body:
        nm = None
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            nm = node.name
        elif isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            nm = node.targets[0].id
        if nm in want:
            exec(compile(ast.Module([node], []), relpath, "exec"), ns)
            got.add(nm)
    missing = want - got
    assert not missing, f"not found in {relpath}: {missing}"


class FakeScalarType:
    def __init__(self, size_bits, lo, hi, bias, signed):
        self.size_bits, self._lo, self._hi, self.bias, self._signed = size_bits, lo, hi, bias, signed

    def min(self):
        return self._lo

    def max(self):
        return self._hi

    def is_integer(self):
        return True

    def is_signed(self):
        return self._signed

    def has_bias(self):
        return self.bias != 0


def _rel_ok(ours, theirs, tol):
    ours, theirs = ours.to(torch.float64), theirs.to(torch.float64)
    nan_o, nan_t = torch.isnan(ours), torch.isnan(theirs)
    assert torch.equal(nan_o, nan_t), "NaN pattern differs"
    fin = ~nan_o
    both = fin & torch.isfinite(ours) & torch.isfinite(theirs)
    d = (ours[both] - theirs[both]).abs()
    scale = theirs[both].abs().clamp(min=1e-300)
    return float((d / scale).max()) if d.numel() else 0.0


def run(vllm_dir=VLLM_DEFAULT, n=240, seed=20260101):
    g = torch.Generator().manual_seed(seed)
    stats = {}

    # ---- torch's own dtypes as an independent oracle for the 8-bit value tables --------------------------------
    allb = torch.arange(256, dtype=torch.uint8)
    t_e4 = allb.view(torch.float8_e4m3fn).to(torch.float64)
    r_e4 = torch.tensor([R.e4m3fn_value(b) for b in range(256)], dtype=torch.float64)
    assert torch.equal(torch.isnan(t_e4), torch.isnan(r_e4)) and torch.equal(t_e4[~torch.isnan(t_e4)], r_e4[~torch.isnan(r_e4)])
    t_e5 = allb.view(torch.float8_e5m2).to(torch.float64)
    r_e5 = torch.tensor([R.e5m2_value(b) for b in range(256)], dtype=torch.float64)
    assert torch.equal(torch.isnan(t_e5), torch.isnan(r_e5)) and torch.equal(t_e5[~torch.isnan(t_e5)], r_e5[~torch.isnan(r_e5)])
    t_e8 = allb.view(torch.float8_e8m0fnu).to(torch.float64)
    for b in range(255):
        assert t_e8[b].item() == R.e8m0_value(b), b
    assert torch.isnan(t_e8[255])  # I3: torch says NaN at 255
    stats["fp8_tables_vs_torch_dtypes"] = {"e4m3fn": 256, "e5m2": 256, "e8m0fnu": 255}

    # ---- NVFP4 vs nvfp4_emulation_utils.dequantize_to_dtype / break_fp4_bytes ------------------------------------
    ns = {"torch": torch, "SimpleNamespace": types.SimpleNamespace}
    lift(vllm_dir, Q + "nvfp4_emulation_utils.py", ["kE2M1ToFloat_handle", "break_fp4_bytes", "convert_swizzled_to_linear", "dequantize_to_dtype"], ns)
    worst_exact = worst_any = 0.0
    for i in range(n):
        m, nblk = int(torch.randint(1, 6, (1,), generator=g)), int(torch.randint(1, 5, (1,), generator=g))
        k = nblk * 16
        packed = torch.randint(0, 256, (m, k // 2), generator=g, dtype=torch.int64).to(torch.uint8)
        sc = torch.randint(0, 256, (m, nblk), generator=g, dtype=torch.int64).to(torch.uint8)
        sc[(sc == 0x7F) | (sc == 0xFF)] = 0x38  # NaN encodings handled separately below
        # (a) global scale with <= 8 significant bits: every float32 product is exact -> must match bit for bit
        gs_exact = float(torch.randint(1, 256, (1,), generator=g)) / 2.0 ** int(torch.randint(0, 10, (1,), generator=g))
        theirs = ns["dequantize_to_dtype"](packed, sc, gs_exact, torch.float32, 16, swizzle=False)
        ours = R.nvfp4_dequant(packed, sc, gs_exact)
        worst_exact = max(worst_exact, _rel_ok(ours, theirs, 0))
        # (b) arbitrary float32 global scale: vLLM rounds twice in float32 (sf*gs, then *fp4)
        gs_any = float(torch.empty(1).uniform_(1e-3, 1e3, generator=g).float())
        theirs2 = ns["dequantize_to_dtype"](packed, sc, gs_any, torch.float32, 16, swizzle=False)
        ours2 = R.nvfp4_dequant(packed, sc, gs_any)
        worst_any = max(worst_any, _rel_ok(ours2, theirs2, 0))
    assert worst_exact <= 1e-10, f"nvfp4 exact-scale disagreement {worst_exact}"
    # EXPLANATION for (b): vLLM computes in float32 (tensor_sf.to(float32) * global_scale, then * fp4), i.e. up to two
    # float32 roundings (<= 2 * 2^-24 = 1.2e-7 relative); our reference is float64 with none. Bound it explicitly.
    assert worst_any <= 2 * 2.0 ** -24 * 1.01, f"nvfp4 arbitrary-scale disagreement {worst_any}"
    # NaN scale bytes propagate as NaN in both
    packed = torch.tensor([[0x21, 0x43, 0x65, 0x87, 0x19, 0x2A, 0x3B, 0x4C]], dtype=torch.uint8)
    sc = torch.tensor([[0x7F]], dtype=torch.uint8)
    th = ns["dequantize_to_dtype"](packed, sc, 1.0, torch.float32, 16, swizzle=False)
    assert torch.isnan(th).all() and torch.isnan(R.nvfp4_dequant(packed, sc, 1.0)).all()
    stats["nvfp4"] = {"cases": 2 * n + 1, "max_rel_exact_gs": worst_exact, "max_rel_any_gs": worst_any,
                      "explanation_any_gs": "vLLM float32 double rounding (<= 2*2^-24); reference is float64"}

    # ---- MXFP8 dequant (E8M0 decode + block-32 broadcast) -------------------------------------------------------
    ns2 = {"torch": torch, "MXFP8_BLOCK_SIZE": 32}
    lift(vllm_dir, Q + "mxfp8_utils.py", ["dequant_mxfp8_to_bf16"], ns2)
    worst = 0.0
    for i in range(n):
        rows, nblk = int(torch.randint(1, 5, (1,), generator=g)), int(torch.randint(1, 4, (1,), generator=g))
        xb = torch.randint(0, 256, (rows, nblk * 32), generator=g, dtype=torch.int64).to(torch.uint8)
        xb[(xb & 0x7F) == 0x7F] = 0x30
        sb = torch.randint(100, 150, (rows, nblk), generator=g, dtype=torch.int64).to(torch.uint8)
        x8 = xb.view(torch.float8_e4m3fn)
        theirs = ns2["dequant_mxfp8_to_bf16"](x8, sb)
        ours = torch.tensor([[R.e4m3fn_value(int(b)) for b in r] for r in xb], dtype=torch.float64)
        sc = torch.tensor([[R.e8m0_value(int(b)) for b in r] for r in sb], dtype=torch.float64)
        ours = (ours.reshape(rows, nblk, 32) * sc.unsqueeze(-1)).reshape(rows, nblk * 32)
        # vLLM returns bfloat16: compare against our float64 rounded the same way (exact equality expected)
        worst = max(worst, _rel_ok(ours.to(torch.bfloat16), theirs, 0))
    assert worst <= 1e-10, f"mxfp8 disagreement {worst}"
    stats["mxfp8_e8m0_decode"] = {"cases": n, "max_rel": worst,
                                  "note": "bf16 output of vLLM equals float64 reference rounded to bf16"}

    # ---- FP8 per-channel / block scaled_dequantize ---------------------------------------------------------------
    ns3 = {"torch": torch, "np": np, "numpy": np}
    from typing import NamedTuple  # noqa: E402
    ns3["NamedTuple"] = NamedTuple
    ns3["ClassVar"] = __import__("typing").ClassVar
    src = open(os.path.join(vllm_dir, Q + "quant_utils.py")).read()
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "_GroupShape":
            exec(compile(ast.Module([node], []), "q", "exec"), ns3)
        if isinstance(node, ast.ClassDef) and node.name == "GroupShape":
            exec(compile(ast.Module([node], []), "q", "exec"), ns3)
    if "_GroupShape" not in ns3:  # may be an assignment/NamedTuple call
        lift(vllm_dir, Q + "quant_utils.py", ["_GroupShape"], ns3)
    lift(vllm_dir, Q + "quant_utils.py", ["_normalize_quant_group_shape", "group_broadcast", "prep_scale_for_group_broadcast", "scaled_dequantize"], ns3)
    # GroupShape.PER_* assignments run after the class; recreate only what scaled_dequantize needs (is_per_tensor)
    GS = ns3["GroupShape"]
    GS.PER_TENSOR, GS.PER_TOKEN, GS.PER_CHANNEL = GS(-1, -1), GS(1, -1), GS(-1, 1)
    worst_c = worst_b = 0.0
    for i in range(n):
        N, K = int(torch.randint(1, 9, (1,), generator=g)), int(torch.randint(1, 9, (1,), generator=g))
        wb = torch.randint(0, 256, (N, K), generator=g, dtype=torch.int64).to(torch.uint8)
        wb[(wb & 0x7F) == 0x7F] = 0x2A
        w8 = wb.view(torch.float8_e4m3fn)
        sc = (torch.rand(N, 1, generator=g) * 4 + 1e-3).float()
        theirs = ns3["scaled_dequantize"](w8, sc, group_shape=None, out_dtype=torch.float32)
        worst_c = max(worst_c, _rel_ok(R.fp8_channel_dequant(wb, sc), theirs, 0))
        bs = int(torch.tensor([1, 2, 4])[int(torch.randint(0, 3, (1,), generator=g))])
        N2, K2 = bs * int(torch.randint(1, 4, (1,), generator=g)), bs * int(torch.randint(1, 4, (1,), generator=g))
        wb2 = torch.randint(0, 256, (N2, K2), generator=g, dtype=torch.int64).to(torch.uint8)
        wb2[(wb2 & 0x7F) == 0x7F] = 0x2A
        sc2 = (torch.rand(N2 // bs, K2 // bs, generator=g) * 4 + 1e-3).float()
        theirs2 = ns3["scaled_dequantize"](wb2.view(torch.float8_e4m3fn), sc2, group_shape=GS(bs, bs), out_dtype=torch.float32)
        worst_b = max(worst_b, _rel_ok(R.fp8_block_dequant(wb2, sc2, bs), theirs2, 0))
    # EXPLANATION: vLLM's product is float32 (x_q.to(float32) * x_s), single rounding of an exact product, vs our float64
    # product: relative difference <= 2^-24.
    for nm, w in (("channel", worst_c), ("block", worst_b)):
        assert w <= 2.0 ** -24 * 1.01, f"fp8 {nm} disagreement {w}"
    stats["fp8_scaled_dequantize"] = {"cases": 2 * n, "max_rel_channel": worst_c, "max_rel_block": worst_b,
                                      "explanation": "vLLM multiplies in float32 (single rounding, <= 2^-24); reference is float64"}

    # ---- int4 pack-quantized + AWQ vs pack/unpack helpers and quantize_weights (the reference vLLM's tests use) ----
    ns4 = {"torch": torch, "numpy": np, "np": np}
    lift(vllm_dir, Q + "quant_utils.py", ["pack_quantized_values_into_int32", "unpack_quantized_values_into_int32", "get_pack_factor", "pack_cols", "unpack_cols", "awq_pack", "quantize_weights"], ns4)
    u4b8 = FakeScalarType(4, -8, 7, 8, False)
    u4 = FakeScalarType(4, 0, 15, 0, False)
    worst_sym = worst_asym = worst_awq = 0.0
    for i in range(n):
        gsz = int(torch.tensor([8, 16, 32])[int(torch.randint(0, 3, (1,), generator=g))])
        K = gsz * int(torch.randint(1, 4, (1,), generator=g))
        N = 8 * int(torch.randint(1, 3, (1,), generator=g))
        w = torch.randn(K, N, generator=g, dtype=torch.float32)
        # symmetric (uint4b8)
        res = ns4["quantize_weights"](w, u4b8, gsz, zero_points=False)
        w_ref, q_w, w_s = res[0], res[1], res[2]
        packed = ns4["pack_quantized_values_into_int32"](q_w.t().contiguous(), types.SimpleNamespace(size_bits=4), packed_dim=1)
        assert torch.equal(ns4["unpack_quantized_values_into_int32"](packed, types.SimpleNamespace(size_bits=4), packed_dim=1).to(torch.int64), q_w.t().to(torch.int64))
        assert torch.equal(R.unpack_int4_words(packed), q_w.t().to(torch.int64))
        ours = R.int4_pack_quantized_dequant(packed, w_s.t().contiguous(), gsz)
        worst_sym = max(worst_sym, _rel_ok(ours.to(torch.float32), w_ref.t(), 0))
        # asymmetric (uint4 + zero point)
        res = ns4["quantize_weights"](w, u4, gsz, zero_points=True)
        w_ref, q_w, w_s, zp = res[0], res[1], res[2], res[3]
        packed = ns4["pack_quantized_values_into_int32"](q_w.t().contiguous(), types.SimpleNamespace(size_bits=4), packed_dim=1)
        zpk = ns4["pack_quantized_values_into_int32"](zp.t().contiguous(), types.SimpleNamespace(size_bits=4), packed_dim=0)  # [N/8, K/g]
        ours = R.int4_pack_quantized_dequant(packed, w_s.t().contiguous(), gsz, zero_packed=zpk)
        worst_asym = max(worst_asym, _rel_ok(ours.to(torch.float32), w_ref.t(), 0))
        # AWQ (same quantiser, awq_pack for qweight/qzeros, scales [K/g, N])
        qw = ns4["awq_pack"](q_w, 4, K, N)
        qz = ns4["awq_pack"](zp, 4, K // gsz, N)
        ours = R.awq_dequant(qw, qz, w_s, gsz)
        worst_awq = max(worst_awq, _rel_ok(ours.to(torch.float32), w_ref, 0))
    # EXPLANATION: w_ref in quantize_weights is `(w_q - zp).to(float32) * w_s` (float32, one rounding of an exact
    # product); ours is float64 and is cast to float32 once, so equality is bit-exact after the cast (rel 0).
    for nm, w in (("sym", worst_sym), ("asym", worst_asym), ("awq", worst_awq)):
        assert w <= 1e-10, f"int4 {nm} disagreement {w}"
    stats["int4"] = {"cases_each": n, "max_rel_sym": worst_sym, "max_rel_asym": worst_asym, "max_rel_awq": worst_awq,
                     "note": "bit-exact after one float32 cast; also pack/unpack round trip equals vLLM's"}
    return stats


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--vllm", default=VLLM_DEFAULT)
    a = ap.parse_args()
    import json
    print(json.dumps(run(a.vllm), indent=2))
