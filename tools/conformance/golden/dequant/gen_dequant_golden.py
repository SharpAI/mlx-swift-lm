#!/usr/bin/env python3
"""Generate the dequant golden fixtures + MANIFEST.json (run once, outputs are committed; tests never run Python).

  VLLM_ROOT=<vLLM checkout at 739e5945d> python gen_dequant_golden.py [--vllm DIR] [--out DIR]

Runs the cross-check against vLLM's own reference code first (dequant_crosscheck.py) and refuses to write anything if it
fails. Fixtures are JSON, deterministic (fixed seeds). Non-finite numbers are the strings "nan" / "inf" / "-inf".
Integers that are float32 values (scales) are emitted as exact float64 reprs of float32 values.

Tolerances in MANIFEST (per dtype, with reasoning) are consumed by the Swift suite via the manifest.
"""

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
import vllm_provenance as VP  # noqa: E402
import dequant_crosscheck as CC  # noqa: E402
import dequant_ref as R  # noqa: E402

VLLM_COMMIT = "739e5945d"
G = np.random.default_rng(20260101)


def num(x):
    x = float(x)
    if math.isnan(x):
        return "nan"
    if math.isinf(x):
        return "inf" if x > 0 else "-inf"
    return x


def flat(t):
    return [num(v) for v in t.to(torch.float64).reshape(-1).tolist()]


def f32(x):
    return float(np.float32(x))


def rand_bytes(shape):
    return torch.from_numpy(G.integers(0, 256, size=shape, dtype=np.int64)).to(torch.uint8)


def rand_scales_f32(shape, lo=0.01, hi=3.0):
    return torch.from_numpy(G.uniform(lo, hi, size=shape).astype(np.float32))


def i32(t):  # int64 holding uint32 patterns -> int32 values for JSON
    return [int(v) - (1 << 32) if v >= (1 << 31) else int(v) for v in t.reshape(-1).tolist()]


def pack_words(q):  # q int [..., K] values 0..15 -> uint32 patterns [..., K/8] (vLLM pack_quantized_values_into_int32, low nibble first)
    q = q.to(torch.int64)
    out = torch.zeros(*q.shape[:-1], q.shape[-1] // 8, dtype=torch.int64)
    for i in range(8):
        out |= q[..., i::8] << (4 * i)
    return out


def mk(fmt, **kw):
    return {"format": fmt, "vllm_commit": VLLM_COMMIT, **kw}


# ------------------------------------------------------------------ fp4 / mxfp4
def fx_fp4_codes():
    codes = [{"code": c, "value": R.fp4_e2m1_value(c)} for c in range(16)]
    return mk("fp4_e2m1_codes", source="Q/utils/nvfp4_emulation_utils.py:18-38", codes=codes)


def fx_e8m0():
    good = [0, 1, 2, 126, 127, 128, 253, 254]
    return mk(
        "e8m0",
        source="Q/utils/mxfp8_utils.py:118, Q/utils/mxfp6_utils.py:78",
        bytes=[{"byte": b, "value": R.e8m0_value(b)} for b in good],
        special={"byte": 255, "vllm_formula_float32": "inf", "torch_float8_e8m0fnu": "nan",
                 "note": "I3: exp2(255-127)=2^128 overflows float32 (formula -> inf); OCP/torch define 255 as NaN"},
    )


def mxfp4_case(cid, packed, scales, note=""):
    exp = R.mxfp4_dequant(packed, scales)
    return {"id": cid, "note": note, "rows": packed.shape[0], "cols": packed.shape[1] * 2,
            "packed": packed.reshape(-1).tolist(), "scales": scales.reshape(-1).tolist(), "expected": flat(exp)}


def fx_mxfp4():
    cases = []
    # every byte value 0..255 once (all 16x16 nibble pairs -> nibble order), unit scale 2^0
    allb = torch.arange(256, dtype=torch.uint8).reshape(8, 32)
    cases.append(mxfp4_case("all_bytes_unit_scale", allb, torch.full((8, 2), 127, dtype=torch.uint8),
                            "8 rows x 64 cols, bytes 0..255 in order; scale byte 127 = 1.0"))
    # nibble order, hand-checkable: byte 0x21 -> low nibble 1 (0.5) then high nibble 2 (1.0)
    p = torch.tensor([[0x21] * 16], dtype=torch.uint8)
    cases.append(mxfp4_case("nibble_order_0x21", p, torch.tensor([[127]], dtype=torch.uint8), "0x21 -> [0.5, 1.0] repeated"))
    # group boundary: 3 groups per row with different scales, element pattern identical in every group
    pat = torch.tensor([[0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE] * 2] * 3, dtype=torch.uint8)  # 32 nibbles 0..15,0..15
    pat = pat.repeat(1, 3)  # [3, 48] bytes -> 96 cols
    sc = torch.tensor([[126, 127, 128], [124, 130, 127], [127, 127, 133]], dtype=torch.uint8)
    cases.append(mxfp4_case("group_boundary_3x32", pat, sc, "scale changes exactly at columns 32 and 64"))
    # one-element-wide boundary: only the LAST element of group 0 and FIRST of group 1 are non-zero
    p2 = torch.zeros(1, 32, dtype=torch.uint8)
    p2[0, 15] = 0x70  # col 31 = nibble hi of byte 15 -> 6.0
    p2[0, 16] = 0x07  # col 32 = nibble lo of byte 16 -> 6.0
    cases.append(mxfp4_case("boundary_last_first", p2, torch.tensor([[129, 125]], dtype=torch.uint8),
                            "col31 uses scale 2^2, col32 uses scale 2^-2"))
    # scale byte edges (kept in separate cases so each failure is localised)
    codes_small = torch.tensor([[0x10, 0x32, 0x91, 0xB3] * 4], dtype=torch.uint8)  # 0.5,1,1.5? -> codes 0,1,2,3,1,9,3,11
    cases.append(mxfp4_case("scale_byte_0", codes_small, torch.tensor([[0]], dtype=torch.uint8), "2^-127 (float32 subnormal)"))
    cases.append(mxfp4_case("scale_byte_1", codes_small, torch.tensor([[1]], dtype=torch.uint8), "2^-126 (smallest normal float32)"))
    cases.append(mxfp4_case("scale_byte_127", codes_small, torch.tensor([[127]], dtype=torch.uint8), "1.0"))
    cases.append(mxfp4_case("scale_byte_254_small_codes", torch.tensor([[0x10, 0x32] * 8], dtype=torch.uint8),
                            torch.tensor([[254]], dtype=torch.uint8), "2^127 x {0,0.5,1,1.5}: finite in float32"))
    cases.append(mxfp4_case("scale_byte_254_overflow", torch.tensor([[0x77] * 16], dtype=torch.uint8),
                            torch.tensor([[254]], dtype=torch.uint8), "6 x 2^127 > float32 max -> inf in float32 (float64 ref finite)"))
    # random matrices, scale bytes 112..142
    for i in range(4):
        pk = rand_bytes((4, 64))
        sb = torch.from_numpy(G.integers(112, 143, size=(4, 4), dtype=np.int64)).to(torch.uint8)
        cases.append(mxfp4_case(f"random_{i}", pk, sb, "4 x 128, random bytes, scales 112..142"))
    return mk("mxfp4", source="Q/utils/ocp_mx_utils.py:9; Q/utils/mxfp8_utils.py:118; Q/utils/nvfp4_emulation_utils.py:23-40 (I1,I2)", cases=cases)


# ------------------------------------------------------------------ nvfp4
def nvfp4_case(cid, packed, sc, gs, note=""):
    exp = R.nvfp4_dequant(packed, sc, gs)
    return {"id": cid, "note": note, "rows": packed.shape[0], "cols": packed.shape[1] * 2, "global_scale": gs,
            "packed": packed.reshape(-1).tolist(), "scales": sc.reshape(-1).tolist(), "expected": flat(exp)}


def fx_nvfp4():
    cases = []
    allb = torch.arange(256, dtype=torch.uint8).reshape(8, 32)
    cases.append(nvfp4_case("all_bytes_unit_scale", allb, torch.full((8, 4), 0x38, dtype=torch.uint8), 1.0, "scale 0x38 = 1.0"))
    pat = torch.tensor([[0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE]] * 2, dtype=torch.uint8).repeat(1, 3)  # [2,24] -> 48 cols = 3 groups of 16
    for cid, sb, gs, note in [
        ("scale_e4m3_boundaries", [0x38, 0x08, 0x78], 1.0, "1.0, min normal 2^-6, 0x78=256"),
        ("scale_e4m3_subnormal", [0x01, 0x04, 0x07], 1.0, "subnormals 2^-9, 2^-7, 7*2^-9"),
        ("scale_e4m3_max", [0x7E, 0x7D, 0x30], 1.0, "448, 416, 0.5"),
        ("scale_zero", [0x00, 0x38, 0x00], 1.0, "scale 0 zeroes its group only"),
        ("global_scale_0p5", [0x38, 0x40, 0x30], 0.5, "global multiplier 0.5"),
        ("global_scale_3", [0x38, 0x40, 0x30], 3.0, "global multiplier 3 (3*e4m3 exact in float32)"),
        ("global_scale_0p15625", [0x34, 0x44, 0x3C], 0.15625, "5/32"),
        ("scale_negative", [0xB8, 0x38, 0xC0], 1.0, "sign bit of the e4m3 scale (not produced by real quantisers)"),
        ("scale_nan", [0x7F, 0x38, 0xFF], 1.0, "e4m3fn NaN encodings 0x7F/0xFF poison their group only"),
    ]:
        cases.append(nvfp4_case(cid, pat, torch.tensor([sb, sb[::-1]], dtype=torch.uint8), gs, note))
    for i in range(3):
        pk = rand_bytes((4, 32))
        sb = torch.from_numpy(G.integers(0x20, 0x50, size=(4, 4), dtype=np.int64)).to(torch.uint8)
        cases.append(nvfp4_case(f"random_{i}", pk, sb, [1.0, 0.25, 2.5][i], "4 x 64"))
    return mk("nvfp4", source="Q/utils/nvfp4_emulation_utils.py:53-76 (swizzle=False); Q/compressed_tensors/schemes/compressed_tensors_w4a16_nvfp4.py:88-90 (I5)", cases=cases)


# ------------------------------------------------------------------ fp8
def fx_fp8_codes():
    return mk(
        "fp8_codes",
        source="torch.float8_e4m3fn / float8_e5m2 (vLLM uses the torch dtypes: nvfp4_emulation_utils.py:68)",
        e4m3fn=[num(R.e4m3fn_value(b)) for b in range(256)],
        e5m2=[num(R.e5m2_value(b)) for b in range(256)],
    )


def fx_fp8_scaled():
    ch, bl = [], []
    for N, K in [(4, 8), (3, 5), (1, 16)]:
        wb = rand_bytes((N, K)); wb[(wb & 0x7F) == 0x7F] = 0x2A
        sc = rand_scales_f32((N, 1))
        ch.append({"id": f"channel_{N}x{K}", "rows": N, "cols": K, "weight": wb.reshape(-1).tolist(),
                   "scale": [f32(v) for v in sc.reshape(-1).tolist()],
                   "expected": flat(R.fp8_channel_dequant(wb, sc))})
    # every code once, per-channel scale 1.0 and 2.0 (NaN codes kept on purpose: NaN * scale = NaN)
    wb = torch.arange(256, dtype=torch.uint8).reshape(8, 32)
    sc = torch.tensor([[1.0], [2.0], [0.5], [1.0], [1.0], [1.0], [1.0], [1.0]])
    ch.append({"id": "all_codes", "rows": 8, "cols": 32, "weight": wb.reshape(-1).tolist(),
               "scale": [f32(v) for v in sc.reshape(-1).tolist()], "expected": flat(R.fp8_channel_dequant(wb, sc))})
    for bs, N, K in [(1, 3, 4), (2, 4, 6), (4, 8, 8), (4, 4, 12), (8, 8, 16)]:
        wb = rand_bytes((N, K)); wb[(wb & 0x7F) == 0x7F] = 0x2A
        sc = rand_scales_f32((N // bs, K // bs))
        bl.append({"id": f"block{bs}_{N}x{K}", "block": bs, "rows": N, "cols": K, "weight": wb.reshape(-1).tolist(),
                   "scale": [f32(v) for v in sc.reshape(-1).tolist()], "scale_shape": [N // bs, K // bs],
                   "expected": flat(R.fp8_block_dequant(wb, sc, bs))})
    # block boundary: scale changes exactly at row/col 4 (unit weight 1.0 = 0x38 everywhere)
    wb = torch.full((8, 8), 0x38, dtype=torch.uint8)
    sc = torch.tensor([[1.0, 2.0], [4.0, 8.0]])
    bl.append({"id": "block4_boundary", "block": 4, "rows": 8, "cols": 8, "weight": wb.reshape(-1).tolist(),
               "scale": [1.0, 2.0, 4.0, 8.0], "scale_shape": [2, 2], "expected": flat(R.fp8_block_dequant(wb, sc, 4))})
    # ue8m0 block scales (I4): byte -> 2^(byte-127)
    ue = []
    for bs, N, K, lo, hi in [(4, 8, 8, 120, 135), (2, 4, 6, 124, 131)]:
        wb = rand_bytes((N, K)); wb[(wb & 0x7F) == 0x7F] = 0x2A
        sbytes = torch.from_numpy(G.integers(lo, hi, size=(N // bs, K // bs), dtype=np.int64)).to(torch.uint8)
        sc = torch.tensor([[R.e8m0_value(int(b)) for b in r] for r in sbytes], dtype=torch.float64)
        ue.append({"id": f"ue8m0_block{bs}_{N}x{K}", "block": bs, "rows": N, "cols": K, "weight": wb.reshape(-1).tolist(),
                   "scale_bytes": sbytes.reshape(-1).tolist(), "scale_shape": [N // bs, K // bs],
                   "expected": flat(R.fp8_block_dequant(wb, sc, bs))})
    return mk("fp8_scaled", source="Q/utils/quant_utils.py:333-343 (scaled_dequantize), :207-219 (group_broadcast); I4, I6",
              channel=ch, block=bl, ue8m0_block=ue)


# ------------------------------------------------------------------ int4 pack-quantized + AWQ
def fx_int4():
    cases = []
    for cid, N, K, g, asym in [("sym_2x16_g8", 2, 16, 8, False), ("sym_4x32_g16", 4, 32, 16, False),
                               ("sym_3x64_g32", 3, 64, 32, False), ("asym_8x16_g8", 8, 16, 8, True),
                               ("sym_2x128_g64", 2, 128, 64, False), ("asym_8x64_g32", 8, 64, 32, True)]:
        q = torch.from_numpy(G.integers(0, 16, size=(N, K), dtype=np.int64))
        q[0, 0], q[0, 1] = 0, 15  # extreme nibbles
        words = pack_words(q)
        sc = rand_scales_f32((N, K // g), 0.01, 0.5)
        entry = {"id": cid, "rows": N, "cols": K, "group": g, "symmetric": not asym,
                 "weight_packed": i32(words), "weight_packed_shape": [N, K // 8], "scale": [f32(v) for v in sc.reshape(-1).tolist()]}
        if asym:
            zp = torch.from_numpy(G.integers(0, 16, size=(N, K // g), dtype=np.int64))
            zpk = pack_words(zp.t().contiguous()).t().contiguous()  # packed along dim 0 -> [N/8, K/g] (I7)
            entry["zero_packed"] = i32(zpk)
            entry["zero_packed_shape"] = [N // 8, K // g]
            entry["expected"] = flat(R.int4_pack_quantized_dequant(torch.tensor(i32(words), dtype=torch.int32).reshape(N, K // 8),
                                                                  sc, g, torch.tensor(i32(zpk), dtype=torch.int32).reshape(N // 8, K // g)))
        else:
            entry["expected"] = flat(R.int4_pack_quantized_dequant(torch.tensor(i32(words), dtype=torch.int32).reshape(N, K // 8), sc, g))
        cases.append(entry)
    # nibble-order probe: word 0x76543210 -> nibbles 0..7, symmetric => (n-8)*scale
    words = torch.tensor([[0x76543210, 0xFEDCBA98]], dtype=torch.int64)
    sc = torch.tensor([[0.5, 0.25]])
    cases.append({"id": "nibble_order_sym_1x16_g8", "rows": 1, "cols": 16, "group": 8, "symmetric": True,
                  "weight_packed": i32(words), "weight_packed_shape": [1, 2], "scale": [0.5, 0.25],
                  "expected": flat(R.int4_pack_quantized_dequant(torch.tensor(i32(words), dtype=torch.int32).reshape(1, 2), sc, 8))})
    return mk("int4_pack_quantized", source="Q/utils/quant_utils.py:404-444, :604, :826-848; Q/compressed_tensors/schemes/compressed_tensors_wNa16.py:35-36 (I7)", cases=cases)


def fx_awq():
    ns = {"torch": torch, "numpy": np}
    CC.lift(VLLM_DIR, CC.Q + "quant_utils.py", ["get_pack_factor", "pack_cols", "awq_pack"], ns)
    cases = []
    for cid, K, N, g in [("awq_32x16_g32", 32, 16, 32), ("awq_64x32_g32", 64, 32, 32), ("awq_128x8_g64", 128, 8, 64)]:
        q = torch.from_numpy(G.integers(0, 16, size=(K, N), dtype=np.int64)).to(torch.int32)
        zp = torch.from_numpy(G.integers(0, 16, size=(K // g, N), dtype=np.int64)).to(torch.int32)
        sc = rand_scales_f32((K // g, N), 0.01, 0.5)
        qw = ns["awq_pack"](q, 4, K, N)  # vLLM's own packer
        qz = ns["awq_pack"](zp, 4, K // g, N)
        cases.append({"id": cid, "in_features": K, "out_features": N, "group": g,
                      "qweight": qw.reshape(-1).tolist(), "qweight_shape": [K, N // 8],
                      "qzeros": qz.reshape(-1).tolist(), "qzeros_shape": [K // g, N // 8],
                      "scales": [f32(v) for v in sc.reshape(-1).tolist()], "scales_shape": [K // g, N],
                      "expected_KxN": flat(R.awq_dequant(qw, qz, sc, g))})
    return mk("awq_int4", source="Q/utils/quant_utils.py:711-734 (pack_cols), :777-796 (awq_pack)", cases=cases)


TOL = {
    "float32_exact": {"rtol": 0.0, "atol": 0.0,
                      "why": "mxfp4 / nvfp4 (exactly representable global scales) / int4 / awq: every product is a small-mantissa value times a power of two or a short-mantissa float32; the float32 result is exact, so equality is required"},
    "float32_product": {"rtol": 0.0, "atol": 0.0,
                        "why": "fp8 * float32 scale: the fp8 value has <= 4 significant bits, so the product has <= 28 bits and is exact in float64; Float(float64 product) and MLX's float32 multiply are the same single correctly rounded result. The matmul-by-identity route only adds exact zeros. Equality is required: a 1-ulp bug must fail"},
    "float32_affine_cancellation": {"rtol": 1.2e-7, "atol": 0.0, "atol_per_scale": 15 * 2.0 ** -23,
                                    "why": "MLX affine dequantize computes q*scale + bias with bias = -zp*scale: two float32 roundings of terms up to 15*scale that cancel, so the absolute error is up to 2 * 2^-24 * 15 * scale (= 15 * 2^-23 * scale, the scale of the element's OWN group) even where the exact result is tiny or zero; vLLM's (q-zp)*scale has one rounding"},
    "bfloat16_values": {"rtol": 0.0, "atol": 0.0, "why": "every fp8 e4m3 value has 3 mantissa bits, exactly representable in bfloat16"},
}


def main():
    global VLLM_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--vllm", default=None, help="vLLM checkout root (commit 739e5945d); or set VLLM_ROOT")
    ap.add_argument("--out", default=os.path.join(HERE, "fixtures"))
    a = ap.parse_args()
    a.vllm = CC.resolve_vllm(a.vllm)
    VLLM_DIR = a.vllm
    head = subprocess.run(["git", "-C", a.vllm, "rev-parse", "--short=9", "HEAD"], capture_output=True, text=True).stdout.strip()
    assert head == VLLM_COMMIT, f"vLLM checkout is {head}, expected {VLLM_COMMIT}"
    stats = CC.run(a.vllm)
    os.makedirs(a.out, exist_ok=True)
    fixtures = {
        "fp4_e2m1_codes.json": fx_fp4_codes(), "e8m0.json": fx_e8m0(), "mxfp4.json": fx_mxfp4(), "nvfp4.json": fx_nvfp4(),
        "fp8_codes.json": fx_fp8_codes(), "fp8_scaled.json": fx_fp8_scaled(), "int4_pack_quantized.json": fx_int4(), "awq_int4.json": fx_awq(),
    }
    files = {}
    for name, obj in fixtures.items():
        data = (json.dumps(obj, sort_keys=True, separators=(",", ":")) + "\n").encode()  # compact: indent=1 put one number per line (~2.5x the size)
        with open(os.path.join(a.out, name), "wb") as f:
            f.write(data)
        files[name] = {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
    manifest = {
        "generator": "tools/conformance/golden/dequant/gen_dequant_golden.py",
        "reference": ["dequant_ref.py", "dequant_crosscheck.py"],
        "vllm_commit": VLLM_COMMIT, "vllm_provenance": VP.statements("739e5945d"), "torch_version": torch.__version__, "numpy_version": np.__version__,
        "tolerances": TOL, "crosscheck": stats, "files": files,
        # the oracle + generator sources the fixtures were produced by; the Swift suite re-hashes them, so editing one
        # without regenerating (and committing) the fixtures is noticed
        "tools": {
            n: {"sha256": hashlib.sha256(open(os.path.join(HERE, n), "rb").read()).hexdigest(),
                "bytes": os.path.getsize(os.path.join(HERE, n))}
            for n in ("dequant_ref.py", "dequant_crosscheck.py", "gen_dequant_golden.py")
        },
    }
    with open(os.path.join(a.out, "MANIFEST.json"), "w") as f:
        json.dump(manifest, f, indent=1, sort_keys=True)
        f.write("\n")
    print("wrote", len(files), "fixtures to", a.out)


if __name__ == "__main__":
    main()
