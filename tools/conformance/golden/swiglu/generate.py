"""Generates Tests/MLXLMTests/Conformance/Golden/swiglu/*.json + MANIFEST.json from swiglu_ref.py (the float64
oracle). Run ONCE, commit the outputs; tests never run Python.

  uv venv venv && uv pip install --python venv/bin/python torch numpy --index-url https://download.pytorch.org/whl/cpu
  VLLM_ROOT=<vLLM checkout at 739e5945d> venv/bin/python -I tools/conformance/golden/swiglu/generate.py

Aborts if the oracle disagrees with vLLM's own forward_native code by > 1e-10 relative (crosscheck()).
"""
import hashlib, json, os, sys, zlib
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import swiglu_ref as ref  # noqa: E402

OUT = os.path.normpath(os.path.join(HERE, "../../../../Tests/MLXLMTests/Conformance/Golden/swiglu"))
VLLM_COMMIT = "739e5945d"
DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
SHAPES = [(1, 1), (3, 1), (1, 3), (2, 5), (4, 7), (3, 33), (5, 129), (2, 64), (70, 3)]  # (rows, d); (70,3) hits SwitchGLU's sorted path (>=64 indices)


def rnd(vals, dt):
    """Round float64 values to dtype and back, so the stored input is exactly representable in the dtype."""
    return torch.tensor(vals, dtype=torch.float64).to(dt).to(torch.float64)


def step(v, dt, up_toward_inf):
    """Neighbouring representable value of v>0 in dtype dt (one ulp up or down)."""
    t = torch.tensor([v], dtype=dt)
    it = {torch.float32: torch.int32, torch.bfloat16: torch.int16, torch.float16: torch.int16}[dt]
    i = t.view(it).clone()
    i += 1 if up_toward_inf else -1
    return float(i.view(dt).to(torch.float64)[0])


def specials(limit, huge, dt):
    L = float(torch.tensor([limit], dtype=dt).to(torch.float64)[0])
    s = [0.0, L, -L, step(L, dt, False), step(L, dt, True), -step(L, dt, False), -step(L, dt, True),
         huge, -huge, -30.0, 0.5, -0.5]
    return [float(v) for v in rnd(s, dt)]


def build(op, layout, limit, huge, dt, shape, rng, special):
    rows, d = shape
    if special:
        S = specials(limit, huge, dt)
        rows = d = len(S)
        gate = np.array([[S[r]] * d for r in range(rows)])         # gate varies by row
        up = np.array([[S[c] for c in range(d)] for r in range(rows)])   # up varies by column -> full cross product
    else:
        scale = rng.choice([1.0, 4.0, 12.0], size=(rows, 2 * d))
        a = rng.standard_normal((rows, 2 * d)) * scale
        hit = rng.random((rows, 2 * d)) < 0.1                        # ~10% land exactly on +-limit
        a = np.where(hit, np.sign(a) * limit, a)
        gate, up = a[:, :d], a[:, d:]
    gate = rnd(gate.tolist(), dt)
    up = rnd(up.tolist(), dt)
    if layout == "chunk":
        x = torch.cat([gate, up], dim=-1)
    else:                                                            # interleave: gate even, up odd (activation.py:322)
        x = torch.stack([gate, up], dim=-1).reshape(rows, 2 * d)
    return x, rows, d


def golden(op, x, limit):
    if op == "silu_and_mul":
        return ref.silu_and_mul(x), None
    if op == "swigluoai_and_mul":
        return ref.swigluoai_and_mul(x, 1.702, limit), None
    if op == "swiglustep_and_mul":
        return ref.swiglustep_and_mul(x, limit), None
    if op == "v4_clamped":
        return ref.v4_clamped(x, limit)
    raise AssertionError(op)


OPS = [
    # name, layout, limits, huge (fp16-safe: silu_and_mul is unclamped so huge*huge must stay < 65504)
    ("silu_and_mul", "chunk", [7.0], 200.0),
    ("swigluoai_and_mul", "interleave", [7.0], 30000.0),
    ("swiglustep_and_mul", "chunk", [7.0], 30000.0),
    ("v4_clamped", "chunk", [7.0, 10.0, 0.5], 30000.0),
]


def main():
    worst = ref.crosscheck(400)
    print("crosscheck vs vLLM forward_native (max relative disagreement, 400 random cases):", worst)
    bad = {k: v for k, v in worst.items() if v > 1e-10}
    assert not bad, f"oracle disagrees with vLLM: {bad}"
    os.makedirs(OUT, exist_ok=True)
    files = {}
    for op, layout, limits, huge in OPS:
        cases = []
        for limit in limits:
            for dname, dt in DTYPES.items():
                for si, shape in enumerate(SHAPES + [None]):
                    name = f"{op}-L{limit}-{dname}-" + ("special" if shape is None else f"{shape[0]}x{shape[1]}")
                    rng = np.random.RandomState(zlib.crc32(name.encode()) & 0xFFFFFFFF)
                    x, rows, d = build(op, layout, limit, huge, dt, shape or (0, 0), rng, shape is None)
                    g, control = golden(op, x, limit)
                    assert torch.isfinite(g).all(), name
                    # the golden must itself be finite in the dtype (no overflow hides inside the comparison)
                    assert torch.isfinite(g.to(dt)).all(), f"{name}: golden overflows {dname}"
                    c = {"name": name, "dtype": dname, "limit": limit, "rows": rows, "d": d,
                         "x": x.reshape(-1).tolist(), "golden": g.reshape(-1).tolist()}
                    if control is not None:       # v4_clamped only: "golden" is max-only (vLLM), this is the labelled control
                        c["golden_gate_symmetric_control"] = control.reshape(-1).tolist()
                    cases.append(c)
        doc = {"op": op, "layout": layout, "cases": cases}
        fn = f"{op}.json"
        with open(os.path.join(OUT, fn), "w") as f:
            json.dump(doc, f, separators=(",", ":"), sort_keys=True)
            f.write("\n")
        files[fn] = {"sha256": hashlib.sha256(open(os.path.join(OUT, fn), "rb").read()).hexdigest(), "cases": len(cases)}
    manifest = {
        "generator": {"scripts": {n: hashlib.sha256(open(os.path.join(HERE, n), "rb").read()).hexdigest()
                                  for n in ("generate.py", "swiglu_ref.py")},
                      "command": "VLLM_ROOT=<vLLM checkout at 739e5945d> venv/bin/python -I tools/conformance/golden/swiglu/generate.py"},
        "vllm": {"commit": VLLM_COMMIT, "path": "vllm/model_executor/layers/activation.py"},
        "torch": torch.__version__, "numpy": np.__version__,
        "oracle_crosscheck_max_relative_disagreement": worst,
        "oracle_provenance": {
            "silu_and_mul": "vLLM activation.py:137-141 (cross-checked)",
            "swigluoai_and_mul": "vLLM activation.py:319-327 (cross-checked)",
            "swiglustep_and_mul": "vLLM activation.py:359-365 (cross-checked; Triton kernel :27-52 emulation also agrees; llama.cpp llama-graph.cpp:1118-1135 corroborates silu-then-clamp)",
            "v4_clamped": "golden = silu(min(gate, limit)) * clamp(up, -limit, limit): gate max-only, per vLLM SiluAndMulWithClamp.forward_native activation.py:239-243 as transcribed by the V4 routing lane at 2eae56c (not in the 739e5945d tree, which has no DeepSeek-V4 model). golden_gate_symmetric_control = the two-sided gate clamp that vLLM rules out, kept only to prove the fixture separates the two readings. INFERRED: chunk layout and the shared `limit` for up.",
        },
        "tolerances": {
            "enforced_in": "ConformanceSwigluGoldenTests.swift SwigluGolden.tolerances (this block only documents them; manifestComplete fails if they differ)",
            "rule": "|swift - golden| <= atol + rtol*|golden|; golden is exact float64 math on dtype-rounded inputs, so the budget is dtype rounding only. Set a priori from the dtype's unit roundoff u and the number of rounded operations (sigmoid, gate*sigmoid, *up, optional clamps which are exact): about 3-5 roundings plus a few ulp of fast-math exp.",
            "float32": {"rtol": 2e-6, "atol": 1e-30, "why": "u=6e-8; 5 roundings + Metal fast exp (<=3 ulp) is ~4e-7; 2e-6 leaves 5x margin. atol 1e-30 only guards denormal flush; the earlier 1e-20 made ~78 swigluoai goldens below 5e-15 effectively unchecked. swigluoai adds an exp-tail term |1.702*gate|*2*2^-24 (INFERRED conditioning argument)."},
            "bfloat16": {"rtol": 2e-2, "atol": 1e-30, "why": "bf16 has 8 significant bits: round-to-nearest u=2^-8=3.9e-3 (same convention as the kda, v4route and v4idx manifests); up to ~5 successive roundings if MLX keeps bf16 intermediates -> 5u=1.95e-2 <= 2e-2; same rtol vLLM's own test_activation.py:90 uses for bf16 (swigluoai/swiglustep)."},
            "float16": {"rtol": 2e-3, "atol": 1e-7, "why": "u=2^-11=4.9e-4; ~5 roundings -> 2.5e-3 worst case, vLLM test uses 2e-3 (test_activation.py:90); atol = 1.7 subnormal steps (6e-8 each) for results below fp16 min normal 6.1e-5."},
        },
        "files": files,
    }
    with open(os.path.join(OUT, "MANIFEST.json"), "w") as f:
        json.dump(manifest, f, indent=1, sort_keys=True)
        f.write("\n")
    print("wrote", OUT, {k: v["cases"] for k, v in files.items()})


if __name__ == "__main__":
    main()
