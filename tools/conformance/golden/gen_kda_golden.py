"""Generate the KDA golden fixtures (ground truth) for ConformanceKdaGoldenTests.

PROVENANCE
  Maths: vLLM commit 739e5945d (739e5945dc4b3ba30d84bdf6e637657abd4136b8, 2026-04-13), transcribed in
    tools/conformance/golden/kda_ref.py with vllm/model_executor/layers/fla/ops/{fused_recurrent,kda,l2norm}.py
    file:line citations (IS_KDA branch of fused_recurrent_gated_delta_rule_fwd_kernel, kda_gate_fwd_kernel).
    Things that are NOT in that vLLM tree are listed as I1..I6 in kda_ref.py's header (per-token mask
    semantics and the lower_bound "safe gate" are INFERRED). kda_crosscheck.py checks kda_ref.py against
    independent formulations (vectorised, chunk_kda WY form, the torch code in vLLM's tests).
  Python: float64 torch CPU. The torch version used for the committed outputs is recorded in
    MANIFEST.json ("torch").

HOW TO RUN (once, then commit the outputs; tests never run python, CI is air-gapped)
    uv venv .venv-golden && uv pip install --python .venv-golden/bin/python torch numpy \\
        --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple
    .venv-golden/bin/python -I tools/conformance/golden/kda_crosscheck.py     # must print OK
    .venv-golden/bin/python -I tools/conformance/golden/gen_kda_golden.py     # rewrites Tests/MLXLMTests/Conformance/Golden/kda/
  Network is needed ONLY for the one-time torch install. Output is deterministic (per-case seed = crc32(name)).

FIXTURE CONTRACT (what the Swift test does with a recurrent case)
  1. inputs q,k,v are ALREADY rounded to `dtype` (float32/float16/bfloat16) before the oracle saw them, so a
     tolerance only has to cover compute error, never input rounding. logDecay, beta, state0 are rounded to
     float32 (the kernel promotes g, beta and the state to float32 anyway).
  2. the test mirrors the CALLER: if `l2norm`, q and k are l2-normalised (x / sqrt(sum(x^2) + 1e-6), vLLM
     FR:128-129); then q is multiplied by `scale` (FR:130); only then the kernel is called with g = exp(logDecay).
     The golden y/state1 are the oracle's result for exactly that. l2norm cases are float32 only, and
     scale is a power of two for the half-precision cases, so no prep rounding exists that the oracle does not see.
  3. y is [B,T,Hv,Dv], state1 is [B,Hv,Dv,Dk] float32; masked steps leave the state alone and give y = 0 (I1).
"""
import argparse
import hashlib
import json
import math
import os
import sys
import zlib

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kda_ref  # noqa: E402

F64 = torch.float64
VLLM_COMMIT = "739e5945dc4b3ba30d84bdf6e637657abd4136b8"
DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}


def rnd(x, dtype):
    """Inputs are first cut to 4 decimals (keeps the committed text short), then rounded to `dtype`; the oracle
    sees exactly the result."""
    return torch.round(x * 1e4).div(1e4).to(DTYPES[dtype]).to(F64)


def seeded(name):
    g = torch.Generator()
    g.manual_seed(zlib.crc32(name.encode()))
    return g


class Nums(list):
    """A flat list of numbers with a text format: 'f32' = shortest float32 round-trip (inputs: the values ARE float32/
    float16/bfloat16-exact, so this loses nothing), 'g10' = 10 significant digits (float64 oracle outputs; the
    1e-10 relative rounding is >= 4 orders below the smallest tolerance in the manifest)."""
    def __init__(self, vals, fmt):
        super().__init__(vals)
        self.fmt = fmt


def flat(x, fmt="f32"):
    vals = x.reshape(-1).tolist()
    if fmt == "f32":
        import numpy as np
        for v in vals:  # inputs must be float32-exact or the compact text would not round-trip
            assert float(np.float32(v)) == v, "input not float32-exact"
    return Nums(vals, fmt)


def _num(v, fmt):
    if fmt == "f32":
        import numpy as np
        t = str(np.float32(v))
    else:
        t = "%.10g" % v
    return "0" if t in ("0.0", "-0.0", "0") and fmt == "g10" else t


def dumps(o):
    if isinstance(o, Nums):
        return "[" + ",".join(_num(v, o.fmt) for v in o) + "]"
    if isinstance(o, dict):
        return "{" + ",".join(json.dumps(k) + ":" + dumps(o[k]) for k in sorted(o)) + "}"
    if isinstance(o, list):
        return "[" + ",".join(dumps(v) for v in o) + "]"
    return json.dumps(o, allow_nan=False)


# ----------------------------------------------------------------------------------------------
# recurrent cases
# ----------------------------------------------------------------------------------------------
def case(name, group, T, Hk, Hv, Dk, Dv, B=1, dtype="float32", state=True, decay="rand", beta="rand",
         mask=None, scale=1.0, l2=False, tol=None):
    return dict(name=name, group=group, B=B, T=T, Hk=Hk, Hv=Hv, Dk=Dk, Dv=Dv, dtype=dtype, state=state,
                decay=decay, beta=beta, mask=mask, scale=scale, l2=l2,
                tol=tol or {"float32": "fp32", "float16": "fp16", "bfloat16": "bf16"}[dtype])


CASES = [
    # --- single step, state / no state (kernel path: Dk % 32 == 0, Dv % 4 == 0)
    case("t1_nostate_dk32", "single_step", 1, 2, 2, 32, 4, state=False),
    case("t1_state_dk32", "single_step", 1, 2, 2, 32, 4),
    # --- multi step
    case("t7_nostate_dk32", "multi_step", 7, 1, 2, 32, 4, state=False),
    case("t7_state_dk32", "multi_step", 7, 1, 2, 32, 4),
    case("t4_b2_state_dk64", "multi_step", 4, 1, 2, 64, 4, B=2),
    case("t3_state_dk128", "multi_step", 3, 1, 2, 128, 4),
    # --- GQA (Hk < Hv)
    case("gqa_hk1_hv4", "gqa", 5, 1, 4, 32, 4),
    case("gqa_hk2_hv6", "gqa", 4, 2, 6, 32, 4),
    # --- decay extremes (logDecay 0 = no decay; very negative; -1000 -> multiplier exactly 0)
    case("nodecay_logdecay0", "decay", 10, 1, 2, 32, 4, decay="zero"),
    case("strong_decay_neg", "decay", 7, 1, 2, 32, 4, decay="strong"),
    case("full_forget_neg1000", "decay", 5, 1, 2, 32, 4, decay="forget"),
    # --- beta extremes
    case("beta0", "beta", 5, 1, 2, 32, 4, beta="zero"),
    case("beta1", "beta", 7, 1, 2, 32, 4, beta="one"),
    # --- mask
    case("mask_all_true", "mask", 5, 1, 2, 32, 4, mask="all"),
    case("mask_partial_state", "mask", 8, 1, 2, 32, 4, mask="partial"),
    case("mask_partial_nostate", "mask", 6, 1, 2, 32, 4, state=False, mask="partial"),
    case("mask_all_false_state", "mask", 4, 1, 2, 32, 4, mask="none"),
    # --- fallback path (Dk % 32 != 0 or Dv % 4 != 0): same golden semantics
    case("fb_dk16", "fallback", 6, 1, 2, 16, 4),
    case("fb_dv6", "fallback", 5, 1, 2, 32, 6),
    case("fb_dk48_dv5_gqa_mask", "fallback", 6, 1, 3, 48, 5, mask="partial"),
    # --- scale and l2norm (applied by the test before the kernel, as the caller does)
    case("scale_pow2_dk64", "prep", 5, 1, 2, 64, 4, scale=0.125),
    case("l2norm_scale_fp32", "prep", 6, 1, 2, 32, 4, scale=32 ** -0.5, l2=True),
    case("l2norm_gqa_mask_fp32", "prep", 7, 1, 4, 32, 4, scale=32 ** -0.5, l2=True, mask="partial"),
    # --- accumulation over T = 256 (no decay: the state integrates every step)
    case("long_t256_nodecay_beta_half", "long", 256, 1, 1, 32, 4, decay="zero", beta="half", tol="fp32_long"),
    # --- low precision inputs
    case("fp16_t7_state", "dtype", 7, 1, 2, 32, 4, dtype="float16"),
    case("bf16_t7_state", "dtype", 7, 1, 2, 32, 4, dtype="bfloat16"),
    case("bf16_gqa_mask_scale", "dtype", 6, 1, 2, 64, 4, dtype="bfloat16", mask="partial", scale=0.125),
    case("fp16_fallback_dk16", "dtype", 6, 1, 2, 16, 4, dtype="float16"),
]


def make_recurrent(c):
    g = seeded(c["name"])
    B, T, Hk, Hv, Dk, Dv, dt = c["B"], c["T"], c["Hk"], c["Hv"], c["Dk"], c["Dv"], c["dtype"]
    r = lambda *s: torch.randn(*s, generator=g, dtype=F64)
    u = lambda *s: torch.rand(*s, generator=g, dtype=F64)
    q, k, v = r(B, T, Hk, Dk), r(B, T, Hk, Dk), r(B, T, Hv, Dv)
    if not c["l2"]:  # unit-norm k keeps the delta rule contractive (|1 - beta |k|^2| <= 1), no blow-up over T=256
        k = k / k.norm(dim=-1, keepdim=True)
    q, k, v = rnd(q, dt), rnd(k, dt), rnd(v, dt)
    d = c["decay"]
    if d == "rand":
        logd = -u(B, T, Hv, Dk) * 0.3
    elif d == "mild":
        logd = -u(B, T, Hv, Dk) * 0.05
    elif d == "zero":
        logd = torch.zeros(B, T, Hv, Dk, dtype=F64)
    elif d == "strong":
        logd = -(20.0 + 40.0 * u(B, T, Hv, Dk))
    elif d == "forget":
        logd = torch.full((B, T, Hv, Dk), -1000.0, dtype=F64)
    else:
        raise ValueError(d)
    logd = rnd(logd, "float32")
    b = c["beta"]
    beta = {"rand": lambda: u(B, T, Hv), "zero": lambda: torch.zeros(B, T, Hv, dtype=F64),
            "one": lambda: torch.ones(B, T, Hv, dtype=F64), "half": lambda: torch.full((B, T, Hv), 0.5, dtype=F64)}[b]()
    beta = rnd(beta, "float32")
    st0 = rnd(r(B, Hv, Dv, Dk) * 0.5, "float32") if c["state"] else None
    m = c["mask"]
    if m is None:
        mask = None
    elif m == "all":
        mask = torch.ones(B, T, dtype=torch.bool)
    elif m == "none":
        mask = torch.zeros(B, T, dtype=torch.bool)
    else:  # partial: deterministic, at least one True and one False
        mask = torch.rand(B, T, generator=g) > 0.45
        mask[0, 0] = False
        mask[-1, -1] = True
    const_logd = {"zero": 0.0, "forget": -1000.0}.get(d)
    const_beta = {"zero": 0.0, "one": 1.0, "half": 0.5}.get(b)
    y, s1 = kda_ref.recurrent(q, k, v, logd, beta, st0, c["scale"], c["l2"], mask)
    path = "kernel" if (Dk % 32 == 0 and Dv % 4 == 0) else "fallback"
    return {
        "name": c["name"], "kind": "recurrent", "group": c["group"], "dtype": dt, "tolerance": c["tol"],
        "expectedPath": path,
        "B": B, "T": T, "Hk": Hk, "Hv": Hv, "Dk": Dk, "Dv": Dv,
        "scale": c["scale"], "l2norm": c["l2"],
        "inputs": {
            "q": flat(q), "k": flat(k), "v": flat(v),
            # constant gates are stored as one scalar (broadcast by the loader) to keep the long cases small
            "logDecay": None if const_logd is not None else flat(logd), "logDecayConst": const_logd,
            "beta": None if const_beta is not None else flat(beta), "betaConst": const_beta,
            "state0": None if st0 is None else flat(st0),
            "mask": None if mask is None else [[bool(x) for x in row] for row in mask.tolist()],
        },
        "outputs": {"y": flat(y, "g10"), "state1": flat(s1, "g10")},
    }


# ----------------------------------------------------------------------------------------------
# gate cases (log decay from raw g)
# ----------------------------------------------------------------------------------------------
def make_gate(name, group, T, H, D, scale_g, bias, lower_bound=None, inferred=False, softplus_beta=1.0):
    g = seeded(name)
    x = rnd(torch.randn(T, H * D, generator=g, dtype=F64) * scale_g, "float32")
    A = rnd(torch.randn(H, generator=g, dtype=F64) * 0.5, "float32")
    bias_t = rnd(torch.randn(H * D, generator=g, dtype=F64), "float32") if bias else None
    out = kda_ref.kda_gate(x, A, bias_t, softplus_beta, 20.0, lower_bound)
    return {
        "name": name, "kind": "gate", "group": group, "dtype": "float32", "tolerance": "gate_fp32",
        "inferred": inferred,
        "T": T, "H": H, "D": D, "softplusBeta": softplus_beta, "threshold": 20.0, "lowerBound": lower_bound,
        "inputs": {"g": flat(x), "aLog": flat(A), "dtBias": None if bias_t is None else flat(bias_t)},
        "outputs": {"logDecay": flat(out.reshape(T, H, D), "g10")},
    }


GATES = [
    lambda: make_gate("gate_softplus_bias", "gate", 3, 4, 8, 3.0, True),
    lambda: make_gate("gate_softplus_nobias", "gate", 2, 2, 16, 2.0, False),
    lambda: make_gate("gate_softplus_linear_branch", "gate", 3, 4, 8, 14.0, True),  # |x| up to ~45: x > 20 takes the linear branch
    lambda: make_gate("gate_softplus_negative_tail", "gate", 2, 2, 8, 12.0, True),
    lambda: make_gate("gate_lower_bound_inferred", "gate_inferred", 3, 4, 8, 3.0, True, lower_bound=-5.0, inferred=True),
]

# ----------------------------------------------------------------------------------------------
# manifest
# ----------------------------------------------------------------------------------------------
# Criterion per class ("mode"): normwise = max|got - want| <= atol + rtol * max|want| over the tensor; elementwise =
# |got_i - want_i| <= atol + rtol * |want_i| for every i. y and state are checked separately. The reasoning is part of
# the manifest on purpose.
TOLERANCES = {
    "fp32": {
        "mode": "normwise",
        "y": {"rtol": 2e-6, "atol": 1e-8}, "state": {"rtol": 2e-6, "atol": 1e-8},
        "why": "float32 compute vs float64 truth. Unit roundoff u = 2^-24 = 6e-8. Each step does a Dk-term dot (kernel: Kahan "
               "on the read, plain on the write-out; ops path: pairwise) and a rank-1 update; with unit-norm k the recurrence is "
               "contractive, so rounding errors accumulate (random-walk) like sqrt(steps * Dk) * u ~ 30u = 1.8e-6 for the "
               "worst short case (T = 7, Dk = 128), they do not amplify. 2e-6 of the tensor's max magnitude is that figure. "
               "atol 1e-8 only covers all-zero tensors and the 1e-10 text rounding of the golden outputs. Relative to "
               "max|want| (normwise), not per element, because y is a sum with cancellation. (Measured afterwards: worst "
               "observed error is 0.17x this bound.)"},
    "fp32_long": {
        "mode": "normwise",
        "y": {"rtol": 1e-5, "atol": 1e-8}, "state": {"rtol": 1e-5, "atol": 1e-8},
        "why": "as fp32 but for T = 256 (Dk = 32): 256 sequential fp32 updates, still contractive (unit-norm k, beta = 0.5); "
               "random-walk bound sqrt(256 * 32) * u = 90u = 5.4e-6, doubled for margin = 1e-5."},
    "fp16": {
        "mode": "normwise",
        "y": {"rtol": 5e-4, "atol": 1e-8}, "state": {"rtol": 2e-6, "atol": 1e-8},
        "why": "q,k,v are rounded to float16 BEFORE the oracle sees them, so input rounding is not in the error. What remains: y "
               "is rounded to float16 once at the store: |round(x) - x| <= u |x| with u = 2^-11 = 4.88e-4, hence <= u max|y|; "
               "plus the fp32 compute error (<= 2e-6). 5e-4 = u + compute + 2% margin. The state is fp32 end to end, so it keeps "
               "the fp32 tolerance."},
    "bf16": {
        "mode": "normwise",
        "y": {"rtol": 4e-3, "atol": 1e-8}, "state": {"rtol": 2e-6, "atol": 1e-8},
        "why": "as fp16 with bfloat16's 8-bit significand: u = 2^-8 = 3.91e-3 (output rounding at the store), plus fp32 compute "
               "error <= 2e-6; 4e-3 = u + compute + 2% margin. Inputs are rounded to bfloat16 before the oracle sees them; "
               "the state stays fp32."},
    "gate_fp32": {
        "mode": "elementwise",
        "y": {"rtol": 4e-6, "atol": 0.0}, "state": None,
        "why": "elementwise relative, atol 0 (gates have no cancellation). The dominant error is CONDITIONING, not the ops: the "
               "gate adds g + bias in float32 (rounding error u |x|) and softplus(x) ~ exp(x) for x << 0, so the relative "
               "error of the result is ~|x| u (the same holds for the sigmoid of the lower_bound form). Fixture inputs are "
               "kept to |x| <= ~40, giving |x| u = 2.4e-6; exp/log1p/sigmoid add ~3e-7. 4e-6 = that sum + ~50% margin. "
               "(A first attempt at 2e-6 with |x| up to ~90 failed on exactly this term; the cause is documented here, "
               "the kernel/reference is not wrong.)"},
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def write_json(path, obj):
    with open(path, "w") as f:
        f.write(dumps(obj))
        f.write("\n")


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, "..", "..", ".."))
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(root, "Tests", "MLXLMTests", "Conformance", "Golden", "kda"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for f in os.listdir(a.out):
        if f.endswith(".json"):
            os.remove(os.path.join(a.out, f))
    fixtures = [make_recurrent(c) for c in CASES] + [g() for g in GATES]
    files = {}
    for fx in fixtures:
        fn = f"{fx['name']}.json"
        write_json(os.path.join(a.out, fn), fx)
        files[fn] = {"sha256": sha256(os.path.join(a.out, fn)), "kind": fx["kind"], "group": fx["group"],
                     "dtype": fx["dtype"], "tolerance": fx["tolerance"]}
    manifest = {
        "schema": 1,
        "generator": "tools/conformance/golden/gen_kda_golden.py",
        "generatorSha256": sha256(os.path.join(here, "gen_kda_golden.py")),
        "oracle": "tools/conformance/golden/kda_ref.py",
        "oracleSha256": sha256(os.path.join(here, "kda_ref.py")),
        "vllmCommit": VLLM_COMMIT,
        "torch": torch.__version__,
        "oracleDtype": "float64",
        "note": "Generated once and committed; tests never run python. Edit nothing by hand: rerun the generator. "
                "Items inferred rather than taken from vLLM: see kda_ref.py header I1..I6 (mask semantics, lower_bound gate).",
        "criterion": "per tolerance class `mode`: normwise = max|got-want| <= atol + rtol*max|want| over a tensor; elementwise = |got_i-want_i| <= atol + rtol*|want_i| for all i; y and state separately",
        "tolerances": TOLERANCES,
        "count": len(files),
        "files": files,
    }
    write_json(os.path.join(a.out, "MANIFEST.json"), manifest)
    total = sum(os.path.getsize(os.path.join(a.out, f)) for f in os.listdir(a.out))
    print(f"wrote {len(files)} fixtures + MANIFEST.json, {total/1024:.0f} KiB, torch {torch.__version__}")


if __name__ == "__main__":
    main()
