"""Dynamic differential harness for the SharpAI KDA kernel (LOCAL ONLY, never in CI: CI is air-gapped).

    <venv>/bin/python -I tools/conformance/differential/kda_diff.py gen     --out DIR --n 60 --seed 1
    KDA_DIFF_DIR=DIR swift test --filter ConformanceKdaDifferentialTests          (see run_local.sh)
    <venv>/bin/python -I tools/conformance/differential/kda_diff.py compare --dir DIR
    <venv>/bin/python -I tools/conformance/differential/kda_diff.py shrink  --dir DIR --case NAME --runner CMD

DATA FLOW (files only, nothing is linked): `gen` draws random shapes/dtypes/seeds, runs the float64 oracle
(tools/conformance/golden/kda_ref.py, the one the golden fixtures are pinned to) and writes DIR/cases/<name>.json in the
golden fixture format of ConformanceKdaGoldenTests plus the index DIR/cases.json. The Swift suite
ConformanceKdaDifferentialTests runs every case through the kernel and the op-composition fallback and writes
DIR/out/<name>.json. `compare` checks those outputs against the float64 truth with the SAME criterion and the SAME
tolerance classes as the golden suite (Tests/.../Golden/kda/MANIFEST.json is read, never copied or edited).

INFERRED (decisions of this file, NOT facts of vLLM or of the manifest; the maths itself is only the oracle's):
    D1  Tolerance class per random case. The manifest derives `fp32` for T*Dk up to 7*128 = 896 (random-walk bound
        sqrt(T*Dk)*u*2, doubled for margin) and `fp32_long` for 256*32 = 8192; `fp16` / `bf16` are derived on top of
        the same 896 compute bound. This file selects: float32 -> fp32 if T*Dk <= 896 else fp32_long if
        T*Dk <= 8192; half precision -> its class only if T*Dk <= 896. Cases beyond those bounds are NOT generated
        (tol_class raises) rather than given a looser tolerance. Consequence: half-precision cases only reach T <= 28
        (Dk = 32), and Dk = 128 only up to T = 64 in float32; T = 65..128 therefore uses Dk <= 64.
    D2  "Seed family": the data of a case is drawn by gen_kda_golden.make_recurrent seeded with crc32(case name), so
        the case name IS the seed. The run seed (--seed) and the case index make the name (kd<seed>_<idx>).
    D3  scale for half-precision cases is a power of two (the golden contract: q * scale must not round).
    D4  maxrel in the table is max|got - want| / max|want| (normwise), the same quantity the tolerance bounds; a
        per-element relative error is meaningless where y cancels to ~0.
    D5  `shrink` is greedy: it only keeps a candidate when Swift says it still fails on a path the original failed on,
        and every candidate keeps the original's dispatch path (kernel / fallback shape). A candidate has its own name,
        hence its own data; "same seed family" means the base name prefix.
"""
import argparse
import importlib.util
import json
import os
import shlex
import shutil
import subprocess
import sys
import zlib  # noqa: F401  (documentation: seeds are crc32(name), see D2)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
GOLDEN_TOOLS = os.path.join(ROOT, "tools", "conformance", "golden")
MANIFEST = os.path.join(ROOT, "Tests", "MLXLMTests", "Conformance", "Golden", "kda", "MANIFEST.json")

T_SWEEP = (1, 2, 31, 32, 33, 64, 65, 127, 128)
HALF_T = (1, 2, 3, 7, 8, 16, 17, 27)
KERNEL_DK = (32, 64, 128)
FALLBACK_DK = (1, 3, 8, 16, 24, 48, 96)
KERNEL_DV = (4, 8, 12, 16, 32)
ODD_DV = (1, 2, 3, 5, 6, 7, 9)
FP32_SHORT_WORK = 896  # 7 * 128, see D1
FP32_LONG_WORK = 8192  # 256 * 32


def tol_class(dtype, T, Dk):
    """D1. Raises instead of loosening."""
    work = T * Dk
    if dtype == "float32":
        if work <= FP32_SHORT_WORK:
            return "fp32"
        if work <= FP32_LONG_WORK:
            return "fp32_long"
        raise ValueError("T*Dk=%d is beyond the manifest's fp32_long bound (%d)" % (work, FP32_LONG_WORK))
    if work > FP32_SHORT_WORK:
        raise ValueError("%s needs T*Dk <= %d (got %d)" % (dtype, FP32_SHORT_WORK, work))
    return {"float16": "fp16", "bfloat16": "bf16"}[dtype]


def dispatch_path(Dk, Dv):
    """Mirrors SharpAIOverlayGatedDeltaKDA.update (and ConformanceKdaGoldenTests): Dk % 32 == 0 && Dv % 4 == 0."""
    return "kernel" if (Dk % 32 == 0 and Dv % 4 == 0) else "fallback"


# ----------------------------------------------------------------------------------------------
# comparison (pure python + numpy; mirrors KdaGolden.check in ConformanceKdaGoldenTests.swift)
# ----------------------------------------------------------------------------------------------
class Check:
    def __init__(self, ratio, maxabs, maxrel, count):
        self.ratio, self.maxabs, self.maxrel, self.count = ratio, maxabs, maxrel, count

    @property
    def ok(self):
        return self.count > 0 and self.ratio <= 1  # NaN <= 1 is False


def check(got, want, tol, mode):
    import numpy as np
    bad = Check(float("inf"), float("inf"), float("inf"), 0)
    if len(got) != len(want) or len(want) == 0:
        return bad
    g = np.asarray([float(x) for x in got], dtype=np.float32).astype(np.float64)  # Swift Float values (D: "NaN" strings ok)
    w = np.asarray(want, dtype=np.float64)
    d = np.abs(g - w)
    if np.isnan(d).any():
        return Check(float("nan"), float("nan"), float("nan"), len(w))
    maxw = float(np.max(np.abs(w)))
    allowed = tol["atol"] + tol["rtol"] * (np.abs(w) if mode == "elementwise" else maxw)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(d == 0, 0.0, d / allowed)
    maxabs = float(np.max(d))
    return Check(float(np.max(r)), maxabs, maxabs / maxw if maxw > 0 else (0.0 if maxabs == 0 else float("inf")), len(w))


def _load_json(path):
    with open(path) as f:
        return json.load(f)


def evaluate(dirpath):
    """Returns (rows, manifest). One row per (case, path)."""
    idx = _load_json(os.path.join(dirpath, "cases.json"))
    tols = _load_json(MANIFEST)["tolerances"]
    rows = []
    for e in idx["cases"]:
        fx = _load_json(os.path.join(dirpath, e["file"]))
        tol = tols[fx["tolerance"]]
        outp = os.path.join(dirpath, "out", e["name"] + ".json")
        out = _load_json(outp) if os.path.exists(outp) else None
        for path in ("kernel", "fallback"):
            label = path if (path == "fallback" or fx["expectedPath"] == "kernel") else "kernel*"
            row = dict(name=e["name"], path=path, label=label, entry=e, tolerance=fx["tolerance"], status="ok", y=None, s=None)
            if out is None:
                row["status"] = "MISSING (no out file: did the Swift suite run?)"
            elif out.get("error"):
                row["status"] = "ERROR " + str(out["error"])[:80]
            elif out.get(path) is None:
                row["status"] = "MISSING " + path
            else:
                row["y"] = check(out[path]["y"], fx["outputs"]["y"], tol["y"], tol["mode"])
                row["s"] = check(out[path]["state"], fx["outputs"]["state1"], tol["state"], tol["mode"])
                if not (row["y"].ok and row["s"].ok):
                    row["status"] = "FAIL"
            rows.append(row)
    return rows


def _fmt(c):
    return "%10.3e %9.2e %6.2f" % (c.maxabs, c.maxrel, c.ratio) if c is not None else "%10s %9s %6s" % ("-", "-", "-")


def compare(dirpath, out=None):
    out = out or sys.stdout
    rows = evaluate(dirpath)
    hdr = "%-14s %-9s %-4s %-5s %-16s %-9s | %-28s | %-28s | %s" % (
        "case", "path", "dt", "tol", "B,T,Hk,Hv,Dk,Dv", "flags", "y: maxabs maxrel ratio", "state: maxabs maxrel ratio", "status")
    print(hdr, file=out)
    for r in rows:
        e = r["entry"]
        shape = "%d,%d,%d,%d,%d,%d" % (e["B"], e["T"], e["Hk"], e["Hv"], e["Dk"], e["Dv"])
        flags = ("s" if e["state"] else "-") + ("m" if e["mask"] else "-") + ("l" if e["l2"] else "-")
        print("%-14s %-9s %-4s %-5s %-16s %-9s | %s | %s | %s" % (
            r["name"], r["label"], e["dtype"][:4], r["tolerance"].replace("fp32_long", "f32L"), shape, flags, _fmt(r["y"]), _fmt(r["s"]), r["status"]), file=out)
    fails = [r for r in rows if r["status"] != "ok"]
    print("", file=out)
    for lab in ("kernel", "kernel*", "fallback"):
        sel = [r for r in rows if r["label"] == lab]
        if not sel:
            continue
        done = [r for r in sel if r["y"] is not None]
        worst = max([max(r["y"].ratio, r["s"].ratio) for r in done if r["y"].ratio == r["y"].ratio and r["s"].ratio == r["s"].ratio] or [0.0])
        print("path %-8s cases=%3d fail=%3d worst ratio (error / allowed)=%.3f" % (lab, len(sel), len([r for r in sel if r["status"] != "ok"]), worst), file=out)
    print("total rows=%d fail=%d  (ratio <= 1 passes; NaN, size mismatch, missing output and ERROR always fail)" % (len(rows), len(fails)), file=out)
    if fails:
        print("\nFAILING CASES (reproduce / minimise with: kda_diff.py shrink --dir %s --case <name> --runner <cmd>)" % dirpath, file=out)
        for r in fails:
            e = r["entry"]
            print("  seed-name=%s run-seed=%s path=%s dtype=%s B=%d T=%d Hk=%d Hv=%d Dk=%d Dv=%d state=%s mask=%s l2=%s scale=%s : %s" % (
                r["name"], e.get("seed"), r["label"], e["dtype"], e["B"], e["T"], e["Hk"], e["Hv"], e["Dk"], e["Dv"], e["state"], e["mask"], e["l2"], e["scale"], r["status"]), file=out)
        return 1
    print("ALL CASES WITHIN TOLERANCE", file=out)
    return 0


# ----------------------------------------------------------------------------------------------
# generation
# ----------------------------------------------------------------------------------------------
def plan(n, seed):
    """The case specs (no oracle run). Deterministic in (n, seed). Sweeps T over T_SWEEP; Dk over {32,64,128} for the
    kernel path and over FALLBACK_DK / odd Dv for the fallback path."""
    import random
    rng = random.Random(seed)
    specs = []
    for i in range(n):
        dtype = rng.choices(["float32", "float16", "bfloat16"], [70, 15, 15])[0]
        half = dtype != "float32"
        want_kernel = rng.random() < 0.65
        T = rng.choice(HALF_T) if half else T_SWEEP[i % len(T_SWEEP)]
        cap = FP32_SHORT_WORK if half else FP32_LONG_WORK
        if want_kernel:
            dks = [d for d in KERNEL_DK if T * d <= cap]
            Dk, Dv = rng.choice(dks), rng.choice(KERNEL_DV)
        elif rng.random() < 0.5:  # odd Dk
            dks = [d for d in FALLBACK_DK if T * d <= cap]
            Dk, Dv = rng.choice(dks), rng.choice(KERNEL_DV + ODD_DV)
        else:  # kernel-friendly Dk, odd Dv
            dks = [d for d in KERNEL_DK if T * d <= cap]
            Dk, Dv = rng.choice(dks), rng.choice(ODD_DV)
        Hk = rng.choice((1, 2, 4))
        Hv = Hk * rng.choice((1, 2) if Hk == 4 else (1, 2, 3))
        l2 = (not half) and rng.random() < 0.3
        specs.append(dict(
            seed=seed, idx=i, name="kd%d_%03d" % (seed, i), dtype=dtype, B=rng.choice((1, 1, 2, 3)), T=T, Hk=Hk, Hv=Hv, Dk=Dk, Dv=Dv,
            state=rng.random() < 0.7,
            mask=rng.choices([None, "partial", "all", "none"], [50, 35, 8, 7])[0],
            decay=rng.choices(["rand", "mild", "zero", "strong", "forget"], [55, 15, 10, 12, 8])[0],
            beta=rng.choices(["rand", "zero", "one", "half"], [70, 5, 10, 15])[0],
            l2=l2,
            scale=rng.choice((1.0, 0.125, 0.5) if half else (1.0, 0.125, Dk ** -0.5)),
            expectedPathHint=dispatch_path(Dk, Dv)))
    return specs


_GEN = None


def load_gen(oracle=None):
    """Imports gen_kda_golden (whose make_recurrent is the generator of the golden fixtures). `oracle` is a path to an
    alternative kda_ref.py (the self-test of the harness uses a deliberately mutated copy); it is registered as the
    module `kda_ref` BEFORE gen_kda_golden imports it."""
    global _GEN
    if oracle:
        spec = importlib.util.spec_from_file_location("kda_ref", oracle)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["kda_ref"] = mod
        spec.loader.exec_module(mod)
        _GEN = None
    if _GEN is None:
        if GOLDEN_TOOLS not in sys.path:
            sys.path.insert(0, GOLDEN_TOOLS)
        import gen_kda_golden
        _GEN = gen_kda_golden
    return _GEN


def build_fixture(spec, oracle=None):
    g = load_gen(oracle)
    tol = tol_class(spec["dtype"], spec["T"], spec["Dk"])
    c = g.case(spec["name"], "diff", spec["T"], spec["Hk"], spec["Hv"], spec["Dk"], spec["Dv"], B=spec["B"], dtype=spec["dtype"],
               state=spec["state"], decay=spec["decay"], beta=spec["beta"], mask=spec["mask"], scale=spec["scale"], l2=spec["l2"], tol=tol)
    return g.dumps(g.make_recurrent(c))


def index_entry(spec):
    keys = ("name", "seed", "idx", "dtype", "B", "T", "Hk", "Hv", "Dk", "Dv", "state", "mask", "decay", "beta", "l2", "scale")
    e = {k: spec[k] for k in keys}
    e["file"] = "cases/%s.json" % spec["name"]
    e["expectedPath"] = dispatch_path(spec["Dk"], spec["Dv"])
    return e


def write_cases(dirpath, specs, oracle=None):
    for sub in ("cases", "out"):
        shutil.rmtree(os.path.join(dirpath, sub), ignore_errors=True)
    os.makedirs(os.path.join(dirpath, "cases"))
    entries = []
    for s in specs:
        e = index_entry(s)
        with open(os.path.join(dirpath, e["file"]), "w") as f:
            f.write(build_fixture(s, oracle) + "\n")
        entries.append(e)
    with open(os.path.join(dirpath, "cases.json"), "w") as f:
        json.dump({"schema": 1, "oracle": oracle or os.path.join(GOLDEN_TOOLS, "kda_ref.py"), "cases": entries}, f, indent=1)
    return entries


def gen(dirpath, n, seed, oracle=None):
    os.makedirs(dirpath, exist_ok=True)
    entries = write_cases(dirpath, plan(n, seed), oracle)
    print("wrote %d cases to %s (seed %d, oracle %s)" % (len(entries), dirpath, seed, oracle or "kda_ref.py (pinned)"))
    return entries


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen")
    g.add_argument("--out", required=True)
    g.add_argument("--n", type=int, default=60)
    g.add_argument("--seed", type=int, default=1)
    g.add_argument("--oracle", help="alternative kda_ref.py (harness self-test with a mutated oracle; scratch only)")
    c = sub.add_parser("compare")
    c.add_argument("--dir", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "gen":
        gen(a.out, a.n, a.seed, a.oracle)
        return 0
    if a.cmd == "compare":
        return compare(a.dir)
    return 2


if __name__ == "__main__":
    sys.exit(main())
