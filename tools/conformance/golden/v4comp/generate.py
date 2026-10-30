"""Generates Tests/MLXLMTests/Conformance/Golden/v4comp/*.json + MANIFEST.json from v4comp_ref.py (float64 oracle).
Run ONCE, commit the outputs; tests never run Python.

  uv venv --python 3.15 venv && uv pip install --python venv/bin/python -r tools/conformance/golden/v4comp/requirements.txt \\
      --extra-index-url https://download.pytorch.org/whl/cpu
  venv/bin/python -B -I tools/conformance/golden/v4comp/generate.py --vllm <path to vllm repo root (commit 2eae56c)>

Pinned inputs: requirements.txt (torch / numpy exactly as used for the committed fixtures; MANIFEST.json records the versions, the vLLM
commit and the sha256 of this file and of v4comp_ref.py, which the Swift suite re-checks). Network is needed once, here; never in tests/CI.

crosscheck() first compares the oracle against vLLM's OWN torch reference code, extracted verbatim from the vLLM test files
(ast) and executed, on >= 200 random cases; any disagreement > 1e-10 relative aborts the run.
"""
import argparse, ast, hashlib, json, math, os, platform, re, sys, textwrap, types
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import v4comp_ref as ref  # noqa: E402

F64 = torch.float64
OUT = os.path.normpath(os.path.join(HERE, "../../../../Tests/MLXLMTests/Conformance/Golden/v4comp"))
VLLM_COMMIT = "2eae56c"
TOL_CROSS = 1e-10


# ---------------------------------------------------------------------------------------------------------------- vLLM extraction
def extract(path, names):
    src = open(path).read()
    tree = ast.parse(src)
    out = {}
    for n in tree.body:
        if isinstance(n, ast.FunctionDef) and n.name in names:
            out[n.name] = ast.get_source_segment(src, n)
    assert set(out) == set(names), (set(names) - set(out))
    return out


PATCHES = []


def patch(src, old, new, why):
    assert src.count(old) == 1, (old, src.count(old))
    PATCHES.append({"replace": old, "with": new, "why": why})
    return src.replace(old, new)


def load_vllm(root):
    kv = os.path.join(root, "tests/kernels/test_compressor_kv_cache.py")
    cpu = os.path.join(root, "tests/kernels/test_deepseek_v4_cpu_kernels.py")
    utl = os.path.join(root, "vllm/models/deepseek_v4/cpu/cpu_utils.py")
    f_tri = extract(kv, ["_reference_kv_compress_norm_rope"])["_reference_kv_compress_norm_rope"]
    f_cpu = extract(cpu, ["_run_compress_eager", "_run_save_partial_states_eager", "_apply_gptj_rope"])
    f_map = extract(utl, ["map_local_to_global_slots_cpu"])["map_local_to_global_slots_cpu"]
    # (1) tests/kernels/test_compressor_kv_cache.py:2432 computes in fp32 (.float()); we need fp64 to compare at 1e-10.
    f_tri64 = f_tri.replace(".float()", ".double()")
    PATCHES.append({"fn": "_reference_kv_compress_norm_rope", "replace": ".float()", "with": ".double()", "why": "run the fp32 reference in fp64"})
    # (2) return the compressed+normed+roped float row before the (fp8 / mxfp4) quantisation, which has no Swift counterpart.
    f_tri64 = patch(f_tri64, "    if return_full_cache:\n", "    return result\n    if return_full_cache:\n",
                    "return before quantisation (kvc:~2510)")
    # (3) capture the pre-quant row of tests/kernels/test_deepseek_v4_cpu_kernels.py:_run_compress_eager (cuk:256).
    f_eager = patch(f_cpu["_run_compress_eager"], "        block_idx_kv = kv_slot // kv_cache_block_size\n",
                    "        CAPTURE.append((normed.clone(), torch.cat([normed[:NOPE_DIM], rotated_rope])))\n        block_idx_kv = kv_slot // kv_cache_block_size\n",
                    "capture normed (pre-RoPE) and the RoPE'd row before the fp8 store")
    return dict(tri=f_tri, tri64=f_tri64, eager=f_eager, save=f_cpu["_run_save_partial_states_eager"],
                rope=f_cpu["_apply_gptj_rope"], map=f_map)


def load_vllm_yarn(root):
    """vLLM's own YaRN code, extracted by ast and executed: common.py yarn_* helpers, deepseek_scaling_rope.py yarn_get_mscale,
    DeepseekScalingRotaryEmbedding._compute_inv_freq and DeepseekV4ScalingRotaryEmbedding._compute_cos_sin_cache. Patched to float64
    (torch.float / torch.float32 -> torch.float64) so the comparison with the float64 oracle can be exact."""
    rd = os.path.join(root, "vllm/model_executor/layers/rotary_embedding")
    helpers = extract(os.path.join(rd, "common.py"), ["yarn_find_correction_dim", "yarn_find_correction_range", "yarn_linear_ramp_mask"])
    mscale = extract(os.path.join(rd, "deepseek_scaling_rope.py"), ["yarn_get_mscale"])["yarn_get_mscale"]
    tree = ast.parse(open(os.path.join(rd, "deepseek_scaling_rope.py")).read())
    methods = {}
    for n in tree.body:
        if isinstance(n, ast.ClassDef):
            for m in n.body:
                if isinstance(m, ast.FunctionDef) and (n.name, m.name) in (("DeepseekScalingRotaryEmbedding", "_compute_inv_freq"),
                                                                           ("DeepseekV4ScalingRotaryEmbedding", "_compute_cos_sin_cache")):
                    methods[m.name] = ast.unparse(m)
    assert set(methods) == {"_compute_inv_freq", "_compute_cos_sin_cache"}, methods.keys()
    def to64(src):
        return re.sub(r"torch\.float\b(?!\d)", "torch.float64", src.replace("torch.float32", "torch.float64"))
    PATCHES.append({"fn": "DeepseekScalingRotaryEmbedding._compute_inv_freq / DeepseekV4ScalingRotaryEmbedding._compute_cos_sin_cache",
                    "replace": "torch.float / torch.float32", "with": "torch.float64", "why": "run the fp32 YaRN cache in fp64"})
    ns = {"torch": torch, "math": math}
    for src in (*helpers.values(), mscale):
        exec(src, ns)
    cls = type("VllmYarnStub", (), {})
    for name, src in methods.items():
        exec(to64(src), ns)
        setattr(cls, name, ns[name])
    return cls, ns["yarn_get_mscale"]


def vllm_yarn_cache(stub_cls, yarn_get_mscale, rd, theta, y, n_rows):
    o = stub_cls()
    o.base, o.rotary_dim, o.max_position_embeddings = theta, rd, y["original_max_position_embeddings"]
    o.beta_fast, o.beta_slow, o.scaling_factor = y["beta_fast"], y["beta_slow"], y["factor"]
    # rope.py:51-52 mscale = mscale_all_dim = 0 -> deepseek_scaling_rope.py:__init__: yarn_get_mscale(f, 0.) / yarn_get_mscale(f, 0.)
    o.mscale = float(yarn_get_mscale(y["factor"], 0.0) / yarn_get_mscale(y["factor"], 0.0))
    assert o.mscale == 1.0
    return o._compute_cos_sin_cache()[:n_rows]


def rel(a, b):
    a, b = a.to(F64), b.to(F64)
    return float((a - b).abs().max() / b.abs().max().clamp(min=1e-300)) if a.numel() else 0.0


def crosscheck(vroot, n_cases=240, seed=2025):
    src = load_vllm(vroot)
    ystub, ymscale = load_vllm_yarn(vroot)
    g = torch.Generator().manual_seed(seed)
    rnd = np.random.default_rng(seed)
    worst = {"entries_vs_triton_ref_fp64": 0.0, "entries_vs_cpu_eager": 0.0, "prerope_vs_cpu_eager": 0.0,
             "state_rows_vs_save_eager": 0.0, "triton_ref_unpatched_fp32_vs_fp64": 0.0, "yarn_cos_sin_vs_vllm_class_fp64": 0.0}
    n_yarn = 0
    n_entries = 0
    shapes = [(24, 8), (32, 16), (64, 16), (128, 64), (80, 16)]
    for case in range(n_cases):
        ratio = 4 if case % 2 == 0 else 128
        overlap = ratio == 4
        hd, rd = shapes[int(rnd.integers(len(shapes)))]
        T = int(rnd.integers(1, 70)) if ratio == 4 else int(rnd.integers(1, 420))
        if case % 5 == 0:
            T = max(T, 2 * ratio + int(rnd.integers(0, ratio)))   # force several complete blocks / the overlap
        hidden = int(rnd.integers(4, 10))
        coff = 1 + int(overlap)
        W = coff * hd
        eps = float(rnd.choice([1e-6, 1e-5]))
        theta = float(rnd.choice([10000.0, 160000.0]))
        sc_scale = float(rnd.choice([0.5, 2.0, 6.0]))
        x = torch.randn(T, hidden, generator=g, dtype=F64)
        wkv = torch.randn(W, hidden, generator=g, dtype=F64) * 0.6
        wg = torch.randn(W, hidden, generator=g, dtype=F64) * sc_scale / np.sqrt(hidden)
        ape = torch.randn(ratio, W, generator=g, dtype=F64)
        nw = 1 + 0.2 * torch.randn(hd, generator=g, dtype=F64)
        yarn = None
        if case % 3 == 0:   # every third case runs with a YaRN rope, its cos/sin cache taken from vLLM's own class code
            yarn = {"factor": float(rnd.choice([4.0, 16.0])), "original_max_position_embeddings": int(rnd.choice([256, 4096, 65536])),
                    "beta_fast": float(rnd.choice([32.0, 16.0])), "beta_slow": 1.0}
            n_yarn += 1
        pre, ent = ref.compress_sequence(x, wkv, wg, ape, nw, eps, ratio, hd, rd, theta, yarn=yarn)
        # --- vLLM state write: tests/kernels/test_deepseek_v4_cpu_kernels.py:205 (_run_save_partial_states_eager)
        bs = 4 if ratio == 4 else 8
        nb_state = -(-T // bs)
        total_blocks = nb_state + 3
        block_table = torch.randperm(total_blocks, generator=g)[:nb_state].to(torch.int64).unsqueeze(0)
        pos = torch.arange(T, dtype=torch.int64)
        slot_mapping = block_table[0, pos // bs] * bs + pos % bs
        state_cache = torch.randn(total_blocks, bs, 2 * W, generator=g, dtype=F64)   # garbage in unused rows
        kv, sc = x @ wkv.T, x @ wg.T
        ns = {"torch": torch}
        exec(src["save"], ns)
        ns["_run_save_partial_states_eager"](state_cache, kv, sc, ape, pos, slot_mapping, ratio)
        mine_kv, mine_sc = ref.save_partial_states(kv, sc, ape, ratio)
        rows = state_cache[slot_mapping // bs, slot_mapping % bs]
        worst["state_rows_vs_save_eager"] = max(worst["state_rows_vs_save_eager"], rel(rows[:, :W], mine_kv), rel(rows[:, W:], mine_sc))
        bp = torch.tensor([p for p in range(T) if (p + 1) % ratio == 0], dtype=torch.int64)
        if len(bp) == 0:
            assert ent.shape[0] == 0
            continue
        n_entries += len(bp)
        if yarn is None:
            cs = ref.make_cos_sin_cache(max(T, 2), rd, theta)
        else:
            cs = vllm_yarn_cache(ystub, ymscale, rd, theta, yarn, max(T, 2))
            worst["yarn_cos_sin_vs_vllm_class_fp64"] = max(worst["yarn_cos_sin_vs_vllm_class_fp64"],
                                                           rel(cs, ref.make_cos_sin_cache(max(T, 2), rd, theta, yarn)))
        # --- vLLM triton-test reference (kvc:2432), fp64-patched
        ns = {"torch": torch}
        exec(src["tri64"], ns)
        v_tri = ns["_reference_kv_compress_norm_rope"](state_cache.clone(), block_table, bp, nw, cs, compress_ratio=ratio,
                                                      overlap=int(overlap), rms_eps=eps)
        worst["entries_vs_triton_ref_fp64"] = max(worst["entries_vs_triton_ref_fp64"], rel(v_tri, ent))
        # --- same function, UNpatched (fp32 inside): sanity that the fp64 patch is only a precision change
        ns = {"torch": torch}
        exec(src["tri"].replace("    if return_full_cache:\n", "    return result\n    if return_full_cache:\n"), ns)
        v_tri32 = ns["_reference_kv_compress_norm_rope"](state_cache.clone(), block_table, bp, nw, cs, compress_ratio=ratio,
                                                        overlap=int(overlap), rms_eps=eps)
        worst["triton_ref_unpatched_fp32_vs_fp64"] = max(worst["triton_ref_unpatched_fp32_vs_fp64"], rel(v_tri32, ent))
        # --- vLLM CPU eager reference (cuk:256), gather_slots exactly as cpu_compressor.py:78-83
        window = coff * ratio
        win_local = bp.unsqueeze(-1) - window + 1 + torch.arange(window)
        ns = {"torch": torch}
        exec(src["map"], ns)
        gather = ns["map_local_to_global_slots_cpu"](win_local, torch.zeros(len(bp), dtype=torch.int64), block_table, bs)
        nope = hd - rd
        qb = 16
        assert nope % qb == 0
        cap = []
        ns = {"torch": torch, "CAPTURE": cap, "NOPE_DIM": nope, "ROPE_DIM": rd, "HEAD_DIM": hd, "QUANT_BLOCK": qb,
              "FP8_MAX": 448.0, "TOKEN_DATA_BYTES": nope + rd * 2, "SCALE_BYTES_PER_TOKEN": nope // qb + 1}
        exec(src["rope"], ns)
        exec(src["eager"], ns)
        kvbs = 16
        nkv = -(-len(bp) // kvbs) + 1
        kv_cache = torch.zeros(nkv, kvbs * ns["TOKEN_DATA_BYTES"] + kvbs * ns["SCALE_BYTES_PER_TOKEN"], dtype=torch.uint8)
        ns["_run_compress_eager"](state_cache, gather, bp, torch.arange(len(bp), dtype=torch.int64), nw, eps, cs, kv_cache, kvbs, ratio)
        assert len(cap) == len(bp)
        worst["prerope_vs_cpu_eager"] = max(worst["prerope_vs_cpu_eager"], rel(torch.stack([c[0] for c in cap]), pre))
        worst["entries_vs_cpu_eager"] = max(worst["entries_vs_cpu_eager"], rel(torch.stack([c[1] for c in cap]), ent))
    gate = {k: v for k, v in worst.items() if k != "triton_ref_unpatched_fp32_vs_fp64"}
    bad = {k: v for k, v in gate.items() if v > TOL_CROSS}
    assert not bad, f"oracle disagrees with vLLM reference code beyond {TOL_CROSS}: {bad}"
    assert worst["triton_ref_unpatched_fp32_vs_fp64"] < 1e-4, worst
    assert n_yarn >= 70, n_yarn
    return {"cases": n_cases, "yarn_cases": n_yarn, "entries_compared": n_entries, "seed": seed, "gate_relative": TOL_CROSS, "max_relative_disagreement": worst}


# ---------------------------------------------------------------------------------------------------------------- fixtures
def f32(t):
    return t.to(torch.float32).to(F64)


def b16(t):
    return t.to(torch.bfloat16).to(F64)


def flat(t):
    return [float(v) for v in t.reshape(-1).tolist()]


# (name, ratio, head_dim, rope_dim, hidden, T, theta)
CASES = [
    ("r4_hd16_rope8_T37", 4, 16, 8, 8, 37, 10000.0),         # 9 complete blocks + 1 pending token; block 0 has no previous block
    ("r4_hd128_rope64_T22", 4, 128, 64, 8, 22, 10000.0),    # the real indexer width (index_head_dim 128, rope 64); 5 blocks + 2 pending
    ("r4_hd32_rope16_T4", 4, 32, 16, 8, 4, 10000.0),        # exactly one block: only the no-previous-block window
    ("r4_hd32_rope16_T3", 4, 32, 16, 8, 3, 10000.0),        # no complete block: nothing emitted
    ("r4_hd16_rope8_T5", 4, 16, 8, 6, 5, 10000.0),          # one block + 1 pending
    ("r4_hd24_rope8_T64_theta160k", 4, 24, 8, 8, 64, 160000.0),  # 16 blocks, compress_rope_theta-like base
    ("r128_hd16_rope8_T300", 128, 16, 8, 8, 300, 10000.0),   # boundaries at 127 and 255, 44 pending
    ("r128_hd128_rope64_T260", 128, 128, 64, 8, 260, 10000.0),  # real main width 128 / rope 64; boundary 127, 255, 4 pending
    ("r128_hd16_rope8_T127", 128, 16, 8, 8, 127, 10000.0),   # one token short of the first boundary
    ("r128_hd16_rope8_T128", 128, 16, 8, 8, 128, 10000.0),   # last token of block 0
    ("r128_hd32_rope16_T385_theta160k", 128, 32, 16, 8, 385, 160000.0),  # 3 entries + 1 pending
]
# Production dtype: hidden / wkv / wgate / norm are bfloat16-exact (the checkpoint dtype), ape stays float32 (comp:261 state_dtype).
# vLLM multiplies bf16 x bf16 with an fp32 OUTPUT (attention.py:698-703 out_dtype=torch.float32) and pools in fp32 (compressor.py:261),
# so the golden is the float64 result computed from the bf16-rounded inputs.
CASES_BF16 = [
    ("r4_hd16_rope8_T37_bf16", 4, 16, 8, 8, 37, 10000.0),
    ("r128_hd16_rope8_T300_bf16", 128, 16, 8, 8, 300, 10000.0),
]


# eps sensitivity: tiny pooled magnitude (wkv scale 0.05 -> mean square ~1e-2) and a large rms_norm_eps, so that eps changes the output
# by ~tens of percent. With the default fixtures (pooled magnitude 2-5, eps 1e-6) eps*10 moves the output by ~5e-7 relative, below the
# tolerance: a wrong or unwired eps would pass. MUTATION CHECK recorded in the manifest ("eps_sensitivity").
CASES_EPS = [
    ("r4_hd16_rope8_T21_eps1e-2", 4, 16, 8, 8, 21, 10000.0),
    ("r128_hd16_rope8_T260_eps1e-2", 128, 16, 8, 8, 260, 10000.0),
]
EPS_LARGE, EPS_WKV_SCALE = 1e-2, 0.05

# YaRN rope on the compressor layers (rope.py:31-37): the REAL V4 config values (scripts/conformance/offline/fixtures/deepseek_v4/*.json:
# rope_scaling {type yarn, factor 16, original_max_position_embeddings 65536, beta_fast 32, beta_slow 1}, compress_rope_theta 160000)
# on tiny and on the real rope width (64), plus one non-default parameter set (factor 4, original 256, beta_fast 8) that moves the ramp.
YARN_FLASH = {"factor": 16.0, "original_max_position_embeddings": 65536, "beta_fast": 32.0, "beta_slow": 1.0}
YARN_SMALL = {"factor": 4.0, "original_max_position_embeddings": 256, "beta_fast": 8.0, "beta_slow": 1.0}
CASES_YARN = [
    (("r4_hd16_rope8_T37_yarn_flash", 4, 16, 8, 8, 37, 160000.0), YARN_FLASH),
    (("r4_hd128_rope64_T64_yarn_flash", 4, 128, 64, 8, 64, 160000.0), YARN_FLASH),
    (("r128_hd16_rope8_T300_yarn_flash", 128, 16, 8, 8, 300, 160000.0), YARN_FLASH),
    (("r128_hd128_rope64_T260_yarn_flash", 128, 128, 64, 8, 260, 160000.0), YARN_FLASH),
    (("r4_hd32_rope16_T40_yarn_small", 4, 32, 16, 8, 40, 10000.0), YARN_SMALL),
]


# Real head width: head_dim 512 / rope 64 (the shipping V4 attention compressor, `hidden` kept at 16 so the fixture stays ~1.5 MB). The hidden
# size is NOT production (4096 / 7168): the matmul here sums 16 terms, so fp32 accumulation-order effects of a 4096-term sum are not covered
# (recorded in the manifest "scope").
CASES_WIDE = [
    (("r4_hd512_rope64_T24_yarn_flash_wide", 4, 512, 64, 16, 24, 160000.0), YARN_FLASH),
    (("r128_hd512_rope64_T260_yarn_flash_wide", 128, 512, 64, 16, 260, 160000.0), YARN_FLASH),
]


def build_case(idx, spec, seed=11, dtype="float32", eps=1e-6, wkv_scale=0.6, yarn=None):
    name, ratio, hd, rd, hidden, T, theta = spec
    rnd_in = b16 if dtype == "bfloat16" else f32
    g = torch.Generator().manual_seed(seed * 1000 + idx)
    overlap = ratio == 4
    W = (1 + int(overlap)) * hd
    x = rnd_in(torch.randn(T, hidden, generator=g, dtype=F64))
    wkv = rnd_in(torch.randn(W, hidden, generator=g, dtype=F64) * wkv_scale)
    wg = rnd_in(torch.randn(W, hidden, generator=g, dtype=F64) * 2.0 / np.sqrt(hidden))
    ape = f32(torch.randn(ratio, W, generator=g, dtype=F64))
    nw = rnd_in(1 + 0.2 * torch.randn(hd, generator=g, dtype=F64))
    pre, ent = ref.compress_sequence(x, wkv, wg, ape, nw, eps, ratio, hd, rd, theta, yarn=yarn)
    nb = T // ratio
    assert ent.shape == (nb, hd)
    # deliberately WRONG readings: the fixtures must be able to tell them apart from vLLM (recorded in the manifest)
    disc = {}
    wrong = {}   # the wrong readings' entries themselves, so the Swift tests can run the shipping code against them
    for v in ("last_token_pos", "swap_halves", "no_overlap_ratio4", "plain_rope"):
        if v in ("swap_halves", "no_overlap_ratio4") and ratio != 4:
            continue
        if v == "plain_rope" and yarn is None:
            continue
        if nb == 0:
            continue
        _, e2 = ref.compress_sequence(x, wkv, wg, ape, nw, eps, ratio, hd, rd, theta, variant=v, yarn=yarn)
        disc[v] = float((e2 - ent).abs().max())
        wrong[v] = flat(e2)
    case = {
        "name": name, "dtype": dtype, "ratio": ratio, "head_dim": hd, "rope_dim": rd, "hidden": hidden, "tokens": T, "eps": eps,
        "rope_theta": theta, "blocks": nb, "pending": T - nb * ratio,
        "x": flat(x), "wkv": flat(wkv), "wgate": flat(wg), "ape": flat(ape), "norm": flat(nw),
        "pooled_normed": flat(pre), "entries": flat(ent),
        "visible": [ref.visible_count(t, ratio) for t in range(T)],
        "wrong_entries": wrong,
    }
    if yarn is not None:
        case["rope_scaling"] = yarn
    return case, disc


INPUT_KEYS = ("x", "wkv", "wgate", "ape", "norm")


def dump_fixture(path, cases):
    """json.dump with the five INPUT arrays (all float32-exact: bf16 / float32 rounded) written as the shortest round-tripping float32 text
    ('%.9g') instead of 17-digit doubles: ~2x smaller files. Swift reads Double and casts to Float, which returns exactly the float32 the
    oracle used (asserted below for every value). Goldens (float64) are written in full."""
    arrays, light = [], []
    for c in cases:
        c2 = dict(c)
        for k in INPUT_KEYS:
            vals = c[k]
            txt = ["%.9g" % v for v in vals]
            for v, t in zip(vals, txt):
                assert float(np.float32(float(t))) == v, (k, v, t)
            c2[k] = "@@%d:%s@@" % (len(arrays), k)
            arrays.append(("[" + ",".join(txt) + "]", c2[k]))
        light.append(c2)
    body = json.dumps({"op": "v4_kv_compressor", "cases": light}, separators=(",", ":"))
    for text, token in arrays:
        assert body.count('"%s"' % token) == 1
        body = body.replace('"%s"' % token, text)
    with open(path, "w") as f:
        f.write(body + "\n")


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vllm", required=True)
    ap.add_argument("--skip-crosscheck", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    xc = None if a.skip_crosscheck else crosscheck(a.vllm)
    print("crosscheck:", json.dumps(xc))
    groups = {"ratio4.json": [], "ratio128.json": [], "bf16.json": [], "eps.json": [], "yarn.json": [], "wide.json": []}
    discs = {}
    for i, spec in enumerate(CASES):
        c, d = build_case(i, spec)
        groups["ratio4.json" if spec[1] == 4 else "ratio128.json"].append(c)
        discs[c["name"]] = d
        for k, v in d.items():
            assert v > 1e-3, (c["name"], k, v)   # the wrong reading must be clearly distinguishable
    for j, spec in enumerate(CASES_BF16):
        c, d = build_case(len(CASES) + j, spec, dtype="bfloat16")
        groups["bf16.json"].append(c)
        discs[c["name"]] = d
    eps_sens = {}
    for k, spec in enumerate(CASES_EPS):
        c, d = build_case(len(CASES) + len(CASES_BF16) + k, spec, eps=EPS_LARGE, wkv_scale=EPS_WKV_SCALE)
        groups["eps.json"].append(c)
        discs[c["name"]] = d
        # how far the output moves if eps is 10x too large (the mutation a reviewer made): must exceed the fp32 tolerance by >> 10x
        x_, wkv_, wg_, ape_, nw_ = (torch.tensor(c[key], dtype=F64) for key in ("x", "wkv", "wgate", "ape", "norm"))
        sp = spec
        W = (1 + int(sp[1] == 4)) * sp[2]
        _, e10 = ref.compress_sequence(x_.reshape(sp[5], sp[4]), wkv_.reshape(W, sp[4]), wg_.reshape(W, sp[4]), ape_.reshape(sp[1], W),
                                       nw_, EPS_LARGE * 10, sp[1], sp[2], sp[3], sp[6])
        e1 = torch.tensor(c["entries"], dtype=F64).reshape(-1, sp[2])
        eps_sens[c["name"]] = float((e10 - e1).abs().max())
        assert eps_sens[c["name"]] > 1e-2, (c["name"], eps_sens[c["name"]])
    for k, (spec, yarn) in enumerate(CASES_YARN):
        c, d = build_case(len(CASES) + len(CASES_BF16) + len(CASES_EPS) + k, spec, yarn=yarn)
        groups["yarn.json"].append(c)
        discs[c["name"]] = d
        assert d["plain_rope"] > 1e-3, (c["name"], d)   # the YaRN fixtures must tell plain RoPE apart
    for k, (spec, yarn) in enumerate(CASES_WIDE):
        c, d = build_case(len(CASES) + len(CASES_BF16) + len(CASES_EPS) + len(CASES_YARN) + k, spec, yarn=yarn)
        groups["wide.json"].append(c)
        discs[c["name"]] = d
        assert d["plain_rope"] > 1e-3, (c["name"], d)
    files = {}
    for fn, cases in groups.items():
        dump_fixture(os.path.join(OUT, fn), cases)
        files[fn] = {"sha256": sha(os.path.join(OUT, fn)), "cases": len(cases)}
    manifest = {
        "vllm_commit": VLLM_COMMIT, "vllm_branch": "main (shallow clone)",
        "torch": torch.__version__, "python": platform.python_version(), "numpy": np.__version__,
        "generator": {"generate.py": sha(os.path.join(HERE, "generate.py")), "v4comp_ref.py": sha(os.path.join(HERE, "v4comp_ref.py"))},
        "files": files,
        "tolerances": {
            "bfloat16": {
                "rtol": 4e-3, "atol": 2e-5, "atol_per_position": 1.5e-7,
                "reasoning": "The module computes in fp32 like vLLM (attention.py:698-703, compressor.py:261) and returns bfloat16 (the model "
                             "dtype), so the output carries bf16 rounding: bf16 keeps 8 significant bits, a rounding is at most half an ulp = "
                             "2^-8 = 3.9e-3 relative (rtol 4e-3; NOT 2^-9). pooled_normed: one rounding, |err| <= rtol |golden|. RoPE'd entries: "
                             "the rotation reads the already rounded pair (xe, xo) and rounds its output, so for a rope channel "
                             "|err| <= rtol (|xe| + |xo| + |golden|) with xe, xo the golden pooled_normed pair (a cancelling output element "
                             "is small relative to its inputs, so a per-element relative bound would be wrong); nope channels keep one "
                             "rounding. atol and the per-position angle term as for float32. Tighter checks live in the Swift suite: the "
                             "bf16 pooled output must equal the float32 result rounded once, exactly. It deliberately does NOT allow bf16 "
                             "rounding of kv / score BEFORE the softmax pooling, which vLLM does not do."},
            "float32": {
                "rtol": 2e-5, "atol": 2e-5, "atol_per_position": 1.5e-7,
                "reasoning": "Inputs and weights are float32-exact, the oracle is float64, Swift runs float32 on Metal. pooled_normed: a "
                             "<=8-term matmul (abs error ~3e-7 on kv/score of size ~3), exp of scores up to ~8 (relative error "
                             "~5e-7 per weight), a sum of <=256 weighted terms and an rsqrt: ~1e-6 observed scale, 2e-5 leaves >10x. "
                             "entries (after RoPE) add the angle error: MLX forms position*freq in float32, so cos/sin carry an "
                             "absolute error of about 2^-23 * angle, angle <= position (frequency of the first pair is 1): "
                             "atol(entry b) = atol + atol_per_position * (b * ratio)."},
        },
        "eps_sensitivity_max_abs_diff_if_eps_x10": eps_sens,
        "scope": {
            "covers": "the compressor algebra (projection, softmax pooling with the ratio-4 overlap, RMSNorm incl. a large eps, RoPE incl. the YaRN of "
                      "the compressor layers with the real V4 config values) at head_dim 16..512, rope_dim 8..64, ratios 4 and 128, float32 and bfloat16",
            "largest_hidden_size": max(c["hidden"] for cs in groups.values() for c in cs),
            "largest_head_dim": max(c["head_dim"] for cs in groups.values() for c in cs),
            "does_not_cover": [
                "production hidden size (4096 / 7168): fp32 accumulation order of a thousands-term projection is not exercised",
                "fp8 / mxfp4 quantisation of the stored entry",
                "batch size > 1",
                "layer-level behaviour (sliding window, indexer top-k, attention sinks): other lanes",
            ],
        },
        "crosscheck": xc,
        "crosscheck_patches_to_vllm_reference_code": PATCHES,
        "discriminators_max_abs_diff_of_wrong_readings_vs_golden": discs,
        "inferred_points_not_from_vllm_source": [
            "head_dim / rope_dim are parameters in the oracle (the C++ hard-codes 512/64 and 128/64; vLLM's torch reference is generic)",
            "cos/sin cache built in float64 (vLLM builds it in float32)",
            "YaRN cos/sin built in float64 (vLLM: float32); yarn.json uses the real config values with mscale forced to 0 (rope.py:51-52), i.e. NO cos/sin scaling; the deepseek_llama_scaling variant (attention_factor == 1.0, rope.py:33) selects the same class and the same maths",
            "ratio4/ratio128/bf16/eps fixtures use plain RoPE (theta only); only yarn.json models the YaRN rope of the compressor layers",
        ],
    }
    with open(os.path.join(OUT, "MANIFEST.json"), "w") as f:
        json.dump(manifest, f, indent=1)
        f.write("\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
