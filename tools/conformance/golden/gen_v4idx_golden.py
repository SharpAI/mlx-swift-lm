"""Generate the V4 sparse-indexer / sparse-attention golden fixtures for ConformanceV4IndexerGoldenTests.

PROVENANCE
  Maths: vLLM commit 2eae56c (main, shallow clone), transcribed in tools/conformance/golden/v4idx_ref.py with
    file:line citations; v4idx_crosscheck.py checks the oracle against vLLM's own torch reference code (extracted from the
    vLLM tree with `ast`, run raw in float32 and mechanically converted to float64) on >= 200 random cases each.
    Items that are inferred rather than in vLLM: I1..I6 in v4idx_ref.py's header.
  Python: float64 torch CPU. The torch version is recorded in MANIFEST.json ("torch").

HOW TO RUN (once, then commit the outputs; tests never run python, CI is air-gapped)
    uv venv .venv-golden && uv pip install --python .venv-golden/bin/python torch numpy \\
        --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple
    .venv-golden/bin/python -I tools/conformance/golden/v4idx_crosscheck.py --vllm <vllm src>   # must print OK
    .venv-golden/bin/python -I tools/conformance/golden/gen_v4idx_golden.py                    # rewrites Tests/.../Golden/v4idx/
  Network is needed ONLY for the one-time torch install. Output is deterministic (per-case seed = crc32(name)).

FIXTURE KINDS (what the Swift test does with each)
  score       x, qr, wq_b, weights_proj, raw compressed keys -> V4Indexer.scores(x, qr:, rope:, keys:, start:) vs `scores`
              (null = -inf), SharpAIOverlay.v4RopeCompressed(raw) vs `keysRoped`, V4Indexer.select(...) vs `select`.
  topk        a score matrix straight into the selection step.
  visibility  SharpAIOverlay.v4Visibility(length:blocks:ratio:start:) vs `visible`.
  swa         the model's local attention visibility, measured by perturbing one token at a time (swa window, golden).
  merge       SharpAIOverlay.v4CompressedAttention(...) vs `out` (window branch + compressed branch + sink in one softmax).
  rope        MLXFast RoPE of a layer's `rope` vs the float64 GPT-J rotation.
  layout      which layers of a compress_ratios list own a compressor / an indexer.
"""
import argparse
import hashlib
import json
import os
import sys
import zlib

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v4idx_ref as ref  # noqa: E402

F64 = torch.float64
VLLM_COMMIT = "2eae56c"
DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
NEG_INF = float("-inf")


def rnd(x, dtype):
    """Cut to 4 decimals (short text), then to `dtype`; the oracle sees exactly the result."""
    return torch.round(x * 1e4).div(1e4).to(DTYPES[dtype]).to(F64)


def to_dtype(x, dtype):
    return x.to(DTYPES[dtype]).to(F64)


def seeded(name):
    g = torch.Generator()
    g.manual_seed(zlib.crc32(name.encode()))
    return g


class Nums(list):
    """Flat numbers: 'f32' = shortest float32 round-trip (inputs are float32/16/bf16-exact), 'g10' = 10 significant digits
    (float64 oracle outputs; 1e-10 relative is >= 4 orders below the smallest tolerance used). None -> null (-inf)."""

    def __init__(self, vals, fmt):
        super().__init__(vals)
        self.fmt = fmt


def flat(x, fmt="f32"):
    vals = x.reshape(-1).tolist() if isinstance(x, torch.Tensor) else list(x)
    if fmt == "f32":
        import numpy as np
        for v in vals:
            assert float(np.float32(v)) == v, "input not float32-exact"
    return Nums(vals, fmt)


def _num(v, fmt):
    if v is None or v == NEG_INF:
        return "null"
    if fmt == "f32":
        import numpy as np
        return str(np.float32(v))
    t = "%.10g" % v
    return "0" if t in ("0.0", "-0.0", "0", "-0") else t


def dumps(o):
    if isinstance(o, Nums):
        return "[" + ",".join(_num(v, o.fmt) for v in o) + "]"
    if isinstance(o, dict):
        return "{" + ",".join(json.dumps(k) + ":" + dumps(o[k]) for k in sorted(o)) + "}"
    if isinstance(o, list):
        return "[" + ",".join(dumps(v) for v in o) + "]"
    return json.dumps(o, allow_nan=False)


def min_gap_ok(scores, k):
    """True when, in every row with more than k visible entries, the k-th score (the threshold) is separated from every
    DIFFERENT value by more than 5e-4 of the row's max |score| (so a float32 implementation cannot legitimately pick a
    different set). Exactly equal values are ties and are fine: the golden encodes them as strict/tie/need."""
    for row in scores.tolist():
        vis = sorted((v for v in row if v != NEG_INF), reverse=True)
        if len(vis) <= k:
            continue
        scale = max(abs(v) for v in vis) or 1.0
        thr = vis[k - 1]
        for v in vis:
            if v != thr and abs(v - thr) <= 5e-4 * scale:
                return False
    return True


# ----------------------------------------------------------------------------------------------
# score cases
# ----------------------------------------------------------------------------------------------
def score_case(name, group, L, nb, topk, **kw):
    """Draws inputs from a seed derived from (name, salt) and keeps the first salt whose selection is robust against float32
    noise (min_gap_ok); the chosen salt is stored in the fixture."""
    for salt in range(200):
        try:
            return _score_case(name, group, L, nb, topk, salt=salt, **kw)
        except _NotRobust:
            continue
    raise AssertionError(f"{name}: no robust draw in 200 salts")


class _NotRobust(Exception):
    pass


def _score_case(name, group, L, nb, topk, salt=0, start=0, theta=10000.0, nh=4, hd=16, rd=8, hid=12, ql=10, dtype="float32",
                tie=None, neg=False, check_select=True, tol=None):
    g = seeded(f"{name}#{salt}")
    r = lambda *s: torch.randn(*s, generator=g, dtype=F64)
    x = rnd(r(L, hid), dtype)
    wp = rnd(r(nh, hid) * 0.7, dtype)
    if neg:  # positive activations x negative weights_proj -> every index weight w[t,h] < 0 -> all scores <= 0
        x, wp = x.abs(), -wp.abs()
    qr = rnd(r(L, ql), dtype)
    wq_b = rnd(r(nh * hd, ql) * 0.6, dtype)
    raw = rnd(r(nb, hd), dtype)
    roped = ref.rope_compressed_keys(raw, theta, rd, 4)
    keys_used = to_dtype(roped, dtype) if dtype != "float32" else rnd(roped, "float32")
    rope_check = True
    if tie == "dup":      # entries 1 and 3 identical -> equal scores in every row that sees both
        keys_used = keys_used.clone()
        keys_used[3] = keys_used[1]
        rope_check = False
    elif tie == "all":    # every entry equal
        keys_used = keys_used[:1].repeat(nb, 1).clone()
        rope_check = False
    scores = ref.index_scores(x, qr, wq_b, wp, keys_used, start, nh, hd, rd, theta)
    sel = [ref.topk_select(row, topk) for row in scores.tolist()]
    if check_select and not min_gap_ok(scores, topk):
        raise _NotRobust()
    if tie is not None and not any(r_["need"] and len(r_["tie"]) > r_["need"] for r_ in sel):
        raise _NotRobust()          # want a draw whose exact tie really straddles the cut
    if neg:
        assert float(scores[torch.isfinite(scores)].max()) <= 0.0
    tolerance = {"float32": "score_fp32", "float16": "score_fp16", "bfloat16": "score_bf16"}[dtype]
    return {
        "name": name, "kind": "score", "group": group, "dtype": dtype, "tolerance": tol or tolerance,
        "nHead": nh, "headDim": hd, "ropeDim": rd, "hidden": hid, "qLora": ql, "ratio": 4, "theta": theta,
        "start": start, "topk": topk, "L": L, "nb": nb, "salt": salt, "checkSelect": check_select, "ropeCheck": rope_check,
        "inputs": {"x": flat(x), "qr": flat(qr), "wqB": flat(wq_b), "wProj": flat(wp), "keysRaw": flat(raw),
                   "keys": flat(keys_used)},
        "outputs": {"keysRoped": flat(roped, "g10"), "scores": flat(scores, "g10"), "select": sel},
    }


SCORE_CASES = [
    lambda: score_case("score_basic", "score", 16, 4, 2),
    lambda: score_case("score_wide_theta160k", "score", 40, 10, 3, theta=160000.0),
    lambda: score_case("score_decode_start37", "decode", 1, 9, 4, start=37),
    lambda: score_case("score_chunk_start13", "decode", 5, 4, 2, start=13),
    lambda: score_case("score_k_gt_blocks", "k_gt_available", 12, 3, 8),
    lambda: score_case("score_rows_without_blocks", "causal", 8, 2, 1),
    lambda: score_case("score_tie_dup_keys", "ties", 20, 5, 2, tie="dup"),
    lambda: score_case("score_tie_all_equal", "ties", 16, 4, 2, tie="all"),
    lambda: score_case("score_all_nonpositive", "negative", 16, 4, 2, neg=True),
    lambda: score_case("score_fp16", "dtype", 16, 4, 2, dtype="float16", check_select=False),
    lambda: score_case("score_bf16", "dtype", 16, 4, 2, dtype="bfloat16", check_select=False),
]


# ----------------------------------------------------------------------------------------------
# topk cases (a score matrix straight into the selection)
# ----------------------------------------------------------------------------------------------
def topk_case(name, group, rows, topk):
    """rows: list of lists, None = -inf."""
    nb = len(rows[0])
    scores = [[NEG_INF if v is None else float(v) for v in row] for row in rows]
    return {
        "name": name, "kind": "topk", "group": group, "dtype": "float32", "tolerance": "exact", "topk": topk,
        "R": len(rows), "nb": nb,
        "inputs": {"scores": flat(scores_flat(scores), "g10")},
        "outputs": {"select": [ref.topk_select(row, topk) for row in scores]},
    }


def scores_flat(scores):
    return [v for row in scores for v in row]


def staircase(name, group, R, nb, topk, quant=None):
    g = seeded(name)
    rows = []
    for t in range(R):
        nvis = min(nb, (t + 1) // 2)      # causal-looking visible counts: 0,1,1,2,2,...
        vals = torch.randn(nb, generator=g, dtype=F64)
        if quant:
            vals = torch.round(vals / quant) * quant
        rows.append([float(vals[j]) if j < nvis else None for j in range(nb)])
    return topk_case(name, group, rows, topk)


TOPK_CASES = [
    lambda: staircase("topk_distinct_staircase", "plain", 14, 8, 3),
    lambda: topk_case("topk_ties_at_threshold", "ties", [[5, 3, 3, 3, 1], [3, 5, 3, 1, 3]], 3),
    lambda: topk_case("topk_all_equal", "ties", [[2, 2, 2, 2, 2]], 2),
    lambda: topk_case("topk_k_gt_visible", "k_gt_available", [[4, 1, 7, None, None, None], [9, None, None, None, None, None]], 5),
    lambda: topk_case("topk_none_visible", "empty", [[None, None, None], [None, None, None], [0.5, None, None]], 2),
    lambda: topk_case("topk_k_gt_columns", "k_gt_available", [[1, 3, 2], [3, 3, 3], [None, 7, 5]], 8),
    lambda: topk_case("topk_negative_ties", "negative", [[-1, -1, -2, -1, -3], [-2, -2, -2, -5, None]], 2),
    lambda: topk_case("topk_zero_scores", "negative", [[0, 0, None, 0], [0, -1, None, 0], [0, 0, 0, 0]], 2),
    lambda: topk_case("topk_k1", "plain", [[1, 9, 4], [-3, -2, -9]], 1),
    lambda: staircase("topk_random_ties_k5", "ties", 24, 16, 5, quant=0.5),
]


# ----------------------------------------------------------------------------------------------
# visibility cases
# ----------------------------------------------------------------------------------------------
def vis_case(name, group, length, blocks, ratio, start):
    return {
        "name": name, "kind": "visibility", "group": group, "dtype": "int", "tolerance": "exact",
        "length": length, "blocks": blocks, "ratio": ratio, "start": start,
        "outputs": {"visible": ref.visibility_matrix(length, blocks, ratio, start)},
    }


VIS_CASES = [
    lambda: vis_case("vis_r4_prefill", "ratio4", 24, 6, 4, 0),
    lambda: vis_case("vis_r4_decode_pos39", "ratio4", 1, 10, 4, 39),
    lambda: vis_case("vis_r4_chunk_start10", "ratio4", 6, 4, 4, 10),
    lambda: vis_case("vis_r4_extra_blocks", "ratio4", 8, 8, 4, 0),
    lambda: vis_case("vis_r128_prefill", "ratio128", 300, 2, 128, 0),
    lambda: vis_case("vis_r128_decode_pos383", "ratio128", 1, 4, 128, 383),
]


# ----------------------------------------------------------------------------------------------
# swa cases
# ----------------------------------------------------------------------------------------------
def swa_case(name, length, window):
    return {
        "name": name, "kind": "swa", "group": "swa", "dtype": "int", "tolerance": "exact",
        "length": length, "window": window,
        "outputs": {"visible": ref.swa_visibility_matrix(length, window)},
    }


SWA_CASES = [
    lambda: swa_case("swa_w4_l12", 12, 4),
    lambda: swa_case("swa_w3_l10", 10, 3),
    lambda: swa_case("swa_w8_l12", 12, 8),
    lambda: swa_case("swa_w8_l6_window_exceeds_length", 6, 8),
    lambda: swa_case("swa_w1_l9", 9, 1),
]


# ----------------------------------------------------------------------------------------------
# merge cases
# ----------------------------------------------------------------------------------------------
def merge_case(name, group, H, L, S, nb, D, window, sink, dtype="float32", comp_vis="rand", local="array", tol=None):
    g = seeded(name)
    r = lambda *s: torch.randn(*s, generator=g, dtype=F64)
    start = S - L
    q = rnd(r(H, L, D), dtype)
    kv = rnd(r(S, D), dtype)
    comp = rnd(r(nb, D), dtype) if nb else torch.zeros(0, D, dtype=F64)
    scale = D ** -0.5
    win_vis = ref.swa_visibility_matrix(L, window, start=start, total=S)
    if local == "causal":
        assert window >= S, "causal local mask needs window >= S"
    if comp_vis == "rand":
        cv = (torch.rand(L, nb, generator=g) > 0.4).tolist() if nb else [[] for _ in range(L)]
    elif comp_vis == "none":
        cv = [[False] * nb for _ in range(L)]
    else:
        cv = [[True] * nb for _ in range(L)]
    sink_t = None if sink is None else rnd(r(H) * 1.5, dtype)
    out = ref.merge_attention(q, kv, comp, win_vis, cv, sink_t, scale)
    tolerance = tol or {"float32": "merge_fp32", "float16": "merge_fp16", "bfloat16": "merge_bf16"}[dtype]
    return {
        "name": name, "kind": "merge", "group": group, "dtype": dtype, "tolerance": tolerance,
        "H": H, "L": L, "S": S, "nb": nb, "D": D, "window": window, "start": start, "scale": scale, "localMask": local,
        "inputs": {"q": flat(q), "kv": flat(kv), "comp": flat(comp), "compVis": cv, "winVis": win_vis,
                   "sink": None if sink_t is None else flat(sink_t)},
        "outputs": {"out": flat(out, "g10")},
    }


MERGE_CASES = [
    lambda: merge_case("merge_window4_sink", "merge", 2, 10, 10, 2, 8, 4, True),
    lambda: merge_case("merge_window4_nosink", "merge", 2, 10, 10, 2, 8, 4, None),
    lambda: merge_case("merge_decode_cache_window8", "cache", 2, 1, 13, 3, 8, 8, True, comp_vis="all"),
    lambda: merge_case("merge_chunk_cache_window5", "cache", 3, 4, 14, 3, 8, 5, True),
    lambda: merge_case("merge_causal_no_window_effect", "merge", 2, 6, 6, 1, 8, 99, True, local="causal", comp_vis="all"),
    lambda: merge_case("merge_window_only", "merge", 2, 8, 8, 2, 8, 3, True, comp_vis="none"),
    lambda: merge_case("merge_window1_self_plus_comp", "merge", 2, 6, 6, 3, 8, 1, True, comp_vis="all"),
    lambda: merge_case("merge_fp16", "dtype", 2, 8, 8, 2, 8, 4, True, dtype="float16"),
    lambda: merge_case("merge_bf16", "dtype", 2, 8, 8, 2, 8, 4, True, dtype="bfloat16"),
]


# ----------------------------------------------------------------------------------------------
# layout cases
# ----------------------------------------------------------------------------------------------
def layout_case(name, ratios):
    return {
        "name": name, "kind": "layout", "group": "layout", "dtype": "int", "tolerance": "exact",
        "ratios": ratios, "outputs": {"layers": ref.layer_layout(ratios)},
    }


LAYOUT_CASES = [
    lambda: layout_case("layout_flash_like", [0, 0, 4, 128, 4, 128, 4, 0]),
    lambda: layout_case("layout_starts_compressed", [4, 128, 0, 4]),
    lambda: layout_case("layout_all_swa", [0, 0, 0]),
]

# ----------------------------------------------------------------------------------------------
# rope cases (which theta a layer must use is checked by the Swift test against vLLM's rule, common/rope.py:30-52)
# ----------------------------------------------------------------------------------------------
def rope_case(name, theta, offset, L, rd=8):
    g = seeded(name)
    x = rnd(torch.randn(L, rd, generator=g, dtype=F64), "float32")
    out = ref.gptj_rope(x, offset + torch.arange(L), theta)
    return {
        "name": name, "kind": "rope", "group": "rope", "dtype": "float32", "tolerance": "rope_fp32",
        "ropeDim": rd, "theta": theta, "offset": offset, "L": L,
        "inputs": {"x": flat(x)}, "outputs": {"out": flat(out, "g10")},
    }


ROPE_CASES = [
    lambda: rope_case("rope_theta10000_offset0", 10000.0, 0, 6),
    lambda: rope_case("rope_theta160000_offset3", 160000.0, 3, 5),
]

# ----------------------------------------------------------------------------------------------
# manifest
# ----------------------------------------------------------------------------------------------
U16, UBF = 2.0 ** -11, 2.0 ** -8
TOLERANCES = {
    "exact": {
        "mode": "exact", "y": {"rtol": 0.0, "atol": 0.0},
        "why": "booleans / indices / selections: integer outcomes, compared exactly (selection through the strict/tie/need "
               "contract, which makes the legitimately unspecified tie-break irrelevant)."},
    "score_fp32": {
        "mode": "normwise", "y": {"rtol": 3e-6, "atol": 1e-7},
        "why": "float32 vs float64, normwise vs max|want| over the fixture. u = 2^-24 = 6e-8. Error sources: q = wq_b(qr) "
               "(10-term dot), RoPE (float32 angle pos * inv_freq, pos <= 47), the 16-term q.k dot, w = weights_proj(x) * "
               "scale, the signed sum over 4 heads (cancellation, <= ~3x). A first analytic bound of 5e-5 was far too "
               "pessimistic: the MEASURED worst case over all fp32 score fixtures is 2.6e-7 (= 4u, score_wide_theta160k). "
               "3e-6 is ~10x that and 12x tighter than the first bound; a formula defect (missing relu, wrong 1/sqrt "
               "factor, wrong RoPE position or theta) moves scores by >= 1e-2, i.e. 3000x this."},
    "score_fp16": {
        "mode": "normwise", "y": {"rtol": 4 * U16, "atol": 1e-6},
        "why": "inputs (x, qr, weights, keys) are rounded to float16 BEFORE the oracle sees them, so input rounding is not in "
               "the error. Matmuls accumulate in float32 and round their output once (u = 2^-11 = 4.9e-4); the RoPE, the "
               "scale multiply, the product and the head sum round again. Measured worst: 0.73u (score_fp16). Budget 4u = "
               "2e-3 (5x measured). A formula defect is >= 1e-2."},
    "score_bf16": {
        "mode": "normwise", "y": {"rtol": 4 * UBF, "atol": 1e-6},
        "why": "as float16 with bfloat16's 8-bit significand: u = 2^-8 = 3.9e-3. Measured worst: 1.3u (score_bf16). Budget "
               "4u = 1.6e-2 (3x measured); a formula defect (a missing 1/sqrt(nHeads), the relu, a wrong position) is >= 5e-2."},
    "rope_fp32": {
        "mode": "normwise", "y": {"rtol": 5e-6, "atol": 1e-7},
        "why": "float32 RoPE vs float64: the rotation angle pos * inv_freq is formed in float32 (<= 8 * 1 * 6e-8 = 5e-7 rad "
               "for pos <= 8 and the largest frequency), cos/sin and the 2-term products add ~2e-7. 5e-6 is ~10x that."},
    "merge_fp32": {
        "mode": "normwise", "y": {"rtol": 2e-6, "atol": 1e-7},
        "why": "float32 softmax attention over <= 14+3 keys, head dim 8, logits |s| <~ 4, normwise vs max|out|. Measured worst "
               "1.9e-7 (3u). 2e-6 is ~10x measured; the first guess of 2e-5 was 100x too loose. A defect (sink dropped or "
               "scaled, window row leaking in, compressed row masked) is >= 1e-3."},
    "merge_fp16": {
        "mode": "normwise", "y": {"rtol": 2 * U16, "atol": 1e-6},
        "why": "inputs rounded to float16 before the oracle; the fused kernel accumulates in float32 and rounds the output once "
               "(u = 4.9e-4), flash-style kernels may also round the probabilities. Measured worst 0.52u (merge_fp16). "
               "Budget 2u = 1e-3 (4x)."},
    "merge_bf16": {
        "mode": "normwise", "y": {"rtol": 3 * UBF, "atol": 1e-6},
        "why": "as float16 with u = 2^-8 = 3.9e-3. Measured worst 0.97u (merge_bf16). Budget 3u = 1.2e-2 (3x)."},
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
    ap.add_argument("--out", default=os.path.join(root, "Tests", "MLXLMTests", "Conformance", "Golden", "v4idx"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for f in os.listdir(a.out):
        if f.endswith(".json"):
            os.remove(os.path.join(a.out, f))
    makers = SCORE_CASES + TOPK_CASES + VIS_CASES + SWA_CASES + MERGE_CASES + LAYOUT_CASES + ROPE_CASES
    fixtures = [m() for m in makers]
    names = [f["name"] for f in fixtures]
    assert len(set(names)) == len(names), "duplicate fixture names"
    files = {}
    for fx in fixtures:
        fn = f"{fx['name']}.json"
        write_json(os.path.join(a.out, fn), fx)
        files[fn] = {"sha256": sha256(os.path.join(a.out, fn)), "kind": fx["kind"], "group": fx["group"],
                     "dtype": fx["dtype"], "tolerance": fx["tolerance"]}
    manifest = {
        "schema": 1,
        "generator": "tools/conformance/golden/gen_v4idx_golden.py",
        "generatorSha256": sha256(os.path.join(here, "gen_v4idx_golden.py")),
        "oracle": "tools/conformance/golden/v4idx_ref.py",
        "oracleSha256": sha256(os.path.join(here, "v4idx_ref.py")),
        "vllmCommit": VLLM_COMMIT,
        "torch": torch.__version__,
        "oracleDtype": "float64",
        "note": "Generated once and committed; tests never run python. Edit nothing by hand: rerun the generator. "
                "Not in vLLM / not modelled: see v4idx_ref.py header I1..I6 (fp8 quantisation of q/k is not modelled; "
                "tie-breaking is unspecified by vLLM so the golden is a strict/tie/need contract; fixtures use plain RoPE; "
                "compressed keys are inputs; sink = extra unscaled logit with zero value, -inf = none).",
        "criterion": "per tolerance class `mode`: normwise = max|got-want| <= atol + rtol*max|want| over a tensor; exact = equality",
        "tolerances": TOLERANCES,
        "count": len(files),
        "files": files,
    }
    write_json(os.path.join(a.out, "MANIFEST.json"), manifest)
    total = sum(os.path.getsize(os.path.join(a.out, f)) for f in os.listdir(a.out))
    print(f"wrote {len(files)} fixtures + MANIFEST.json, {total/1024:.0f} KiB, torch {torch.__version__}")


if __name__ == "__main__":
    main()
