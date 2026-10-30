"""Cross-check v4route_ref.py (our float64 transcription) against vLLM's OWN torch reference code.

vLLM's references are plain torch but are written for bf16/fp32 (they assert dtypes, call .float()/.bfloat16()) and
live in files that import vllm. So this script extracts the function sources with `ast` (nothing from vllm is
imported or executed), rewrites ONLY the dtype spelling (torch.float32 / torch.bfloat16 / .float() / .bfloat16() ->
float64) and executes them. The maths is vLLM's text, run in float64, so the comparison is to ~1e-15.

  [TK]  tests/kernels/moe/test_topk_softplus_sqrt.py::_torch_topk_softplus_sqrt
  [MT]  vllm/model_executor/kernels/mhc/torch.py::mhc_pre_torch, mhc_post_torch
  [MK]  tests/kernels/mhc/test_mhc_kernels.py::sinkhorn_normalize_ref, mhc_pre_ref, mhc_post_ref, hc_head_ref
  [AC]  vllm/model_executor/layers/activation.py::SiluAndMulWithClamp.forward_native
  [FU]  vllm/model_executor/layers/fused_moe/utils.py::_swiglu_limit_torch (torch.compile decorator dropped)

Needs a vLLM source tree (it is NOT part of this repo; one-time download, never from tests/CI). The oracle was
transcribed from commit 2eae56c of https://github.com/vllm-project/vllm (the commit the lane was given; the tree it
was read from had no .git, so that id is not re-verified). To reproduce:
    git clone https://github.com/vllm-project/vllm /some/dir/vllm && git -C /some/dir/vllm checkout 2eae56c
The tree root is passed with --vllm or the VLLM_ROOT environment variable; there is no default path.

Run (python reads downloaded data, so -I):
    .venv-golden/bin/python -I tools/conformance/golden/v4route_crosscheck.py --vllm /some/dir/vllm
Prints one line per check with the number of cases and the worst relative disagreement, then OK. Exit code 1 on
any disagreement above REL_TOL (1e-10) that is not one of the EXPLAINED cases (there is exactly one, below).

EXPLAINED disagreement: torch.nn.functional.softplus (used by [TK:32]) returns x itself for x > 20 (its default
`threshold`), while the kernels [CU:116] use the exact max(x,0) + log1p(exp(-|x|)). The exact value is
x + log1p(exp(-x)) > x by up to e^-20 = 2.1e-9, so the relative difference of the SCORE is <= e^-20 / (2 x) ~ 5e-11
at x = 20 and falls from there. The check "softplus threshold" measures this on logits in [20, 40] and requires the
measured value to be below that bound; every other check uses |logit| < 20 where the two agree to ~1e-16.
"""
import argparse
import ast
import math
import os
import sys
import types

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v4route_ref as R  # noqa: E402

F64 = torch.float64
REL_TOL = 1e-10
SUBS = [("torch.float32", "torch.float64"), ("torch.bfloat16", "torch.float64"), (".float()", ".double()"),
        (".bfloat16()", ".double()")]


def extract(path, names, class_name=None):
    """Return {name: function} executed from the source text of `names` in `path` (dtype spelling rewritten)."""
    src = open(path).read()
    tree = ast.parse(src)
    scope = tree
    if class_name:
        scope = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    ns = {"torch": torch, "F": F, "SimpleNamespace": types.SimpleNamespace}
    out = {}
    for node in scope.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            node.decorator_list = []  # drop @torch.compile / @CustomOp...
            text = ast.get_source_segment(src, node)
            # get_source_segment starts at 'def' (decorators are not part of it); dedent a method body
            lines = text.split("\n")
            ind = len(lines[0]) - len(lines[0].lstrip())
            text = "\n".join(l[ind:] if l[:ind].strip() == "" else l for l in lines)
            for a, b in SUBS:
                text = text.replace(a, b)
            exec(compile(text, f"{path}::{node.name}", "exec"), ns)
            out[node.name] = ns[node.name]
    missing = set(names) - set(out)
    assert not missing, f"{path}: functions not found: {missing}"
    return out


def rel(a, b):
    a, b = a.to(F64), b.to(F64)
    return float((a - b).abs().max() / max(float(b.abs().max()), 1e-30))


class Report:
    def __init__(self):
        self.rows = []
        self.bad = []

    def add(self, name, n, worst, tol=REL_TOL):
        self.rows.append((name, n, worst, tol))
        flag = "ok " if worst <= tol else "BAD"
        print(f"  {flag} {name:<44} cases={n:<5} worst relative disagreement={worst:.3e} (tol {tol:.0e})")
        if worst > tol:
            self.bad.append(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vllm", default=os.environ.get("VLLM_ROOT"),
                    help="vLLM source tree root (commit 2eae56c); or set VLLM_ROOT")
    args = ap.parse_args()
    root = args.vllm
    if not root or not os.path.isdir(os.path.join(root, "vllm")):
        ap.error("need a vLLM source tree: pass --vllm <root> or set VLLM_ROOT (see the docstring for how to fetch "
                 "commit 2eae56c)")
    g = torch.Generator()
    g.manual_seed(20261029)
    rep = Report()

    def randn(*s, scale=1.0):
        return torch.randn(*s, generator=g, dtype=F64) * scale

    # ---- routing: [TK] _torch_topk_softplus_sqrt ------------------------------------------------------------
    tk = extract(f"{root}/tests/kernels/moe/test_topk_softplus_sqrt.py", {"_torch_topk_softplus_sqrt"})["_torch_topk_softplus_sqrt"]
    worst_w, ids_equal, n = 0.0, True, 0
    for case in range(240):
        T = int(torch.randint(1, 9, (1,), generator=g))
        E = int(torch.randint(8, 65, (1,), generator=g))
        k = int(torch.randint(1, min(E, 12) + 1, (1,), generator=g))
        renorm = bool(case % 2)
        scale = [1.0, 1.5, 2.5][case % 3]
        mode = case % 4  # 0: bias, 1: no bias, 2: hash, 3: bias + exact ties
        logits = randn(T, E, scale=3.0)
        if mode == 3:
            logits[:, 1] = logits[:, 2]  # exact tie between experts 1 and 2 (same score, same biased score)
        bias = randn(E) if mode in (0, 3) else None
        if mode == 3:
            bias[2] = bias[1]
        table = ids = None
        if mode == 2:
            vocab = 20
            table = torch.stack([torch.randperm(E, generator=g)[:k] for _ in range(vocab)]).to(torch.int32)
            ids = torch.randint(0, vocab, (T,), generator=g)
        w_ref, i_ref = tk(logits, k, renorm, scale, e_score_correction_bias=bias, input_ids=ids, hash_indices_table=table)
        w, i = R.route(logits, k, renorm, scale, bias, ids, table)
        ids_equal &= bool((i == i_ref.long()).all())
        worst_w = max(worst_w, rel(w, w_ref))
        n += 1
    rep.add("routing weights vs [TK] (bias/no-bias/hash/ties)", n, worst_w)
    if not ids_equal:
        rep.bad.append("routing ids differ")
        print("  BAD routing ids differ")
    else:
        print(f"  ok  routing ids identical on all {n} cases (including exact ties)")

    # explained: the softplus threshold
    x = torch.rand(4000, generator=g, dtype=F64) * 20 + 20  # [20, 40]
    d = rel(F.softplus(x).sqrt(), R.sqrtsoftplus(x))
    bound = math.exp(-20) / (2 * 20)
    rep.add("softplus threshold (explained, x in [20,40])", 4000, d, tol=bound * 1.01)
    xs = randn(4000, scale=6.0).clamp(-19.9, 19.9)
    rep.add("sqrtsoftplus vs F.softplus.sqrt (|x|<20)", 4000, rel(F.softplus(xs).sqrt(), R.sqrtsoftplus(xs)))

    # ---- mHC: [MT] and [MK] ---------------------------------------------------------------------------------
    mt = extract(f"{root}/vllm/model_executor/kernels/mhc/torch.py", {"mhc_pre_torch", "mhc_post_torch"})
    mk = extract(f"{root}/tests/kernels/mhc/test_mhc_kernels.py",
                 {"sinkhorn_normalize_ref", "mhc_pre_ref", "mhc_post_ref", "hc_head_ref"})
    w_pre = {"post": 0.0, "comb": 0.0, "y": 0.0}
    w_pre2 = dict(w_pre)
    w_sink, w_post, w_post2, w_head, n = 0.0, 0.0, 0.0, 0.0, 0
    for case in range(220):
        T = int(torch.randint(1, 5, (1,), generator=g))
        hc = [2, 3, 4][case % 3]
        D = int(torch.randint(2, 13, (1,), generator=g))
        mix = (2 + hc) * hc
        rep_n = [1, 2, 3, 5, 20][case % 5]
        rms_eps = [1e-6, 1e-3, 1e-2][case % 3]
        hc_eps = [1e-6, 1e-4, 1e-2][(case // 3) % 3]
        res = randn(T, hc, D, scale=1.5)
        fn = randn(mix, hc * D, scale=0.3)
        scale = randn(3, scale=0.8)
        base = randn(mix, scale=0.5)
        y, post, comb = R.mhc_pre(res, fn, scale, base, rms_eps, hc_eps, rep_n)
        p_mt, c_mt, y_mt = mt["mhc_pre_torch"](res, fn, scale, base, rms_eps, hc_eps, hc_eps, 2.0, rep_n)
        p_mk, c_mk, y_mk = mk["mhc_pre_ref"](res, fn, scale, base, rms_eps, hc_eps, hc_eps, 2.0, rep_n)
        for acc, (pp, cc, yy) in ((w_pre, (p_mt, c_mt, y_mt)), (w_pre2, (p_mk, c_mk, y_mk))):
            acc["post"] = max(acc["post"], rel(post, pp.reshape(T, hc)))
            acc["comb"] = max(acc["comb"], rel(comb, cc.reshape(T, hc, hc)))
            acc["y"] = max(acc["y"], rel(y, yy.reshape(T, D)))
        sl = randn(T, hc, hc, scale=2.0)
        w_sink = max(w_sink, rel(R.sinkhorn(sl, rep_n, hc_eps), mk["sinkhorn_normalize_ref"](sl, rep_n, hc_eps)))
        x = randn(T, D)
        w_post = max(w_post, rel(R.mhc_post(x, res, post, comb), mt["mhc_post_torch"](x, res, post.reshape(T, hc, 1), comb)))
        w_post2 = max(w_post2, rel(R.mhc_post(x, res, post, comb), mk["mhc_post_ref"](x, res, post.reshape(T, hc, 1), comb)))
        hf = randn(hc, hc * D, scale=0.3)
        hs = randn(1, scale=0.8)
        hb = randn(hc, scale=0.5)
        w_head = max(w_head, rel(R.hc_head(res, hf, hs, hb, rms_eps, hc_eps), mk["hc_head_ref"](res, hf, hs, hb, rms_eps, hc_eps)))
        n += 1
    for k_, v in w_pre.items():
        rep.add(f"mhc_pre {k_} vs [MT] mhc_pre_torch", n, v)
    for k_, v in w_pre2.items():
        rep.add(f"mhc_pre {k_} vs [MK] mhc_pre_ref", n, v)
    rep.add("sinkhorn vs [MK] sinkhorn_normalize_ref", n, w_sink)
    rep.add("mhc_post vs [MT] mhc_post_torch", n, w_post)
    rep.add("mhc_post vs [MK] mhc_post_ref", n, w_post2)
    rep.add("hc_head vs [MK] hc_head_ref", n, w_head)

    # ---- clamped SwiGLU: [AC] and [FU] -----------------------------------------------------------------------
    ac = extract(f"{root}/vllm/model_executor/layers/activation.py", {"forward_native"}, class_name="SiluAndMulWithClamp")["forward_native"]
    fu = extract(f"{root}/vllm/model_executor/layers/fused_moe/utils.py", {"_swiglu_limit_torch"})["_swiglu_limit_torch"]
    w_ac, w_fu, n = 0.0, 0.0, 0
    for case in range(220):
        T, d = int(torch.randint(1, 6, (1,), generator=g)), int(torch.randint(1, 9, (1,), generator=g))
        limit = [0.5, 1.5, 3.0, 10.0][case % 4]
        gu = randn(T, 2 * d, scale=4.0)
        me = R.swiglu_clamp(gu[:, :d], gu[:, d:], limit)
        w_ac = max(w_ac, rel(me, ac(types.SimpleNamespace(swiglu_limit=limit, alpha=1.0, beta=0.0), gu)))
        out = torch.empty(T, d, dtype=F64)
        fu(out, gu, limit)
        w_fu = max(w_fu, rel(me, out))
        n += 1
    rep.add("swiglu_clamp vs [AC] SiluAndMulWithClamp", n, w_ac)
    rep.add("swiglu_clamp vs [FU] _swiglu_limit_torch", n, w_fu)

    if rep.bad:
        print("DISAGREEMENT:", rep.bad)
        sys.exit(1)
    print("OK")


if __name__ == "__main__":
    main()
