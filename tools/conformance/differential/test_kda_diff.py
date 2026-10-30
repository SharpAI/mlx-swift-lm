"""Self-tests of kda_diff.py (LOCAL ONLY, needs the golden venv: torch + numpy; no Swift, no network).

    <venv>/bin/python -I tools/conformance/differential/test_kda_diff.py

They prove the comparison is not vacuous: exact Swift-like outputs pass, a perturbed / NaN / missing / mis-sized
output fails, and the generated cases honour the shape sweep and the tolerance-class rule.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import kda_diff as D  # noqa: E402


def write_truth_as_swift_output(d, perturb=None, drop=None, nan=None):
    """Write DIR/out/<case>.json the way the Swift suite does, from the float64 truth cast to float32."""
    import numpy as np
    idx = json.load(open(os.path.join(d, "cases.json")))
    os.makedirs(os.path.join(d, "out"), exist_ok=True)
    for e in idx["cases"]:
        if e["name"] == drop:
            continue
        f = json.load(open(os.path.join(d, e["file"])))
        y = [float(np.float32(v)) for v in f["outputs"]["y"]]
        s = [float(np.float32(v)) for v in f["outputs"]["state1"]]
        if perturb == e["name"]:
            s = [v + 1e-3 * (abs(max(s, key=abs)) + 1) if i == 0 else v for i, v in enumerate(s)]
        if nan == e["name"]:
            y[0] = "NaN"
        out = {"name": e["name"], "kernel": {"y": y, "state": s}, "fallback": {"y": y, "state": s}}
        json.dump(out, open(os.path.join(d, "out", e["name"] + ".json"), "w"))


class DiffTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="kda-diff-test-")
        cls.dir = os.path.join(cls.tmp, "g")
        D.gen(cls.dir, 12, 7)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def clean_out(self):
        shutil.rmtree(os.path.join(self.dir, "out"), ignore_errors=True)

    def test_check_matches_swift_semantics(self):
        tol = {"rtol": 1e-3, "atol": 0.0}
        self.assertTrue(D.check([1.0, 2.0], [1.0, 2.0], tol, "normwise").ok)
        self.assertFalse(D.check([1.0, 2.1], [1.0, 2.0], tol, "normwise").ok)
        self.assertFalse(D.check([1.0], [1.0, 2.0], tol, "normwise").ok)  # size mismatch
        self.assertFalse(D.check([float("nan")], [1.0], tol, "normwise").ok)
        self.assertFalse(D.check([], [], tol, "normwise").ok)  # empty never passes
        # normwise allows the small element to be off by rtol*max; elementwise does not
        self.assertTrue(D.check([100.0, 0.05], [100.0, 0.0], tol, "normwise").ok)
        self.assertFalse(D.check([100.0, 0.05], [100.0, 0.0], tol, "elementwise").ok)

    def test_generated_cases_cover_the_sweep(self):
        idx = json.load(open(os.path.join(self.dir, "cases.json")))
        self.assertEqual(len(idx["cases"]), 12)
        self.assertEqual({e["T"] for e in D.plan(60, 1)} >= set(D.T_SWEEP), True)
        for e in D.plan(60, 1):
            self.assertIn(e["Dk"] if e["expectedPathHint"] == "kernel" else 32, (32, 64, 128) if e["expectedPathHint"] == "kernel" else (32,))
            self.assertLessEqual(e["T"] * e["Dk"], 896 if e["dtype"] != "float32" else 8192)

    def test_exact_outputs_pass(self):
        self.clean_out()
        write_truth_as_swift_output(self.dir)
        self.assertEqual(D.compare(self.dir, out=open(os.devnull, "w")), 0)

    def test_perturbed_output_fails(self):
        self.clean_out()
        name = json.load(open(os.path.join(self.dir, "cases.json")))["cases"][3]["name"]
        write_truth_as_swift_output(self.dir, perturb=name)
        self.assertNotEqual(D.compare(self.dir, out=open(os.devnull, "w")), 0)

    def test_nan_missing_and_absent_out_dir_fail(self):
        self.clean_out()
        self.assertNotEqual(D.compare(self.dir, out=open(os.devnull, "w")), 0)  # nothing ran
        names = [e["name"] for e in json.load(open(os.path.join(self.dir, "cases.json")))["cases"]]
        self.clean_out()
        write_truth_as_swift_output(self.dir, nan=names[0])
        self.assertNotEqual(D.compare(self.dir, out=open(os.devnull, "w")), 0)
        self.clean_out()
        write_truth_as_swift_output(self.dir, drop=names[1])
        self.assertNotEqual(D.compare(self.dir, out=open(os.devnull, "w")), 0)

    def test_tolerance_class_rule(self):
        self.assertEqual(D.tol_class("float32", 7, 128), "fp32")
        self.assertEqual(D.tol_class("float32", 256, 32), "fp32_long")
        self.assertEqual(D.tol_class("bfloat16", 7, 128), "bf16")
        with self.assertRaises(ValueError):
            D.tol_class("float32", 128, 128)  # beyond the documented bound: never silently loosened
        with self.assertRaises(ValueError):
            D.tol_class("float16", 128, 32)


FAKE_RUNNER = r"""
import json, os, sys
import numpy as np
d = os.environ["KDA_DIFF_DIR"]
idx = json.load(open(os.path.join(d, "cases.json")))
os.makedirs(os.path.join(d, "out"), exist_ok=True)
for e in idx["cases"]:
    f = json.load(open(os.path.join(d, e["file"])))
    y = [float(np.float32(v)) for v in f["outputs"]["y"]]
    s = [float(np.float32(v)) for v in f["outputs"]["state1"]]
    if FAULT(e):  # injected "kernel bug"
        s = [v + 0.1 for v in s]
    json.dump({"name": e["name"], "kernel": {"y": y, "state": s}, "fallback": {"y": y, "state": [float(np.float32(v)) for v in f["outputs"]["state1"]]}},
              open(os.path.join(d, "out", e["name"] + ".json"), "w"))
"""


class ShrinkTests(unittest.TestCase):
    def test_candidates_are_smaller_valid_and_keep_the_dispatch_path(self):
        spec = dict(D.plan(60, 1)[52])  # a kernel-path float16 case, T=7, Dk=128
        self.assertEqual(D.dispatch_path(spec["Dk"], spec["Dv"]), "kernel")
        cands = D.candidates(spec)
        self.assertGreater(len(cands), 5)
        names = set()
        for c in cands:
            self.assertEqual(D.dispatch_path(c["Dk"], c["Dv"]), "kernel")
            D.tol_class(c["dtype"], c["T"], c["Dk"])  # does not raise
            self.assertLessEqual(D.size(c), D.size(spec))
            self.assertNotEqual([c[k] for k in D.SHAPE_KEYS + D.FEATURE_KEYS], [spec[k] for k in D.SHAPE_KEYS + D.FEATURE_KEYS])
            self.assertEqual(c["Hv"] % c["Hk"], 0)
            self.assertNotIn(c["name"], names)
            names.add(c["name"])

    def run_shrink(self, fault_src):
        tmp = tempfile.mkdtemp(prefix="kda-shrink-test-")
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        runner = os.path.join(tmp, "fake_runner.py")
        open(runner, "w").write("FAULT = " + fault_src + "\n" + FAKE_RUNNER)
        d = os.path.join(tmp, "g")
        D.gen(d, 30, 3)
        env = dict(os.environ)
        # the original dir needs outputs too: run the fake runner on it
        env["KDA_DIFF_DIR"] = d
        subprocess.check_call([sys.executable, "-I", runner], env=env)
        return d, runner

    def test_shrink_finds_the_minimal_masked_case(self):
        d, runner = self.run_shrink("lambda e: e['mask'] is not None")
        victim = next(e["name"] for e in json.load(open(os.path.join(d, "cases.json")))["cases"] if e["mask"] and e["T"] > 7)
        res = D.shrink(d, victim, [sys.executable, "-I", runner], log=open(os.devnull, "w"))
        m = res["spec"]
        self.assertTrue(m["mask"] is not None)  # the fault needs a mask, so shrink must keep it
        self.assertEqual((m["B"], m["T"], m["Hk"], m["Hv"]), (1, 1, 1, 1))
        self.assertEqual(D.dispatch_path(m["Dk"], m["Dv"]), res["orig_path"])
        self.assertTrue(os.path.exists(res["fixture"]))
        self.assertIn('case("%s"' % m["name"], res["case_line"])
        fx = json.load(open(res["fixture"]))
        self.assertEqual(fx["name"], m["name"])

    def test_shrink_refuses_a_passing_case(self):
        d, runner = self.run_shrink("lambda e: False")
        name = json.load(open(os.path.join(d, "cases.json")))["cases"][0]["name"]
        with self.assertRaises(SystemExit):
            D.shrink(d, name, [sys.executable, "-I", runner], log=open(os.devnull, "w"))

    def test_shrink_reports_when_the_round_cap_stops_it(self):
        import io
        d, runner = self.run_shrink("lambda e: e['mask'] is not None")
        victim = next(e["name"] for e in json.load(open(os.path.join(d, "cases.json")))["cases"] if e["mask"] and e["T"] > 7)
        log = io.StringIO()
        res = D.shrink(d, victim, [sys.executable, "-I", runner], max_rounds=1, log=log)
        self.assertTrue(res["capped"])
        self.assertIn("NOT proven minimal", log.getvalue())
        self.assertNotIn("SMALLEST FAILING CASE:", log.getvalue())
        full = io.StringIO()
        res = D.shrink(d, victim, [sys.executable, "-I", runner], log=full)
        self.assertFalse(res["capped"])
        self.assertIn("SMALLEST FAILING CASE:", full.getvalue())
        self.assertNotIn("NOT proven minimal", full.getvalue())


class RelativeDirTests(unittest.TestCase):
    """A relative --dir / dirpath must mean the same directory to python (caller's cwd) and to swift (run_local.sh cd's to the root)."""

    def test_run_local_passes_an_absolute_dir_to_swift_and_python(self):
        tmp = os.path.realpath(tempfile.mkdtemp(prefix="kda-rel-test-"))
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        rec = os.path.join(tmp, "rec.txt")
        os.makedirs(os.path.join(tmp, "bin"))
        fake_py = os.path.join(tmp, "fakepy.sh")
        open(fake_py, "w").write('#!/usr/bin/env bash\nprintf "py %s\\n" "$*" >> "$REC"\nwhile [ $# -gt 0 ]; do case "$1" in --out) mkdir -p "$2";; esac; shift; done\nexit 0\n')
        fake_swift = os.path.join(tmp, "bin", "swift")
        open(fake_swift, "w").write('#!/usr/bin/env bash\nprintf "swift %s\\n" "$KDA_DIFF_DIR" >> "$REC"\nexit 0\n')
        for f in (fake_py, fake_swift):
            os.chmod(f, 0o755)
        env = dict(os.environ, REC=rec, KDA_DIFF_PY=fake_py, PATH=os.path.join(tmp, "bin") + os.pathsep + os.environ["PATH"])
        subprocess.check_call(["bash", os.path.join(HERE, "run_local.sh"), "--n", "1", "--dir", "out1"], cwd=tmp, env=env,
                              stdout=subprocess.DEVNULL)
        lines = open(rec).read().splitlines()
        want = os.path.join(tmp, "out1")
        self.assertTrue(os.path.isdir(want))
        self.assertIn("swift " + want, lines)  # absolute, equal to where python wrote it
        self.assertTrue(any(l.startswith("py ") and ("--out " + want) in l for l in lines), lines)

    def test_run_passes_an_absolute_dir_to_the_runner(self):
        tmp = os.path.realpath(tempfile.mkdtemp(prefix="kda-rel-test-"))
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        os.makedirs(os.path.join(tmp, "rel"))
        rec = os.path.join(tmp, "rec.txt")
        runner = [sys.executable, "-I", "-c", "import os,sys; open(%r,'w').write(os.environ['KDA_DIFF_DIR'])" % rec]
        old = os.getcwd()
        os.chdir(tmp)
        try:
            self.assertEqual(D._run(runner, "rel"), 0)
        finally:
            os.chdir(old)
        self.assertEqual(open(rec).read(), os.path.join(tmp, "rel"))


class LoadGenTests(unittest.TestCase):
    def test_oracle_switches_do_not_reuse_a_stale_generator_module(self):
        tmp = tempfile.mkdtemp(prefix="kda-oracle-test-")
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        alt = os.path.join(tmp, "kda_ref.py")
        shutil.copy(os.path.join(D.GOLDEN_TOOLS, "kda_ref.py"), alt)
        pinned = os.path.join(D.GOLDEN_TOOLS, "kda_ref.py")
        self.assertEqual(os.path.realpath(D.load_gen(None).kda_ref.__file__), os.path.realpath(pinned))
        self.assertEqual(os.path.realpath(D.load_gen(alt).kda_ref.__file__), os.path.realpath(alt))  # not the cached pinned one
        self.assertEqual(os.path.realpath(D.load_gen(None).kda_ref.__file__), os.path.realpath(pinned))  # mutated oracle does not leak


class HygieneTests(unittest.TestCase):
    def test_no_duplicate_or_unused_imports_in_kda_diff(self):
        import ast
        src = open(os.path.join(HERE, "kda_diff.py")).read()
        tree = ast.parse(src)
        seen = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                seen += [a.asname or a.name for a in node.names]
        self.assertEqual(sorted(n for n in set(seen) if seen.count(n) > 1), [], "duplicate imports")
        used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        self.assertEqual([n for n in set(seen) if n.split(".")[0] not in used], [], "unused imports")


if __name__ == "__main__":
    unittest.main(verbosity=2)
