"""Self-tests of kda_diff.py (LOCAL ONLY, needs the golden venv: torch + numpy; no Swift, no network).

    <venv>/bin/python -I tools/conformance/differential/test_kda_diff.py

They prove the comparison is not vacuous: exact Swift-like outputs pass, a perturbed / NaN / missing / mis-sized
output fails, and the generated cases honour the shape sweep and the tolerance-class rule.
"""
import json
import os
import shutil
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
