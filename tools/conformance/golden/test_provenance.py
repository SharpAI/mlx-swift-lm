"""One vLLM provenance statement for every golden lane: each MANIFEST.json carries `vllm_provenance` (a list with the
statement of every vLLM commit the lane cites), taken from vllm_provenance.py, and no source still carries the old,
contradictory notes. Run:  python -I tools/conformance/golden/test_provenance.py
"""
import glob
import json
import os
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "../../.."))
GOLDEN = os.path.join(REPO, "Tests/MLXLMTests/Conformance/Golden")
FULL = {"739e5945d": "739e5945dc4b3ba30d84bdf6e637657abd4136b8", "2eae56c": "2eae56c4ba0cf12087fec035514eedede83c257d"}

# (manifest path, abbreviated commits it cites)
LANES = {
    "kda": (os.path.join(GOLDEN, "kda/MANIFEST.json"), ["739e5945d"]),
    "swiglu": (os.path.join(GOLDEN, "swiglu/MANIFEST.json"), ["739e5945d", "2eae56c"]),   # v4_clamped cites 2eae56c
    "dequant": (os.path.join(HERE, "dequant/fixtures/MANIFEST.json"), ["739e5945d"]),
    "v4route": (os.path.join(GOLDEN, "v4route/MANIFEST.json"), ["2eae56c"]),
    "v4idx": (os.path.join(GOLDEN, "v4idx/MANIFEST.json"), ["2eae56c", "739e5945d"]),     # YaRN helpers: 739e594
    "v4comp": (os.path.join(GOLDEN, "v4comp/MANIFEST.json"), ["2eae56c"]),
}
STALE = ["could not be fetched", "not re-verified", "the commit is the one the lane was given", "re-fetchable",
         "a fresh depth-1 clone had HEAD"]


def load_module():
    import importlib.util
    spec = importlib.util.spec_from_file_location("vllm_provenance", os.path.join(HERE, "vllm_provenance.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class Provenance(unittest.TestCase):
    def test_every_manifest_carries_the_shared_statement(self):
        vp = load_module()
        for lane, (path, commits) in LANES.items():
            with open(path) as f:
                m = json.load(f)
            self.assertEqual(m.get("vllm_provenance"), vp.statements(*commits), lane)

    def test_statements_name_the_full_commit_and_how_to_get_it(self):
        vp = load_module()
        for short, full in FULL.items():
            (text,) = vp.statements(short)
            self.assertIn(full, text)
            self.assertIn("git clone https://github.com/vllm-project/vllm", text)
            self.assertIn("checkout " + short, text)
        self.assertIn("VERIFIED", vp.statements("739e5945d")[0])
        self.assertIn("git cat-file -t 2eae56c", vp.statements("2eae56c")[0])

    def test_recorded_commit_matches_the_statement(self):
        for lane, (path, commits) in LANES.items():
            with open(path) as f:
                m = json.load(f)
            rec = m.get("vllmCommit") or m.get("vllm_commit") or (m.get("vllm") or {}).get("commit")
            self.assertTrue(rec and any(FULL[c].startswith(rec) or rec.startswith(c) for c in commits), (lane, rec))

    def test_no_contradictory_notes_left_in_sources(self):
        for path in glob.glob(os.path.join(HERE, "**/*.py"), recursive=True) + glob.glob(os.path.join(HERE, "**/*.txt"), recursive=True):
            if os.path.basename(path) == "test_provenance.py":
                continue
            with open(path, encoding="utf-8") as fh:
                text = fh.read()
            for bad in STALE:
                self.assertTrue(bad not in text, f"{os.path.relpath(path, REPO)} still says {bad!r}")


if __name__ == "__main__":
    unittest.main()
