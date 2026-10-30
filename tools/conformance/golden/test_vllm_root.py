"""Every tool that needs a vLLM checkout takes it from --vllm or the VLLM_ROOT environment variable, has NO default
path, and says so clearly when neither is given. Run with the golden venv (the scripts import torch):
    .venv-golden/bin/python -I tools/conformance/golden/test_vllm_root.py
"""
import os
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = [
    ("dequant/dequant_crosscheck.py", []),
    ("dequant/gen_dequant_golden.py", ["--out", "{tmp}/out"]),
    ("v4route_crosscheck.py", []),
    ("v4idx/v4idx_crosscheck.py", []),
    ("v4comp/generate.py", []),
    ("swiglu/generate.py", []),
]


def run(script, extra, env_add, tmp):
    env = {k: v for k, v in os.environ.items() if not k.startswith("VLLM_")}
    env.update(env_add)
    args = [a.replace("{tmp}", tmp) for a in extra]
    # cwd is a scratch dir so no relative default (../../3rd_party/vllm) can accidentally resolve
    p = subprocess.run([sys.executable, "-I", os.path.join(HERE, script)] + args, env=env, cwd=tmp,
                       capture_output=True, text=True, timeout=120)
    return p.returncode, p.stdout + p.stderr


class VllmRoot(unittest.TestCase):
    def test_no_argument_no_env_is_a_clear_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            for script, extra in SCRIPTS:
                rc, out = run(script, extra, {}, tmp)
                self.assertNotEqual(rc, 0, script)
                self.assertIn("VLLM_ROOT", out, f"{script}: error must name VLLM_ROOT\n{out[-300:]}")
                self.assertNotIn("Traceback", out, f"{script}: must be a clean error, not a traceback\n{out[-300:]}")

    def test_old_env_name_is_not_honoured(self):
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, "fake", "vllm"))
            for script, extra in SCRIPTS:
                rc, out = run(script, extra, {"VLLM_SRC": os.path.join(tmp, "fake")}, tmp)
                self.assertNotEqual(rc, 0, script)
                self.assertIn("VLLM_ROOT", out, f"{script}\n{out[-300:]}")

    def test_missing_directory_is_a_clear_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            for script, extra in SCRIPTS:
                rc, out = run(script, extra, {"VLLM_ROOT": os.path.join(tmp, "nope")}, tmp)
                self.assertNotEqual(rc, 0, script)
                self.assertIn("VLLM_ROOT", out, f"{script}\n{out[-300:]}")
                self.assertNotIn("Traceback", out, f"{script}\n{out[-300:]}")

    def test_no_personal_or_missing_default_paths_in_sources(self):
        for root, _, files in os.walk(HERE):
            for f in files:
                if f.endswith((".py", ".md", ".txt")) and f != os.path.basename(__file__):
                    with open(os.path.join(root, f), encoding="utf-8") as fh:
                        text = fh.read()
                    self.assertTrue("VLLM_SRC" not in text, f"{f} still mentions VLLM_SRC")
                    self.assertTrue('"3rd_party"' not in text, f"{f} still builds a 3rd_party default path")


if __name__ == "__main__":
    unittest.main()
