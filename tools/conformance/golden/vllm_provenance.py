"""The single statement of which vLLM commits the golden lanes cite, whether they are verified, and how to obtain them.
Every generator writes `statements(...)` into its MANIFEST.json as `vllm_provenance`; test_provenance.py keeps the
manifests and this file equal. Change a statement HERE, then re-run the generators.
"""

_CLONE = "git clone https://github.com/vllm-project/vllm && git -C vllm checkout "

_S = {
    "739e5945d": (
        "vLLM 739e5945dc4b3ba30d84bdf6e637657abd4136b8 (2026-04-13): VERIFIED. Read from a git checkout whose HEAD is exactly "
        "this commit (git rev-parse HEAD); it has no DeepSeek-V4 model. Obtain: " + _CLONE + "739e5945d"),
    "2eae56c": (
        "vLLM 2eae56c4ba0cf12087fec035514eedede83c257d: EXISTS in vllm-project/vllm (checked 2026-10-29: "
        "git clone --bare --filter=tree:0 https://github.com/vllm-project/vllm, then git cat-file -t 2eae56c -> commit). "
        "Earlier notes that it was unfetchable came from `git fetch origin 2eae56c`, which cannot work with an "
        "abbreviated id; a shallow clone's HEAD (e.g. 335dd77) is just a different commit of a moving main. The oracles' "
        "file:line citations were transcribed from a source tree that had no .git and are NOT individually re-read against "
        "this commit, except vllm/model_executor/layers/activation.py:239-243 (SiluAndMulWithClamp.forward_native: gate "
        "clamped with max only, up two-sided), which was. The v4route, v4idx and v4comp cross-check scripts were re-run on "
        "2026-10-29 against a tree extracted from this commit (git archive of vllm/, tests/kernels, tests/v1/attention) "
        "and print OK. Obtain: " + _CLONE + "2eae56c"),
}


def statements(*commits):
    """The provenance statements for the given abbreviated commits, in the order given."""
    return [_S[c] for c in commits]
