# KDA differential harness (local only)

Random-case comparison of the SharpAI KDA Metal kernel and its op-composition fallback against the float64
oracle `tools/conformance/golden/kda_ref.py`. It complements the fixed golden suite
(`ConformanceKdaGoldenTests`): the golden cases are hand-picked and committed, this draws new shapes, dtypes and
seeds on demand and exchanges data through files.

**It is local-only because CI is air-gapped.** Generating cases needs torch and numpy (a one-time network install),
and CI has neither network nor python in the test path. Nothing here runs in CI: the Swift suite is skipped
unless `KDA_DIFF_DIR` is set, the python files are never invoked by a test, and `Package.swift` is untouched
(no new target, the Swift file lives in the existing `MLXLMTests` target).

## Setup (once)

    uv venv .venv-golden && uv pip install --python .venv-golden/bin/python torch numpy \
        --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple

## Run

    tools/conformance/differential/run_local.sh --n 60 --seed 1
    # with the GPU lock and a prebuilt test bundle:
    KDA_DIFF_WRAP="python3 /path/with-lock.py gpu 1 --" KDA_DIFF_SWIFT_ARGS="--skip-build" \
        tools/conformance/differential/run_local.sh --n 60 --seed 1 --dir /tmp/kda-diff-1

Steps (also usable by hand):

1. `kda_diff.py gen --out DIR --n N --seed S` writes `DIR/cases.json` and `DIR/cases/<name>.json` (golden fixture
   format, float64 truth included). The case name is the data seed (`kd<S>_<idx>` -> crc32).
2. `KDA_DIFF_DIR=DIR swift test --filter ConformanceKdaDifferentialTests` runs every case through the kernel request
   and the fallback and writes `DIR/out/<name>.json`. The suite does not judge.
3. `kda_diff.py compare --dir DIR` prints one row per case and path (`kernel`, `kernel*` = a fallback-shaped case
   that asked for the kernel, `fallback`) with max abs error, max rel error (normwise, vs max|truth|) and
   error/allowed ratio for y and state, a header line with the python/torch/numpy versions the fixtures were generated with, then the failing seed names and shapes. Exit 1 if any ratio is above 1,
   or a value is NaN, a size is wrong, an output is missing or the case raised.
4. `kda_diff.py shrink --dir DIR --case NAME --runner "tools/conformance/differential/run_local.sh --swift-only"`
   greedily shrinks a failing case (T, B, Hk, Hv, Dk, Dv, then each feature: mask, state, l2norm, dtype, decay, beta,
   scale), re-running the Swift suite per round, and writes `DIR/shrunk/<name>.json` (a golden fixture) and
   `<name>.case.txt` (a `case(...)` line to paste into `CASES` in `gen_kda_golden.py`; the name is its own seed, so
   the line reproduces the fixture). Promoting it means regenerating the golden fixtures and MANIFEST.json.

`test_kda_diff.py` self-tests the python side (compare logic, shrink with a fake runner) without Swift:
`<venv>/bin/python -I tools/conformance/differential/test_kda_diff.py`.

## Reproducing a failure

The case name is the torch seed (`crc32(name)`), but torch does not promise the same CPU random stream across versions, and some
shape draws in `plan()` are conditional on earlier draws. So a name alone is not a recipe. `cases.json` records
`env` (python, torch, numpy) and a `sha256` per fixture; `compare` prints the versions and fails a row whose
fixture file changed since `gen` (`FIXTURE CHANGED`). To reproduce, keep the whole `DIR` (or the file from `DIR/shrunk/`), or regenerate
with the same torch version.

## What is covered

T in {1, 2, 31, 32, 33, 64, 65, 127, 128}; kernel path Dk in {32, 64, 128} with Dv in {4, 8, 12, 16, 32};
fallback shapes (Dk in {1, 3, 8, 16, 24, 48, 96}, or Dv not a multiple of 4); B in 1..3; GQA (Hv = 1..3 x Hk);
float32 / float16 / bfloat16; state or none; mask none / partial / all / none-true; decay rand / mild / zero /
strong / forget; beta rand / 0 / 1 / 0.5; l2norm (float32 only) and scale.

## Tolerances and what is inferred

Tolerances are the golden manifest's classes (`Tests/.../Golden/kda/MANIFEST.json`), read at compare time and never
loosened. The class of a random case is a decision of `kda_diff.py` (item D1 in its header): float32 uses `fp32` up
to T*Dk = 896 and `fp32_long` up to 8192, half precision its own class only up to T*Dk = 896; cases beyond those
bounds are not generated, because the manifest's error derivations do not cover them. So half-precision cases reach
only T <= 27, and Dk = 128 only T <= 64. Everything about the maths that is not in vLLM (per-token mask, lower_bound
gate) is listed as I1..I6 in `kda_ref.py` and is inherited here.
