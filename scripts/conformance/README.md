# Conformance harness (air-gapped model coverage)

Config-only tests that track which architectures this fork can run, without downloading any
weights. Tracking issue and survey method: SharpAI/SwiftLM#254.

## What is in here

| File | Role |
|---|---|
| `model-coverage.csv` | Snapshot of the static vLLM-vs-SharpAI survey (225 rows). Input only. |
| `gen_expectations.py` | Greps the three type registries and writes `Tests/MLXLMTests/Conformance/ConformanceExpectations.swift`. |
| `Tests/MLXLMTests/Conformance/ConformanceRegistryTests.swift` | Tier 0: committed expectations == real registries (no GPU, no weights). |
| `Tests/MLXLMTests/Conformance/ConformanceForwardTests.swift` | Tiers 1-2: build from config JSON with random weights, forward, check numeric properties. |
| `Tests/MLXLMTests/Conformance/ConformanceSynthWeightsTests.swift` | Synthetic checkpoint (random weights saved/loaded as safetensors) + `sanitize` key parity for llama / qwen3_moe / gemma3_text. |
| `Tests/MLXLMTests/Conformance/ConformanceOfflineBundleTests.swift` | Loads `offline/` config fixtures (manifest consistency + one decode through the registry). |
| `offline/` (`make_bundle.py`, `MANIFEST.json`, `fixtures/`) | 169 config-only `config.json` files of the surveyed repos (no weights). Rebuild: `make_bundle.py --src DIR --csv-only`. |
| `Tests/MLXLMTests/Conformance/ConformanceExtrapolation*.swift`, `extrapolate/README.md` | Slope extrapolation of step time from reduced-layer random-weight models (dense + MoE). Maths is unit-tested; timing outcomes are report-only. |
| `run-test.sh`, `test-scripts.sh` | Run one named test filter and print the process `EXIT=`; self-test of these scripts. |
| `upstream/sentinel.sh` | After an upstream sync: regenerate expectations, run the conformance tests with `CONFORMANCE_STRICT=1`, print a report. |
| `upstream/shadow-check.sh` (+ `shadow-allowlist.txt`, `test-shadow-check.sh`) | `--upstream DIR`: fails on a COLLISION between our `SharpAIOverlay`/sharpai-prefixed declarations and an upstream declaration, and on a direct `registerModelType(` call outside the allow-list. Self-test: `test-shadow-check.sh`. |
| `upstream/UPSTREAM_WATCH.md` | Per overlay item: upstream symbol / `model_type` to look for, and the action when it LANDED. |
| `Tests/MLXLMTests/Conformance/OverlayRegistrationTests.swift` | `SharpAIOverlay.registerIfAbsent` (atomic, existing registration wins) + lint of direct `registerModelType` calls in `Libraries/`. |

## Reading a failure

* `REGRESSION` -- a `model_type` that used to be registered is gone. A real bug.
* `LANDED` -- a registry holds a `model_type` the expectations do not list. After an **upstream sync**
  this means upstream implemented it: re-run `gen_expectations.py`, and delete any local overlay for it.
  By default this is only a printed warning (`[conformance][LANDED]`), so a correct upstream sync is
  never blocked by the safety net. Run with `CONFORMANCE_STRICT=1` (the upstream sentinel does) to make
  it fail.
* A `withKnownIssue` test going red with "known issue not recorded" means the gap was fixed:
  remove the wrapper in the same change. A fix always lands together with the test that proves it.

## Rules for adding ops / models (so upstream syncs stay painless)

1. **Red first.** Add a failing case (or a `withKnownIssue` wrapper) that demonstrates the gap, then fix.
2. **New files only.** Put new models/ops in new files under their own directory; do not edit
   upstream-owned files except for a minimal hook.
3. **Namespaced types.** Never reuse an upstream type name (`enum SharpAIOverlay { ... }`). A same-named
   public type or function breaks the build when upstream adds it -- this is exactly how the
   `compile(shapeless:)` shims broke mlx-swift#28 (fixed by mlx-swift-lm#89).
4. **Register if absent.** Overlay registration must check `ModelTypeRegistry.contains(_:)` first, so
   an upstream registration wins automatically.
5. **Upstream watch.** Record, next to each overlay, the upstream symbol / `model_type` to look for;
   the periodic upstream check runs this harness after every sync.

## After an upstream sync: order of steps

1. **Diff the harness before merging upstream.** Before running the merge, look at what the sync changes in
   the harness's own inputs, so a red result is attributable: `git diff HEAD...upstream/main --stat -- Libraries Tests/MLXLMTests/Conformance scripts/conformance` (look at the three `*TypeRegistry.swift` files first)
   (and `git diff HEAD...upstream/main -- Package.swift`). Also run
   `upstream/shadow-check.sh --upstream <upstream checkout>`: a COLLISION must be fixed (rename ours) before
   the merge can build at all. `sentinel.sh` does the merge itself on a throwaway branch and never
   edits the harness, so this diff is the only place the harness-side changes are reviewed by a person.
2. Run `upstream/sentinel.sh` and read `LANDED` / `REGRESSION` / conflicts.
3. Walk `upstream/UPSTREAM_WATCH.md` for every LANDED name.

## Regenerating

```bash
python3 scripts/conformance/gen_expectations.py   # then: swift test --filter Conformance
```
