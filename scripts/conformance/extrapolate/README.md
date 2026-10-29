# Slope extrapolation of speed (SL-29 phase 3)

Air-gapped method: estimate how a model's step time scales with depth from tiny random-weight models, then
extrapolate to the real layer count read from a config-only fixture. No weights, no network.

## Method

1. Build the model from config JSON (public `LLMTypeRegistry`) at N1 < N2 < N3 layers (2, 4, 8) with
   shrunk hidden / expert sizes so it runs in milliseconds.
2. Time a **prefill** step (64 tokens, fresh cache) and a **decode** step (1 token on a prefilled
   cache). 3 warm-up iterations, then the median of 9.
3. Least-squares fit `time = a + b * layers` (`SharpAIOverlayExtrapolation.fit`), evaluate at the real
   layer count (`extrapolate`).

Code: `Tests/MLXLMTests/Conformance/ConformanceExtrapolationMath.swift` (pure maths, unit-tested with
hand-made numbers in `ConformanceExtrapolationMathTests`) and `ConformanceExtrapolationTimingTests.swift`
(dense `qwen3` -> Qwen3-8B fixture, 36 layers; MoE `qwen3_moe` -> Ovis2.6-30B-A3B `llm_config`, 48 layers).

## What is asserted, what is only reported

Hard (deterministic): fit maths; models really have the requested layer counts; the timed steps advanced
the KV cache; times are finite and positive; the fixture layer count was found.

Report-only (printed as `[extrapolation] ...`, never fails): R^2, monotonicity, slope, the extrapolated
milliseconds. Wall-clock depends on machine load, so asserting on it would be flaky. Read the printed R^2
(a clean run is >= 0.95) before trusting a number; a low R^2 or a non-monotonic line means a loaded machine,
re-run.

## What it CAN claim

* The per-layer cost trend: the slope `b` and whether step time grows linearly with depth for this
  architecture's code path (no accidental super-linear behaviour, e.g. a cache that copies per layer).
* A relative comparison between two architectures or two code revisions measured the same way
  (slope ratio), on this machine.
* That the build -> forward -> cache path works at several depths from config alone.

## What it CANNOT claim

* Absolute speed. The shrunk models are dispatch/launch-bound, not compute-bound; the fit's intercept can
  even come out negative. Two runs of the same test differ by tens of percent in the extrapolated number
  (observed: 75 ms vs 94 ms for dense prefill).
* Output accuracy or quality: weights are random.
* Memory-bandwidth effects of real weights: real decode is dominated by reading GBs of weights per token
  (cache misses, unified-memory pressure, quantised kernels, expert paging with `--stream-experts`), none
  of which exist in a few-MB model. Do not read the extrapolation as tokens/s.
* Width effects: only depth is varied; hidden size, head count and expert count are shrunk, so per-layer
  cost is not the real per-layer cost.
