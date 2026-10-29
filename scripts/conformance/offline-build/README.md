# Sealed (network-blocked) build and test

SL-28 step A4, part of the conformance harness (SharpAI/SwiftLM#254). Question answered: does a CLEAN
`swift build --build-tests` plus the offline conformance tests work with outbound network blocked?
Answer on macOS (see `run-blocked.sh`): yes, provided the SwiftPM cache is warm.

## What SwiftPM needs offline

The dependency graph is small: `../mlx-swift` (local path), `swift-numerics` 1.1.1 (via mlx-swift) and
`swift-syntax` 603.0.2 (macros; `swift-syntax` also has signed prebuilts that SwiftPM caches under
`~/Library/Caches/org.swift.swiftpm/prebuilts`). Only these two URLs are remote:

* `https://github.com/apple/swift-numerics`
* `https://github.com/swiftlang/swift-syntax.git`

Offline, SwiftPM resolves both **from its repository cache** (`~/Library/Caches/org.swift.swiftpm/repositories`,
log line `Fetching ... from cache`). With an empty cache (`--cache-path <empty>`) the same command fails on exactly
those two URLs (observed). `Package.resolved` is git-ignored and not needed: resolution from the cache picks the
same versions. No weights, tokenizers or HF access are involved: the tests use `scripts/conformance/offline/`
config fixtures and synthetic weights.

What would vendor them, if the cache cannot be guaranteed: commit-pinned bare clones restored into the SwiftPM
cache, or a SwiftPM mirror (`swift package config set-mirror`) pointing at an internal copy. Not needed locally.

## run-blocked.sh

`scripts/conformance/offline-build/run-blocked.sh [--self-test-only]` prints a final `EXIT=<n>`.

0. Self-test (so a pass is not vacuous): a local TCP listener is reachable outside `sandbox-exec -p
   '(version 1)(allow default)(deny network*)'` and unreachable inside; `curl https://1.1.1.1` and `git ls-remote`
   of the SwiftPM remote fail inside. Any miss gives `EXIT=70` and nothing else runs.
1. `swift package resolve` into a fresh scratch dir, inside the sandbox; fails (naming the URL) if a dependency
   is not in the cache; also fails (71) if a line says Fetching without `from cache`.
2. `swift build --build-tests` into the fresh scratch dir (default `$TMPDIR/swiftlm-blocked-build`), inside the
   sandbox. `--disable-sandbox` is required because SwiftPM's own sandbox cannot nest.
3. `swift test --skip-build --filter <f>` per filter (default `ConformanceOfflineGuard`,
   `ConformanceOfflineBundleTests`) inside the sandbox with `SWIFTLM_OFFLINE=1 HF_HUB_OFFLINE=1`.

`LOCK_BUILD` / `LOCK_GPU` env vars take an optional command prefix (a lock wrapper) for shared machines.

## Design of the future sealed CI leg (no workflow is created here)

A macOS runner job, run after the normal online job so the SwiftPM cache is already warm:

1. Checkout; make `../mlx-swift` available (same step as the existing legs).
2. **Prefetch (network allowed):** `swift package resolve` (fills the cache). Cache `~/Library/Caches/org.swift.swiftpm`
   between runs if the runner is not persistent.
3. **Seal (network denied):** `scripts/conformance/offline-build/run-blocked.sh`. The job passes only on
   `EXIT=0`; the self-test inside it guards against the seal being a no-op (for example a runner image where
   `sandbox-exec` is deprecated or a profile that silently allows network).
4. Gate: the step fails the leg on any non-zero exit. A resolve failure there means a new remote dependency
   appeared: add it to the prefetch step on purpose, do not weaken the seal.

Limits: sandbox-exec is macOS only and officially deprecated (still functional on this host); Linux legs would
need `unshare -n` or a network-less container. The seal covers build and the named offline tests, not tests
that need Metal weights from the network (those are `.directory`-sourced or synthetic by design).
