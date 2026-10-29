# Offline config bundle

Config-only fixtures (`config.json` of public Hugging Face repos) plus `MANIFEST.json`
(`path`, `model_type`, `repo`, `repo_source`, `sha256`, `size`, `architectures`). **No weights, tokenizers or
processors.** Part of the conformance harness (SharpAI/SwiftLM#254).

* `fixtures/<model_type>/<org>__<name>.json` -- raw bytes as fetched (`_none` = the file has no top-level
  `model_type`, e.g. nested-only configs). Treat them as untrusted data.
* `MANIFEST.json` -- `unfetched` lists survey repos with no config here (gated, restricted, not public, or never
  fetched: the survey could not read gated repos, so **gated models' configs are not in the bundle**);
  `skipped` lists input files that were rejected (error pages, no repo mapping).
* Rebuild (no network, reads local dirs only):
  `python3 -I scripts/conformance/offline/make_bundle.py --src <dir-of-fetched-json> [--src ...]`
* Loading from Swift: `ConformanceBundle` in `Tests/MLXLMTests/Conformance/ConformanceOfflineBundleTests.swift`
  finds this directory through `#filePath`. It lives outside every SwiftPM target, so `Package.swift` is not touched
  and SwiftPM reports no unhandled files.

## Offline switch (SL-28 A1)

`SWIFTLM_OFFLINE=1` (or `HF_HUB_OFFLINE=1`; accepted truthy values: 1/true/yes/on) makes every `Downloader.download`
call issued by mlx-swift-lm throw `SharpAIOverlay.OfflineError.networkBlocked(url)` before any I/O. Unset = identity
(no behaviour change). Code: `Libraries/MLXLMCommon/SharpAIOverlay/OfflineGuard.swift`; hooks (one line each):
`resolve` (ModelFactory.swift), `resolveForModelConversion` (ModelConversion.swift), `ModelAdapterFactory.load`.
Not covered: URLSession in BenchmarkHelpers / IntegrationTestHelpers, and the internals of external packages
(swift-huggingface, swift-transformers) when a caller supplies its own non-`Downloader` path. Cache hits in a
Downloader are blocked too; offline runs use `.directory` sources. Tests: `ConformanceOfflineGuardTests`.
