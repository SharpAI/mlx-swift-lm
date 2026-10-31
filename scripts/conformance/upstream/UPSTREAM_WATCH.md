# Upstream watch (SL-30)

What to look for in `ml-explore/mlx-swift-lm` after each sync, per overlay item this fork carries, and
what to do when it **LANDED**. README rule 5: add a row here in the same change that adds an overlay.

How to use it: run `upstream/sentinel.sh` (see the order of steps in the README), then
`upstream/shadow-check.sh --upstream <ml-explore checkout>` (the sentinel now runs it itself, before the merge) and
`upstream/mlx-swift-shadow.sh --mlx-swift <ml-explore/mlx-swift checkout>`. Sentinel prints LANDED/REGRESSION for the
three registries; shadow-check prints COLLISION for symbol names. Or run all of it with `upstream/run-periodic.sh`. Then walk this table.

Status column: `verified@<sha>` = checked by grep against a real upstream tree at that commit (the sha is the one
that was checked, so the claim can be re-run); `unverified` = no upstream tree was available. Last full pass:
**2026-10-29**, `ml-explore/mlx-swift-lm` `f5b0677f6b52b81d068871611cfcb82d1e7a0005` and `ml-explore/mlx-swift`
`78d3cc1d60617205d30da2c50db3c57f019cab88` (scratch clones, read-only fetch). Re-run with
`upstream/run-periodic.sh`; edit a row only with the sha you checked.

## Overlay symbols (name collisions break the build)

| Our overlay symbol | Upstream symbol to look for | Action on LANDED | Status |
|---|---|---|---|
| `enum SharpAIOverlay` (`Libraries/MLXLMCommon/SharpAIOverlay/`) | any declaration named `SharpAIOverlay` (shadow-check reports COLLISION) | Rename ours (e.g. `SharpAIForkOverlay`), update call sites, the lint allow-list and this table | verified@f5b0677 (no `SharpAIOverlay` declaration in upstream Libraries; shadow-check PASS) |
| `SharpAIOverlay.registerIfAbsent(registry:type:creator:)` | a `registerIfAbsent` function in upstream `ModelTypeRegistry` or a registry helper | If upstream's has the same semantics (atomic, existing wins), delete ours and call upstream's; otherwise keep ours under a distinct name | verified@f5b0677 (`grep -r registerIfAbsent` in upstream Libraries and Tests: no hit; upstream `ModelTypeRegistry` has only `registerModelType` x2, `createModel`, `contains`) |
| `ModelTypeRegistry.sharpAIRegisterIfAbsent` (internal extension, same file) | a member of `ModelTypeRegistry` with that exact name (vanishingly unlikely), or a change making `creators` accessible so the atomic form can move | Rename or drop together with the row above; keep `OverlayRegistrationTests.concurrentRegistrationIsAtomic` green | verified@f5b0677 (name absent upstream; `creators` is still `private var` in upstream `ModelTypeRegistry.swift`, so the atomic form still cannot move) |
| `compile(shapeless:)` shims (mlx-swift#28, fixed by mlx-swift-lm#89): `Libraries/MLXLMCommon/CompileOverloads.swift`, public `compile(shapeless:_:)` x N | upstream mlx-swift providing `compile(shapeless:_ @Sendable ...)` itself | **LANDED upstream.** Delete `CompileOverloads.swift` once the `SharpAI/mlx-swift` fork has synced past mlx-swift `78d3cc1d`. `CompileOverloads.swift` is already absent here and its `COLLISION`/`OVERLAP` allow-list entries were removed (SL-34); `mlx-swift-shadow.sh` still passes | verified@mlx-swift 78d3cc1d: `Source/MLX/Transforms+Compile.swift` declares `compile(shapeless:_:)` with `@Sendable` bodies and `compile(inputs:outputs:shapeless:_:)`; `ml-explore/mlx-swift-lm` f5b0677 has no `CompileOverloads.swift`. Correction: the earlier note "not present in `Libraries/`" was wrong, the file is there. Checked mechanically by `upstream/mlx-swift-shadow.sh` |
| `MLXArray.l2Normalized(axis:eps:)` in `Libraries/MLXEmbedders/Extensions/MLXArray+Helper.swift` (an upstream-lm file, not ours) | the same member in mlx-swift `Source/MLX/MLXArray+Normalizer.swift` | Not ours. Both upstreams declare it, so watch for ambiguous `.l2Normalized()` calls in MLXEmbedders when the mlx-swift pin advances; report to ml-explore if it breaks. Allow-listed in `mlx-swift-shadow-allowlist.txt` | verified@mlx-swift 78d3cc1d and lm f5b0677 (found by `mlx-swift-shadow.sh`) |

## DeepseekV4 (SL-41): ours wins over a future upstream V4

Policy (owner, SL-41): when upstream adds DeepseekV4, **KEEP OURS**. The fork-only V4 code is isolated so that
this is a decision, not a build break:

* Files: `Libraries/MLXLLM/SharpAI/SharpAIDeepseekV4{,Compressor,CompressorState,Indexer,Registration}.swift`.
* Types and module-level helpers carry the `SharpAI` prefix (`SharpAIDeepseekV4Model`, `...Configuration`,
  `...Block`, `...ModelInner`, `...Attention`, `...Expert`, `...Gate`, `...MoE`, `SharpAIHCParams`,
  `sharpAIHCSplitSinkhorn`, `sharpAIHCPre`, `sharpAIHCPost`, `sharpAIHCHead`), so an upstream `DeepseekV4*` type or
  `DeepseekV4.swift` file cannot collide with them. `shadow-check.sh` covers them through its sharpai-prefix rule.
* Registration: the `"deepseek_v4"` literal entry is gone from `LLMModelFactory.swift`. The one hook left in that
  upstream-owned file is the last line of the `LLMTypeRegistry.shared` literal:
  `].merging(SharpAIDeepseekV4Overlay.creators) { _, ours in ours })`. The merge closure is the explicit override:
  if upstream's literal ever has a `"deepseek_v4"` key, ours replaces it. Registration is synchronous, so no
  bootstrap call is needed by any caller.
* External names are unchanged: model_type `deepseek_v4`, HF architectures `DeepseekV4ForCausalLM` /
  `DeepseekV4ForConditionalGeneration`.

| Our symbol | Upstream symbol to look for | Action on LANDED | Status |
|---|---|---|---|
| `SharpAIDeepseekV4*` types, `SharpAIDeepseekV4Overlay` | an upstream `deepseek_v4` key in `LLMTypeRegistry`, a `DeepseekV4*` type, or `Libraries/MLXLLM/Models/DeepseekV4*.swift` | **Keep ours.** Nothing to delete. The merge closure keeps our creator on the key; the upstream model is simply unused. Review the upstream implementation for fixes worth porting, and record the decision in this row. The sentinel should flag the appearance (SL-42 implements that check; until then the `registry-diff.py` LANDED/`deepseek_v4` line and a manual `git grep -i deepseekv4` on the upstream tree are the signal). | verified@128f49f (no V4 in upstream/main Libraries) |
| The merge hook line at the end of the `LLMTypeRegistry.shared` literal in `Libraries/MLXLLM/LLMModelFactory.swift` | upstream reshaping the literal or `ModelTypeRegistry.init(creators:)` | Re-apply the hook on the new shape; `SharpAIV4OverlayTests` fails if it is lost (`deepseek_v4` stops resolving to `SharpAIDeepseekV4Model`). | in force |

## Registrations that are direct `registerModelType` calls today (allow-listed)

Counts are pinned by `OverlayRegistrationTests.allowList` and the paths by `shadow-allowlist.txt`.

| Registration (registry) | Upstream state | Action | Status |
|---|---|---|---|
| `qwen3_5_text`, `qwen3_8_text`, `qwen3_5`, `qwen3_5_moe`, `qwen3_8`, `qwen3_8_moe` into `MTPDrafterTypeRegistry.shared` / `.visionLanguage` (`Qwen35TextMTPRegistration.swift`, `Qwen35VLMMTPRegistration.swift`) | **Already upstream.** `ml-explore/mlx-swift-lm` has `enum MTPDrafterTypeRegistry` (`Libraries/MLXLMCommon/MTPDrafterModelFactory.swift`) and both registration files at the same paths with the same keys. These are upstream-owned files, allow-listed because upstream itself calls `registerModelType` directly, not overlays of ours | Nothing to delete. Keep the allow-list entries so shadow-check does not flag upstream's own calls; do not edit these files (merge conflicts) | verified@f5b0677 |
| `qwen3_5_mtp`, `qwen3_8_mtp` (same two files; drafter model types) | Already upstream: `Libraries/MLXLLM/Models/Qwen35MTP.swift` (`Qwen35MTPDraftModel`) and the registrations exist upstream | As above | verified@f5b0677 |
| `gemma4_assistant`, `gemma4_unified_assistant` (`Libraries/MLXVLM/Gemma4AssistantRegistration.swift`) | Already upstream: `Gemma4AssistantRegistration.swift` and `Models/Gemma4Assistant.swift` exist upstream and register into `MTPDrafterTypeRegistry.shared`. Note `gen_expectations.py` lists `gemma4_assistant` as registered in ours but not in upstream's three type registries because upstream keeps it in the drafter registry, not LLM/VLM | As above | verified@f5b0677 |
| `Libraries/MLXLMCommon/Registries/ModelTypeRegistry.swift` (allow-listed: defines `registerModelType`) | upstream-owned | none | verified@f5b0677 |

## Model types the survey lists as missing or partial

| Our status | Upstream `model_type` to look for | Action on LANDED | Status |
|---|---|---|---|
| `missing` / `partial` rows of `model-coverage.csv` (147 / 59 at seed time) | the CSV's `model_type` column appearing as a new key in `LLMTypeRegistry`, `VLMTypeRegistry` or `EmbedderTypeRegistry` | The sentinel's `LANDED` line names it: run `gen_expectations.py`, update the CSV row (`our_status`), delete any local overlay for that type | verified@f5b0677 (mechanism); see the findings below |

Findings from the 2026-10-29 pass (upstream f5b0677 registries via `gen_expectations.py` on a copy of the upstream tree, against this branch's expectations):

* Upstream registers `nemotron_h_puzzle` (`LLMModelFactory.swift`, `Models/NemotronH.swift`); this branch does not, and its CSV row says `partial, only on unmerged branch`. It will show as `LANDED nemotron_h_puzzle -> .llm` once the upstream merge is real. Update the CSV row then.
* Registered here but not in upstream's three registries: `deepseek_v4`, `glm_moe_dsa` (ours, no upstream equivalent at f5b0677; since SL-41 registered by the overlay `SharpAIDeepseekV4Overlay`, ours wins if upstream adds it, see the DeepseekV4 section) and `gemma4_assistant` (upstream keeps it in the drafter registry, see above). These are not regressions; a sentinel `REGRESSION` for them would mean ours disappeared.
* CSV rows `partial`/`missing` whose `model_type` upstream already registers (the key exists, the gap is feature-level): `bert`, `xlm-roberta`, `roberta`, `qwen3_5`, `qwen3`, `qwen2`, `lfm2`, `lfm2_vl`, `mistral3`, `embedding_gemma2`, `gemma4_text`, `gemma3_text`, `glm_ocr`, `glm4_moe`, `mellum`, `mimo_v2_flash`, `nemotron_h_puzzle`. `glm4_moe_lite` has a `missing` row (`Glm4MoeLiteMTPModel`, the MTP head) while the base model is `supported`, so a registry key does not flip that row. Registry presence only proves the base type; read `our_gaps` before changing a status.

## When a row fires

1. Reproduce: sentinel (`CONFORMANCE_STRICT=1`) and shadow-check on the same upstream commit.
2. Prefer upstream. Delete our code only after the tests that covered it pass against upstream's.
3. Remove or update the row here, the allow-list and `OverlayRegistrationTests.allowList` in the same change.
