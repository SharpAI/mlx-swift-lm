# Upstream watch (SL-30)

What to look for in `ml-explore/mlx-swift-lm` after each sync, per overlay item this fork carries, and
what to do when it **LANDED**. README rule 5: add a row here in the same change that adds an overlay.

How to use it: run `upstream/sentinel.sh` (see the order of steps in the README), then
`upstream/shadow-check.sh --upstream <ml-explore checkout>`. Sentinel prints LANDED/REGRESSION for the
three registries; shadow-check prints COLLISION for symbol names. Then walk this table.

Status column: `verified` = the symbol exists in this repo today; `unverified` = no upstream tree was
available when the row was seeded, so "what to look for" is a prediction, not an observation. Update it
when you first diff against a real upstream checkout.

## Overlay symbols (name collisions break the build)

| Our overlay symbol | Upstream symbol to look for | Action on LANDED | Status |
|---|---|---|---|
| `enum SharpAIOverlay` (`Libraries/MLXLMCommon/SharpAIOverlay/`) | any declaration named `SharpAIOverlay` (shadow-check reports COLLISION) | Rename ours (e.g. `SharpAIForkOverlay`), update call sites, the lint allow-list and this table | verified |
| `SharpAIOverlay.registerIfAbsent(registry:type:creator:)` | a `registerIfAbsent` function in upstream `ModelTypeRegistry` or a registry helper | If upstream's has the same semantics (atomic, existing wins), delete ours and call upstream's; otherwise keep ours under a distinct name | verified |
| `ModelTypeRegistry.sharpAIRegisterIfAbsent` (internal extension, same file) | a member of `ModelTypeRegistry` with that exact name (vanishingly unlikely), or a change making `creators` accessible so the atomic form can move | Rename or drop together with the row above; keep `OverlayRegistrationTests.concurrentRegistrationIsAtomic` green | verified |
| `compile(shapeless:)` shims (mlx-swift#28, fixed by mlx-swift-lm#89) | upstream mlx-swift providing `compile(shapeless:)` itself | Remove the shim; shadow-check cannot see the mlx-swift package, check it by hand | unverified (history only; not present in `Libraries/`) |

## Registrations that are direct `registerModelType` calls today (allow-listed)

Counts are pinned by `OverlayRegistrationTests.allowList` and the paths by `shadow-allowlist.txt`.

| Our overlay `model_type` (registry) | Upstream `model_type` / symbol to look for | Action on LANDED | Status |
|---|---|---|---|
| `qwen3_5_text`, `qwen3_8_text`, `qwen3_5`, `qwen3_5_moe`, `qwen3_8`, `qwen3_8_moe` (MTP drafter registries: `MTPDrafterTypeRegistry.shared` / `.visionLanguage`, files `Qwen35TextMTPRegistration.swift`, `Qwen35VLMMTPRegistration.swift`) | upstream `MTPDrafterTypeRegistry` or any speculative/MTP drafter registry with these keys | Compare protocols; if upstream's registry exists, move our creators onto it via `registerIfAbsent`, then delete ours | unverified |
| `qwen3_5_mtp`, `qwen3_8_mtp` (same files, drafter model types) | `Qwen35MTP` or an `*_mtp` model type in upstream `LLMTypeRegistry`/`VLMTypeRegistry` | Delete `Models/Qwen35MTP.swift` and the registration, update the allow-list and its counts | unverified |
| `gemma4_assistant`, `gemma4_unified_assistant` (`Gemma4AssistantRegistration.swift`) | `Gemma4Assistant` / `gemma4_assistant` in upstream | Delete `Models/Gemma4Assistant.swift` and the registration, update the allow-list | unverified |

## Model types the survey lists as missing or partial

| Our status | Upstream `model_type` to look for | Action on LANDED | Status |
|---|---|---|---|
| `missing` / `partial` rows of `model-coverage.csv` (147 / 59 at seed time) | the CSV's `model_type` column appearing as a new key in `LLMTypeRegistry`, `VLMTypeRegistry` or `EmbedderTypeRegistry` | The sentinel's `LANDED` line names it: run `gen_expectations.py`, update the CSV row (`our_status`), delete any local overlay for that type | verified (mechanism); individual rows are found by the sentinel, not listed here |

## When a row fires

1. Reproduce: sentinel (`CONFORMANCE_STRICT=1`) and shadow-check on the same upstream commit.
2. Prefer upstream. Delete our code only after the tests that covered it pass against upstream's.
3. Remove or update the row here, the allow-list and `OverlayRegistrationTests.allowList` in the same change.
