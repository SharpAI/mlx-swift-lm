// SharpAI fork overlay aggregate for the MLXLLM target. Not part of upstream mlx-swift-lm.
//
// Collects the SL-47 per-model overlays that are wired into `LLMTypeRegistry.shared`.
// Policy (owner decision, SL-50): UPSTREAM WINS. `LLMModelFactory` merges `SharpAIModelOverlays.llm`
// with a closure that keeps the upstream creator for any `model_type` the upstream-owned literal
// already registers, so a future upstream implementation replaces ours automatically.
// This is the opposite of the SL-41 `deepseek_v4` hook, which stays ours-wins and is merged last.
//
// Only overlays whose model can be created, loaded through `sanitize(weights:)` and used for
// generation are listed in `llmWiredKeys`. Left out on purpose (not registered, so
// `ModelTypeRegistry.contains` stays false for them):
//   - creators that always throw (dots3_note_mtp, k3_dspark, kimi_k3_mtp, minimax_m2,
//     minimax_m3_dspark, minimax_m3_mtp, muse_spark_dspark, sarvam_mla, unknown, params.json)
//   - models that build but cannot generate (inkling_model, inkling_mm_model, pangu_ultra_moe_mtp)
//   - conditional or experimental ones (beam, step5)
//   - encoder-style models that the generation path does not serve (modernbert, llama_bidirec,
//     deepseek_v3_bidirec, openai_privacy_filter, jina_embeddings_v5)
// Keys are listed one per line: scripts/conformance/gen_expectations.py and
// scripts/conformance/missing-ops/gen_report.py read them from here.

import Foundation
import MLXLMCommon

public enum SharpAIModelOverlays {

    /// Merge closure for `Dictionary.merging(_:uniquingKeysWith:)`: when both sides have a key, keep the
    /// upstream value. `LLMModelFactory` uses this one function so a test can pin the policy.
    public static func upstreamWins<T>(_ upstream: T, _ ours: T) -> T { upstream }

    /// `model_type` keys registered in `LLMTypeRegistry.shared` through this aggregate.
    public static let llmWiredKeys: Set<String> = [
        "PanguEmbedded",
        "alice_ai",
        "axk1",
        "bailing_hybrid",
        "bailing_moe_v3",
        "cohere2_moe",
        "deepseek_v32",
        "domynedge",
        "dory",
        "exaone_moe",
        "flint",
        "granite_swa",
        "granitemoe_swa",
        "hrm_text",
        "hy_v3",
        "hy_v4",
        "hyperclovax",
        "iquestcoder",
        "iquestloopcoder",
        "jais2",
        "k2_horizon",
        "kolibri1",
        "laguna",
        "longcat_flash",
        "mimo_v2",
        "minimax_m3_text",
        "muse_glimmer_text",
        "nemotron_h_mtp",
        "olmo_hybrid",
        "pangu_ultra_moe",
        "param2moe",
        "plamo3",
        "qwen3_5_moe_text",
        "sarvam_moe",
        "step3p5",
        "telechat3",
        "vaultgemma",
    ]

    /// Creators for `llmWiredKeys`, same closure shape as `LLMTypeRegistry`.
    nonisolated(unsafe) public static let llm: [String: (Data) throws -> any LanguageModel] = {
        var all: [String: (Data) throws -> any LanguageModel] = [:]
        func add(_ source: [String: (Data) throws -> any LanguageModel]) {
            for (key, creator) in source where llmWiredKeys.contains(key) { all[key] = creator }
        }
            add(SharpAIAliceAiOverlay.creators)
            add(SharpAIAxk1Overlay.creators)
            add(SharpAIBailingHybridOverlay.creators)
            add(SharpAIBailingMoeV3Overlay.creators)
            add(SharpAICohere2MoeOverlay.creators)
            add(SharpAIDeepseekV32Overlay.creators)
            add(SharpAIDomynedgeOverlay.creators)
            add(SharpAIDoryOverlay.creators)
            add(SharpAIExaoneMoeOverlay.creators)
            add(SharpAIFlintOverlay.creators)
            add(SharpAIGraniteSwaOverlay.creators)
            add(SharpAIGranitemoeSwaOverlay.creators)
            add(SharpAIHrmTextOverlay.creators)
            add(SharpAIHyV3Overlay.creators)
            add(SharpAIHyV4Overlay.creators)
            add(SharpAIHyperclovaxOverlay.creators)
            add(SharpAIIquestcoderOverlay.creators)
            add(SharpAIIquestloopcoderOverlay.creators)
            add(SharpAIJais2Overlay.creators)
            add(SharpAIK2HorizonOverlay.creators)
            add(SharpAIKolibri1Overlay.creators)
            add(SharpAILagunaOverlay.creators)
            add(SharpAILongcatFlashOverlay.creators)
            // MimoV2 declares its creators `@Sendable`. Passing that dictionary straight to `add` would
            // make Swift cast it lazily, and the cast traps at run time ("Could not cast value of type
            // '@Sendable ...'"), so each closure is re-wrapped.
            add(SharpAIMimoV2Overlay.creators.mapValues { creator in { data in try creator(data) } })
            add(SharpAIMinimaxM3TextOverlay.creators)
            add(SharpAIMuseGlimmerTextOverlay.creators)
            add(SharpAINemotronHMtpOverlay.creators)
            add(SharpAIOlmoHybridOverlay.creators)
            add(SharpAIPanguUltraMoeOverlay.creators)
            add(SharpAIPanguembeddedOverlay.creators)
            add(SharpAIParam2moeOverlay.creators)
            add(SharpAIPlamo3Overlay.creators)
            add(SharpAIQwen35MoeTextOverlay.creators)
            add(SharpAISarvamMoeOverlay.creators)
            add(SharpAIStep3p5Overlay.creators)
            add(SharpAITelechat3Overlay.creators)
            add(SharpAIVaultgemmaOverlay.creators)
        return all
    }()
}
