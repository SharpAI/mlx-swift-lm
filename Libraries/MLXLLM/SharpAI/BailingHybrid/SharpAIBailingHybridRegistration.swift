// SharpAI fork overlay: bailing_hybrid creator. Not part of upstream mlx-swift-lm.
//
// Decodes and validates the config, then builds SharpAIBailingHybridModel. Configs that lack the
// MLA or grouping fields fail with ModelFactoryError.invalidConfiguration before any weights are built.
// Wiring into LLMTypeRegistry is left to the overlay consumer; LLMModelFactory is not edited.

import Foundation
import MLXLMCommon

public enum SharpAIBailingHybridOverlay {

    /// This overlay implements the Ling/Ring 2.5 lightning + MLA layout only. The Ling 3.x KDA variant
    /// (`BailingMoeV3ForCausalLM`, `linear_silu`, `up_proj_norm`, ...) and FP8 checkpoints would build
    /// without error and then fail late or give wrong logits, so they are refused up front.
    static func rejectUnsupportedVariants(_ data: Data) throws {
        guard let raw = try JSONSerialization.jsonObject(with: data) as? [String: Any] else { return }
        let architectures = raw["architectures"] as? [String] ?? []
        if architectures.contains("BailingMoeV3ForCausalLM") || (raw["linear_silu"] as? Bool) == true
            || raw["up_proj_norm"] != nil
        {
            throw ModelFactoryError.unsupportedModelType(
                "bailing_hybrid: the Ling 3.x KDA variant is not implemented; only Ling/Ring 2.5 is")
        }
        if raw["quantization_config"] != nil {
            throw ModelFactoryError.unsupportedModelType(
                "bailing_hybrid: checkpoints with quantization_config (for example FP8) are not supported")
        }
    }

    /// Creators keyed by `model_type`, same closure shape as `LLMTypeRegistry`.
    /// Not wired into LLMTypeRegistry by this overlay.
    nonisolated(unsafe) public static let creators: [String: (Data) throws -> any LanguageModel] = [
        "bailing_hybrid": { data in
            try rejectUnsupportedVariants(data)
            let configuration = try JSONDecoder.json5().decode(
                SharpAIBailingHybridConfiguration.self, from: data)
            try configuration.validateModelConfiguration()
            return try SharpAIBailingHybridModel(configuration)
        }
    ]
}
