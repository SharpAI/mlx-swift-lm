// SharpAI fork overlay: deepseek_v32 creator. Not part of upstream mlx-swift-lm.
//
// The upstream registry has only "deepseek_v3_2" and "glm_moe_dsa". This adds the
// "deepseek_v32" key used by the spec. LLMModelFactory is not edited; the overlay consumer
// merges these creators into its own registry.

import Foundation
import MLXLMCommon

public enum SharpAIDeepseekV32Overlay {

    /// Creators keyed by `model_type`, same closure shape as `LLMTypeRegistry`.
    nonisolated(unsafe) public static let creators: [String: (Data) throws -> any LanguageModel] = [
        "deepseek_v32": { data in
            try SharpAIDeepseekV32ModelOptProbe.rejectIfModelOpt(data)
            let configuration = try JSONDecoder.json5().decode(
                DeepseekV32Configuration.self, from: data)
            return SharpAIDeepseekV32Model(configuration)
        }
    ]
}
