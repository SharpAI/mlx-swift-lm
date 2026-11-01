// SharpAI fork overlay: deepseek_vl_v2 creator. Not part of upstream mlx-swift-lm.
//
// The model class is not written yet: the SigLIP vision tower, MlpProjector and non-MLA text
// attention (kv_lora_rank null) are missing. Reference: deepseek-ai/DeepSeek-VL2. This creator decodes and validates the config, then fails.
// Wiring into VLMTypeRegistry is left to the overlay consumer; VLMModelFactory is not edited.

import Foundation
import MLXLMCommon

public enum SharpAIDeepseekVlV2Overlay {

    /// Creators keyed by `model_type`, same closure shape as `VLMTypeRegistry`.
    nonisolated(unsafe) public static let creators: [String: (Data) throws -> any LanguageModel] = [
        "deepseek_vl_v2": { data in
            let configuration = try JSONDecoder.json5().decode(
                SharpAIDeepseekVlV2Configuration.self, from: data)
            try configuration.validateModelConfiguration()
            throw SharpAIDeepseekVlV2Error.forwardNotImplemented(configuration.modelType)
        }
    ]
}
