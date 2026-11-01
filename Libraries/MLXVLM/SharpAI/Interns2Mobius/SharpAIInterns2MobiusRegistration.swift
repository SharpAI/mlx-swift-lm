// SharpAI fork overlay: interns2_mobius creator. Not part of upstream mlx-swift-lm.
//
// The creator decodes and validates the config, then builds the text model
// (SharpAIInterns2MobiusModel in MLXLLM). VLMModelFactory is not edited; the overlay consumer
// merges these creators into its own registry.
//
// Known gaps (not implemented):
// - Vision tower and image/video processor (model.visual.*, vision_config). The HF modeling file
//   defines InternS2MobiusVisionModel; it has not been ported.
// - MTP drafter head (mtp.* keys, mtp_num_experts). Neither the HF modeling file nor the SGLang
//   model file implements it. See the registration test for details.

import Foundation
import MLXLLM
import MLXLMCommon

public enum SharpAIInterns2MobiusOverlay {

    /// Creators keyed by `model_type`, same closure shape as `VLMTypeRegistry`.
    nonisolated(unsafe) public static let creators: [String: (Data) throws -> any LanguageModel] = [
        "interns2_mobius": { data in
            let configuration = try JSONDecoder.json5().decode(
                SharpAIInterns2MobiusConfiguration.self, from: data)
            try configuration.validateModelConfiguration()
            return SharpAIInterns2MobiusModel(configuration.textConfiguration)
        }
    ]
}
