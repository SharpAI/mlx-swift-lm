// SharpAI fork overlay: pangu_ultra_moe_mtp creator. Not part of upstream mlx-swift-lm.
//
// Same closure shape as LLMTypeRegistry: decode, validate, construct. LLMModelFactory is not
// edited. A consumer merges these creators into its registry.
// The construct step builds SharpAIPanguUltraMoeMtpModel. Its MTP block is the base
// pangu_ultra_moe decoder layer from SharpAI/PanguUltraMoe, so this overlay depends on that
// overlay's configuration and decoder layer types. The model is a drafter: its LanguageModel
// entry points trap, and callers use draftLogits.

import Foundation
import MLXLMCommon

public enum SharpAIPanguUltraMoeMtpOverlay {

    /// Creators keyed by `model_type`.
    nonisolated(unsafe) public static let creators: [String: (Data) throws -> any LanguageModel] = [
        "pangu_ultra_moe_mtp": { data in
            let configuration = try JSONDecoder.json5().decode(
                SharpAIPanguUltraMoeMtpConfiguration.self, from: data)
            try configuration.validateModelConfiguration()
            guard let nextnPredictLayers = configuration.nextnPredictLayers,
                nextnPredictLayers > 0
            else {
                throw SharpAIPanguUltraMoeMtpError.invalidConfiguration(
                    "num_nextn_predict_layers must be a positive integer")
            }
            let base = try SharpAIPanguUltraMoeMtpConfiguration.baseConfiguration(from: data)
            return SharpAIPanguUltraMoeMtpModel(
                base, numNextnPredictLayers: nextnPredictLayers)
        }
    ]
}
