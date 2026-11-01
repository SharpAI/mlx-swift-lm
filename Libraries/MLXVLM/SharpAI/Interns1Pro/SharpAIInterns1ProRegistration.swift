// SharpAI fork overlay: interns1_pro creator. Not part of upstream mlx-swift-lm.
//
// VLMModelFactory is not edited; the overlay consumer merges these creators into its own registry.
// The creator decodes and validates the config, then builds the text decoder
// (SharpAIInterns1ProModel). It needs text_config. Vision and time-series inputs are not handled
// (see SharpAIInterns1ProTextModel.swift).
//
// Verified against: https://huggingface.co/internlm/Intern-S1-Pro/resolve/main/config.json

import Foundation
import MLXLMCommon

public enum SharpAIInterns1ProOverlay {

    /// Creators keyed by `model_type`, same closure shape as `VLMTypeRegistry`.
    nonisolated(unsafe) public static let creators: [String: (Data) throws -> any LanguageModel] = [
        "interns1_pro": { data in
            let configuration = try JSONDecoder.json5().decode(
                SharpAIInterns1ProConfiguration.self, from: data)
            try configuration.validateModelConfiguration()
            guard let textConfiguration = configuration.textConfig else {
                throw ModelFactoryError.invalidConfiguration(
                    "interns1_pro: text_config is required to build the model")
            }
            return SharpAIInterns1ProModel(
                textConfiguration, tieWordEmbeddings: configuration.tieWordEmbeddings)
        }
    ]
}
