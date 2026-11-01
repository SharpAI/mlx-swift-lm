//
//  SharpAIIquestloopcoderRegistration.swift
//  MLXLLM
//
//  Overlay creators for `iquestloopcoder`. Not wired into LLMTypeRegistry; the owner wires it.
//  The creator decodes and validates the config, then builds the model.
//

import Foundation
import MLXLMCommon

public enum SharpAIIquestloopcoderOverlay {

    /// Creators keyed by `model_type`, same closure shape as `LLMTypeRegistry`.
    nonisolated(unsafe) public static let creators: [String: (Data) throws -> any LanguageModel] = [
        "iquestloopcoder": { data in
            let configuration = try JSONDecoder.json5().decode(
                SharpAIIquestloopcoderConfiguration.self, from: data)
            try configuration.validateModelConfiguration()
            return SharpAIIquestloopcoderModel(configuration)
        }
    ]
}
