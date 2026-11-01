//
//  SharpAIStep5Registration.swift
//  MLXLLM
//
//  Overlay creators for `step5`. Not wired into LLMTypeRegistry. The owner wires it.
//  The creator decodes and validates the config, refuses features that are not ported
//  (DSA sparse/full-attention layers, fp8 experts, non-identity rope scaling), and then
//  builds the text model. The forward path covers sliding-window layers only.
//

import Foundation
import MLXLMCommon

public enum SharpAIStep5Overlay {

    /// Creators keyed by `model_type`.
    nonisolated(unsafe) public static let creators: [String: (Data) throws -> any LanguageModel] = [
        "step5": { data in
            let configuration = try JSONDecoder.json5().decode(
                SharpAIStep5Configuration.self, from: data)
            try configuration.validateModelConfiguration()
            try configuration.validateSupportedArchitecture()
            return SharpAIStep5Model(configuration)
        }
    ]
}
