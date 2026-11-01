//
//  SharpAISarvamMoeRegistration.swift
//  MLXLLM
//
//  Overlay creators for `sarvam_moe`. Not wired into LLMTypeRegistry; the owner
//  wires it. Each creator decodes, validates, then constructs, as in LLMModelFactory.
//

import Foundation
import MLXLMCommon

public enum SharpAISarvamMoeOverlay {
    nonisolated(unsafe) public static let creators: [String: (Data) throws -> any LanguageModel] = [
        "sarvam_moe": { data in
            let configuration = try JSONDecoder.json5().decode(
                SharpAISarvamMoeConfiguration.self, from: data)
            try configuration.validateModelConfiguration()
            return SharpAISarvamMoeModel(configuration)
        }
    ]
}
