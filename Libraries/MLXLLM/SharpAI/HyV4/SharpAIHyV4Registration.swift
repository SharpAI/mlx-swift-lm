//
//  SharpAIHyV4Registration.swift
//  MLXLLM
//
//  Overlay creator for `hy_v4`. Not wired into LLMTypeRegistry. The owner wires it.
//  The creator decodes and validates the config, then constructs SharpAIHyV4Model,
//  as in LLMModelFactory. Multi-token-prediction weights are not loaded.
//

import Foundation
import MLXLMCommon

public enum SharpAIHyV4Overlay {
    nonisolated(unsafe) public static let creators: [String: (Data) throws -> any LanguageModel] = [
        "hy_v4": { data in
            let configuration = try JSONDecoder.json5().decode(
                SharpAIHyV4Configuration.self, from: data)
            try configuration.validateModelConfiguration()
            return SharpAIHyV4Model(configuration)
        }
    ]
}
