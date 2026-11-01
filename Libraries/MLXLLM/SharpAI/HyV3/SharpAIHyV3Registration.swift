//
//  SharpAIHyV3Registration.swift
//  MLXLLM
//
//  Overlay creators for `hy_v3`. Not wired into LLMTypeRegistry. The owner
//  wires it. Each creator decodes, validates, then constructs, as in
//  LLMModelFactory.
//

import Foundation
import MLXLMCommon

public enum SharpAIHyV3Overlay {
    nonisolated(unsafe) public static let creators: [String: (Data) throws -> any LanguageModel] = [
        "hy_v3": { data in
            let configuration = try JSONDecoder.json5().decode(
                SharpAIHyV3Configuration.self, from: data)
            try configuration.validateModelConfiguration()
            return SharpAIHyV3Model(configuration)
        }
    ]
}
