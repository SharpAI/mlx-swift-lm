//
//  SharpAIBailingMoeV3Registration.swift
//  MLXLLM
//
//  Overlay creators for `bailing_moe_v3`. Not wired into LLMTypeRegistry. The
//  owner wires it. Each creator decodes, validates, then constructs, as in
//  LLMModelFactory.
//

import Foundation
import MLXLMCommon

public enum SharpAIBailingMoeV3Overlay {
    nonisolated(unsafe) public static let creators: [String: (Data) throws -> any LanguageModel] = [
        "bailing_moe_v3": { data in
            let configuration = try JSONDecoder.json5().decode(
                SharpAIBailingMoeV3Configuration.self, from: data)
            try configuration.validateModelConfiguration()
            return SharpAIBailingMoeV3Model(configuration)
        }
    ]
}
