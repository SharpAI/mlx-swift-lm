//
//  SharpAIMimoV2Registration.swift
//  MLXLLM
//
//  Overlay creators for `mimo_v2`. Not wired into LLMTypeRegistry. The owner
//  wires it. Each creator decodes, validates, then constructs, as in
//  LLMModelFactory. Omni (VLM) checkpoints are not registered here.
//

import Foundation
import MLXLMCommon

public enum SharpAIMimoV2Overlay {
    public static let creators: [String: @Sendable (Data) throws -> LanguageModel] = [
        "mimo_v2": { data in
            let configuration = try JSONDecoder.json5().decode(
                SharpAIMimoV2Configuration.self, from: data)
            try configuration.validateModelConfiguration()
            return SharpAIMimoV2Model(configuration)
        }
    ]
}
