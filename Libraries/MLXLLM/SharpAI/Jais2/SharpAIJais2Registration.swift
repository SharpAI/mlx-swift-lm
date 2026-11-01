//
//  SharpAIJais2Registration.swift
//  MLXLLM
//
//  Overlay creators for `jais2`. Not wired into LLMTypeRegistry. The owner wires it.
//  The creator decodes and validates the config, then builds `SharpAIJais2Model`.
//

import Foundation
import MLXLMCommon

public enum SharpAIJais2Overlay {

    /// Creators keyed by `model_type`.
    nonisolated(unsafe) public static let creators: [String: (Data) throws -> any LanguageModel] = [
        "jais2": { data in
            try sharpAIJais2Create(data)
        }
    ]
}

private func sharpAIJais2Create(_ data: Data) throws -> any LanguageModel {
    let configuration = try JSONDecoder.json5().decode(
        SharpAIJais2Configuration.self, from: data)
    try configuration.validateModelConfiguration()
    // transformers Jais2Config defaults hidden_act to "relu2". Only that is ported.
    if let act = configuration.hiddenAct, act != "relu2" {
        throw SharpAIJais2Error.unsupported("hidden_act \"\(act)\" is not implemented; only relu2")
    }
    return SharpAIJais2Model(configuration)
}
