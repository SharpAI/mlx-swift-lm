//
//  SharpAILongcatFlashRegistration.swift
//  MLXLLM
//
//  Overlay creators for `longcat_flash`. Not wired into LLMTypeRegistry. The owner wires it.
//  Each creator decodes, validates, then constructs, as in LLMModelFactory.
//

import Foundation
import MLXLMCommon

public enum SharpAILongcatFlashOverlay {

    /// Creators keyed by `model_type`, same closure shape as `LLMTypeRegistry`.
    nonisolated(unsafe) public static let creators: [String: (Data) throws -> any LanguageModel] = [
        "longcat_flash": { data in
            let configuration = try JSONDecoder.json5().decode(
                SharpAILongcatFlashConfiguration.self, from: data)
            try configuration.validateModelConfiguration()
            return SharpAILongcatFlashModel(configuration)
        }
    ]
}
