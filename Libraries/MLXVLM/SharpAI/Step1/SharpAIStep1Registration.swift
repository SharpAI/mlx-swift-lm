//
//  SharpAIStep1Registration.swift
//  MLXVLM
//
//  Overlay creators for `step1`. Not wired into VLMTypeRegistry. The owner wires it.
//  The creator decodes and validates the config, then builds the text model
//  (SharpAIStep1Model), which follows stepfun-ai/Step-Audio-EditX modeling_step1.py.
//  The audio tokenizer and audio tower are not built here.
//

import Foundation
import MLXLMCommon

public enum SharpAIStep1Overlay {

    /// Creators keyed by `model_type`.
    nonisolated(unsafe) public static let creators: [String: (Data) throws -> any LanguageModel] = [
        "step1": { data in
            let configuration = try JSONDecoder.json5().decode(
                SharpAIStep1Configuration.self, from: data)
            try configuration.validateModelConfiguration()
            return SharpAIStep1Model(configuration)
        }
    ]
}
