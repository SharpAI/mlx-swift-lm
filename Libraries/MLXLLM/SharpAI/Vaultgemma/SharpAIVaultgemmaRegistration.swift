//
//  SharpAIVaultgemmaRegistration.swift
//  MLXLLM
//
//  SharpAI fork overlay: vaultgemma creator. Not part of upstream mlx-swift-lm.
//  Decode, validate, then construct, as in LLMModelFactory. LLMModelFactory is not edited.
//  The overlay consumer wires this creator in.
//

import Foundation
import MLXLMCommon

public enum SharpAIVaultgemmaOverlay {

    /// Creators keyed by `model_type`, same closure shape as `LLMTypeRegistry`.
    nonisolated(unsafe) public static let creators: [String: (Data) throws -> any LanguageModel] = [
        "vaultgemma": { data in
            let configuration = try JSONDecoder.json5().decode(
                SharpAIVaultgemmaConfiguration.self, from: data)
            try configuration.validateModelConfiguration()
            return SharpAIVaultgemmaModel(configuration)
        }
    ]
}
