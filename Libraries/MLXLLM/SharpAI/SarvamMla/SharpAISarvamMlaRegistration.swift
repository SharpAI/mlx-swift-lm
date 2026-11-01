//
//  SharpAISarvamMlaRegistration.swift
//  MLXLLM
//
//  Overlay creator for `sarvam_mla`. Not wired into LLMTypeRegistry. The owner
//  wires it. The creator decodes and validates the config, as in
//  LLMModelFactory, then stops: the forward model is not implemented yet.
//  UNVERIFIED: no forward model exists, so this creator never returns a model.
//

import Foundation
import MLXLMCommon

public enum SharpAISarvamMlaOverlay {
    nonisolated(unsafe) public static let creators: [String: (Data) throws -> LanguageModel] = [
        "sarvam_mla": { data in
            let configuration = try JSONDecoder.json5().decode(
                SharpAISarvamMlaConfiguration.self, from: data)
            try configuration.validateModelConfiguration()
            throw SharpAISarvamMlaError.unsupported(
                "forward model not implemented; MLA attention, use_qk_norm and router details are unverified")
        }
    ]
}
