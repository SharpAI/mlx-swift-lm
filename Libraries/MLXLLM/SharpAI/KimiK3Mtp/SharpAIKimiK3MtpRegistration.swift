//
//  SharpAIKimiK3MtpRegistration.swift
//  MLXLLM
//
//  Overlay creators for `kimi_k3_mtp`. Not wired into LLMTypeRegistry; the owner wires it.
//  Config decoding is in SharpAIKimiK3MtpConfiguration (verified against vLLM mtp.py).
//  The MTP layer body and the base KimiK3 text backbone are not in the repo, and the released
//  config declares num_nextn_predict_layers = 0, so the creator throws and does not build a model.
//

import Foundation
import MLXLMCommon

public enum SharpAIKimiK3MtpError: LocalizedError {
    case blockedOnBase

    public var errorDescription: String? {
        switch self {
        case .blockedOnBase:
            return
                "Model type kimi_k3_mtp is not loadable yet: the released config declares no MTP "
                + "layers (num_nextn_predict_layers = 0) and the KimiK3 MTP layer and base backbone are not implemented."
        }
    }
}

public enum SharpAIKimiK3MtpOverlay {

    /// Creators keyed by `model_type`, same closure shape as `LLMTypeRegistry`.
    nonisolated(unsafe) public static let creators: [String: (Data) throws -> any LanguageModel] = [
        "kimi_k3_mtp": { _ in
            // Model construction needs the KimiK3 MTP layer body and base backbone (not implemented).
            throw SharpAIKimiK3MtpError.blockedOnBase
        }
    ]
}
