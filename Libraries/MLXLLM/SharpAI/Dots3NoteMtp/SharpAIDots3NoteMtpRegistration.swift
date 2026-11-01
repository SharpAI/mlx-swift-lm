// SharpAI fork overlay: dots3_note_mtp creator. Not part of upstream mlx-swift-lm.
//
// dots3_note_mtp is an MTP drafter head for the dots3_note base model. The MTP head itself is
// implemented in SharpAIDots3NoteMtpModel.swift (config, key mapping and forward verified against
// the dots-studio checkpoint index and sglang's NextN draft). The dots3_note target model is still
// not in the repo, and the head needs its hidden states and lm_head, so this creator keeps
// failing at load time. LLMModelFactory is not edited; the overlay consumer wires this in.

import Foundation
import MLXLMCommon

public enum SharpAIDots3NoteMtpError: LocalizedError {
    case blockedOnBase

    public var errorDescription: String? {
        switch self {
        case .blockedOnBase:
            return
                "Model type dots3_note_mtp is not loadable yet: it depends on the dots3_note base "
                + "model, which is not implemented. The MTP head is available as SharpAIDots3NoteMtpHead."
        }
    }
}

public enum SharpAIDots3NoteMtpOverlay {

    /// Creators keyed by `model_type`, same closure shape as `LLMTypeRegistry`.
    nonisolated(unsafe) public static let creators: [String: (Data) throws -> any LanguageModel] = [
        "dots3_note_mtp": { _ in
            throw SharpAIDots3NoteMtpError.blockedOnBase
        }
    ]
}
