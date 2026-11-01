// SharpAI fork overlay: step5_mtp creator. Not part of upstream mlx-swift-lm.
//
// The MTP module is blocked. Its decoder block (Step5DecoderLayer: sparse attention, indexer,
// MoE) is not in the repo, so this creator decodes and validates the config, then fails with a
// clear error at load time. VLMTypeRegistry and VLMModelFactory are not edited. The Step5 base
// types (step5, step5_vl) are not registered here.

import Foundation
import MLXLMCommon

public enum SharpAIStep5MtpOverlay {

    /// Creators keyed by `model_type`, same closure shape as `VLMTypeRegistry`.
    nonisolated(unsafe) public static let creators: [String: (Data) throws -> any LanguageModel] = [
        "step5_mtp": { data in
            let configuration = try SharpAIStep5MtpConfiguration.decode(from: data)
            try configuration.validateModelConfiguration()
            throw SharpAIStep5MtpError.forwardNotImplemented(configuration.modelType)
        }
    ]
}
