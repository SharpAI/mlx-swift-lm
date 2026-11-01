// SharpAI fork overlay: nemotron_h_mtp creator. Not part of upstream.
//
// Same closure shape as LLMTypeRegistry: decode, validate, construct. LLMModelFactory is
// not edited. A consumer merges these creators into its registry.

import Foundation
import MLXLMCommon

public enum SharpAINemotronHMtpOverlay {

    /// Creators keyed by `model_type`.
    nonisolated(unsafe) public static let creators: [String: (Data) throws -> any LanguageModel] = [
        "nemotron_h_mtp": { data in
            // NemotronHConfiguration has no validation step, so decode and construct only.
            let configuration = try JSONDecoder.json5().decode(
                NemotronHConfiguration.self, from: data)
            return SharpAINemotronHMtpModel(configuration)
        }
    ]
}
