// SharpAI fork overlay: nemotron_parse creator. Not part of upstream mlx-swift-lm.
//
// Builds the RADIO encoder, the mBART-style decoder, and the untied lm_head. VLMTypeRegistry and
// VLMModelFactory are not edited. No image processor is registered: the caller supplies pixels
// already resized to a multiple of 16 and normalized.

import Foundation
import MLXLMCommon

public enum SharpAINemotronParseOverlay {

    /// Creators keyed by `model_type`, same closure shape as `VLMTypeRegistry`.
    nonisolated(unsafe) public static let creators: [String: (Data) throws -> any LanguageModel] = [
        "nemotron_parse": { data in
            let configuration = try JSONDecoder.json5().decode(
                SharpAINemotronParseConfiguration.self, from: data)
            try configuration.validateModelConfiguration()
            return SharpAINemotronParse(configuration)
        }
    ]
}
