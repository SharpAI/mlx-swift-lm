// SharpAI fork overlay: openvla creator. Not part of upstream mlx-swift-lm.
//
// Same decode, validate, construct sequence as `create` in VLMModelFactory. VLMModelFactory and
// VLMTypeRegistry are not edited, so the consumer merges `creators` itself. No processor is
// registered: OpenVLA's image processor is not implemented.

import Foundation
import MLXLMCommon

public enum SharpAIOpenvlaOverlay {

    /// Creators keyed by `model_type`, same closure shape as `VLMTypeRegistry`.
    nonisolated(unsafe) public static let creators: [String: (Data) throws -> any LanguageModel] = [
        "openvla": { data in
            let configuration = try JSONDecoder.json5().decode(
                SharpAIOpenvlaConfiguration.self, from: data)
            try configuration.validateModelConfiguration()
            return SharpAIOpenvla(configuration)
        }
    ]
}
