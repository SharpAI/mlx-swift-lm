// SharpAI fork overlay: bagel creator. Not part of upstream mlx-swift-lm.
//
// Same decode, validate, construct sequence as the VLMTypeRegistry creators. VLMModelFactory is
// not edited, so the overlay consumer wires this dictionary in.

import Foundation
import MLXLMCommon

private func sharpAIBagelCreator<C: Decodable, M: LanguageModel>(
    _ configurationType: C.Type, _ modelInit: @escaping (C) -> M
) -> (Data) throws -> any LanguageModel {
    { data in
        let configuration = try JSONDecoder.json5().decode(C.self, from: data)
        if let validating = configuration as? ModelConfigurationValidating {
            try validating.validateModelConfiguration()
        }
        return modelInit(configuration)
    }
}

public enum SharpAIBagelOverlay {

    /// Creators keyed by `model_type`, same closure shape as `VLMTypeRegistry`.
    nonisolated(unsafe) public static let creators: [String: (Data) throws -> any LanguageModel] = [
        "bagel": sharpAIBagelCreator(SharpAIBagelConfiguration.self, SharpAIBagel.init)
    ]
}
