// SharpAI fork overlay: nemotron_h_omni creator. Not part of upstream mlx-swift-lm.
//
// Same creator shape and decode, validate, construct order as VLMTypeRegistry.shared.
// VLMModelFactory is not edited; the consumer merges this overlay.

import Foundation
import MLXLMCommon

public enum SharpAINemotronHOmniOverlay {

    /// Keyed by `model_type`. The arch string is not used for dispatch.
    nonisolated(unsafe) public static let creators: [String: (Data) throws -> any LanguageModel] = [
        "nemotron_h_omni": sharpAINemotronHOmniCreate(
            SharpAINemotronHOmniConfiguration.self
        ) { configuration in
            SharpAINemotronHOmniModel(configuration)
        }
    ]
}

private func sharpAINemotronHOmniCreate<C: Decodable>(
    _ configurationType: C.Type, _ modelInit: @escaping (C) -> any LanguageModel
) -> (Data) throws -> any LanguageModel {
    { data in
        let configuration = try JSONDecoder.json5().decode(C.self, from: data)
        if let validating = configuration as? ModelConfigurationValidating {
            try validating.validateModelConfiguration()
        }
        return modelInit(configuration)
    }
}
