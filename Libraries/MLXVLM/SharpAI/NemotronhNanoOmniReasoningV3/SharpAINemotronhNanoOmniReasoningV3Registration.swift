// SharpAI fork overlay: NemotronH_Nano_Omni_Reasoning_V3 creator. Not part of upstream mlx-swift-lm.
//
// Same creator shape and decode, validate, construct order as VLMTypeRegistry.shared.
// VLMModelFactory is not edited; the consumer merges this overlay. No processor is registered,
// so image and audio inputs have no processor path yet.

import Foundation
import MLXLMCommon

public enum SharpAINemotronhNanoOmniReasoningV3Overlay {

    /// Keyed by `model_type`. The arch string is not used for dispatch.
    nonisolated(unsafe) public static let creators: [String: (Data) throws -> any LanguageModel] = [
        "NemotronH_Nano_Omni_Reasoning_V3": sharpAINemotronhOmniCreate(
            SharpAINemotronhNanoOmniReasoningV3Configuration.self
        ) { configuration in
            SharpAINemotronhNanoOmniReasoningV3Model(configuration)
        }
    ]
}

private func sharpAINemotronhOmniCreate<C: Decodable>(
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
