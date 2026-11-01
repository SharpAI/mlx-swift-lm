// SharpAI fork overlay: hyperclovax_vision_v2 creator. Not part of upstream mlx-swift-lm.
//
// The model class is not written yet: the text decoder has no reference source in the repo, and the
// ViT is fileprivate upstream. This creator decodes and validates the config, then fails.
// UNVERIFIED: no code in this repo reads `creators` yet. The key is not in VLMTypeRegistry, so
// load still reports unsupportedModelType. Wiring needs a VLMModelFactory edit, which this overlay
// does not make.

import Foundation
import MLXLMCommon

public enum SharpAIHyperclovaxVisionV2Overlay {

    /// Creators keyed by `model_type`, same closure shape as `VLMTypeRegistry`.
    nonisolated(unsafe) public static let creators: [String: (Data) throws -> any LanguageModel] = [
        "hyperclovax_vision_v2": { data in
            let configuration = try JSONDecoder.json5().decode(
                SharpAIHyperclovaxVisionV2Configuration.self, from: data)
            try configuration.validateModelConfiguration()
            throw SharpAIHyperclovaxVisionV2Error.forwardNotImplemented(configuration.modelType)
        }
    ]
}
