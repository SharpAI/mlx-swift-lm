// SharpAI fork overlay: muse_spark_dspark creator. Not part of upstream mlx-swift-lm.
//
// A DSpark drafter has no standalone LanguageModel. It needs the frozen Muse Spark target for
// embeddings and target hidden states, and that target is not implemented. The creator decodes
// and validates the config, then throws. SharpAIMuseSparkDsparkDraftModel is the class to attach
// once the target exists. LLMModelFactory is not edited; the overlay consumer merges these
// creators.

import Foundation
import MLXLMCommon

public enum SharpAIMuseSparkDsparkOverlay {

    /// Creators keyed by `model_type`, same closure shape as `LLMTypeRegistry`.
    nonisolated(unsafe) public static let creators: [String: (Data) throws -> any LanguageModel] = [
        "muse_spark_dspark": { data in
            let configuration = try JSONDecoder.json5().decode(
                SharpAIMuseSparkDsparkConfiguration.self, from: data)
            try configuration.validateModelConfiguration()
            throw SharpAIMuseSparkDsparkError.targetNotImplemented(configuration.modelType)
        }
    ]
}
