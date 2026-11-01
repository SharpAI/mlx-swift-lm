import Foundation
import MLXLMCommon

/// Creators for the Beam model type, keyed by `config.json` `model_type`.
///
/// Uses the same decode, validate, and construct sequence as `LLMModelFactory`. The factory
/// itself is not edited. A caller merges these creators into its registry.
public enum SharpAIBeamOverlay {
    public static let modelType = "beam"

    public static var creators: [String: (Data) throws -> any LanguageModel] {
        [
            modelType: { data in
                let config = try JSONDecoder.json5().decode(
                    SharpAIBeamConfiguration.self, from: data)
                try config.validateModelConfiguration()
                return SharpAIBeamForCausalLM(config)
            }
        ]
    }
}
