// SharpAI fork overlay: PanguEmbedded configuration. Not part of upstream mlx-swift-lm.
//
// Source: openPangu-Embedded-7B-V1.1 config.json (scripts/conformance/offline/fixtures/PanguEmbedded)
// and llama.cpp src/models/pangu-embedded.cpp. The checkpoint names its attention bias flag
// "bias" (not "attention_bias"). q, k, v and o all carry bias. The MLP carries none.

import Foundation
import MLXLMCommon

public struct SharpAIPanguembeddedConfiguration: Decodable, Sendable, ModelConfigurationValidating {

    /// The Llama configuration the model is built from. Its `attentionBias` is resolved here.
    public let llamaConfiguration: LlamaConfiguration

    private enum BiasKeys: String, CodingKey {
        case attentionBias = "attention_bias"
        case bias
    }

    public init(from decoder: Decoder) throws {
        var llama = try LlamaConfiguration(from: decoder)

        // Fall back to the checkpoint's "bias" key. An explicit "attention_bias" wins.
        let keys = try decoder.container(keyedBy: BiasKeys.self)
        if let attentionBias = try keys.decodeIfPresent(Bool.self, forKey: .attentionBias) {
            llama.attentionBias = attentionBias
        } else if let bias = try keys.decodeIfPresent(Bool.self, forKey: .bias) {
            llama.attentionBias = bias
        }

        self.llamaConfiguration = llama
    }

    public func validateModelConfiguration() throws {
        let config = llamaConfiguration
        guard config.hiddenLayers > 0, config.vocabularySize > 0,
            config.attentionHeads > 0, config.kvHeads > 0,
            config.attentionHeads % config.kvHeads == 0
        else {
            throw ModelFactoryError.invalidConfiguration(
                "PanguEmbedded needs positive layer, vocabulary and head counts, "
                    + "with attention heads divisible by kv heads")
        }
    }
}
