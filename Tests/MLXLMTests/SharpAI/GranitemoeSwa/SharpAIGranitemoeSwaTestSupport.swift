// SharpAI fork overlay test support for granitemoe_swa. Not part of upstream.

import Foundation
import MLXLMCommon

@testable import MLXLLM

enum SharpAIGranitemoeSwaTestFixtures {

    /// Two layers: layer 0 slides with NoPE, layer 1 is full attention with RoPE.
    /// Sinks are on through `layer_types`. Dimensions are tiny.
    static let tinyJSON = """
        {
            "model_type": "granitemoe_swa",
            "vocab_size": 64,
            "hidden_size": 16,
            "intermediate_size": 8,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "num_local_experts": 4,
            "num_experts_per_tok": 2,
            "shared_intermediate_size": 8,
            "max_position_embeddings": 128,
            "rope_theta": 10000.0,
            "layer_types": ["sliding_attention", "full_attention"],
            "layer_rope_theta": [0.0, 10000.0],
            "sliding_window": 2,
            "tie_word_embeddings": true,
            "attention_multiplier": 0.25,
            "embedding_multiplier": 2.0,
            "residual_multiplier": 0.5,
            "logits_scaling": 2.0
        }
        """

    /// Decodes `tinyJSON` with `overrides` applied. A value of `NSNull()` removes the key.
    static func configuration(
        overriding overrides: [String: Any] = [:]
    ) throws -> SharpAIGranitemoeSwaConfiguration {
        var dict = (try JSONSerialization.jsonObject(with: Data(tinyJSON.utf8)) as? [String: Any])
            ?? [:]
        for (key, value) in overrides {
            dict[key] = value
        }
        let data = try JSONSerialization.data(withJSONObject: dict)
        return try JSONDecoder.json5().decode(SharpAIGranitemoeSwaConfiguration.self, from: data)
    }
}
