// SharpAI fork overlay test support for domynedge. Not part of upstream.

import Foundation
import MLXLMCommon

@testable import MLXLLM

enum SharpAIDomynedgeTestFixtures {

    /// A 4-layer config with the same layout as the real checkpoint: nope full layers and
    /// rope sliding layers, with per-layer head sizes. Dimensions are tiny.
    static let tinyJSON = """
        {
            "model_type": "domynedge",
            "architectures": ["DomynEdgeForCausalLM"],
            "hidden_size": 16,
            "num_hidden_layers": 4,
            "intermediate_size": 32,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_sizes": [16, 8, 8, 16],
            "position_layers": ["nope", "rope", "rope", "nope"],
            "window_sizes": [[-1, 0], [4, 0], [4, 0], [-1, 0]],
            "rms_norm_eps": 1e-6,
            "vocab_size": 64,
            "max_position_embeddings": 128,
            "rope_theta": 10000.0,
            "hidden_act": "silu",
            "tie_word_embeddings": true
        }
        """

    static func tinyConfiguration(tied: Bool = true) throws -> SharpAIDomynedgeConfiguration {
        let json =
            tied
            ? tinyJSON
            : tinyJSON.replacingOccurrences(
                of: "\"tie_word_embeddings\": true", with: "\"tie_word_embeddings\": false")
        return try JSONDecoder.json5().decode(
            SharpAIDomynedgeConfiguration.self, from: Data(json.utf8))
    }
}
