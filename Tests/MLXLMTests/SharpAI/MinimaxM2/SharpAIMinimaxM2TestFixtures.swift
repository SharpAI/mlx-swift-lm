import Foundation
import MLXLMCommon

@testable import MLXLLM

enum SharpAIMinimaxM2TestFixtures {

    /// Tiny EAGLE3-style draft config. Values are test-only, not from a real checkpoint.
    static let tinyJSON = """
        {
          "model_type": "minimax_m2",
          "architectures": ["Eagle3MiniMaxM2ForCausalLM"],
          "hidden_size": 16,
          "intermediate_size": 32,
          "num_hidden_layers": 1,
          "num_attention_heads": 4,
          "num_key_value_heads": 2,
          "vocab_size": 64,
          "draft_vocab_size": 32,
          "rms_norm_eps": 1e-6,
          "rope_theta": 10000,
          "max_position_embeddings": 128
        }
        """

    static func decode(_ json: String = tinyJSON) throws -> SharpAIMinimaxM2Configuration {
        try JSONDecoder.json5().decode(
            SharpAIMinimaxM2Configuration.self, from: Data(json.utf8))
    }
}
