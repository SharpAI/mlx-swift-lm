// SharpAI fork overlay test helpers for alice_ai. Not part of upstream.

import Foundation
import XCTest

enum SharpAIAliceAiTestSupport {

    /// A 4-layer model (3 KDA, 1 full attention) with tiny dimensions for random-weight tests.
    static func tinyDictionary() -> [String: Any] {
        [
            "model_type": "alice_ai",
            "hidden_size": 16,
            "num_hidden_layers": 4,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 8,
            "layer_types": [
                "linear_attention", "linear_attention", "linear_attention", "full_attention",
            ],
            "hidden_act": "silu",
            "rms_norm_eps": 1e-6,
            "vocab_size": 32,
            "rope_theta": 10000.0,
            "partial_rotary_factor": 0.5,
            "max_position_embeddings": 64,
            "tie_word_embeddings": false,
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 2,
            "linear_key_head_dim": 4,
            "linear_value_head_dim": 4,
            "linear_conv_kernel_dim": 4,
            "block_attn_res_block_size": 2,
            "num_experts": 4,
            "num_experts_per_tok": 2,
            "moe_intermediate_size": 8,
            "shared_expert_intermediate_size": 8,
            "norm_topk_prob": true,
            "router_score_function": "sigmoid",
        ]
    }

    static func tinyData(overrides: [String: Any] = [:]) throws -> Data {
        var dictionary = tinyDictionary()
        for (key, value) in overrides {
            dictionary[key] = value
        }
        return try JSONSerialization.data(withJSONObject: dictionary)
    }

    static func assertClose(
        _ actual: [Float], _ expected: [Float], tolerance: Float = 1e-4,
        file: StaticString = #filePath, line: UInt = #line
    ) {
        XCTAssertEqual(actual.count, expected.count, file: file, line: line)
        for (a, e) in zip(actual, expected) {
            XCTAssertEqual(a, e, accuracy: tolerance, file: file, line: line)
        }
    }
}
