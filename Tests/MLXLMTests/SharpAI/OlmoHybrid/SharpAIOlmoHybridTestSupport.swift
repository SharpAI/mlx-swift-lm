// SharpAI fork overlay test helpers for olmo_hybrid. Not part of upstream.

import Foundation
import XCTest

enum SharpAIOlmoHybridTestSupport {

    /// A 4-layer model (3 GatedDeltaNet, 1 full attention) with tiny dimensions for random-weight tests.
    static func tinyDictionary() -> [String: Any] {
        [
            "model_type": "olmo_hybrid",
            "hidden_size": 16,
            "num_hidden_layers": 4,
            "intermediate_size": 24,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 8,
            "layer_types": [
                "linear_attention", "linear_attention", "linear_attention", "full_attention",
            ],
            "hidden_act": "silu",
            "rms_norm_eps": 1e-6,
            "vocab_size": 32,
            "max_position_embeddings": 64,
            "attention_bias": false,
            "tie_word_embeddings": false,
            "rope_parameters": ["rope_type": "default", "rope_theta": 10000.0],
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 2,
            "linear_key_head_dim": 4,
            "linear_value_head_dim": 4,
            "linear_conv_kernel_dim": 4,
            "linear_allow_neg_eigval": true,
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
