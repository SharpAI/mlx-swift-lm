// SharpAI fork overlay tests for axk1 config decoding. Not part of upstream.

import Foundation
import MLXLMCommon
import XCTest

@testable import MLXLLM

/// Shared inline fixture. Layer 0 is dense, layer 1 is MoE (first_k_dense_replace = 1).
enum SharpAIAxk1TestFixtures {
    static func json(numHiddenLayers: Int = 2) -> String {
        """
            {
                "model_type": "axk1",
                "architectures": ["AXK1ForCausalLM"],
                "vocab_size": 32,
                "hidden_size": 8,
                "intermediate_size": 16,
                "moe_intermediate_size": 8,
                "num_hidden_layers": \(numHiddenLayers),
                "num_attention_heads": 2,
                "num_key_value_heads": 2,
                "n_routed_experts": 4,
                "n_shared_experts": 1,
                "num_experts_per_tok": 2,
                "n_group": 2,
                "topk_group": 1,
                "norm_topk_prob": true,
                "routed_scaling_factor": 1.0,
                "first_k_dense_replace": 1,
                "moe_layer_freq": 1,
                "q_lora_rank": 4,
                "kv_lora_rank": 4,
                "qk_rope_head_dim": 2,
                "qk_nope_head_dim": 2,
                "v_head_dim": 4,
                "rms_norm_eps": 1e-6,
                "rope_theta": 10000.0,
                "max_position_embeddings": 128,
                "attention_bias": false
            }
            """
    }

    static func configuration(numHiddenLayers: Int = 2) throws -> SharpAIAxk1Configuration {
        try JSONDecoder.json5().decode(
            SharpAIAxk1Configuration.self, from: Data(json(numHiddenLayers: numHiddenLayers).utf8))
    }
}

final class SharpAIAxk1ConfigurationTests: XCTestCase {

    func testDecodesModelTypeAndBaseFields() throws {
        let config = try SharpAIAxk1TestFixtures.configuration(numHiddenLayers: 3)

        XCTAssertEqual(config.modelType, "axk1")
        XCTAssertEqual(config.base.numHiddenLayers, 3)
        XCTAssertEqual(config.base.nRoutedExperts, 4)
        XCTAssertEqual(config.base.firstKDenseReplace, 1)
        XCTAssertEqual(config.base.qkRopeHeadDim, 2)
    }

    func testMissingModelTypeUsesDefault() throws {
        // UNVERIFIED: the default model_type is an assumption, pinned here so a change is visible.
        let json = SharpAIAxk1TestFixtures.json().replacingOccurrences(
            of: "\"model_type\": \"axk1\",", with: "")
        let config = try JSONDecoder.json5().decode(
            SharpAIAxk1Configuration.self, from: Data(json.utf8))

        XCTAssertEqual(config.modelType, "axk1")
    }

    func testRegistryCreatorDecodesConfigurationIntoModel() throws {
        let creator = try XCTUnwrap(SharpAIAxk1Overlay.creators["axk1"])
        let model = try creator(Data(SharpAIAxk1TestFixtures.json().utf8))

        XCTAssertTrue(model is SharpAIAxk1Model)
    }
}
