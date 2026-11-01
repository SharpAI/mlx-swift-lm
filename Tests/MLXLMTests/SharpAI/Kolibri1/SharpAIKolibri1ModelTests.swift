// SharpAI fork overlay tests for kolibri1 registration and the text model.
// Not part of upstream.

import Foundation
import MLX
import MLXLMCommon
import XCTest

@testable import MLXLLM

/// Tiny config for forward-pass tests. The full fixture (hidden 2560, 384 experts) is too large
/// to instantiate in a unit test.
enum SharpAIKolibri1TinyFixture {
    static let json = """
        {
            "architectures": ["Kolibri1ForCausalLM"],
            "model_type": "kolibri1",
            "hidden_size": 16,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 4,
            "hidden_act": "silu",
            "max_position_embeddings": 64,
            "rms_norm_eps": 1e-06,
            "vocab_size": 32,
            "rope_theta": 10000.0,
            "num_experts": 4,
            "num_experts_per_tok": 2,
            "moe_intermediate_size": 8,
            "shared_expert_intermediate_size": 8,
            "norm_topk_prob": true,
            "attention_bias": false,
            "tie_word_embeddings": false,
            "use_sliding_window": true,
            "sliding_window": 3,
            "layer_types": ["sliding_attention", "full_attention"],
            "eos_token_id": 1,
            "pad_token_id": 0
        }
        """
}

final class SharpAIKolibri1ModelTests: XCTestCase {

    func testRegistryCreatorBuildsModel() throws {
        let creator = try XCTUnwrap(SharpAIKolibri1Overlay.creators["kolibri1"])
        let data = Data(SharpAIKolibri1TinyFixture.json.utf8)

        let model = try creator(data)
        XCTAssertTrue(model is SharpAIKolibri1Model)
    }

    func testRegistryCreatorRejectsInvalidConfigBeforeThrowing() throws {
        let creator = try XCTUnwrap(SharpAIKolibri1Overlay.creators["kolibri1"])
        // 4 attention heads are not a multiple of 3 KV heads.
        let broken = SharpAIKolibri1TinyFixture.json
            .replacingOccurrences(of: "\"num_key_value_heads\": 2", with: "\"num_key_value_heads\": 3")

        XCTAssertThrowsError(try creator(Data(broken.utf8))) { error in
            guard case SharpAIKolibri1Error.invalidConfiguration = error else {
                return XCTFail("expected invalidConfiguration, got \(error)")
            }
        }
    }

    func testTinyShapeForwardPass() throws {
        let config = try JSONDecoder.json5().decode(
            SharpAIKolibri1Configuration.self, from: Data(SharpAIKolibri1TinyFixture.json.utf8))
        let model = SharpAIKolibri1Model(config)

        let tokens = MLXArray([1, 2, 3, 4, 5] as [Int32]).reshaped([1, 5])
        let logits = model(tokens)

        XCTAssertEqual(logits.shape, [1, 5, 32])
        XCTAssertTrue(logits.asArray(Float.self).allSatisfy { $0.isFinite })
    }
}
