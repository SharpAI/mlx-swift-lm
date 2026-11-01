// SharpAI fork overlay tests for the cohere2_moe creator. Not part of upstream.

import Foundation
import MLX
import MLXLMCommon
import XCTest

@testable import MLXLLM

final class SharpAICohere2MoeRegistrationTests: XCTestCase {

    private let validJSON = """
        {
            "model_type": "cohere2_moe",
            "hidden_size": 64,
            "num_hidden_layers": 2,
            "intermediate_size": 128,
            "num_attention_heads": 8,
            "num_key_value_heads": 2,
            "vocab_size": 512
        }
        """

    func testCreatorIsRegisteredForModelType() {
        XCTAssertNotNil(SharpAICohere2MoeOverlay.creators["cohere2_moe"])
    }

    func testValidConfigBuildsModel() throws {
        guard let creator = SharpAICohere2MoeOverlay.creators["cohere2_moe"] else {
            XCTFail("cohere2_moe creator missing")
            return
        }

        let model = try creator(Data(validJSON.utf8))
        guard let cohere = model as? SharpAICohere2MoeModel else {
            XCTFail("creator returned \(type(of: model)), expected SharpAICohere2MoeModel")
            return
        }
        XCTAssertEqual(cohere.vocabularySize, 512)
        XCTAssertEqual(cohere.kvHeads, [2, 2])
    }

    func testInvalidConfigFailsBeforeConstruction() {
        guard let creator = SharpAICohere2MoeOverlay.creators["cohere2_moe"] else {
            XCTFail("cohere2_moe creator missing")
            return
        }
        let json = """
            { "hidden_size": 64 }
            """

        XCTAssertThrowsError(try creator(Data(json.utf8)))
    }

    func testHeadMismatchFailsValidation() {
        guard let creator = SharpAICohere2MoeOverlay.creators["cohere2_moe"] else {
            XCTFail("cohere2_moe creator missing")
            return
        }
        let json = """
            {
                "hidden_size": 64,
                "num_hidden_layers": 2,
                "intermediate_size": 128,
                "num_attention_heads": 8,
                "num_key_value_heads": 3,
                "vocab_size": 512
            }
            """

        XCTAssertThrowsError(try creator(Data(json.utf8)))
    }

    /// Tiny MoE config. Layer 0 is a dense prefix layer, so it is full attention with RoPE forced.
    /// Layers 1 and 2 are sliding (window 4, seq 6). Layer 3 is full attention.
    func testTinyForwardPass() throws {
        let json = """
            {
                "model_type": "cohere2_moe",
                "hidden_size": 16,
                "num_hidden_layers": 4,
                "intermediate_size": 24,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 8,
                "vocab_size": 512,
                "sliding_window": 4,
                "sliding_window_pattern": 4,
                "prefix_dense_sliding_window_pattern": 1,
                "first_k_dense_replace": 1,
                "num_experts": 4,
                "num_experts_per_tok": 2,
                "num_shared_experts": 1,
                "tie_word_embeddings": true
            }
            """
        let config = try JSONDecoder.json5().decode(
            SharpAICohere2MoeConfiguration.self, from: Data(json.utf8))
        try config.validateModelConfiguration()
        XCTAssertEqual(
            config.layerTypes,
            ["full_attention", "sliding_attention", "sliding_attention", "full_attention"])

        let model = SharpAICohere2MoeModel(config)
        let tokens = MLXArray([1, 2, 3, 4, 5, 6] as [Int32]).reshaped([1, 6])
        let logits = model(tokens, cache: nil)
        eval(logits)

        XCTAssertEqual(logits.shape, [1, 6, 512])
        XCTAssertTrue(logits.asArray(Float.self).allSatisfy { $0.isFinite })
    }
}
