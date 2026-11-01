// SharpAI fork overlay tests for the plamo3 forward pass. Not part of upstream.
// Tiny random-weight model, no checkpoint. Shapes, cache path, tied embeddings, sanitize, layer kinds.

import Foundation
import MLX
import MLXLMCommon
import XCTest

@testable import MLXLLM

final class SharpAIPlamo3ModelTests: XCTestCase {

    /// Sliding layer 0, full layer 1, window 4, so a 5-token prompt exercises the windowed mask.
    private func tinyJSON(tied: Bool = false) -> String {
        """
        {
          "architectures": ["Plamo3ForCausalLM"],
          "model_type": "plamo3",
          "hidden_size": 16,
          "intermediate_size": 32,
          "num_hidden_layers": 2,
          "num_attention_heads": 2,
          "num_key_value_heads": 1,
          "head_dim": 8,
          "vocab_size": 64,
          "layer_types": ["sliding_attention", "full_attention"],
          "sliding_window": 4,
          "rope_theta": 1000000.0,
          "rope_local_theta": 10000.0,
          "tie_word_embeddings": \(tied)
        }
        """
    }

    private func makeModel(tied: Bool = false) throws -> SharpAIPlamo3Model {
        let configuration = try JSONDecoder.json5().decode(
            SharpAIPlamo3Configuration.self, from: Data(tinyJSON(tied: tied).utf8))
        return try SharpAIPlamo3Model(configuration)
    }

    private func tokens(_ count: Int) -> MLXArray {
        MLXArray((0 ..< count).map { Int32($0 + 1) }).reshaped(1, count)
    }

    func testTinyForwardPassShape() throws {
        let model = try makeModel()

        let logits = model(tokens(5), cache: nil)

        XCTAssertEqual(logits.shape, [1, 5, 64])
        XCTAssertTrue(logits.asArray(Float.self).allSatisfy { $0.isFinite })
    }

    func testForwardPassWithCacheShapeAndKinds() throws {
        let model = try makeModel()

        let cache = try model.newCache(parameters: nil)
        XCTAssertEqual(cache.count, 2)

        let logits = model(tokens(5), cache: cache)

        XCTAssertEqual(logits.shape, [1, 5, 64])
        XCTAssertTrue(logits.asArray(Float.self).allSatisfy { $0.isFinite })
    }

    func testTiedEmbeddingsUseEmbeddingAsHead() throws {
        let model = try makeModel(tied: true)
        XCTAssertNil(model.lmHead)

        let logits = model(tokens(3), cache: nil)

        XCTAssertEqual(logits.shape, [1, 3, 64])
    }

    func testUntiedEmbeddingsHaveSeparateHead() throws {
        let model = try makeModel(tied: false)
        XCTAssertNotNil(model.lmHead)
    }

    func testSanitizeDropsLMHeadOnlyWhenTied() throws {
        let weights: [String: MLXArray] = [
            "model.embed_tokens.weight": MLXArray.zeros([2]),
            "lm_head.weight": MLXArray.zeros([2]),
        ]

        let tied = try makeModel(tied: true).sanitize(weights: weights)
        XCTAssertEqual(Set(tied.keys), ["model.embed_tokens.weight"])

        let untied = try makeModel(tied: false).sanitize(weights: weights)
        XCTAssertEqual(Set(untied.keys), Set(weights.keys))
    }

    func testResolvedLayerTypesFollowPatternWhenAbsent() throws {
        let json = """
            {
              "model_type": "plamo3",
              "hidden_size": 16,
              "num_hidden_layers": 8,
              "num_attention_heads": 2,
              "vocab_size": 64,
              "intermediate_size": 32
            }
            """
        let configuration = try JSONDecoder.json5().decode(
            SharpAIPlamo3Configuration.self, from: Data(json.utf8))

        let kinds = configuration.resolvedLayerTypes
        XCTAssertEqual(kinds.count, 8)
        // Default pattern 8: only the 8th (1-indexed) layer is full attention.
        XCTAssertEqual(kinds.last, "full_attention")
        XCTAssertEqual(kinds.filter { $0 == "full_attention" }.count, 1)
    }
}
