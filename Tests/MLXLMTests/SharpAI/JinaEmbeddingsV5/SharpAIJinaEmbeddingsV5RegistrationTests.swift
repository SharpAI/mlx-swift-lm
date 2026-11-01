// SharpAI fork overlay tests for the jina_embeddings_v5 creator and a tiny forward pass.
// Not part of upstream. Random weights only, no network.

import Foundation
import MLX
import MLXLMCommon
import XCTest

@testable import MLXLLM

final class SharpAIJinaEmbeddingsV5RegistrationTests: XCTestCase {

    private let tinyJSON = """
        {
          "model_type": "jina_embeddings_v5",
          "hidden_size": 8,
          "num_hidden_layers": 1,
          "intermediate_size": 16,
          "num_attention_heads": 2,
          "num_key_value_heads": 1,
          "head_dim": 4,
          "rms_norm_eps": 1e-06,
          "vocab_size": 32,
          "tie_word_embeddings": true
        }
        """

    func testCreatorIsRegisteredForModelType() {
        XCTAssertNotNil(SharpAIJinaEmbeddingsV5Overlay.creators["jina_embeddings_v5"])
    }

    func testCreatorBuildsModelFromValidConfig() throws {
        let creator = try XCTUnwrap(SharpAIJinaEmbeddingsV5Overlay.creators["jina_embeddings_v5"])
        let model = try creator(Data(tinyJSON.utf8))
        XCTAssertTrue(model is SharpAIJinaEmbeddingsV5Model)
    }

    func testCreatorRejectsInvalidConfigBeforeConstruction() throws {
        let creator = try XCTUnwrap(SharpAIJinaEmbeddingsV5Overlay.creators["jina_embeddings_v5"])
        XCTAssertThrowsError(try creator(Data(#"{ "hidden_size": 8 }"#.utf8)))
    }

    func testTinyForwardPassProducesVocabularyLogits() throws {
        let config = try JSONDecoder.json5().decode(
            SharpAIJinaEmbeddingsV5Configuration.self, from: Data(tinyJSON.utf8))
        let model = SharpAIJinaEmbeddingsV5Model(config)

        let tokens = MLXArray([1, 2, 3] as [Int32]).reshaped(1, 3)
        let logits = model.callAsFunction(tokens, cache: nil)

        XCTAssertEqual(logits.shape, [1, 3, 32])
    }

    func testUntiedConfigBuildsLMHead() throws {
        let untied = tinyJSON.replacingOccurrences(
            of: "\"tie_word_embeddings\": true", with: "\"tie_word_embeddings\": false")
        let config = try JSONDecoder.json5().decode(
            SharpAIJinaEmbeddingsV5Configuration.self, from: Data(untied.utf8))
        let model = SharpAIJinaEmbeddingsV5Model(config)

        XCTAssertNotNil(model.lmHead)
        let logits = model.callAsFunction(MLXArray([1, 2] as [Int32]).reshaped(1, 2), cache: nil)
        XCTAssertEqual(logits.shape, [1, 2, 32])
    }
}
