// SharpAI fork overlay tests for the jina_embeddings_v5 configuration. Not part of upstream.

import Foundation
import MLXLMCommon
import XCTest

@testable import MLXLLM

final class SharpAIJinaEmbeddingsV5ConfigurationTests: XCTestCase {

    // Field values from the offline fixture jinaai/jina-embeddings-v5-text-small.
    private let fixtureJSON = """
        {
          "architectures": ["JinaEmbeddingsV5Model"],
          "task_names": ["retrieval", "text-matching", "clustering", "classification"],
          "attention_bias": false,
          "attention_dropout": 0.0,
          "bos_token_id": 151643,
          "eos_token_id": 151645,
          "head_dim": 128,
          "hidden_act": "silu",
          "hidden_size": 1024,
          "initializer_range": 0.02,
          "intermediate_size": 3072,
          "max_position_embeddings": 32768,
          "max_window_layers": 28,
          "model_type": "jina_embeddings_v5",
          "num_attention_heads": 16,
          "num_hidden_layers": 28,
          "num_key_value_heads": 8,
          "rms_norm_eps": 1e-06,
          "rope_scaling": null,
          "rope_theta": 3500000,
          "sliding_window": null,
          "tie_word_embeddings": true,
          "use_cache": true,
          "use_sliding_window": false,
          "vocab_size": 151936
        }
        """

    func testDecodesFixtureBackboneFields() throws {
        let config = try JSONDecoder.json5().decode(
            SharpAIJinaEmbeddingsV5Configuration.self, from: Data(fixtureJSON.utf8))

        XCTAssertEqual(config.base.hiddenSize, 1024)
        XCTAssertEqual(config.base.hiddenLayers, 28)
        XCTAssertEqual(config.base.intermediateSize, 3072)
        XCTAssertEqual(config.base.attentionHeads, 16)
        XCTAssertEqual(config.base.kvHeads, 8)
        XCTAssertEqual(config.base.headDim, 128)
        XCTAssertEqual(config.base.vocabularySize, 151936)
        XCTAssertEqual(config.base.ropeTheta, 3_500_000)
        XCTAssertTrue(config.base.tieWordEmbeddings)
        XCTAssertNil(config.base.ropeScaling)
    }

    func testDecodesTaskNamesAndDefaultsJinaTask() throws {
        let config = try JSONDecoder.json5().decode(
            SharpAIJinaEmbeddingsV5Configuration.self, from: Data(fixtureJSON.utf8))

        XCTAssertEqual(
            config.taskNames, ["retrieval", "text-matching", "clustering", "classification"])
        XCTAssertEqual(config.jinaTask, "retrieval")
    }

    func testTaskNamesFallBackToSupportedTasksWhenAbsent() throws {
        let json = """
            {
              "model_type": "jina_embeddings_v5",
              "hidden_size": 8,
              "num_hidden_layers": 1,
              "intermediate_size": 16,
              "num_attention_heads": 2,
              "num_key_value_heads": 1,
              "head_dim": 4,
              "rms_norm_eps": 1e-06,
              "vocab_size": 32
            }
            """
        let config = try JSONDecoder.json5().decode(
            SharpAIJinaEmbeddingsV5Configuration.self, from: Data(json.utf8))

        XCTAssertEqual(config.taskNames, SharpAIJinaEmbeddingsV5Adapter.supportedTasks)
        XCTAssertFalse(config.base.tieWordEmbeddings)
    }

    func testMissingBackboneFieldFailsDecoding() {
        let json = """
            { "model_type": "jina_embeddings_v5", "hidden_size": 8 }
            """
        XCTAssertThrowsError(
            try JSONDecoder.json5().decode(
                SharpAIJinaEmbeddingsV5Configuration.self, from: Data(json.utf8)))
    }
}
