// SharpAI fork overlay tests for the ColPali configuration. Not part of upstream.
//
// Offline checks only: inline JSON shaped like vidore/colpali-v1.3-hf, no weights.

import Foundation
import MLXLMCommon
import XCTest

@testable import MLXVLM

final class SharpAIColpaliConfigurationTests: XCTestCase {

    private let fixtureJSON = """
        {
          "architectures": ["ColPaliForRetrieval"],
          "embedding_dim": 128,
          "is_composition": false,
          "model_type": "colpali",
          "text_config": {
            "hidden_size": 2048,
            "intermediate_size": 16384,
            "model_type": "gemma",
            "num_attention_heads": 8,
            "num_hidden_layers": 18,
            "num_image_tokens": 1024,
            "num_key_value_heads": 1,
            "torch_dtype": "float32",
            "vocab_size": 257216
          },
          "torch_dtype": "bfloat16",
          "transformers_version": "4.48.0.dev0",
          "vlm_config": {
            "_name_or_path": "google/paligemma-3b-mix-448",
            "_vocab_size": 257216,
            "bos_token_id": 2,
            "eos_token_id": 1,
            "image_token_index": 257152,
            "model_type": "paligemma",
            "pad_token_id": 0,
            "text_config": {
              "hidden_size": 2048,
              "intermediate_size": 16384,
              "num_attention_heads": 8,
              "num_hidden_layers": 18,
              "num_image_tokens": 1024,
              "num_key_value_heads": 1,
              "torch_dtype": "float32",
              "vocab_size": 257216
            },
            "torch_dtype": "bfloat16",
            "vision_config": {
              "hidden_size": 1152,
              "image_size": 448,
              "intermediate_size": 4304,
              "num_attention_heads": 16,
              "num_hidden_layers": 27,
              "num_image_tokens": 1024,
              "patch_size": 14,
              "projection_dim": 2048,
              "projector_hidden_act": "gelu_fast",
              "vision_use_head": false
            }
          }
        }
        """

    private func decode(_ json: String) throws -> SharpAIColpaliConfiguration {
        try JSONDecoder.json5().decode(SharpAIColpaliConfiguration.self, from: Data(json.utf8))
    }

    func testDecodesNestedVLMConfigFromFixtureShape() throws {
        let config = try decode(fixtureJSON)

        XCTAssertEqual(config.modelType, "colpali")
        XCTAssertEqual(config.embeddingDimensions, 128)
        XCTAssertEqual(config.textConfiguration.hiddenSize, 2048)
        XCTAssertEqual(config.textConfiguration.hiddenLayers, 18)
        XCTAssertEqual(config.textConfiguration.kvHeads, 1)
        XCTAssertEqual(config.textConfiguration.resolvedHeadDim, 256)
        XCTAssertEqual(config.textConfiguration.vocabularySize, 257216)
        XCTAssertEqual(config.textConfiguration.rmsNormEps, 1e-6)
        XCTAssertEqual(config.textConfiguration.ropeTheta, 10_000)

        XCTAssertEqual(config.imageTokenIndex, 257152)
        XCTAssertEqual(config.padTokenId, 0)
        XCTAssertEqual(config.visionConfiguration.imageSize, 448)
        XCTAssertEqual(config.visionConfiguration.patchSize, 14)
        XCTAssertEqual(config.visionConfiguration.hiddenLayers, 27)
        XCTAssertEqual(config.visionConfiguration.projectionDimensions, 2048)
        XCTAssertEqual(config.visionConfiguration.channels, 3)
        XCTAssertEqual(config.visionConfiguration.layerNormEps, 1e-6)
    }

    func testFixtureShapeValidates() throws {
        try decode(fixtureJSON).validateModelConfiguration()
    }

    func testFallsBackToVLMNestedTextConfig() throws {
        let json = """
            {
              "model_type": "colpali",
              "embedding_dim": 16,
              "vlm_config": {
                "image_token_index": 60,
                "text_config": {
                  "hidden_size": 32, "num_hidden_layers": 2, "intermediate_size": 64,
                  "num_attention_heads": 2, "num_key_value_heads": 1, "vocab_size": 64
                },
                "vision_config": {
                  "hidden_size": 16, "num_hidden_layers": 1, "intermediate_size": 32,
                  "num_attention_heads": 2, "patch_size": 14, "image_size": 28,
                  "projection_dim": 32
                }
              }
            }
            """
        let config = try decode(json)

        XCTAssertEqual(config.textConfiguration.hiddenSize, 32)
        XCTAssertEqual(config.padTokenId, 0)
        try config.validateModelConfiguration()
    }

    func testEmbeddingDimensionDefaultsTo128() throws {
        let json = fixtureJSON.replacingOccurrences(
            of: "\"embedding_dim\": 128,", with: "")
        XCTAssertEqual(try decode(json).embeddingDimensions, 128)
    }

    func testRejectsProjectionThatDoesNotMatchTextWidth() throws {
        let json = fixtureJSON.replacingOccurrences(
            of: "\"projection_dim\": 2048,", with: "\"projection_dim\": 1024,")
        let config = try decode(json)

        XCTAssertThrowsError(try config.validateModelConfiguration()) { error in
            guard case SharpAIColpaliError.invalidConfiguration = error else {
                XCTFail("unexpected error: \(error)")
                return
            }
        }
    }

    func testRejectsNonPositiveEmbeddingDimension() throws {
        let json = fixtureJSON.replacingOccurrences(
            of: "\"embedding_dim\": 128,", with: "\"embedding_dim\": 0,")
        XCTAssertThrowsError(try decode(json).validateModelConfiguration())
    }
}
