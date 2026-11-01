// SharpAI fork overlay tests for the ModernVBERT configuration. Not part of upstream.
//
// Offline checks only: inline JSON, no weights, no network.

import Foundation
import MLXLMCommon
import XCTest

@testable import MLXVLM

final class SharpAIModernvbertConfigurationTests: XCTestCase {

    /// Text width 16, vision width 8, 32x32 image, 8-pixel patches, 2x2 pixel shuffle.
    private func configJSON(
        architectures: String = "\"ColModernVBertForRetrieval\"",
        imageSize: Int = 32,
        shuffleFactor: Int = 2
    ) -> String {
        """
        {
          "architectures": [\(architectures)],
          "model_type": "modernvbert",
          "embedding_dim": 16,
          "image_token_id": 60,
          "pixel_shuffle_factor": \(shuffleFactor),
          "text_config": {
            "vocab_size": 64,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 3,
            "num_attention_heads": 2,
            "global_attn_every_n_layers": 2,
            "local_attention": 4,
            "global_rope_theta": 160000.0,
            "local_rope_theta": 10000.0
          },
          "vision_config": {
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "patch_size": 8,
            "image_size": \(imageSize),
            "num_channels": 3
          }
        }
        """
    }

    private func decode(_ json: String) throws -> SharpAIModernvbertConfiguration {
        try JSONDecoder.json5().decode(SharpAIModernvbertConfiguration.self, from: Data(json.utf8))
    }

    func testDecodesTextVisionAndTopLevelFields() throws {
        let config = try decode(configJSON())

        XCTAssertEqual(config.modelType, "modernvbert")
        XCTAssertEqual(config.imageTokenIndex, 60)
        XCTAssertEqual(config.pixelShuffleFactor, 2)
        XCTAssertEqual(config.embeddingDimensions, 16)
        XCTAssertTrue(config.hasRetrievalHead)

        XCTAssertEqual(config.textConfiguration.vocabularySize, 64)
        XCTAssertEqual(config.textConfiguration.hiddenSize, 16)
        XCTAssertEqual(config.textConfiguration.hiddenLayers, 3)
        XCTAssertEqual(config.textConfiguration.headDim, 8)
        XCTAssertEqual(config.textConfiguration.localAttention, 4)
        XCTAssertEqual(config.textConfiguration.globalAttnEveryNLayers, 2)
        XCTAssertEqual(config.textConfiguration.globalRopeTheta, 160_000, accuracy: 1)
        XCTAssertEqual(config.textConfiguration.localRopeTheta, 10_000, accuracy: 1)
        XCTAssertFalse(config.textConfiguration.attentionBias)
        XCTAssertFalse(config.textConfiguration.mlpBias)

        XCTAssertEqual(config.visionConfiguration.hiddenSize, 8)
        XCTAssertEqual(config.visionConfiguration.patchSize, 8)
        XCTAssertEqual(config.visionConfiguration.channels, 3)
        XCTAssertEqual(config.visionConfiguration.layerNormEps, 1e-6, accuracy: 1e-9)
    }

    func testImageTokenCountFollowsPatchGridAndShuffleFactor() throws {
        // 32 / 8 = 4 patches per side, shuffled 2x2 gives 2 per side, so 4 image tokens.
        let config = try decode(configJSON())
        XCTAssertEqual(config.imageTokenCount, 4)
    }

    func testPlainModernVBertModelHasNoRetrievalHead() throws {
        let config = try decode(configJSON(architectures: "\"ModernVBertModel\""))
        XCTAssertFalse(config.hasRetrievalHead)
    }

    func testMissingArchitecturesDefaultsToRetrievalHead() throws {
        let json = configJSON().replacingOccurrences(
            of: "\"architectures\": [\"ColModernVBertForRetrieval\"],", with: "")
        let config = try decode(json)
        XCTAssertTrue(config.hasRetrievalHead)
    }

    func testDefaultsForOptionalTopLevelFields() throws {
        let json = """
            {
              "image_token_id": 60,
              "text_config": {
                "vocab_size": 64, "hidden_size": 16, "intermediate_size": 32,
                "num_hidden_layers": 2, "num_attention_heads": 2
              },
              "vision_config": {
                "hidden_size": 8, "intermediate_size": 16, "num_hidden_layers": 1,
                "num_attention_heads": 2, "patch_size": 8, "image_size": 32
              }
            }
            """
        let config = try decode(json)

        XCTAssertEqual(config.modelType, "modernvbert")
        XCTAssertEqual(config.pixelShuffleFactor, 4)
        XCTAssertEqual(config.embeddingDimensions, 128)
        XCTAssertEqual(config.textConfiguration.normEps, 1e-5, accuracy: 1e-12)
        XCTAssertEqual(config.textConfiguration.localAttention, 128)
        XCTAssertEqual(config.textConfiguration.globalAttnEveryNLayers, 3)
        XCTAssertEqual(config.textConfiguration.globalRopeTheta, 160_000, accuracy: 1)
        XCTAssertEqual(config.visionConfiguration.channels, 3)
    }

    func testValidationAcceptsConsistentConfig() throws {
        try decode(configJSON()).validateModelConfiguration()
    }

    func testValidationRejectsImageSizeNotDivisibleByPatch() throws {
        let config = try decode(configJSON(imageSize: 30))
        XCTAssertThrowsError(try config.validateModelConfiguration()) { error in
            guard case SharpAIModernvbertError.invalidConfiguration = error else {
                return XCTFail("unexpected error \(error)")
            }
        }
    }

    func testValidationRejectsGridNotDivisibleByShuffleFactor() throws {
        // 32 / 8 = 4 patches per side, which 3 does not divide.
        let config = try decode(configJSON(shuffleFactor: 3))
        XCTAssertThrowsError(try config.validateModelConfiguration())
    }
}
