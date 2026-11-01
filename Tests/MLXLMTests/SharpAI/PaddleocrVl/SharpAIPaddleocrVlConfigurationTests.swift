// SharpAI fork overlay tests for the paddleocr_vl configuration. Not part of upstream.
// Inline JSON only, no network, no weights.
// UNVERIFIED: this file has not been compiled or run.

import Foundation
import MLXLMCommon
@testable import MLXVLM
import XCTest

final class SharpAIPaddleocrVlConfigurationTests: XCTestCase {

    /// Text fields sit at the top level, as in the real PaddleOCR-VL config.json.
    static let tinyJSON = """
        {
          "model_type": "paddleocr_vl",
          "image_token_id": 9,
          "vocab_size": 128,
          "hidden_size": 16,
          "num_hidden_layers": 2,
          "intermediate_size": 32,
          "num_attention_heads": 2,
          "num_key_value_heads": 1,
          "head_dim": 8,
          "rms_norm_eps": 1e-5,
          "rope_theta": 10000,
          "rope_scaling": { "rope_type": "default", "mrope_section": [1, 1, 2] },
          "tie_word_embeddings": true,
          "use_bias": false,
          "vision_config": {
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_channels": 3,
            "image_size": 56,
            "patch_size": 14,
            "spatial_merge_size": 2,
            "layer_norm_eps": 1e-6
          }
        }
        """

    /// Same field values as the repo fixture PaddlePaddle__PaddleOCR-VL.json.
    static let fixtureJSON = """
        {
          "model_type": "paddleocr_vl",
          "image_token_id": 100295,
          "vocab_size": 103424,
          "hidden_size": 1024,
          "num_hidden_layers": 18,
          "intermediate_size": 3072,
          "num_attention_heads": 16,
          "num_key_value_heads": 2,
          "head_dim": 128,
          "rms_norm_eps": 1e-05,
          "rope_theta": 500000,
          "rope_scaling": {
            "mrope_section": [16, 24, 24],
            "rope_type": "default",
            "type": "default"
          },
          "tie_word_embeddings": false,
          "use_bias": false,
          "vision_config": {
            "hidden_size": 1152,
            "image_size": 384,
            "intermediate_size": 4304,
            "layer_norm_eps": 1e-06,
            "num_attention_heads": 16,
            "num_channels": 3,
            "num_hidden_layers": 27,
            "patch_size": 14,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2
          }
        }
        """

    private func decode(_ json: String) throws -> SharpAIPaddleocrVlConfiguration {
        try JSONDecoder.json5().decode(
            SharpAIPaddleocrVlConfiguration.self, from: Data(json.utf8))
    }

    func testDecodesTextAndVisionFields() throws {
        let config = try decode(Self.tinyJSON)

        XCTAssertEqual(config.imageTokenId, 9)
        XCTAssertEqual(config.textConfiguration.vocabularySize, 128)
        XCTAssertEqual(config.textConfiguration.hiddenLayers, 2)
        XCTAssertEqual(config.textConfiguration.kvHeads, 1)
        XCTAssertEqual(config.textConfiguration.headDim, 8)
        XCTAssertEqual(config.textConfiguration.mropeSection, [1, 1, 2])
        XCTAssertEqual(config.visionConfiguration.patchesPerSide, 4)
        XCTAssertEqual(config.visionConfiguration.spatialMergeSize, 2)
    }

    func testDecodesFixtureLayout() throws {
        let config = try decode(Self.fixtureJSON)

        XCTAssertEqual(config.imageTokenId, 100_295)
        XCTAssertEqual(config.textConfiguration.vocabularySize, 103_424)
        XCTAssertEqual(config.textConfiguration.hiddenSize, 1024)
        XCTAssertEqual(config.textConfiguration.hiddenLayers, 18)
        XCTAssertEqual(config.textConfiguration.headDim, 128)
        XCTAssertEqual(config.textConfiguration.mropeSection, [16, 24, 24])
        XCTAssertFalse(config.textConfiguration.tieWordEmbeddings)
        XCTAssertEqual(config.visionConfiguration.hiddenSize, 1152)
        XCTAssertEqual(config.visionConfiguration.hiddenLayers, 27)
        XCTAssertEqual(config.visionConfiguration.patchesPerSide, 27)
    }

    func testFixtureLayoutValidates() throws {
        let config = try decode(Self.fixtureJSON)

        XCTAssertNoThrow(try config.validateModelConfiguration())
    }

    func testNestedTextConfigIsStillAccepted() throws {
        let json = """
            {
              "text_config": { "hidden_size": 16, "num_hidden_layers": 3, "head_dim": 8,
                "num_attention_heads": 2, "num_key_value_heads": 1,
                "rope_scaling": { "mrope_section": [1, 1, 2] } },
              "vision_config": {}
            }
            """
        let config = try decode(json)

        XCTAssertEqual(config.textConfiguration.hiddenSize, 16)
        XCTAssertEqual(config.textConfiguration.hiddenLayers, 3)
        XCTAssertEqual(config.textConfiguration.mropeSection, [1, 1, 2])
    }

    /// Fallbacks match the repo fixture PaddlePaddle__PaddleOCR-VL.json.
    func testMissingFieldsUseFallbackDefaults() throws {
        let config = try decode("{ \"vision_config\": {} }")

        XCTAssertEqual(config.imageTokenId, 100_295)
        XCTAssertEqual(config.textConfiguration.hiddenLayers, 18)
        XCTAssertEqual(config.textConfiguration.mropeSection, [16, 24, 24])
        XCTAssertEqual(config.textConfiguration.ropeTheta, 500_000)
        XCTAssertFalse(config.textConfiguration.tieWordEmbeddings)
        XCTAssertEqual(config.visionConfiguration.hiddenLayers, 27)
        XCTAssertEqual(config.visionConfiguration.patchesPerSide, 27)
    }

    func testValidationAcceptsTinyConfig() throws {
        let config = try decode(Self.tinyJSON)

        XCTAssertNoThrow(try config.validateModelConfiguration())
    }

    /// image_size 384 is not a multiple of patch 14, and that is valid (27x27 position grid).
    func testValidationAcceptsFixtureSizedConfig() throws {
        let config = try decode("{ \"vision_config\": {} }")

        XCTAssertNoThrow(try config.validateModelConfiguration())
    }

    func testValidationRejectsMropeSectionThatDoesNotMatchHeadDim() throws {
        // Sections sum to 3, but head_dim 8 needs 4 half-dim frequencies.
        let json = Self.tinyJSON.replacingOccurrences(
            of: "[1, 1, 2]", with: "[1, 1, 1]")
        let config = try decode(json)

        XCTAssertThrowsError(try config.validateModelConfiguration())
    }

    func testMissingVisionConfigFailsDecoding() {
        XCTAssertThrowsError(
            try decode("{ \"model_type\": \"paddleocr_vl\", \"hidden_size\": 16 }"))
    }
}
