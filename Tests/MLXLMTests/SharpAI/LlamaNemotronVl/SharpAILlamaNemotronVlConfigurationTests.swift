// SharpAI fork overlay tests for the LlamaNemotronVL config. Not part of upstream.
//
// Offline checks only: inline JSON, no checkpoint is read.

import Foundation
import MLXLMCommon
import XCTest

@testable import MLXVLM

final class SharpAILlamaNemotronVlConfigurationTests: XCTestCase {

    private let tinyJSON = """
        {
          "model_type": "llama_nemotron_vl",
          "llm_config": {
            "hidden_size": 32,
            "num_hidden_layers": 2,
            "intermediate_size": 64,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "vocab_size": 64,
            "rope_theta": 500000,
            "rope_scaling": {"type": "linear", "factor": 2.0}
          },
          "vision_config": {
            "hidden_size": 16,
            "num_hidden_layers": 1,
            "intermediate_size": 32,
            "num_attention_heads": 2,
            "patch_size": 14,
            "image_size": 28
          },
          "downsample_ratio": 0.5,
          "ps_version": "v2",
          "img_context_token_id": 60
        }
        """

    private func decode(_ json: String) throws -> SharpAILlamaNemotronVlConfiguration {
        try JSONDecoder.json5().decode(
            SharpAILlamaNemotronVlConfiguration.self, from: Data(json.utf8))
    }

    func testDecodesTextVisionAndTopLevelFields() throws {
        let config = try decode(tinyJSON)

        XCTAssertEqual(config.modelType, "llama_nemotron_vl")
        XCTAssertEqual(config.textConfiguration.hiddenSize, 32)
        XCTAssertEqual(config.textConfiguration.hiddenLayers, 2)
        XCTAssertEqual(config.textConfiguration.kvHeads, 1)
        XCTAssertEqual(config.textConfiguration.headDim, 16)
        XCTAssertEqual(config.textConfiguration.ropeTheta, 500_000)
        XCTAssertEqual(config.textConfiguration.ropeScaling?["type"], .string("linear"))
        XCTAssertEqual(config.visionConfiguration.hiddenSize, 16)
        XCTAssertEqual(config.visionConfiguration.patchSize, 14)
        XCTAssertEqual(config.visionConfiguration.channels, 3)
        XCTAssertEqual(config.downsampleRatio, 0.5)
        XCTAssertEqual(config.psVersion, "v2")
        XCTAssertEqual(config.imageTokenId, 60)
        XCTAssertEqual(config.pixelShuffleFactor, 2)
    }

    func testTextConfigKeyIsAcceptedAsFallback() throws {
        let json = tinyJSON.replacingOccurrences(of: "\"llm_config\"", with: "\"text_config\"")
        let config = try decode(json)

        XCTAssertEqual(config.textConfiguration.vocabularySize, 64)
    }

    func testDefaultsApplyWhenOptionalKeysAreAbsent() throws {
        let json = """
            {
              "text_config": {
                "hidden_size": 32,
                "num_hidden_layers": 2,
                "intermediate_size": 64,
                "num_attention_heads": 4,
                "vocab_size": 64
              },
              "vision_config": {
                "hidden_size": 16,
                "num_hidden_layers": 1,
                "intermediate_size": 32,
                "num_attention_heads": 2,
                "patch_size": 14,
                "image_size": 28
              },
              "img_context_token_id": 60
            }
            """
        let config = try decode(json)

        XCTAssertEqual(config.textConfiguration.kvHeads, 4)
        XCTAssertEqual(config.textConfiguration.ropeTheta, 10_000)
        XCTAssertEqual(config.textConfiguration.rmsNormEps, 1e-5)
        XCTAssertEqual(config.downsampleRatio, 0.5)
        XCTAssertEqual(config.psVersion, "v2")
        XCTAssertEqual(config.visionConfiguration.layerNormEps, 1e-6)
        XCTAssertNoThrow(try config.validateModelConfiguration())
    }

    func testMissingImageTokenIdFailsToDecode() {
        let json = tinyJSON.replacingOccurrences(
            of: "\"img_context_token_id\": 60", with: "\"unused\": 60")

        XCTAssertThrowsError(try decode(json))
    }

    func testNonIntegerReciprocalDownsampleRatioFailsValidation() throws {
        let json = tinyJSON.replacingOccurrences(
            of: "\"downsample_ratio\": 0.5", with: "\"downsample_ratio\": 0.3")
        let config = try decode(json)

        XCTAssertThrowsError(try config.validateModelConfiguration()) { error in
            guard case SharpAILlamaNemotronVlError.invalidConfiguration = error else {
                XCTFail("unexpected error: \(error)")
                return
            }
        }
    }

    func testHeadCountNotMultipleOfKVHeadsFailsValidation() throws {
        let json = tinyJSON.replacingOccurrences(
            of: "\"num_key_value_heads\": 1", with: "\"num_key_value_heads\": 3")
        let config = try decode(json)

        XCTAssertThrowsError(try config.validateModelConfiguration()) { error in
            guard case SharpAILlamaNemotronVlError.invalidConfiguration = error else {
                XCTFail("unexpected error: \(error)")
                return
            }
        }
    }
}
