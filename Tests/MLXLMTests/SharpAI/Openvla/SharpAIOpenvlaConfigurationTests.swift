// SharpAI fork overlay tests for the OpenVLA configuration. Not part of upstream.
//
// Offline checks only: inline JSON shaped like openvla/openvla-7b, no weights.

import Foundation
import MLXLMCommon
import XCTest

@testable import MLXVLM

final class SharpAIOpenvlaConfigurationTests: XCTestCase {

    /// Shaped like the OpenVLA-7B fixture. `text_config` lacks the Llama sizes, so defaults apply.
    private let fixtureJSON = """
        {
          "architectures": ["OpenVLAForActionPrediction"],
          "arch_specifier": "no-align+fused-gelu-mlp",
          "image_sizes": [224, 224],
          "model_type": "openvla",
          "n_action_bins": 256,
          "pad_token_id": 32000,
          "text_config": {
            "model_type": "llama",
            "pad_token_id": 32000,
            "torch_dtype": "bfloat16",
            "vocab_size": 32064
          },
          "use_fused_vision_backbone": true,
          "vision_backbone_id": "dinosiglip-vit-so-224px"
        }
        """

    func testDecodesFixtureShapeWithLlama2Defaults() throws {
        let config = try JSONDecoder.json5().decode(
            SharpAIOpenvlaConfiguration.self, from: Data(fixtureJSON.utf8))
        try config.validateModelConfiguration()

        XCTAssertEqual(config.modelType, "openvla")
        XCTAssertEqual(config.padTokenId, 32000)
        XCTAssertEqual(config.visionBackboneId, "dinosiglip-vit-so-224px")
        XCTAssertEqual(config.archSpecifier, "no-align+fused-gelu-mlp")
        XCTAssertEqual(config.imageSizes, [224, 224])

        let text = config.textConfiguration
        XCTAssertEqual(text.vocabularySize, 32064)
        // UNVERIFIED: these defaults are Llama-2-7B values, not read from the fixture.
        XCTAssertEqual(text.hiddenSize, 4096)
        XCTAssertEqual(text.hiddenLayers, 32)
        XCTAssertEqual(text.intermediateSize, 11008)
        XCTAssertEqual(text.attentionHeads, 32)
        XCTAssertEqual(text.kvHeads, 32)
        XCTAssertEqual(text.resolvedHeadDim, 128)
        XCTAssertEqual(text.ropeTheta, 10_000)
        XCTAssertEqual(text.rmsNormEps, 1e-6, accuracy: 1e-12)
    }

    func testDecodesExplicitTextConfiguration() throws {
        let json = """
            {
              "model_type": "openvla",
              "text_config": {
                "hidden_size": 32,
                "num_hidden_layers": 2,
                "intermediate_size": 64,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "vocab_size": 64,
                "rms_norm_eps": 0.000001
              }
            }
            """
        let config = try JSONDecoder.json5().decode(
            SharpAIOpenvlaConfiguration.self, from: Data(json.utf8))
        try config.validateModelConfiguration()

        let text = config.textConfiguration
        XCTAssertEqual(text.hiddenSize, 32)
        XCTAssertEqual(text.hiddenLayers, 2)
        XCTAssertEqual(text.attentionHeads, 4)
        XCTAssertEqual(text.kvHeads, 2)
        XCTAssertEqual(text.vocabularySize, 64)
        XCTAssertEqual(text.resolvedHeadDim, 8)
        XCTAssertEqual(text.rmsNormEps, 1e-6, accuracy: 1e-12)
    }

    func testRejectsUnsupportedArchSpecifier() throws {
        let json = fixtureJSON.replacingOccurrences(
            of: "no-align+fused-gelu-mlp", with: "linear-projector")
        let config = try JSONDecoder.json5().decode(
            SharpAIOpenvlaConfiguration.self, from: Data(json.utf8))

        XCTAssertThrowsError(try config.validateModelConfiguration()) { error in
            guard case SharpAIOpenvlaError.invalidConfiguration = error else {
                return XCTFail("unexpected error \(error)")
            }
        }
    }

    func testRejectsUnsupportedVisionBackbone() throws {
        let json = fixtureJSON.replacingOccurrences(
            of: "dinosiglip-vit-so-224px", with: "clip-vit-l-336px")
        let config = try JSONDecoder.json5().decode(
            SharpAIOpenvlaConfiguration.self, from: Data(json.utf8))

        XCTAssertThrowsError(try config.validateModelConfiguration())
    }

    func testRejectsKVHeadsThatDoNotDivideAttentionHeads() throws {
        let json = """
            {
              "model_type": "openvla",
              "text_config": {
                "hidden_size": 32,
                "num_hidden_layers": 2,
                "intermediate_size": 64,
                "num_attention_heads": 4,
                "num_key_value_heads": 3,
                "vocab_size": 64
              }
            }
            """
        let config = try JSONDecoder.json5().decode(
            SharpAIOpenvlaConfiguration.self, from: Data(json.utf8))

        XCTAssertThrowsError(try config.validateModelConfiguration())
    }
}
