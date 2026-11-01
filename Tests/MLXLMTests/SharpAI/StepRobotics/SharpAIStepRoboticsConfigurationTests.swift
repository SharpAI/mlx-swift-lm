// SharpAI fork overlay tests for the StepVL (step_robotics) configuration. Not part of upstream.
//
// Offline checks only: inline JSON, no weights, no network.
// UNVERIFIED: the vision fields are shaped by the Perception Encoder, not by a checkpoint.

import Foundation
import MLXLMCommon
import XCTest

@testable import MLXVLM

final class SharpAIStepRoboticsConfigurationTests: XCTestCase {

    private let tinyJSON = """
        {
          "model_type": "step_robotics",
          "projector_bias": true,
          "image_token_id": 63,
          "vision_config": {
            "width": 16,
            "layers": 1,
            "heads": 2,
            "patch_size": 14,
            "image_size": 28,
            "mlp_ratio": 2.0,
            "hidden_act": "gelu"
          },
          "text_config": {
            "hidden_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "intermediate_size": 48,
            "vocab_size": 64,
            "head_dim": 8,
            "share_q_dim": 16,
            "rms_norm_eps": 1e-6,
            "moe_num_experts": 4,
            "moe_top_k": 2,
            "moe_intermediate_size": 16,
            "share_expert_dim": 16,
            "moe_layers_enum": "1"
          }
        }
        """

    private func decode(_ json: String) throws -> SharpAIStepRoboticsConfiguration {
        try JSONDecoder.json5().decode(
            SharpAIStepRoboticsConfiguration.self, from: Data(json.utf8))
    }

    func testDecodesTinyConfigAndValidates() throws {
        let config = try decode(tinyJSON)
        try config.validateModelConfiguration()

        XCTAssertEqual(config.modelType, "step_robotics")
        XCTAssertTrue(config.projectorBias)
        XCTAssertEqual(config.imageTokenId, 63)

        let text = config.textConfiguration
        XCTAssertEqual(text.hiddenSize, 32)
        XCTAssertEqual(text.hiddenLayers, 2)
        XCTAssertEqual(text.resolvedQDim, 16)
        XCTAssertEqual(text.headDim, 8)
        XCTAssertEqual(text.moeExperts, 4)
        XCTAssertEqual(text.moeTopK, 2)
        XCTAssertEqual(text.moeLayers, [1])
        XCTAssertEqual(text.ropeType, "default")
        // Not in the JSON, so the Step3TextConfig class default applies.
        XCTAssertEqual(text.ropeTheta, 500_000)

        let vision = config.visionConfiguration
        XCTAssertEqual(vision.width, 16)
        XCTAssertEqual(vision.heads, 2)
        XCTAssertEqual(vision.patchSize, 14)
        XCTAssertEqual(vision.imageSize, 28)
        XCTAssertEqual(vision.mlpRatio, 2.0)
        XCTAssertNil(vision.layerScaleInit)
        XCTAssertFalse(vision.useClsToken)
        XCTAssertTrue(vision.useRope2d)
    }

    func testMoELayersAcceptListForm() throws {
        let json = tinyJSON.replacingOccurrences(
            of: "\"moe_layers_enum\": \"1\"", with: "\"moe_layers_enum\": [0, 1]")
        let config = try decode(json)

        XCTAssertEqual(config.textConfiguration.moeLayers, [0, 1])
    }

    func testRopeThetaReadsFromRopeParameters() throws {
        let json = tinyJSON.replacingOccurrences(
            of: "\"share_expert_dim\": 16,",
            with: "\"share_expert_dim\": 16, \"rope_parameters\": {\"rope_type\": \"default\", \"rope_theta\": 1000},")
        let config = try decode(json)
        try config.validateModelConfiguration()

        XCTAssertEqual(config.textConfiguration.ropeTheta, 1000)
    }

    func testRejectsScaledRope() throws {
        let json = tinyJSON.replacingOccurrences(
            of: "\"share_expert_dim\": 16,",
            with: "\"share_expert_dim\": 16, \"rope_scaling\": {\"rope_type\": \"linear\", \"factor\": 2.0},")
        let config = try decode(json)

        XCTAssertThrowsError(try config.validateModelConfiguration()) { error in
            guard case SharpAIStepRoboticsError.invalidConfiguration = error else {
                return XCTFail("unexpected error \(error)")
            }
        }
    }

    func testRejectsUnsupportedVisionActivation() throws {
        let json = tinyJSON.replacingOccurrences(
            of: "\"hidden_act\": \"gelu\"", with: "\"hidden_act\": \"swish_unknown\"")
        let config = try decode(json)

        XCTAssertThrowsError(try config.validateModelConfiguration())
    }

    func testRejectsOddTextHeadDim() throws {
        let json = tinyJSON.replacingOccurrences(of: "\"head_dim\": 8", with: "\"head_dim\": 7")
        let config = try decode(json)

        XCTAssertThrowsError(try config.validateModelConfiguration())
    }

    func testRejectsVisionHeadDimNotDivisibleByFour() throws {
        // width 12 over 2 heads gives head_dim 6, which 2D RoPE cannot split into quarters.
        let json = tinyJSON.replacingOccurrences(of: "\"width\": 16", with: "\"width\": 12")
        let config = try decode(json)

        XCTAssertThrowsError(try config.validateModelConfiguration())
    }

    func testRejectsTopKLargerThanExpertCount() throws {
        let json = tinyJSON.replacingOccurrences(
            of: "\"moe_top_k\": 2", with: "\"moe_top_k\": 5")
        let config = try decode(json)

        XCTAssertThrowsError(try config.validateModelConfiguration())
    }
}
