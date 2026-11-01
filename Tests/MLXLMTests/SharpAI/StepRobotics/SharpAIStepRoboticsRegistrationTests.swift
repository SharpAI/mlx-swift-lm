// SharpAI fork overlay tests for the step_robotics creator. Not part of upstream.
//
// Failure paths and the key set, plus one tiny valid build with random weights.

import Foundation
import MLXLMCommon
import XCTest

@testable import MLXVLM

final class SharpAIStepRoboticsRegistrationTests: XCTestCase {

    private let tinyJSON = """
        {
          "model_type": "step_robotics",
          "vision_config": {
            "width": 16, "layers": 1, "heads": 2, "patch_size": 14, "image_size": 28,
            "mlp_ratio": 2.0
          },
          "text_config": {
            "hidden_size": 32, "num_hidden_layers": 2, "num_attention_heads": 2,
            "intermediate_size": 48, "vocab_size": 64, "head_dim": 8, "share_q_dim": 16,
            "moe_num_experts": 4, "moe_top_k": 2, "moe_intermediate_size": 16,
            "share_expert_dim": 16, "moe_layers_enum": "1"
          }
        }
        """

    func testRegistersOnlyTheStepRoboticsKey() {
        XCTAssertEqual(Array(SharpAIStepRoboticsOverlay.creators.keys), ["step_robotics"])
    }

    func testCreatorBuildsTinyModelFromValidConfig() throws {
        let creator = try XCTUnwrap(SharpAIStepRoboticsOverlay.creators["step_robotics"])

        let model = try creator(Data(tinyJSON.utf8))

        XCTAssertTrue(model is SharpAIStepRoboticsModel)
    }

    func testCreatorRejectsUnsupportedActivationBeforeBuildingModel() throws {
        let creator = try XCTUnwrap(SharpAIStepRoboticsOverlay.creators["step_robotics"])
        let json = tinyJSON.replacingOccurrences(
            of: "\"mlp_ratio\": 2.0", with: "\"mlp_ratio\": 2.0, \"hidden_act\": \"swish_unknown\"")

        XCTAssertThrowsError(try creator(Data(json.utf8)))
    }
}
