// SharpAI fork overlay tests for the nemotron_h_mtp config decoding, creator and forward pass.
// Not part of upstream.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import XCTest

final class SharpAINemotronHMtpRegistrationTests: XCTestCase {

    private let validJSON = """
        {
            "model_type": "nemotron_h_mtp",
            "vocab_size": 100,
            "hidden_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "mamba_num_heads": 4,
            "mamba_head_dim": 16,
            "ssm_state_size": 16,
            "conv_kernel": 4,
            "n_groups": 2,
            "intermediate_size": 128,
            "moe_intermediate_size": 64,
            "moe_shared_expert_intermediate_size": 64,
            "n_routed_experts": 4,
            "num_experts_per_tok": 2,
            "hybrid_override_pattern": "M*",
            "layer_norm_epsilon": 1e-5,
            "n_group": 2,
            "topk_group": 1
        }
        """

    func testConfigurationDecodesModelType() throws {
        let config = try JSONDecoder.json5().decode(
            NemotronHConfiguration.self, from: Data(validJSON.utf8))

        XCTAssertEqual(config.modelType, "nemotron_h_mtp")
        XCTAssertEqual(config.vocabSize, 100)
        XCTAssertEqual(config.hybridOverridePattern, "M*")
    }

    func testCreatorIsRegisteredForModelType() {
        XCTAssertNotNil(SharpAINemotronHMtpOverlay.creators["nemotron_h_mtp"])
    }

    func testCreatorBuildsMTPModel() throws {
        let creator = try XCTUnwrap(SharpAINemotronHMtpOverlay.creators["nemotron_h_mtp"])
        let model = try creator(Data(validJSON.utf8))

        XCTAssertTrue(model is SharpAINemotronHMtpModel)
    }

    func testCreatorRejectsMalformedJSON() throws {
        let creator = try XCTUnwrap(SharpAINemotronHMtpOverlay.creators["nemotron_h_mtp"])

        XCTAssertThrowsError(try creator(Data("{\"model_type\": \"nemotron_h_mtp\"}".utf8)))
    }

    func testForwardPassWithRandomWeightsHasTinyShape() throws {
        let config = try JSONDecoder.json5().decode(
            NemotronHConfiguration.self, from: Data(validJSON.utf8))
        let model = SharpAINemotronHMtpModel(config)

        let input = MLXArray([1, 2, 3])[.newAxis, .ellipsis]
        let output = model.callAsFunction(input, cache: nil)

        XCTAssertEqual(output.shape, [1, 3, 100])
    }
}
