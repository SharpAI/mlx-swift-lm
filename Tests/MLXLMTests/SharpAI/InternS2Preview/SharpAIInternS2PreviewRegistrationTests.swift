// SharpAI fork overlay tests for the intern_s2_preview creator. Not part of upstream.

import Foundation
import MLXLMCommon
import MLXVLM
import XCTest

final class SharpAIInternS2PreviewRegistrationTests: XCTestCase {

    private let validJSON = """
        {
            "model_type": "intern_s2_preview",
            "architectures": ["InternS2PreviewForConditionalGeneration"],
            "vocab_size": 512,
            "text_config": {
                "model_type": "qwen3_5_moe_text", "hidden_size": 64,
                "num_hidden_layers": 2, "intermediate_size": 128,
                "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 32,
                "vocab_size": 512, "full_attention_interval": 2,
                "linear_num_value_heads": 2, "linear_num_key_heads": 1,
                "linear_key_head_dim": 32, "linear_value_head_dim": 32,
                "linear_conv_kernel_dim": 4, "num_experts": 0, "num_experts_per_tok": 0,
                "max_position_embeddings": 4096,
                "rope_parameters": {"mrope_section": [8,4,4], "rope_theta": 100000,
                    "partial_rotary_factor": 1.0, "type": "default"}
            },
            "vision_config": {"model_type": "qwen3_vl", "depth": 1,
                "hidden_size": 32, "intermediate_size": 64, "out_hidden_size": 64,
                "num_heads": 2, "patch_size": 16, "spatial_merge_size": 2,
                "temporal_patch_size": 2, "num_position_embeddings": 64}
        }
        """

    func testCreatorIsRegisteredForModelType() {
        XCTAssertNotNil(SharpAIInternS2PreviewOverlay.creators["intern_s2_preview"])
    }

    func testValidConfigBuildsInternS2PreviewModel() throws {
        guard let creator = SharpAIInternS2PreviewOverlay.creators["intern_s2_preview"] else {
            XCTFail("intern_s2_preview creator missing")
            return
        }

        let model = try creator(Data(validJSON.utf8))

        XCTAssertTrue(model is SharpAIInternS2PreviewModel)
    }

    func testWrongModelTypeFailsBeforeConstruction() throws {
        guard let creator = SharpAIInternS2PreviewOverlay.creators["intern_s2_preview"] else {
            XCTFail("intern_s2_preview creator missing")
            return
        }
        let json = """
            { "model_type": "llama" }
            """

        XCTAssertThrowsError(try creator(Data(json.utf8))) { error in
            guard case ModelFactoryError.invalidConfiguration = error else {
                XCTFail("unexpected error: \(error)")
                return
            }
        }
    }
}
