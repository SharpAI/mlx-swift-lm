// SharpAI fork overlay tests for the deepseek_vl_v2 creator. Not part of upstream.

import Foundation
import MLXLMCommon
import MLXVLM
import XCTest

final class SharpAIDeepseekVlV2RegistrationTests: XCTestCase {

    private let validJSON = """
        {
            "model_type": "deepseek_vl_v2",
            "language_config": {
                "model_type": "deepseek_v2",
                "vocab_size": 1024,
                "hidden_size": 256,
                "num_hidden_layers": 2,
                "num_attention_heads": 8,
                "num_key_value_heads": 8,
                "kv_lora_rank": null
            }
        }
        """

    func testCreatorIsRegisteredForModelType() {
        XCTAssertNotNil(SharpAIDeepseekVlV2Overlay.creators["deepseek_vl_v2"])
    }

    func testValidConfigFailsUntilForwardIsImplemented() throws {
        guard let creator = SharpAIDeepseekVlV2Overlay.creators["deepseek_vl_v2"] else {
            XCTFail("deepseek_vl_v2 creator missing")
            return
        }

        XCTAssertThrowsError(try creator(Data(validJSON.utf8))) { error in
            guard case SharpAIDeepseekVlV2Error.forwardNotImplemented(let type) = error else {
                XCTFail("unexpected error: \(error)")
                return
            }
            XCTAssertEqual(type, "deepseek_vl_v2")
        }
    }

    func testInvalidConfigFailsBeforeConstruction() throws {
        guard let creator = SharpAIDeepseekVlV2Overlay.creators["deepseek_vl_v2"] else {
            XCTFail("deepseek_vl_v2 creator missing")
            return
        }

        // Flat config (no language_config). 8 attention heads are not divisible by 3 KV heads,
        // so validateModelConfiguration() must throw before the forward-pass error is reached.
        let invalidJSON = """
            { "num_attention_heads": 8, "num_key_value_heads": 3 }
            """

        XCTAssertThrowsError(try creator(Data(invalidJSON.utf8))) { error in
            guard case ModelFactoryError.invalidConfiguration = error else {
                XCTFail("expected invalidConfiguration from validation, got: \(error)")
                return
            }
        }
    }

    func testTinyForwardPassKnownGap() {
        guard let creator = SharpAIDeepseekVlV2Overlay.creators["deepseek_vl_v2"] else {
            XCTFail("deepseek_vl_v2 creator missing")
            return
        }

        // Reference found: DeepSeek-VL2 config (deepseek-ai/deepseek-vl2-tiny) and
        // modeling_deepseek_vl_v2.py (deepseek-ai/DeepSeek-VL2). The vision tower is a
        // SigLIP VisionTransformer, not SAM. The text stack needs non-MLA attention
        // (kv_lora_rank = null), which DeepseekV2 in this repo does not build.
        XCTExpectFailure(
            "KNOWN GAP: deepseek_vl_v2 model class not written; needs the SigLIP vision tower, "
                + "MlpProjector, and non-MLA DeepseekV2 attention (kv_lora_rank null). "
                + "Reference: deepseek-ai/DeepSeek-VL2 modeling_deepseek_vl_v2.py"
        ) {
            let model = try? creator(Data(validJSON.utf8))
            XCTAssertNotNil(model)
        }
    }
}
