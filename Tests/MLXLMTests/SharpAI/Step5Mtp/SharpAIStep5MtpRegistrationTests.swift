// SharpAI fork overlay tests for the step5_mtp creator. Not part of upstream.

import Foundation
import MLXLMCommon
import MLXVLM
import XCTest

final class SharpAIStep5MtpRegistrationTests: XCTestCase {

    private let validJSON = """
        {
            "model_type": "step5_mtp",
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_attention_heads": 8,
            "num_attention_groups": 2,
            "num_hidden_layers": 2,
            "num_nextn_predict_layers": 1,
            "vocab_size": 512,
            "layer_types_with_mtp": ["full_attention", "sliding_attention", "full_attention"]
        }
        """

    func testCreatorIsRegisteredForModelType() {
        XCTAssertNotNil(SharpAIStep5MtpOverlay.creators["step5_mtp"])
    }

    func testBaseStep5TypesAreNotRegistered() {
        XCTAssertNil(SharpAIStep5MtpOverlay.creators["step5"])
        XCTAssertNil(SharpAIStep5MtpOverlay.creators["step5_vl"])
    }

    func testValidConfigFailsUntilForwardIsImplemented() throws {
        guard let creator = SharpAIStep5MtpOverlay.creators["step5_mtp"] else {
            XCTFail("step5_mtp creator missing")
            return
        }

        XCTAssertThrowsError(try creator(Data(validJSON.utf8))) { error in
            guard case SharpAIStep5MtpError.forwardNotImplemented(let type) = error else {
                XCTFail("unexpected error: \(error)")
                return
            }
            XCTAssertEqual(type, "step5_mtp")
        }
    }

    func testInvalidConfigFailsValidationBeforeForward() throws {
        guard let creator = SharpAIStep5MtpOverlay.creators["step5_mtp"] else {
            XCTFail("step5_mtp creator missing")
            return
        }
        let noMTPJSON = #"{"model_type": "step5_mtp", "hidden_size": 64}"#

        XCTAssertThrowsError(try creator(Data(noMTPJSON.utf8))) { error in
            guard case ModelFactoryError.invalidConfiguration = error else {
                XCTFail("expected validation error, got \(error)")
                return
            }
        }
    }

    /// No model class exists yet, so the tiny forward pass cannot run. Sources read so far cover
    /// the MTP wrapper (vLLM PR #61837, amd/mtp.py) but not the full Step5 decoder block: CSA
    /// sparse attention (csa_attention.py), the sparse indexer math and the MoE router. That keeps
    /// this a known gap, reported as an expected failure until the forward pass exists.
    func testTinyShapeForwardPassIsBlocked() throws {
        guard let creator = SharpAIStep5MtpOverlay.creators["step5_mtp"] else {
            XCTFail("step5_mtp creator missing")
            return
        }

        try XCTExpectFailure(
            "KNOWN GAP: step5_mtp forward pass needs the Step5 decoder block (Step5DecoderLayer used as mtp_block). Missing artifact: the verbatim source of vllm/models/step5/amd/csa_attention.py (vllm-project/vllm PR #61837) plus the Step5 sparse indexer and MoE router. WebFetch of raw.githubusercontent.com returned only a summary (it refused verbatim reproduction), and the CSA path depends on Triton/SM90 kernels (paged_mqa_logits_sm90, sparse_attention_paged) with no MLX equivalent in this repo. No transformers modeling_step5 source was found."
        ) {
            XCTAssertNoThrow(try creator(Data(validJSON.utf8)))
        }
    }
}
