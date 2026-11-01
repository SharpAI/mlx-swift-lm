// SharpAI fork overlay tests for the interns2_mobius creator. Not part of upstream.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXVLM
import XCTest

final class SharpAIInterns2MobiusRegistrationTests: XCTestCase {

    private let validJSON = """
        {
            "model_type": "interns2_mobius",
            "architectures": ["InternS2MobiusForConditionalGeneration"],
            "mtp_num_experts": 8
        }
        """

    /// A tiny text_config with the same layout as the real model: one hybrid period of
    /// linear_attention layers and one full_attention layer, a 2-bank meta_mlp, and partial RoPE.
    /// The real 40-layer, 2560-expert config is far too large to build inside a unit test.
    private let tinyJSON = """
        {
            "model_type": "interns2_mobius",
            "architectures": ["InternS2MobiusForConditionalGeneration"],
            "text_config": {
                "model_type": "interns2_mobius_text",
                "vocab_size": 64,
                "hidden_size": 32,
                "num_hidden_layers": 4,
                "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"],
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 16,
                "rms_norm_eps": 1e-6,
                "linear_num_key_heads": 2,
                "linear_num_value_heads": 4,
                "linear_key_head_dim": 8,
                "linear_value_head_dim": 8,
                "linear_conv_kernel_dim": 4,
                "num_experts": 4,
                "num_experts_per_tok": 2,
                "moe_intermediate_size": 8,
                "shared_expert_intermediate_size": 8,
                "num_blocks": 2,
                "mtp_num_experts": 4,
                "rope_parameters": {
                    "rope_type": "default",
                    "rope_theta": 10000000,
                    "partial_rotary_factor": 0.25
                }
            }
        }
        """

    func testCreatorIsRegisteredForModelType() {
        XCTAssertNotNil(SharpAIInterns2MobiusOverlay.creators["interns2_mobius"])
    }

    func testValidConfigBuildsTextModel() throws {
        guard let creator = SharpAIInterns2MobiusOverlay.creators["interns2_mobius"] else {
            XCTFail("interns2_mobius creator missing")
            return
        }

        let model = try creator(Data(tinyJSON.utf8))
        XCTAssertTrue(model is SharpAIInterns2MobiusModel)
    }

    func testInvalidConfigFailsBeforeConstruction() throws {
        guard let creator = SharpAIInterns2MobiusOverlay.creators["interns2_mobius"] else {
            XCTFail("interns2_mobius creator missing")
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

    func testErrorDescriptionNamesModelType() {
        let description =
            SharpAIInterns2MobiusError.forwardNotImplemented("interns2_mobius").errorDescription
            ?? ""
        XCTAssertTrue(description.contains("interns2_mobius"))
    }

    // Real architecture: InternS2MobiusMTP head (1 MTP layer, 256 experts, top-8) per the HF config
    // summary. The "interns2_mobius_mtp" registry key below is a planned name, not a verified one.
    // Gap, checked against: HF modeling_interns2_mobius.py sets _keys_to_ignore_on_load_unexpected =
    // [r"^mtp.*"], so it never loads or defines the MTP head. The SGLang model file
    // (python/sglang/srt/models/interns2_mobius.py, sgl-project/sglang PR #33691) also skips every
    // "mtp." weight in _is_intentional_mobius_skip. The oMLX PR #2996 (closed unmerged) is the only
    // MLX-side attempt found, and its diff was not fetched.
    func testMTPDrafterRegistrationKnownIssue() {
        XCTExpectFailure(
            "KNOWN GAP: interns2_mobius MTP drafter is not registered. No public source defines the MTP head: HF modeling_interns2_mobius.py ignores mtp.* keys, SGLang skips them as intentional, and oMLX PR #2996 (closed unmerged) was not fetched. The key 'interns2_mobius_mtp' is a planned name."
        ) {
            XCTAssertNotNil(SharpAIInterns2MobiusOverlay.creators["interns2_mobius_mtp"])
        }
    }

    // The text stack builds from a tiny config and runs a forward pass that returns vocab logits.
    func testTinyForwardPassProducesVocabLogits() throws {
        guard let creator = SharpAIInterns2MobiusOverlay.creators["interns2_mobius"] else {
            XCTFail("interns2_mobius creator missing")
            return
        }
        let model = try creator(Data(tinyJSON.utf8))
        guard let lm = model as? SharpAIInterns2MobiusModel else {
            XCTFail("creator did not return SharpAIInterns2MobiusModel")
            return
        }

        let tokens = MLXArray([Int32(1), 5, 9, 13]).reshaped(1, 4)
        let logits = lm(tokens, cache: nil)

        XCTAssertEqual(logits.shape, [1, 4, 64])
    }
}
