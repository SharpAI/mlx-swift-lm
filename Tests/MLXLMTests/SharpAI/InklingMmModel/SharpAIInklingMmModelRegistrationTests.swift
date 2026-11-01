// SharpAI fork overlay tests for the inkling_mm_model creator. Not part of upstream.

import Foundation
import MLX
import MLXLMCommon
import MLXVLM
import XCTest

final class SharpAIInklingMmModelRegistrationTests: XCTestCase {

    // Tiny fixture shape for tests only. The key names are the published config.json names.
    private let validJSON = """
        {
            "model_type": "inkling_mm_model",
            "text_config": {
                "vocab_size": 32,
                "hidden_size": 16,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 4,
                "swa_num_attention_heads": 4,
                "swa_num_key_value_heads": 2,
                "swa_head_dim": 4,
                "sliding_window_size": 8,
                "d_rel": 2,
                "rel_extent": 16,
                "log_scaling_alpha": 0.1,
                "local_layer_ids": [0],
                "dense_mlp_idx": 1,
                "rms_norm_eps": 0.000001,
                "sconv_kernel_size": 4,
                "dense_intermediate_size": 32,
                "intermediate_size": 8,
                "n_routed_experts": 4,
                "num_experts_per_tok": 2,
                "n_shared_experts": 1,
                "shared_expert_sink": true,
                "route_scale": 1.0,
                "logits_mup_width_multiplier": 1.0
            },
            "audio_config": {
                "n_mel_bins": 8,
                "mel_vocab_size": 16,
                "decoder_dmodel": 16
            },
            "vision_config": {
                "vision_encoder_type": "hmlp",
                "patch_size": 4,
                "temporal_patch_size": 2,
                "n_channels": 3,
                "n_layers": 1,
                "decoder_dmodel": 16
            }
        }
        """

    func testCreatorIsRegisteredForModelType() {
        XCTAssertNotNil(SharpAIInklingMmModelOverlay.creators["inkling_mm_model"])
    }

    func testTextOnlyTypeIsNotRegistered() {
        XCTAssertNil(SharpAIInklingMmModelOverlay.creators["inkling_model"])
    }

    func testValidConfigBuildsModel() throws {
        guard let creator = SharpAIInklingMmModelOverlay.creators["inkling_mm_model"] else {
            XCTFail("inkling_mm_model creator missing")
            return
        }

        // Building the model must not throw. The forward-pass test below checks its output shape.
        _ = try creator(Data(validJSON.utf8))
    }

    func testInvalidConfigFailsBeforeConstruction() throws {
        guard let creator = SharpAIInklingMmModelOverlay.creators["inkling_mm_model"] else {
            XCTFail("inkling_mm_model creator missing")
            return
        }
        let json = """
            { "model_type": "inkling_mm_model", "text_config": { "num_experts_per_tok": 512 } }
            """

        XCTAssertThrowsError(try creator(Data(json.utf8)))
    }

    // Previously a KNOWN GAP: the forward pass was not implemented. The text path now runs.
    func testTinyForwardPassProducesLogits() throws {
        guard let creator = SharpAIInklingMmModelOverlay.creators["inkling_mm_model"] else {
            XCTFail("inkling_mm_model creator missing")
            return
        }
        let model = try creator(Data(validJSON.utf8))

        let tokens = MLXArray([Int32(1), Int32(2), Int32(3)]).reshaped([1, 3])
        let logits = model(tokens, cache: nil)

        XCTAssertEqual(logits.shape, [1, 3, 32])
    }
}
