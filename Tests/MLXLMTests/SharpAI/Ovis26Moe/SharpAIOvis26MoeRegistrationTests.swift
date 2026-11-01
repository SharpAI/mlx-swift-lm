// SharpAI fork overlay tests for the ovis2_6_moe creator. Not part of upstream.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXVLM
import XCTest

final class SharpAIOvis26MoeRegistrationTests: XCTestCase {

    private let validJSON = """
        {
            "model_type": "ovis2_6_moe",
            "llm_config": {
                "vocab_size": 1024,
                "hidden_size": 256,
                "num_hidden_layers": 2,
                "num_attention_heads": 8,
                "num_key_value_heads": 8,
                "num_experts": 8,
                "num_experts_per_tok": 2
            },
            "vit_config": {
                "model_type": "siglip2_navit"
            }
        }
        """

    func testCreatorIsRegisteredForModelType() {
        XCTAssertNotNil(SharpAIOvis26MoeOverlay.creators["ovis2_6_moe"])
    }

    func testValidConfigFailsUntilForwardIsImplemented() throws {
        guard let creator = SharpAIOvis26MoeOverlay.creators["ovis2_6_moe"] else {
            XCTFail("ovis2_6_moe creator missing")
            return
        }

        XCTAssertThrowsError(try creator(Data(validJSON.utf8))) { error in
            guard case SharpAIOvis26MoeError.forwardNotImplemented(let type) = error else {
                XCTFail("unexpected error: \(error)")
                return
            }
            XCTAssertEqual(type, "ovis2_6_moe")
        }
    }

    func testInvalidConfigFailsBeforeConstruction() throws {
        guard let creator = SharpAIOvis26MoeOverlay.creators["ovis2_6_moe"] else {
            XCTFail("ovis2_6_moe creator missing")
            return
        }

        XCTAssertThrowsError(try creator(Data("{ \"hidden_size\": 64 }".utf8))) { error in
            if case SharpAIOvis26MoeError.forwardNotImplemented = error {
                XCTFail("config validation should fail first")
            }
        }
    }

    /// Text block of the Ovis2.6-30B-A3B `llm_config` (model_type qwen3_moe), with the real
    /// Qwen3-30B-A3B layout scaled down. Source: AIDC-AI/Ovis2.6-30B-A3B config.json.
    private let tinyTextJSON = """
        {
            "model_type": "qwen3_moe",
            "hidden_size": 256,
            "num_hidden_layers": 2,
            "intermediate_size": 64,
            "num_attention_heads": 8,
            "num_key_value_heads": 4,
            "head_dim": 32,
            "num_experts": 8,
            "num_experts_per_tok": 2,
            "decoder_sparse_step": 1,
            "mlp_only_layers": [],
            "moe_intermediate_size": 32,
            "norm_topk_prob": true,
            "rms_norm_eps": 1e-6,
            "vocab_size": 1024
        }
        """

    /// The reference Ovis2_6_MoeForCausalLM forward with no pixel_values feeds
    /// `wte(input_ids)` straight into `llm` (modeling_ovis2_6.py, merge_multimodal). That is the
    /// Qwen3MoE text decoder on token ids, so a tiny Qwen3MoEModel run is the text-only forward.
    /// The visual path (SigLIP2 NaViT tower, visual tokenizer, vte) is not covered here.
    func testTinyTextForwardPass() throws {
        let config = try JSONDecoder().decode(
            Qwen3MoEConfiguration.self, from: Data(tinyTextJSON.utf8))
        let model = Qwen3MoEModel(config)

        let tokens = MLXArray([1, 2, 3, 4, 5] as [Int32]).reshaped(1, 5)
        let logits = model(tokens, cache: nil)

        XCTAssertEqual(logits.shape, [1, 5, 1024])
        XCTAssertTrue(all(isFinite(logits)).item(Bool.self))

        // Causal attention: the first three positions must not depend on the later tokens.
        let prefixLogits = model(tokens[0..., 0 ..< 3], cache: nil)
        let headLogits = logits[0..., 0 ..< 3, 0...]
        XCTAssertTrue(
            allClose(prefixLogits, headLogits, rtol: 1e-4, atol: 1e-4).item(Bool.self))
    }
}
