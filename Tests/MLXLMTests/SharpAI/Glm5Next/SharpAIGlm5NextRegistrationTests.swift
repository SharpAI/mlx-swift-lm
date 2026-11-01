// SharpAI fork overlay tests for the glm5_next creator. Not part of upstream.

import Foundation
import MLX
import MLXLMCommon
import MLXVLM
import XCTest

final class SharpAIGlm5NextRegistrationTests: XCTestCase {

    private let minimalJSON = """
        {
            "model_type": "glm5_next",
            "architectures": ["Glm5NextForConditionalGeneration"],
            "text_config": { "model_type": "glm5_next_text", "num_hidden_layers": 2 },
            "vision_config": { "model_type": "glm5_next_vision" }
        }
        """

    func testRegistryHasGlm5NextKeyOnly() {
        XCTAssertEqual(Array(SharpAIGlm5NextOverlay.creators.keys), ["glm5_next"])
    }

    func testCreatorFailsWithForwardNotImplemented() throws {
        let creator = try XCTUnwrap(SharpAIGlm5NextOverlay.creators["glm5_next"])

        XCTAssertThrowsError(try creator(Data(minimalJSON.utf8))) { error in
            guard case SharpAIGlm5NextError.forwardNotImplemented(let type) = error else {
                XCTFail("unexpected error: \(error)")
                return
            }
            XCTAssertEqual(type, "glm5_next")
        }
    }

    func testCreatorRejectsConfigWithoutTextConfig() throws {
        let creator = try XCTUnwrap(SharpAIGlm5NextOverlay.creators["glm5_next"])

        XCTAssertThrowsError(try creator(Data(#"{ "model_type": "glm5_next" }"#.utf8))) { error in
            guard case ModelFactoryError.invalidConfiguration = error else {
                XCTFail("unexpected error: \(error)")
                return
            }
        }
    }

    /// Tiny random-weight prefill through the text decoder (KDA, NoPE MLA with the kpool
    /// indexer, mHC, dense and sparse MoE). Checks shape, finiteness, and causality: changing
    /// only the last token must leave every earlier logit unchanged.
    func testTinyForwardPassWithRandomWeightsIsPending() throws {
        let configuration = try JSONDecoder.json5().decode(
            SharpAIGlm5NextConfiguration.self, from: Data(tinyJSON.utf8))
        let text = try XCTUnwrap(configuration.textConfig)
        let model = try SharpAIGlm5NextTextModel(text)

        let tokens: [Int32] = [1, 5, 9, 2, 7, 3, 11, 4, 6, 8, 10, 12]
        let logits = model(MLXArray(tokens, [1, tokens.count]))
        XCTAssertEqual(logits.shape, [1, tokens.count, 64])
        XCTAssertTrue(logits.asArray(Float.self).allSatisfy { $0.isFinite })

        var perturbed = tokens
        perturbed[perturbed.count - 1] = 13
        let shifted = model(MLXArray(perturbed, [1, tokens.count]))
        let prefix = logits[0..., ..<(tokens.count - 1)].asArray(Float.self)
        let prefixShifted = shifted[0..., ..<(tokens.count - 1)].asArray(Float.self)
        XCTAssertEqual(prefix.count, prefixShifted.count)
        for (a, b) in zip(prefix, prefixShifted) {
            XCTAssertEqual(a, b, accuracy: 1e-4)
        }
    }

    private let tinyJSON = """
        {
            "model_type": "glm5_next",
            "architectures": ["Glm5NextForConditionalGeneration"],
            "text_config": {
                "model_type": "glm5_next_text",
                "vocab_size": 64,
                "hidden_size": 32,
                "intermediate_size": 48,
                "moe_intermediate_size": 16,
                "num_hidden_layers": 4,
                "num_attention_heads": 4,
                "num_key_value_heads": 4,
                "rms_norm_eps": 1e-5,
                "attention_bias": false,
                "hidden_act": "silu",
                "swiglu_limit": 10.0,
                "q_lora_rank": 16,
                "kv_lora_rank": 8,
                "qk_nope_head_dim": 8,
                "qk_rope_head_dim": 0,
                "qk_head_dim": 8,
                "v_head_dim": 8,
                "mla_use_nope": true,
                "index_n_heads": 2,
                "index_head_dim": 8,
                "index_topk": 8,
                "index_kpool": 4,
                "index_kpool_always_select_tail": true,
                "indexer_types": ["full", "full", "full", "full"],
                "layer_types": [
                    "linear_attention", "deepseek_sparse_attention",
                    "linear_attention", "deepseek_sparse_attention"
                ],
                "mlp_layer_types": ["dense", "sparse", "sparse", "sparse"],
                "first_k_dense_replace": 1,
                "n_routed_experts": 4,
                "n_shared_experts": 1,
                "num_experts_per_tok": 2,
                "n_group": 1,
                "topk_group": 1,
                "routed_scaling_factor": 2.5,
                "norm_topk_prob": true,
                "scoring_func": "sigmoid",
                "topk_method": "noaux_tc",
                "mhc": true,
                "hc_mult": 2,
                "hc_sinkhorn_iters": 3,
                "hc_eps": 1e-6,
                "linear_attn_config": {
                    "num_heads": 4,
                    "head_dim": 8,
                    "short_conv_kernel_size": 4,
                    "gate_lower_bound": -5.0
                }
            }
        }
        """
}
