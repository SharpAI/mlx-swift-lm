// SharpAI fork overlay tests for the deepseek_v41 text model. Not part of upstream.

import Foundation
import MLX
import MLXLMCommon
import MLXVLM
import XCTest

/// A tiny deepseek_v41 text_config. Layer 0 is window-only; layer 1 compresses KV (ratio 2),
/// indexes and is the candidate source; layer 2 compresses with ratio 2 and only indexes, so it
/// reads the shared compressed KV and index keys. Compressed layers use YaRN.
enum DeepseekV41TinyFixture {
    static let vocabSize = 64

    static let json = """
        {
            "model_type": "deepseek_v41",
            "architectures": ["DeepseekV41ForCausalLM"],
            "text_config": {
                "model_type": "deepseek_v41_text",
                "vocab_size": 64,
                "hidden_size": 32,
                "num_hidden_layers": 3,
                "num_attention_heads": 4,
                "head_dim": 16,
                "qk_rope_head_dim": 8,
                "q_lora_rank": 16,
                "o_lora_rank": 8,
                "o_groups": 2,
                "moe_intermediate_size": 16,
                "n_routed_experts": 4,
                "n_shared_experts": 1,
                "num_experts_per_tok": 2,
                "sliding_window": 4,
                "compress_ratios": [0, 2, 2],
                "kv_source_layer_ids": [1],
                "index_source_layer_ids": [1, 2],
                "index_n_heads": 2,
                "index_head_dim": 8,
                "index_topk": 3,
                "candidate_source_layer_id": 1,
                "candidate_topk_blocks": 2,
                "candidate_block_size": 2,
                "hc_mult": 2,
                "hc_sinkhorn_iters": 3,
                "max_position_embeddings": 64,
                "rope_theta": 10000,
                "compress_rope_theta": 160000,
                "rope_scaling": {
                    "rope_type": "yarn",
                    "factor": 4,
                    "beta_fast": 32,
                    "beta_slow": 1,
                    "original_max_position_embeddings": 16
                }
            }
        }
        """
}

final class SharpAIDeepseekV41ModelTests: XCTestCase {

    private func makeModel() throws -> SharpAIDeepseekV41Model {
        let config = try JSONDecoder.json5().decode(
            SharpAIDeepseekV41Configuration.self, from: Data(DeepseekV41TinyFixture.json.utf8))
        let text = try XCTUnwrap(config.text)
        return SharpAIDeepseekV41Model(text)
    }

    func testForwardPassReturnsVocabularyLogits() throws {
        let model = try makeModel()
        let tokens = MLXArray([Int32(3), 7, 1, 9, 4]).reshaped(1, 5)

        let logits = model(tokens, cache: model.newCache(parameters: nil))

        XCTAssertEqual(logits.shape, [1, 5, DeepseekV41TinyFixture.vocabSize])
    }

    /// Window rows, closed compressed groups, index keys and the open group must all carry over
    /// between calls: an incremental decode has to match one prefill over the same tokens.
    func testIncrementalDecodeMatchesPrefill() throws {
        let model = try makeModel()
        let tokens = MLXArray([Int32(3), 7, 1, 9, 4, 2, 8]).reshaped(1, 7)

        let prefill = model(tokens, cache: model.newCache(parameters: nil))

        let cache = model.newCache(parameters: nil)
        var steps: [MLXArray] = []
        for t in 0 ..< 7 {
            steps.append(model(tokens[0..., t ..< (t + 1)], cache: cache))
        }
        let incremental = concatenated(steps, axis: 1)

        XCTAssertEqual(incremental.shape, prefill.shape)
        XCTAssertTrue(allClose(prefill, incremental, rtol: 1e-3, atol: 1e-3).item(Bool.self))
    }

    func testCheckpointNamesMapToModuleNames() {
        XCTAssertEqual(
            SharpAIDeepseekV41Sanitizer.renameKey("layers.0.attn.wq_a.weight"),
            "model.layers.0.self_attn.q_a_proj.weight")
        XCTAssertEqual(
            SharpAIDeepseekV41Sanitizer.renameKey("layers.0.attn.attn_sink"),
            "model.layers.0.self_attn.sinks")
        XCTAssertEqual(
            SharpAIDeepseekV41Sanitizer.renameKey("layers.0.hc_attn_fn"),
            "model.layers.0.attn_hc.fn")
        XCTAssertEqual(
            SharpAIDeepseekV41Sanitizer.renameKey("layers.2.ffn.gate.bias"),
            "model.layers.2.mlp.gate.e_score_correction_bias")
        XCTAssertEqual(
            SharpAIDeepseekV41Sanitizer.renameKey("layers.2.attn.compressor.norm.weight"),
            "model.layers.2.self_attn.compressor.kv_norm.weight")
        XCTAssertEqual(SharpAIDeepseekV41Sanitizer.renameKey("head.weight"), "lm_head.weight")
        XCTAssertEqual(
            SharpAIDeepseekV41Sanitizer.renameKey("embed.weight"),
            "model.embed_tokens.weight")
    }

    func testPerExpertWeightsStackIntoSwitchMLP() throws {
        var weights: [String: MLXArray] = [:]
        for expert in 0 ..< 2 {
            weights["layers.1.ffn.experts.\(expert).w1.weight"] =
                MLXArray([Float](repeating: Float(expert + 1), count: 4), [2, 2])
        }

        let result = SharpAIDeepseekV41Sanitizer.toModuleKeys(weights, numExperts: 2)
        let stacked = try XCTUnwrap(
            result["model.layers.1.mlp.experts.switch_mlp.gate_proj.weight"])

        XCTAssertEqual(stacked.shape, [2, 2, 2])
        XCTAssertEqual(stacked[1].sum().item(Float.self), 8)
        XCTAssertNil(result["layers.1.ffn.experts.0.w1.weight"])
    }
}
