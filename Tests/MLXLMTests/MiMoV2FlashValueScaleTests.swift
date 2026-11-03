import Foundation
import MLX
import MLXNN
import XCTest

@testable import MLXLLM

/// attention_value_scale for MiMoV2Flash: v is multiplied by the scale before SDPA.
/// Assumed formula (unverified air-gapped; mirrors SharpAIMimoV2Attention in this repo):
/// `v = v_proj(x) * attention_value_scale`, i.e. equivalent to folding the scale into v_proj.
final class MiMoV2FlashValueScaleTests: XCTestCase {

    private func makeConfig(valueScale: String?) throws -> MiMoV2FlashConfiguration {
        let extra = valueScale.map { ", \"attention_value_scale\": \($0)" } ?? ""
        let json = """
            {
                "model_type": "mimo_v2_flash", "num_experts_per_tok": 2,
                "hybrid_layer_pattern": [1, 0], "moe_layer_freq": [0, 0],
                "add_swa_attention_sink_bias": true, "add_full_attention_sink_bias": true,
                "sliding_window_size": 8, "vocab_size": 32, "hidden_size": 8,
                "intermediate_size": 16, "moe_intermediate_size": 8, "num_hidden_layers": 2,
                "num_attention_heads": 2, "num_key_value_heads": 1,
                "topk_method": "greedy", "scoring_func": "softmax", "norm_topk_prob": true,
                "n_group": 1, "topk_group": 1, "max_position_embeddings": 128,
                "layernorm_epsilon": 1e-6, "rope_theta": 10000.0, "swa_rope_theta": 10000.0,
                "swa_num_attention_heads": 2, "swa_num_key_value_heads": 1,
                "head_dim": 4, "v_head_dim": 4, "swa_head_dim": 4, "swa_v_head_dim": 4,
                "partial_rotary_factor": 1.0\(extra)
            }
            """
        return try JSONDecoder().decode(MiMoV2FlashConfiguration.self, from: Data(json.utf8))
    }

    private func run(_ attn: MiMoV2FlashAttention, _ x: MLXArray) -> MLXArray {
        let out = attn(x, mask: .causal, cache: nil)
        eval(out)
        return out
    }

    func testValueScaleMatchesFoldingScaleIntoVProj() throws {
        for sliding in [false, true] {
            MLXRandom.seed(7)
            let ref = MiMoV2FlashAttention(try makeConfig(valueScale: nil), isSlidingWindow: sliding)
            let scaled = MiMoV2FlashAttention(
                try makeConfig(valueScale: "2.0"), isSlidingWindow: sliding)
            eval(ref)
            scaled.update(parameters: ref.parameters())
            eval(scaled)
            let x = MLXRandom.normal([1, 5, 8])

            let unscaled = run(ref, x)
            // Fold the scale into v_proj of the reference.
            ref.wv.update(
                parameters: ModuleParameters.unflattened(["weight": ref.wv.weight * 2.0]))
            eval(ref)
            let folded = run(ref, x)
            let actual = run(scaled, x)

            XCTAssertFalse(
                allClose(unscaled, folded, rtol: 1e-3, atol: 1e-5).item(Bool.self),
                "test would be vacuous: scale must change the output")
            XCTAssertTrue(
                allClose(actual, folded, rtol: 1e-4, atol: 1e-5).item(Bool.self),
                "sliding=\(sliding): attention_value_scale not applied to v")
        }
    }

    func testAbsentAndUnitScaleAreBitIdentical() throws {
        MLXRandom.seed(11)
        let a = MiMoV2FlashAttention(try makeConfig(valueScale: nil), isSlidingWindow: false)
        let b = MiMoV2FlashAttention(try makeConfig(valueScale: "1.0"), isSlidingWindow: false)
        eval(a)
        b.update(parameters: a.parameters())
        eval(b)
        let x = MLXRandom.normal([1, 5, 8])
        XCTAssertTrue(arrayEqual(run(a, x), run(b, x)).item(Bool.self))
    }
}
