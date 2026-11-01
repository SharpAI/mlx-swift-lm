// SharpAI fork overlay tests for inkling_model checkpoint key sanitization. Not part of upstream.
//
// Key names come from the thinkingmachines/Inkling safetensors index and the transformers
// `inkling_mm_model` conversion mapping. The split of fused gate-up rows follows transformers'
// `Interleave` + `Chunk` (row 2a is gate, row 2a + 1 is up).

import Foundation
import MLX
import MLXLLM
import XCTest

final class SharpAIInklingModelSanitizeTests: XCTestCase {

    func testSanitizePreservesKeysAndValues() {
        let weights: [String: MLXArray] = [
            "model.embed_tokens.weight": MLXArray(Float(1)),
            "lm_head.weight": MLXArray(Float(2)),
        ]

        let sanitized = SharpAIInklingModelSanitizer.sanitize(weights)

        XCTAssertEqual(Set(sanitized.keys), Set(weights.keys))
        XCTAssertNotNil(sanitized["lm_head.weight"])
    }

    func testSanitizeOfEmptyWeightsIsEmpty() {
        XCTAssertTrue(SharpAIInklingModelSanitizer.sanitize([:]).isEmpty)
    }

    func testCheckpointKeyRemappingMatchesTransformersNames() {
        // Expert w13 [E=2, 2I=4, H=3]: value at (e, r, h) is e*100 + r*10 + h.
        var expertValues: [Float] = []
        for e in 0 ..< 2 {
            for r in 0 ..< 4 {
                for h in 0 ..< 3 {
                    expertValues.append(Float(e * 100 + r * 10 + h))
                }
            }
        }
        let expertW13 = MLXArray(expertValues).reshaped([2, 4, 3])

        // Dense w13 [2I=4, H=3]: value at (r, h) is r*10 + h.
        let denseW13 = MLXArray((0 ..< 4).flatMap { r in (0 ..< 3).map { Float(r * 10 + $0) } })
            .reshaped([4, 3])

        let placeholder = MLXArray(Float(0))
        let weights: [String: MLXArray] = [
            "model.llm.embed.weight": placeholder,
            "model.llm.embed_norm.weight": placeholder,
            "model.llm.norm.weight": placeholder,
            "model.llm.unembed.weight": placeholder,
            "model.llm.layers.0.attn_norm.weight": placeholder,
            "model.llm.layers.0.mlp_norm.weight": placeholder,
            "model.llm.layers.0.attn.wq_du.weight": placeholder,
            "model.llm.layers.0.attn.wr_du.weight": placeholder,
            "model.llm.layers.0.attn.rel_logits_proj.proj": placeholder,
            "model.llm.layers.0.attn_sconv.weight": placeholder,
            "model.llm.layers.0.mlp.w13_dn.weight": denseW13,
            "model.llm.layers.0.mlp.w2_md.weight": placeholder,
            "model.llm.layers.0.mlp.global_scale": placeholder,
            "model.llm.layers.2.mlp.experts.w13_weight": expertW13,
            "model.llm.layers.2.mlp.experts.w2_weight": placeholder,
            "model.llm.layers.2.mlp.gate.bias": placeholder,
            "model.llm.layers.2.mlp.gate.weight": placeholder,
            "model.llm.layers.2.mlp.shared_experts.shared_w2_weight": placeholder,
            "model.audio.encoder.weight": placeholder,
            "model.visual.final_norm.weight": placeholder,
            "model.mtp.layers.0.embed_norm.weight": placeholder,
        ]

        let sanitized = SharpAIInklingModelSanitizer.sanitize(weights)

        let expectedKeys: Set<String> = [
            "model.embed_tokens.weight",
            "model.embed_norm.weight",
            "model.norm.weight",
            "lm_head.weight",
            "model.layers.0.input_layernorm.weight",
            "model.layers.0.post_attention_layernorm.weight",
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.r_proj.weight",
            "model.layers.0.self_attn.rel_logits_proj.proj",
            "model.layers.0.attn_sconv.weight",
            "model.layers.0.mlp.gate_proj.weight",
            "model.layers.0.mlp.up_proj.weight",
            "model.layers.0.mlp.down_proj.weight",
            "model.layers.0.mlp.global_scale",
            "model.layers.2.moe.experts.gate_proj.weight",
            "model.layers.2.moe.experts.up_proj.weight",
            "model.layers.2.moe.experts.down_proj.weight",
            "model.layers.2.moe.gate.weight",
            "model.layers.2.moe.gate.e_score_correction_bias",
            "model.layers.2.moe.shared_experts.down_proj.weight",
        ]
        XCTAssertEqual(Set(sanitized.keys), expectedKeys)

        // Gate takes even rows of each expert, and up takes odd rows.
        var expectedGate: [Float] = []
        var expectedUp: [Float] = []
        for e in 0 ..< 2 {
            for i in 0 ..< 2 {
                for h in 0 ..< 3 {
                    expectedGate.append(Float(e * 100 + (2 * i) * 10 + h))
                    expectedUp.append(Float(e * 100 + (2 * i + 1) * 10 + h))
                }
            }
        }
        XCTAssertEqual(
            sanitized["model.layers.2.moe.experts.gate_proj.weight"]?.shape, [2, 2, 3])
        XCTAssertEqual(
            sanitized["model.layers.2.moe.experts.gate_proj.weight"]?.asArray(Float.self),
            expectedGate)
        XCTAssertEqual(
            sanitized["model.layers.2.moe.experts.up_proj.weight"]?.asArray(Float.self),
            expectedUp)

        XCTAssertEqual(
            sanitized["model.layers.0.mlp.gate_proj.weight"]?.asArray(Float.self),
            [0, 1, 2, 20, 21, 22])
        XCTAssertEqual(
            sanitized["model.layers.0.mlp.up_proj.weight"]?.asArray(Float.self),
            [10, 11, 12, 30, 31, 32])
    }
}
