// SharpAI ParamsJson sanitize tests. Key mapping and expert stacking on zero tensors only.
// The real-layout tests use the key names read from the safetensors header of
// mistralai/Mistral-Large-3-675B-Instruct-2512-Eagle (consolidated.safetensors), with tiny shapes.

import Foundation
import MLX
import XCTest

@testable import MLXLLM

final class SharpAIParamsJsonSanitizeTests: XCTestCase {

    private func config(tied: Bool = false) throws -> SharpAIParamsJsonConfiguration {
        var json = SharpAIParamsJsonTestFixtures.tinyParamsJSON
        if tied {
            json = json.replacingOccurrences(
                of: "\"tied_embeddings\": false", with: "\"tied_embeddings\": true")
        }
        return try SharpAIParamsJsonConfiguration.decode(Data(json.utf8))
    }

    func testRenamesNativeAttentionAndNormKeys() throws {
        let weights: [String: MLXArray] = [
            "layers.0.attention.wq_a.weight": MLXArray.zeros([16, 32]),
            "layers.0.attention.q_a_norm.weight": MLXArray.zeros([16]),
            "layers.0.attention.wq_b.weight": MLXArray.zeros([48, 16]),
            "layers.0.attention.wkv_a_with_mqa.weight": MLXArray.zeros([16, 32]),
            "layers.0.attention.kv_a_norm.weight": MLXArray.zeros([12]),
            "layers.0.attention.wkv_b.weight": MLXArray.zeros([64, 12]),
            "layers.0.attention.wo.weight": MLXArray.zeros([32, 32]),
            "layers.0.attention_norm.weight": MLXArray.zeros([32]),
            "layers.0.ffn_norm.weight": MLXArray.zeros([32]),
        ]
        let out = SharpAIParamsJsonSanitize.sanitize(weights, config: try config())

        XCTAssertNotNil(out["layers.0.self_attn.q_a_proj.weight"])
        XCTAssertNotNil(out["layers.0.self_attn.q_a_layernorm.weight"])
        XCTAssertNotNil(out["layers.0.self_attn.q_b_proj.weight"])
        XCTAssertNotNil(out["layers.0.self_attn.kv_a_proj_with_mqa.weight"])
        XCTAssertNotNil(out["layers.0.self_attn.kv_a_layernorm.weight"])
        XCTAssertNotNil(out["layers.0.self_attn.kv_b_proj.weight"])
        XCTAssertNotNil(out["layers.0.self_attn.o_proj.weight"])
        XCTAssertNotNil(out["layers.0.input_layernorm.weight"])
        XCTAssertNotNil(out["layers.0.post_attention_layernorm.weight"])
        XCTAssertNil(out["layers.0.attention.wq_a.weight"])
        XCTAssertNil(out["layers.0.attention_norm.weight"])
    }

    func testRenamesTopLevelKeys() throws {
        let weights: [String: MLXArray] = [
            "tok_embeddings.weight": MLXArray.zeros([64, 32]),
            "norm.weight": MLXArray.zeros([32]),
            "output.weight": MLXArray.zeros([64, 32]),
            "eagle_linear.weight": MLXArray.zeros([32, 64]),
        ]
        let out = SharpAIParamsJsonSanitize.sanitize(weights, config: try config())

        XCTAssertNotNil(out["embed_tokens.weight"])
        XCTAssertNotNil(out["norm.weight"])
        XCTAssertNotNil(out["lm_head.weight"])
        XCTAssertNotNil(out["eagle_proj.weight"])
        XCTAssertNil(out["tok_embeddings.weight"])
    }

    func testStacksPerExpertTensorsIntoSwitchMLP() throws {
        var weights: [String: MLXArray] = [:]
        for e in 0..<4 {
            weights["layers.0.feed_forward.experts.\(e).w1.weight"] = MLXArray.zeros([16, 32])
            weights["layers.0.feed_forward.experts.\(e).w2.weight"] = MLXArray.zeros([32, 16])
            weights["layers.0.feed_forward.experts.\(e).w3.weight"] = MLXArray.zeros([16, 32])
        }
        let out = SharpAIParamsJsonSanitize.sanitize(weights, config: try config())

        XCTAssertEqual(out["layers.0.mlp.switch_mlp.gate_proj.weight"]?.shape, [4, 16, 32])
        XCTAssertEqual(out["layers.0.mlp.switch_mlp.down_proj.weight"]?.shape, [4, 32, 16])
        XCTAssertEqual(out["layers.0.mlp.switch_mlp.up_proj.weight"]?.shape, [4, 16, 32])
        XCTAssertNil(out["layers.0.feed_forward.experts.0.w1.weight"])
    }

    func testExpertsWithoutFeedForwardPrefixAlsoStack() throws {
        var weights: [String: MLXArray] = [:]
        for e in 0..<4 {
            weights["layers.0.experts.\(e).w1.weight"] = MLXArray.zeros([16, 32])
        }
        let out = SharpAIParamsJsonSanitize.sanitize(weights, config: try config())

        XCTAssertEqual(out["layers.0.mlp.switch_mlp.gate_proj.weight"]?.shape, [4, 16, 32])
    }

    func testSharedExpertAndGateKeysRenamed() throws {
        let weights: [String: MLXArray] = [
            "layers.0.feed_forward.gate.weight": MLXArray.zeros([4, 32]),
            "layers.0.feed_forward.shared_experts.w1.weight": MLXArray.zeros([16, 32]),
            "layers.0.feed_forward.shared_experts.w2.weight": MLXArray.zeros([32, 16]),
        ]
        let out = SharpAIParamsJsonSanitize.sanitize(weights, config: try config())

        XCTAssertNotNil(out["layers.0.mlp.gate.weight"])
        XCTAssertNotNil(out["layers.0.mlp.shared_experts.gate_proj.weight"])
        XCTAssertNotNil(out["layers.0.mlp.shared_experts.down_proj.weight"])
    }

    func testRotaryAndInvFreqKeysAreDropped() throws {
        let weights: [String: MLXArray] = [
            "layers.0.attention.rotary_emb.inv_freq": MLXArray.zeros([4]),
            "rotary_emb.inv_freq": MLXArray.zeros([4]),
            "norm.weight": MLXArray.zeros([32]),
        ]
        let out = SharpAIParamsJsonSanitize.sanitize(weights, config: try config())

        XCTAssertEqual(Set(out.keys), ["norm.weight"])
    }

    func testHFModelPrefixIsStripped() throws {
        let weights: [String: MLXArray] = [
            "model.layers.0.self_attn.q_a_proj.weight": MLXArray.zeros([16, 32])
        ]
        let out = SharpAIParamsJsonSanitize.sanitize(weights, config: try config())

        XCTAssertNotNil(out["layers.0.self_attn.q_a_proj.weight"])
    }

    /// Real drafter names, in the layout of the EAGLE safetensors header: no feed_forward prefix,
    /// experts as layers.0.experts.N.wK.weight, FP8 weights with qscale_weight and qscale_act,
    /// eagle_linear.weight for the fusion, and no tok_embeddings or output.weight in the file.
    func testRealEagleCheckpointLayoutMapsToModelKeys() throws {
        let dim = 32
        var weights: [String: MLXArray] = [
            "eagle_linear.weight": MLXArray.zeros([dim, 2 * dim]),
            "eagle_linear.qscale_weight": MLXArray(Float(1)),
            "eagle_linear.qscale_act": MLXArray(Float(1)),
            "norm.weight": MLXArray.zeros([dim]),
            "layers.0.attention_norm.weight": MLXArray.zeros([dim]),
            "layers.0.ffn_norm.weight": MLXArray.zeros([dim]),
            "layers.0.attention.wq_a.weight": MLXArray.zeros([16, dim]),
            "layers.0.attention.wq_a.qscale_weight": MLXArray(Float(1)),
            "layers.0.attention.wq_a.qscale_act": MLXArray(Float(1)),
            "layers.0.attention.wkv_a_with_mqa.weight": MLXArray.zeros([16, dim]),
            "layers.0.attention.wo.weight": MLXArray.zeros([dim, 32]),
            "layers.0.gate.weight": MLXArray.zeros([4, dim]),
            "layers.0.shared_experts.w1.weight": MLXArray.zeros([16, dim]),
            "layers.0.shared_experts.w2.weight": MLXArray.zeros([dim, 16]),
            "layers.0.shared_experts.w3.weight": MLXArray.zeros([16, dim]),
        ]
        for e in 0..<4 {
            weights["layers.0.experts.\(e).w1.weight"] = MLXArray.zeros([16, dim])
            weights["layers.0.experts.\(e).w2.weight"] = MLXArray.zeros([dim, 16])
            weights["layers.0.experts.\(e).w3.weight"] = MLXArray.zeros([16, dim])
        }
        let out = SharpAIParamsJsonSanitize.sanitize(weights, config: try config())

        XCTAssertNotNil(out["eagle_proj.weight"])
        XCTAssertNotNil(out["norm.weight"])
        XCTAssertNotNil(out["layers.0.input_layernorm.weight"])
        XCTAssertNotNil(out["layers.0.post_attention_layernorm.weight"])
        XCTAssertNotNil(out["layers.0.self_attn.q_a_proj.weight"])
        XCTAssertNotNil(out["layers.0.self_attn.kv_a_proj_with_mqa.weight"])
        XCTAssertNotNil(out["layers.0.self_attn.o_proj.weight"])
        XCTAssertNotNil(out["layers.0.mlp.gate.weight"])
        XCTAssertNotNil(out["layers.0.mlp.shared_experts.gate_proj.weight"])
        XCTAssertNotNil(out["layers.0.mlp.shared_experts.down_proj.weight"])
        XCTAssertNotNil(out["layers.0.mlp.shared_experts.up_proj.weight"])
        XCTAssertEqual(out["layers.0.mlp.switch_mlp.gate_proj.weight"]?.shape, [4, 16, dim])
        XCTAssertEqual(out["layers.0.mlp.switch_mlp.down_proj.weight"]?.shape, [4, dim, 16])
        XCTAssertEqual(out["layers.0.mlp.switch_mlp.up_proj.weight"]?.shape, [4, 16, dim])

        // Scale tensors are consumed; none survive under any name.
        XCTAssertFalse(out.keys.contains { $0.contains("qscale") })
        // The file has no embedding or lm_head. vLLM's load_weights reports both as loaded from
        // the target model, so the sanitized drafter must not invent them.
        XCTAssertNil(out["embed_tokens.weight"])
        XCTAssertNil(out["lm_head.weight"])
    }

    /// qscale_weight is a per-tensor multiplicative dequant scale: stored fp8 * scale.
    func testFP8WeightScaleIsAppliedBeforeExpertStacking() throws {
        var weights: [String: MLXArray] = [:]
        for e in 0..<4 {
            weights["layers.0.experts.\(e).w1.weight"] = MLXArray.ones([16, 32]) * Float(e + 1)
            weights["layers.0.experts.\(e).w1.qscale_weight"] = MLXArray(Float(0.5))
        }
        let out = SharpAIParamsJsonSanitize.sanitize(weights, config: try config())
        let stacked = out["layers.0.mlp.switch_mlp.gate_proj.weight"]

        XCTAssertEqual(stacked?.shape, [4, 16, 32])
        // Expert e holds the value (e + 1) * 0.5 in every element.
        let values = stacked?.reshaped(4, -1)[.ellipsis, 0].asArray(Float.self) ?? []
        XCTAssertEqual(values, [0.5, 1.0, 1.5, 2.0])
        XCTAssertFalse(out.keys.contains { $0.contains("qscale") })
    }

    func testTiedEmbeddingsDropsLMHead() throws {
        let weights: [String: MLXArray] = [
            "output.weight": MLXArray.zeros([64, 32]),
            "tok_embeddings.weight": MLXArray.zeros([64, 32]),
        ]
        let out = SharpAIParamsJsonSanitize.sanitize(weights, config: try config(tied: true))

        XCTAssertNil(out["lm_head.weight"])
        XCTAssertNotNil(out["embed_tokens.weight"])
    }
}
