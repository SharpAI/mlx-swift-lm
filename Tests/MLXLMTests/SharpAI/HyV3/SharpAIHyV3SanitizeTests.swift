// SharpAI fork overlay tests for hy_v3 checkpoint key sanitization. Not part of upstream.

import Foundation
import MLX
import MLXLMCommon
import XCTest

@testable import MLXLLM

final class SharpAIHyV3SanitizeTests: XCTestCase {

    private func model(tied: Bool = false) throws -> SharpAIHyV3Model {
        let config = try JSONDecoder.json5().decode(
            SharpAIHyV3Configuration.self,
            from: Data(SharpAIHyV3TestFixture.tinyJSON(tied: tied).utf8))
        return SharpAIHyV3Model(config)
    }

    private func weights(_ keys: [String]) -> [String: MLXArray] {
        Dictionary(uniqueKeysWithValues: keys.map { ($0, MLXArray([Float(1)])) })
    }

    func testStacksRoutedExpertsIntoSwitchMLP() throws {
        let out = try model().sanitize(
            weights: weights([
                "model.layers.1.mlp.experts.0.gate_proj.weight",
                "model.layers.1.mlp.experts.1.gate_proj.weight",
            ]))

        XCTAssertNil(out["model.layers.1.mlp.experts.0.gate_proj.weight"])
        XCTAssertNil(out["model.layers.1.mlp.experts.1.gate_proj.weight"])
        XCTAssertEqual(out["model.layers.1.mlp.switch_mlp.gate_proj.weight"]?.shape, [2, 1])
    }

    func testKeepsDenseFirstLayerMLPUnchanged() throws {
        let keys = [
            "model.layers.0.mlp.gate_proj.weight",
            "model.layers.0.mlp.up_proj.weight",
            "model.layers.0.mlp.down_proj.weight",
        ]
        let out = try model().sanitize(weights: weights(keys))

        for key in keys {
            XCTAssertNotNil(out[key], key)
        }
        XCTAssertNil(out["model.layers.0.mlp.switch_mlp.gate_proj.weight"])
    }

    func testKeepsMTPLayerKeysAtIndexNumHiddenLayers() throws {
        let out = try model().sanitize(
            weights: weights([
                "model.layers.3.eh_proj.weight",
                "model.layers.3.enorm.weight",
                "model.layers.3.hnorm.weight",
                "model.layers.3.final_layernorm.weight",
                "model.layers.3.self_attn.q_proj.weight",
                "model.layers.3.mlp.experts.0.gate_proj.weight",
                "model.layers.3.mlp.experts.1.gate_proj.weight",
                "model.layers.2.self_attn.q_proj.weight",
            ]))

        XCTAssertNotNil(out["model.layers.3.eh_proj.weight"])
        XCTAssertNotNil(out["model.layers.3.enorm.weight"])
        XCTAssertNotNil(out["model.layers.3.hnorm.weight"])
        XCTAssertNotNil(out["model.layers.3.final_layernorm.weight"])
        XCTAssertNotNil(out["model.layers.3.self_attn.q_proj.weight"])
        XCTAssertNil(out["model.layers.3.mlp.experts.0.gate_proj.weight"])
        XCTAssertEqual(out["model.layers.3.mlp.switch_mlp.gate_proj.weight"]?.shape, [2, 1])
        XCTAssertNotNil(out["model.layers.2.self_attn.q_proj.weight"])
    }

    /// vLLM hy_v3_mtp.py skips embed_tokens and shared_head weights under the spec layer,
    /// because the MTP step reuses the main embedding and lm_head.
    func testDropsMTPEmbedAndSharedHeadCopies() throws {
        let out = try model().sanitize(
            weights: weights([
                "model.layers.3.embed_tokens.weight",
                "model.layers.3.shared_head.head.weight",
                "model.layers.3.shared_head.norm.weight",
            ]))

        XCTAssertTrue(out.isEmpty)
    }

    func testDropsRotaryInvFreq() throws {
        let out = try model().sanitize(
            weights: weights([
                "model.rotary_emb.inv_freq",
                "model.layers.0.self_attn.rotary_emb.inv_freq",
            ]))

        XCTAssertTrue(out.isEmpty)
    }

    func testKeepsHFAttentionAndNormKeyNames() throws {
        let keys = [
            "model.embed_tokens.weight",
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.k_norm.weight",
            "model.layers.0.self_attn.q_norm.weight",
            "model.layers.1.mlp.gate.weight",
            "model.layers.1.mlp.e_score_correction_bias",
            "model.layers.1.mlp.shared_experts.down_proj.weight",
            "model.norm.weight",
        ]
        let out = try model().sanitize(weights: weights(keys))

        for key in keys {
            XCTAssertNotNil(out[key], key)
        }
    }

    func testDropsLmHeadWhenEmbeddingsAreTied() throws {
        let out = try model(tied: true).sanitize(
            weights: weights(["model.embed_tokens.weight", "lm_head.weight"]))

        XCTAssertNotNil(out["model.embed_tokens.weight"])
        XCTAssertNil(out["lm_head.weight"])
    }

    func testKeepsLmHeadWhenEmbeddingsAreNotTied() throws {
        let out = try model(tied: false).sanitize(weights: weights(["lm_head.weight"]))

        XCTAssertNotNil(out["lm_head.weight"])
    }
}
