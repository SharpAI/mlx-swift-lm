// SharpAI fork overlay tests for glm5_next checkpoint sanitization. Not part of upstream.

import Foundation
import MLX
import MLXVLM
import XCTest

final class SharpAIGlm5NextSanitizeTests: XCTestCase {

    private func sample() -> [String: MLXArray] {
        [
            "model.layers.0.mlp.experts.0.gate_proj.weight": MLXArray.ones([2, 3]),
            "model.layers.0.mlp.experts.1.gate_proj.weight": MLXArray.zeros([2, 3]),
            "model.layers.0.self_attn.q_proj.weight": MLXArray.ones([4, 4]),
            "model.layers.0.self_attn.rotary_emb.inv_freq": MLXArray.ones([4]),
            "model.layers.0.self_attn.indexer.wk.weight": MLXArray.ones([4, 4]),
            "model.layers.2.eh_proj.weight": MLXArray.ones([4, 8]),
            "model.layers.2.enorm.weight": MLXArray.ones([4]),
            "model.layers.2.embed_tokens.weight": MLXArray.ones([4, 4]),
            "visual.blocks.0.attn.qkv.weight": MLXArray.ones([4, 4]),
        ]
    }

    func testStacksExpertsIntoSwitchMLP() {
        let out = SharpAIGlm5NextSanitizer.sanitize(
            sample(), numHiddenLayers: 2, numNextnPredictLayers: 1, expertCount: 2,
            retainMTP: false)

        XCTAssertEqual(
            out["model.layers.0.mlp.switch_mlp.gate_proj.weight"]?.shape, [2, 2, 3])
        XCTAssertNil(out["model.layers.0.mlp.experts.0.gate_proj.weight"])
        XCTAssertNil(out["model.layers.0.mlp.experts.1.gate_proj.weight"])
    }

    func testKeepsHFNamesForMainLayerAndVisionKeys() {
        let out = SharpAIGlm5NextSanitizer.sanitize(
            sample(), numHiddenLayers: 2, numNextnPredictLayers: 1, expertCount: 2,
            retainMTP: false)

        XCTAssertNotNil(out["model.layers.0.self_attn.q_proj.weight"])
        XCTAssertNotNil(out["model.layers.0.self_attn.indexer.wk.weight"])
        XCTAssertNotNil(out["visual.blocks.0.attn.qkv.weight"])
    }

    func testDropsRotaryBuffers() {
        let out = SharpAIGlm5NextSanitizer.sanitize(
            sample(), numHiddenLayers: 2, numNextnPredictLayers: 1, expertCount: 2,
            retainMTP: false)

        XCTAssertNil(out["model.layers.0.self_attn.rotary_emb.inv_freq"])
    }

    func testDropsNextnLayersUnlessRetained() {
        let dropped = SharpAIGlm5NextSanitizer.sanitize(
            sample(), numHiddenLayers: 2, numNextnPredictLayers: 1, expertCount: 2,
            retainMTP: false)
        XCTAssertNil(dropped["model.layers.2.eh_proj.weight"])
        XCTAssertNil(dropped["model.layers.2.enorm.weight"])

        let kept = SharpAIGlm5NextSanitizer.sanitize(
            sample(), numHiddenLayers: 2, numNextnPredictLayers: 1, expertCount: 2,
            retainMTP: true)
        XCTAssertNotNil(kept["model.layers.2.eh_proj.weight"])
        XCTAssertNotNil(kept["model.layers.2.enorm.weight"])
        // The MTP block reuses the main embedding, so its copy is dropped.
        XCTAssertNil(kept["model.layers.2.embed_tokens.weight"])
    }

    func testDequantizesBlockScaleInvAndDropsScale() {
        let weights: [String: MLXArray] = [
            "model.layers.0.self_attn.o_proj.weight": MLXArray.ones([128, 128]),
            "model.layers.0.self_attn.o_proj.weight_scale_inv": MLXArray([Float(2)], [1, 1]),
        ]
        let out = SharpAIGlm5NextSanitizer.sanitize(
            weights, numHiddenLayers: 1, numNextnPredictLayers: 0, expertCount: 0,
            retainMTP: false)

        XCTAssertNil(out["model.layers.0.self_attn.o_proj.weight_scale_inv"])
        let values = out["model.layers.0.self_attn.o_proj.weight"]?.asArray(Float.self) ?? []
        XCTAssertEqual(values.count, 128 * 128)
        XCTAssertTrue(values.allSatisfy { $0 == 2 })
    }
}
