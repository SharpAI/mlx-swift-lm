// SharpAI fork overlay tests for mimo_v2 checkpoint sanitization. Not part of upstream.

import Foundation
import MLX
import MLXLMCommon
import XCTest

@testable import MLXLLM

final class SharpAIMimoV2SanitizeTests: XCTestCase {

    private func model(retainMTP: Bool = false, tied: Bool = false) throws -> SharpAIMimoV2Model {
        let config = try JSONDecoder.json5().decode(
            SharpAIMimoV2Configuration.self,
            from: Data(SharpAIMimoV2TestFixture.tinyJSON(tied: tied).utf8))
        return SharpAIMimoV2Model(config, retainMTP: retainMTP)
    }

    private func weights(_ keys: [String]) -> [String: MLXArray] {
        Dictionary(uniqueKeysWithValues: keys.map { ($0, MLXArray([Float(1)])) })
    }

    /// Fused qkv rows 0..<64 with 4 columns, valued by row index.
    private func fusedQKV(rows: Int = 64, cols: Int = 4) -> MLXArray {
        MLXArray((0 ..< (rows * cols)).map { Float($0) }).reshaped(rows, cols)
    }

    func testSplitsFusedQKVIntoChunkMajorSections() throws {
        // Layer 0 is sliding-window. 2 chunks, each [Q 16 | K 8 | V 8] rows.
        let out = try model().sanitize(
            weights: ["model.layers.0.self_attn.qkv_proj.weight": fusedQKV()])

        XCTAssertNil(out["model.layers.0.self_attn.qkv_proj.weight"])
        XCTAssertEqual(out["model.layers.0.self_attn.q_proj.weight"]?.shape, [32, 4])
        XCTAssertEqual(out["model.layers.0.self_attn.k_proj.weight"]?.shape, [16, 4])
        XCTAssertEqual(out["model.layers.0.self_attn.v_proj.weight"]?.shape, [16, 4])

        // Q row 16 is chunk 1's first Q row, which is fused row 32.
        XCTAssertEqual(out["model.layers.0.self_attn.q_proj.weight"]?[16, 0].item(Float.self), 128)
        // K row 0 is chunk 0's first K row, which is fused row 16.
        XCTAssertEqual(out["model.layers.0.self_attn.k_proj.weight"]?[0, 0].item(Float.self), 64)
        // V row 0 is chunk 0's first V row, which is fused row 24.
        XCTAssertEqual(out["model.layers.0.self_attn.v_proj.weight"]?[0, 0].item(Float.self), 96)
    }

    func testDequantizesPerChunkScalesBeforeSplitting() throws {
        let weight = MLXArray.ones([64, 4])
        let scale = MLXArray([2.0, 3.0] as [Float]).reshaped(2, 1)
        let out = try model().sanitize(weights: [
            "model.layers.0.self_attn.qkv_proj.weight": weight,
            "model.layers.0.self_attn.qkv_proj.weight_scale_inv": scale,
        ])

        XCTAssertNil(out["model.layers.0.self_attn.qkv_proj.weight_scale_inv"])
        // Chunk 0 rows scale by 2. Chunk 1 rows scale by 3.
        XCTAssertEqual(out["model.layers.0.self_attn.q_proj.weight"]?[0, 0].item(Float.self), 2)
        XCTAssertEqual(out["model.layers.0.self_attn.q_proj.weight"]?[16, 0].item(Float.self), 3)
    }

    func testDequantizesContinuousBlockScales() throws {
        // Non-fused weight: 256 rows give two 128-row scale blocks.
        let weight = MLXArray.ones([256, 128])
        let scale = MLXArray([2.0, 3.0] as [Float]).reshaped(2, 1)
        let out = try model().sanitize(weights: [
            "model.layers.1.mlp.up_proj.weight": weight,
            "model.layers.1.mlp.up_proj.weight_scale_inv": scale,
        ])

        XCTAssertNil(out["model.layers.1.mlp.up_proj.weight_scale_inv"])
        XCTAssertEqual(out["model.layers.1.mlp.up_proj.weight"]?[0, 0].item(Float.self), 2)
        XCTAssertEqual(out["model.layers.1.mlp.up_proj.weight"]?[200, 5].item(Float.self), 3)
    }

    func testStacksRoutedExpertsIntoSwitchMLP() throws {
        let out = try model().sanitize(
            weights: weights(
                (0 ..< 4).map { "model.layers.1.mlp.experts.\($0).gate_proj.weight" }))

        XCTAssertNil(out["model.layers.1.mlp.experts.0.gate_proj.weight"])
        XCTAssertNil(out["model.layers.1.mlp.experts.3.gate_proj.weight"])
        XCTAssertEqual(out["model.layers.1.mlp.switch_mlp.gate_proj.weight"]?.shape, [4, 1])
    }

    func testKeepsDenseLayerMLPUnchanged() throws {
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

    func testDropsRotaryCachesAndLayersPastCount() throws {
        let out = try model().sanitize(
            weights: weights([
                "model.layers.0.self_attn.rotary_emb.inv_freq",
                "model.layers.4.self_attn.q_proj.weight",
                "model.layers.3.self_attn.q_proj.weight",
            ]))

        XCTAssertNil(out["model.layers.0.self_attn.rotary_emb.inv_freq"])
        XCTAssertNil(out["model.layers.4.self_attn.q_proj.weight"])
        XCTAssertNotNil(out["model.layers.3.self_attn.q_proj.weight"])
    }

    func testDropsMTPKeysWhenNotRetained() throws {
        let out = try model(retainMTP: false).sanitize(
            weights: weights([
                "model.mtp.layers.0.enorm.weight",
                "model.mtp.layers.0.eh_proj.weight",
            ]))

        XCTAssertTrue(out.isEmpty)
    }

    func testKeepsMTPLayerZeroWhenRetainedAndSplitsItsQKV() throws {
        let out = try model(retainMTP: true).sanitize(weights: [
            "model.mtp.layers.0.enorm.weight": MLXArray([Float(1)]),
            "model.mtp.layers.1.enorm.weight": MLXArray([Float(1)]),
            "model.mtp.layers.0.self_attn.qkv_proj.weight": fusedQKV(),
        ])

        XCTAssertNotNil(out["model.mtp.layers.0.enorm.weight"])
        XCTAssertNil(out["model.mtp.layers.1.enorm.weight"])
        XCTAssertEqual(out["model.mtp.layers.0.self_attn.q_proj.weight"]?.shape, [32, 4])
    }

    func testDropsLMHeadWhenEmbeddingsAreTied() throws {
        let out = try model(tied: true).sanitize(
            weights: weights([
                "lm_head.weight",
                "model.embed_tokens.weight",
            ]))

        XCTAssertNil(out["lm_head.weight"])
        XCTAssertNotNil(out["model.embed_tokens.weight"])
    }
}
