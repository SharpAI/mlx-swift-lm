// SharpAI fork overlay tests for qwen4_exp checkpoint key sanitization. Not part of upstream.

import Foundation
import MLX
import MLXVLM
import XCTest

final class SharpAIQwen4ExpSanitizeTests: XCTestCase {

    private func vector(_ value: Float) -> MLXArray {
        MLXArray([value, value + 1])
    }

    func testDropsRotaryBuffers() {
        let weights: [String: MLXArray] = [
            "model.layers.0.self_attn.rotary_emb.inv_freq": vector(0),
            "model.layers.3.self_attn.q_proj.weight": vector(1),
        ]

        let result = SharpAIQwen4ExpSanitizer.sanitize(weights)

        XCTAssertNil(result["model.layers.0.self_attn.rotary_emb.inv_freq"])
        XCTAssertNotNil(result["model.layers.3.self_attn.q_proj.weight"])
        XCTAssertEqual(result.count, 1)
    }

    /// Keys with no remap rule pass through unchanged.
    func testPassesThroughUnremappedGapKeys() {
        let names = [
            "model.layers.2.ple.conv.weight",
            "model.layers.0.hc.mixer.weight",
            "model.layers.0.ngram_embed.weight",
            "model.layers.3.indexer.wq.weight",
        ]
        var weights: [String: MLXArray] = [:]
        for name in names {
            weights[name] = vector(2)
        }

        let result = SharpAIQwen4ExpSanitizer.sanitize(weights)

        XCTAssertEqual(Set(result.keys), Set(names))
    }

    func testDropsVisionMTPAndConfigDerivedBuffers() {
        let weights: [String: MLXArray] = [
            "model.visual.blocks.0.attn.qkv.weight": vector(0),
            "mtp.layers.0.mlp.gate.weight": vector(0),
            "model.language_model.layers.1.ple.ple_embedding.layer_multipliers": vector(0),
            "model.language_model.layers.1.ple.ple_embedding.ngram_heads_vocab_sizes": vector(0),
            "model.language_model.layers.1.ple.ple_embedding.ngram_heads_offsets": vector(0),
            "model.language_model.embed_tokens.weight": vector(1),
        ]

        let result = SharpAIQwen4ExpSanitizer.sanitize(weights)

        XCTAssertEqual(Set(result.keys), ["model.language_model.embed_tokens.weight"])
    }

    func testConcatenatesNgramShardsInIndexOrder() {
        let shard0 = MLXArray.zeros([2, 3])
        let shard1 = MLXArray.ones([2, 3])
        let prefix = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding"
        let weights: [String: MLXArray] = [
            "\(prefix).shard_1.weight": shard1,
            "\(prefix).shard_0.weight": shard0,
        ]

        let result = SharpAIQwen4ExpSanitizer.sanitize(weights)

        XCTAssertNil(result["\(prefix).shard_0.weight"])
        XCTAssertNil(result["\(prefix).shard_1.weight"])
        let table = result["\(prefix).weight"]
        XCTAssertEqual(table?.shape, [4, 3])
        XCTAssertEqual(table?[0, 0].item(Float.self), 0)
        XCTAssertEqual(table?[3, 0].item(Float.self), 1)
    }

    func testSplitsFusedExpertsIntoSwitchGLUWeights() {
        // gate_up_proj is (E, 2I, H) with E = 2, I = 2, H = 3; down_proj is (E, H, I).
        let fused = MLXArray((0 ..< 24).map { Float($0) }, [2, 4, 3])
        let down = MLXArray((0 ..< 12).map { Float($0) }, [2, 3, 2])
        let prefix = "model.language_model.layers.3.mlp"
        let weights: [String: MLXArray] = [
            "\(prefix).experts.gate_up_proj": fused,
            "\(prefix).experts.down_proj": down,
        ]

        let result = SharpAIQwen4ExpSanitizer.sanitize(weights)

        XCTAssertEqual(result["\(prefix).switch_mlp.gate_proj.weight"]?.shape, [2, 2, 3])
        XCTAssertEqual(result["\(prefix).switch_mlp.up_proj.weight"]?.shape, [2, 2, 3])
        XCTAssertEqual(result["\(prefix).switch_mlp.down_proj.weight"]?.shape, [2, 3, 2])
        // The gate half is rows 0..<2 of each expert, and the up half is rows 2..<4.
        XCTAssertEqual(result["\(prefix).switch_mlp.gate_proj.weight"]?[1, 0, 0].item(Float.self), 12)
        XCTAssertEqual(result["\(prefix).switch_mlp.up_proj.weight"]?[0, 0, 0].item(Float.self), 6)
    }

    func testMovesConvWeightToMLXLayout() {
        let torchConv = MLXArray.zeros([5, 1, 4])
        let weights: [String: MLXArray] = [
            "model.language_model.layers.0.linear_attn.conv1d.weight": torchConv
        ]

        let result = SharpAIQwen4ExpSanitizer.sanitize(weights)

        XCTAssertEqual(
            result["model.language_model.layers.0.linear_attn.conv1d.weight"]?.shape, [5, 4, 1])
    }
}
