// SharpAI fork overlay tests for deepseek_v41 checkpoint key sanitization. Not part of upstream.

import Foundation
import MLX
import MLXVLM
import XCTest

final class SharpAIDeepseekV41SanitizeTests: XCTestCase {

    private func vector(_ value: Float) -> MLXArray {
        MLXArray([value, value + 1])
    }

    func testDropsRotaryBuffersAndMTPHeads() {
        let weights: [String: MLXArray] = [
            "model.layers.0.self_attn.rotary_emb.inv_freq": vector(0),
            "mtp.0.eh_proj.weight": vector(1),
            "model.layers.0.self_attn.q_proj.weight": vector(2),
        ]

        let result = SharpAIDeepseekV41Sanitizer.sanitize(weights)

        XCTAssertNil(result["model.layers.0.self_attn.rotary_emb.inv_freq"])
        XCTAssertNil(result["mtp.0.eh_proj.weight"])
        XCTAssertNotNil(result["model.layers.0.self_attn.q_proj.weight"])
    }

    func testKeepsCompressorAndIndexerKeys() {
        let names = [
            "model.layers.2.self_attn.compressor.wkv.weight",
            "model.layers.2.self_attn.indexer.wq_b.weight",
            "model.layers.2.ffn.gate.weight",
        ]
        var weights: [String: MLXArray] = [:]
        for name in names {
            weights[name] = vector(3)
        }

        let result = SharpAIDeepseekV41Sanitizer.sanitize(weights)

        XCTAssertEqual(Set(result.keys), Set(names))
    }

    func testEmptyInputStaysEmpty() {
        XCTAssertTrue(SharpAIDeepseekV41Sanitizer.sanitize([:]).isEmpty)
    }

    /// Engram tables and DSpark drafter weights are not loaded by the text model.
    /// Engram keys are dropped by the `.engram.` filter in mlx-lm; DSpark keys by `mtp.`.
    func testEngramAndDSparkKeysDropped() {
        let dropped = [
            "model.layers.1.engram.embed.weight",
            "model.layers.14.engram.gate.weight",
            "mtp.0.dspark.block.weight",
        ]
        let kept = "model.layers.1.ffn.gate.weight"
        var weights: [String: MLXArray] = [kept: vector(4)]
        for name in dropped {
            weights[name] = vector(5)
        }

        let result = SharpAIDeepseekV41Sanitizer.sanitize(weights)

        XCTAssertEqual(Set(result.keys), [kept])
    }
}
