// SharpAI fork overlay tests for openai_privacy_filter checkpoint key mapping. Not part of upstream.

import Foundation
import MLX
import XCTest

@testable import MLXLLM

final class SharpAIOpenaiPrivacyFilterSanitizeTests: XCTestCase {

    func testExpertTensorsGetWeightSuffixAndSwappedAxes() {
        let weights: [String: MLXArray] = [
            "model.layers.0.mlp.experts.gate_up_proj": MLXArray.zeros([2, 3, 4]),
            "model.layers.0.mlp.experts.gate_up_proj_bias": MLXArray.zeros([2, 4]),
            "model.layers.0.mlp.experts.down_proj": MLXArray.zeros([2, 4, 3]),
            "model.layers.0.mlp.experts.down_proj_bias": MLXArray.zeros([2, 3]),
        ]

        let out = SharpAIOpenaiPrivacyFilterSanitizer.sanitize(weights)

        // HF [experts, in, out] becomes SwitchLinear [experts, out, in].
        XCTAssertEqual(
            out["model.layers.0.mlp.experts.gate_up_proj.weight"]?.shape, [2, 4, 3])
        XCTAssertEqual(
            out["model.layers.0.mlp.experts.down_proj.weight"]?.shape, [2, 3, 4])
        XCTAssertEqual(
            out["model.layers.0.mlp.experts.gate_up_proj.bias"]?.shape, [2, 4])
        XCTAssertEqual(
            out["model.layers.0.mlp.experts.down_proj.bias"]?.shape, [2, 3])
        XCTAssertNil(out["model.layers.0.mlp.experts.gate_up_proj"])
        XCTAssertNil(out["model.layers.0.mlp.experts.gate_up_proj_bias"])
    }

    func testKeepsModelAndScoreKeysAndDropsOthers() {
        let weights: [String: MLXArray] = [
            "model.embed_tokens.weight": MLXArray.zeros([4, 2]),
            "model.layers.0.self_attn.sinks": MLXArray.zeros([2]),
            "model.layers.0.mlp.router.weight": MLXArray.zeros([4, 2]),
            "score.weight": MLXArray.zeros([3, 2]),
            "score.bias": MLXArray.zeros([3]),
            "model.rotary_emb.inv_freq": MLXArray.zeros([1]),
            "lm_head.weight": MLXArray.zeros([4, 2]),
        ]

        let out = SharpAIOpenaiPrivacyFilterSanitizer.sanitize(weights)

        XCTAssertNotNil(out["model.embed_tokens.weight"])
        XCTAssertNotNil(out["model.layers.0.self_attn.sinks"])
        XCTAssertNotNil(out["model.layers.0.mlp.router.weight"])
        XCTAssertNotNil(out["score.weight"])
        XCTAssertNotNil(out["score.bias"])
        XCTAssertNil(out["model.rotary_emb.inv_freq"])
        XCTAssertNil(out["lm_head.weight"])
    }
}
