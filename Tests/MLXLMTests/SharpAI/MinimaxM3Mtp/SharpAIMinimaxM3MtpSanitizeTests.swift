// SharpAI fork overlay tests for minimax_m3_mtp key sanitization. Not part of upstream.
// Expectations follow vLLM `MiniMaxM3MTP._map_checkpoint_name`
// (https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/models/minimax_m3/amd/mtp.py):
// MTP layer tensors are renamed to model.layers.<i>.*, and every other key is dropped.

import MLX
import XCTest

@testable import MLXLLM

final class SharpAIMinimaxM3MtpSanitizeTests: XCTestCase {

    func testMtpLayerKeysLoseTheMtpSegment() {
        let weights: [String: MLXArray] = [
            "model.mtp.layers.0.eh_proj.weight": MLXArray([Float(1), 2]),
            "model.mtp.layers.1.transformer_layer.mlp.gate.weight": MLXArray([Float(3)]),
        ]

        let sanitized = SharpAIMinimaxM3MtpSanitizer.sanitize(weights)

        XCTAssertEqual(
            Set(sanitized.keys),
            [
                "model.layers.0.eh_proj.weight",
                "model.layers.1.transformer_layer.mlp.gate.weight",
            ])
    }

    func testBundledPrefixIsStripped() {
        let weights: [String: MLXArray] = [
            "language_model.model.mtp.layers.0.enorm.weight": MLXArray([Float(1)])
        ]

        let sanitized = SharpAIMinimaxM3MtpSanitizer.sanitize(weights)

        XCTAssertEqual(Set(sanitized.keys), ["model.layers.0.enorm.weight"])
    }

    func testBlockScaleIsRenamed() {
        let weights: [String: MLXArray] = [
            "model.mtp.layers.0.transformer_layer.self_attn.q_proj.weight_scale_inv":
                MLXArray([Float(1)])
        ]

        let sanitized = SharpAIMinimaxM3MtpSanitizer.sanitize(weights)

        XCTAssertEqual(
            Set(sanitized.keys),
            ["model.layers.0.transformer_layer.self_attn.q_proj.weight_scale"])
    }

    func testNonMtpKeysAreDropped() {
        let weights: [String: MLXArray] = [
            "lm_head.weight": MLXArray([Float(1)]),
            "model.embed_tokens.weight": MLXArray([Float(2)]),
            "model.layers.0.self_attn.q_proj.weight": MLXArray([Float(3)]),
            "model.mtp.enorm.weight": MLXArray([Float(4)]),
        ]

        XCTAssertEqual(SharpAIMinimaxM3MtpSanitizer.sanitize(weights).count, 0)
    }
}
