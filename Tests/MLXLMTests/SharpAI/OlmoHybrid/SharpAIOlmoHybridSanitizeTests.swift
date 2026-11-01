// SharpAI fork overlay tests for olmo_hybrid checkpoint key mapping. Not part of upstream.

import Foundation
import MLX
import MLXLLM
import XCTest

final class SharpAIOlmoHybridSanitizeTests: XCTestCase {

    func testDropsMtpAndRotaryBuffers() {
        let weights: [String: MLXArray] = [
            "model.embed_tokens.weight": MLXArray.zeros([32, 16]),
            "model.layers.0.linear_attn.q_proj.weight": MLXArray.zeros([8, 16]),
            "model.rotary_emb.inv_freq": MLXArray.zeros([4]),
            "mtp.layers.0.weight": MLXArray.zeros([4]),
            "lm_head.weight": MLXArray.zeros([32, 16]),
        ]

        let out = SharpAIOlmoHybridSanitize.sanitize(weights, tiedWordEmbeddings: false)

        XCTAssertEqual(
            Set(out.keys),
            [
                "model.embed_tokens.weight",
                "model.layers.0.linear_attn.q_proj.weight",
                "lm_head.weight",
            ])
    }

    func testDropsLmHeadWhenEmbeddingsAreTied() {
        let weights: [String: MLXArray] = [
            "model.embed_tokens.weight": MLXArray.zeros([32, 16]),
            "lm_head.weight": MLXArray.zeros([32, 16]),
        ]

        let out = SharpAIOlmoHybridSanitize.sanitize(weights, tiedWordEmbeddings: true)

        XCTAssertEqual(Set(out.keys), ["model.embed_tokens.weight"])
    }

    /// Torch depthwise conv weights are [C, 1, K]. The MLX Conv1d expects [C, K, 1].
    func testMovesTorchConvWeightToMlxLayout() throws {
        let values = (0 ..< 12).map { Float($0) }
        let torchWeight = MLXArray(values).reshaped(3, 1, 4)
        let key = "model.layers.0.linear_attn.conv1d.weight"

        let out = SharpAIOlmoHybridSanitize.sanitize(
            [key: torchWeight], tiedWordEmbeddings: false)

        let weight = try XCTUnwrap(out[key])
        XCTAssertEqual(weight.shape, [3, 4, 1])
        XCTAssertEqual(weight.asType(.float32).asArray(Float.self), values)
    }

    func testLeavesMlxConvLayoutUnchanged() throws {
        let key = "model.layers.1.linear_attn.conv1d.weight"
        let mlxWeight = MLXArray.zeros([3, 4, 1])

        let out = SharpAIOlmoHybridSanitize.sanitize([key: mlxWeight], tiedWordEmbeddings: false)

        XCTAssertEqual(try XCTUnwrap(out[key]).shape, [3, 4, 1])
    }

    func testKeepsNormAndGateWeightsUnshifted() throws {
        // OLMo Hybrid uses plain RMSNorm (weight * x), so no +1 shift applies.
        let key = "model.layers.0.post_attention_layernorm.weight"
        let norm = MLXArray([0.5, 1.5] as [Float])

        let out = SharpAIOlmoHybridSanitize.sanitize([key: norm], tiedWordEmbeddings: false)

        XCTAssertEqual(
            try XCTUnwrap(out[key]).asType(.float32).asArray(Float.self), [0.5, 1.5])
    }
}
