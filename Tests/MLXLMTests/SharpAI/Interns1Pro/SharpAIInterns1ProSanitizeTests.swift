// SharpAI fork overlay tests for interns1_pro checkpoint key sanitization. Not part of upstream.

import Foundation
import MLX
import MLXVLM
import XCTest

final class SharpAIInterns1ProSanitizeTests: XCTestCase {

    private func vector(_ value: Float) -> MLXArray {
        MLXArray([value, value + 1])
    }

    func testDropsRotaryBuffers() {
        let weights: [String: MLXArray] = [
            "model.layers.0.self_attn.rotary_emb.inv_freq": vector(0),
            "model.layers.0.self_attn.q_proj.weight": vector(1),
        ]

        let result = SharpAIInterns1ProSanitizer.sanitize(weights)

        XCTAssertNil(result["model.layers.0.self_attn.rotary_emb.inv_freq"])
        XCTAssertNotNil(result["model.layers.0.self_attn.q_proj.weight"])
    }

    /// The FoPE coefficients are learned parameters and the checkpoint stores them. They must
    /// survive sanitization, unlike inv_freq.
    func testKeepsFoPECoefficients() {
        let names = [
            "model.language_model.rotary_emb.sin_coef",
            "model.language_model.rotary_emb.cos_coef",
        ]
        var weights: [String: MLXArray] = [:]
        for name in names {
            weights[name] = vector(3)
        }

        let result = SharpAIInterns1ProSanitizer.sanitize(weights)

        XCTAssertEqual(Set(result.keys), Set(names))
    }

    func testKeepsHFKeyNamesUnchanged() {
        let names = [
            "language_model.model.embed_tokens.weight",
            "language_model.lm_head.weight",
            "visual.patch_embed.proj.weight",
        ]
        var weights: [String: MLXArray] = [:]
        for name in names {
            weights[name] = vector(2)
        }

        let result = SharpAIInterns1ProSanitizer.sanitize(weights)

        XCTAssertEqual(Set(result.keys), Set(names))
        for name in names {
            XCTAssertEqual(result[name]?.shape, [2], "value for \(name) was altered")
        }
    }

    func testEmptyInputStaysEmpty() {
        XCTAssertTrue(SharpAIInterns1ProSanitizer.sanitize([:]).isEmpty)
    }

    // Real checkpoint module names from internlm/Intern-S1-Pro config.json
    // quantization_config.modules_to_not_convert (not FP8-converted). Names are checked against
    // model.safetensors.index.json: the non-expert tensors use the ".weight" suffix shown here.
    func testConfigListedCheckpointKeysPassThrough() {
        let names = [
            "model.language_model.layers.31.mlp.gate.weight",
            "model.language_model.layers.23.self_attn.k_norm.weight",
            "model.language_model.layers.30.post_attention_layernorm.weight",
            "model.language_model.layers.7.self_attn.q_norm.weight",
        ]
        var weights: [String: MLXArray] = [:]
        for name in names {
            weights[name] = vector(4)
        }

        let result = SharpAIInterns1ProSanitizer.sanitize(weights)

        XCTAssertEqual(Set(result.keys), Set(names))
    }

    /// Verified against: model.safetensors.index.json of internlm/Intern-S1-Pro. Expert FP8 tensors
    /// are named `<name>.weight` with `<name>.weight_scale_inv`. The scale grid is one value per
    /// 128x128 tile (quantization_config.weight_block_size [128, 128]).
    func testFP8ExpertScaleTensorsAreConsumed() {
        let weightKey = "model.language_model.layers.0.mlp.experts.0.gate_proj.weight"
        let scaleKey = weightKey + "_scale_inv"
        // One tile row, two tile columns: left tile scale 6, right tile scale 7.
        let weights: [String: MLXArray] = [
            weightKey: MLXArray.ones([128, 256]) * Float(5),
            scaleKey: MLXArray([Float(6), Float(7)]).reshaped(1, 2),
        ]

        let result = SharpAIInterns1ProSanitizer.sanitize(weights)

        XCTAssertNil(result[scaleKey], "the scale tensor must be consumed")
        guard let dequantized = result[weightKey] else {
            XCTFail("dequantized weight missing")
            return
        }
        XCTAssertEqual(dequantized.shape, [128, 256])
        let values = dequantized.asArray(Float.self)
        XCTAssertEqual(values[0], 30, accuracy: 1e-4)  // row 0, col 0: 5 * 6
        XCTAssertEqual(values[127], 30, accuracy: 1e-4)  // row 0, col 127: 5 * 6
        XCTAssertEqual(values[128], 35, accuracy: 1e-4)  // row 0, col 128: 5 * 7
        XCTAssertEqual(values[255], 35, accuracy: 1e-4)  // row 0, col 255: 5 * 7
    }

    /// Padding: a 130x130 weight has a 2x2 tile grid, so the last tile row and column are partial.
    func testFP8PartialTilesUseTheirOwnScale() {
        let weightKey = "model.language_model.layers.0.self_attn.q_proj.weight"
        let scaleKey = weightKey + "_scale_inv"
        let weights: [String: MLXArray] = [
            weightKey: MLXArray.ones([130, 130]) * Float(2),
            scaleKey: MLXArray([Float(1), 2, 3, 4]).reshaped(2, 2),
        ]

        let result = SharpAIInterns1ProSanitizer.sanitize(weights)

        XCTAssertNil(result[scaleKey])
        guard let dequantized = result[weightKey] else {
            XCTFail("dequantized weight missing")
            return
        }
        XCTAssertEqual(dequantized.shape, [130, 130])
        let values = dequantized.asArray(Float.self)
        XCTAssertEqual(values[0], 2, accuracy: 1e-4)  // tile (0,0): 2 * 1
        XCTAssertEqual(values[129], 4, accuracy: 1e-4)  // row 0, col 129: tile (0,1), 2 * 2
        XCTAssertEqual(values[129 * 130], 6, accuracy: 1e-4)  // row 129, col 0: tile (1,0), 2 * 3
        XCTAssertEqual(values[129 * 130 + 129], 8, accuracy: 1e-4)  // tile (1,1): 2 * 4
    }

    /// A scale grid that does not match the tile grid is left alone, so the loader reports it.
    func testFP8ScaleShapeMismatchIsLeftUntouched() {
        let weightKey = "model.language_model.layers.0.self_attn.k_proj.weight"
        let scaleKey = weightKey + "_scale_inv"
        let weights: [String: MLXArray] = [
            weightKey: MLXArray.ones([128, 256]),
            scaleKey: MLXArray.ones([2, 2]),
        ]

        let result = SharpAIInterns1ProSanitizer.sanitize(weights)

        XCTAssertNotNil(result[scaleKey])
        XCTAssertEqual(result[weightKey]?.shape, [128, 256])
    }

    /// Per-expert tensors are stacked into switch_mlp only when the expert count is given.
    func testExpertsAreStackedWhenCountIsGiven() {
        var weights: [String: MLXArray] = [:]
        for e in 0 ..< 2 {
            weights["model.language_model.layers.0.mlp.experts.\(e).gate_proj.weight"] =
                MLXArray.ones([3, 4]) * Float(e + 1)
        }

        let unstacked = SharpAIInterns1ProSanitizer.sanitize(weights)
        XCTAssertNotNil(unstacked["model.language_model.layers.0.mlp.experts.0.gate_proj.weight"])

        let stacked = SharpAIInterns1ProSanitizer.sanitize(weights, numExperts: 2)
        XCTAssertNil(stacked["model.language_model.layers.0.mlp.experts.0.gate_proj.weight"])
        XCTAssertEqual(
            stacked["model.language_model.layers.0.mlp.switch_mlp.gate_proj.weight"]?.shape,
            [2, 3, 4])
    }
}
