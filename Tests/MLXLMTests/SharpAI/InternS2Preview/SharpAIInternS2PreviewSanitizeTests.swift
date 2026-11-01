// SharpAI fork overlay tests for intern_s2_preview checkpoint key handling. Not part of upstream.

import Foundation
import MLX
import XCTest

@testable import MLXVLM

final class SharpAIInternS2PreviewSanitizeTests: XCTestCase {

    func testTimeSeriesKeysAreDropped() {
        let dummy = MLXArray.zeros([1, 1])
        let weights: [String: MLXArray] = [
            "model.time_series.encoder.layers.0.weight": dummy,
            "time_series.projector.weight": dummy,
            "model.language_model.layers.0.mlp.gate.weight": dummy,
            "model.visual.blocks.0.attn.qkv.weight": dummy,
        ]

        let kept = SharpAIInternS2PreviewSanitizer.dropTimeSeries(weights)

        XCTAssertEqual(
            Set(kept.keys),
            [
                "model.language_model.layers.0.mlp.gate.weight",
                "model.visual.blocks.0.attn.qkv.weight",
            ])
    }

    // A key that only contains "time_series" in the middle must not match the prefix rule.
    func testMidPathTimeSeriesNameIsKept() {
        let dummy = MLXArray.zeros([1, 1])
        let weights: [String: MLXArray] = [
            "model.language_model.layers.0.time_series.weight": dummy
        ]

        let kept = SharpAIInternS2PreviewSanitizer.dropTimeSeries(weights)

        XCTAssertEqual(kept.count, 1)
    }

    func testGateUpExpertsAreSplitIntoSwitchMLP() {
        // Shape [experts, 2 * intermediate, hidden]: gate and up each take half of axis -2.
        let gateUp = MLXArray.zeros([2, 4, 3])
        let downProj = MLXArray.zeros([2, 3, 2])
        let weights: [String: MLXArray] = [
            "model.language_model.layers.0.mlp.experts.gate_up_proj": gateUp,
            "model.language_model.layers.0.mlp.experts.down_proj": downProj,
        ]

        let fused = SharpAIInternS2PreviewSanitizer.fuseExperts(weights, hiddenLayers: 1)

        XCTAssertNil(fused["model.language_model.layers.0.mlp.experts.gate_up_proj"])
        XCTAssertNil(fused["model.language_model.layers.0.mlp.experts.down_proj"])
        XCTAssertEqual(
            fused["model.language_model.layers.0.mlp.switch_mlp.gate_proj.weight"]?.shape,
            [2, 2, 3])
        XCTAssertEqual(
            fused["model.language_model.layers.0.mlp.switch_mlp.up_proj.weight"]?.shape,
            [2, 2, 3])
        XCTAssertEqual(
            fused["model.language_model.layers.0.mlp.switch_mlp.down_proj.weight"]?.shape,
            [2, 3, 2])
    }

    func testFusionLeavesDenseWeightsUntouched() {
        let dummy = MLXArray.zeros([4, 4])
        let weights: [String: MLXArray] = [
            "model.language_model.layers.0.mlp.gate_proj.weight": dummy
        ]

        let fused = SharpAIInternS2PreviewSanitizer.fuseExperts(weights, hiddenLayers: 1)

        XCTAssertEqual(Set(fused.keys), Set(weights.keys))
    }
}
