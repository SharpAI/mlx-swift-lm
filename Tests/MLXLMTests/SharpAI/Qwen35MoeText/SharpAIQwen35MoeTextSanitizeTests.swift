// SharpAI fork overlay tests for qwen3_5_moe_text expert stacking. Not part of upstream.
//
// Key and shape checks only. The weights are zeros, and no checkpoint is loaded.

import Foundation
import MLX
import MLXLMCommon
import XCTest

@testable import MLXLLM

final class SharpAIQwen35MoeTextSanitizeTests: XCTestCase {

    private func perExpertWeights(layer: Int = 0, numExperts: Int = 4) -> [String: MLXArray] {
        var weights = [String: MLXArray]()
        for expert in 0 ..< numExperts {
            let prefix = "model.layers.\(layer).mlp.experts.\(expert)"
            weights["\(prefix).gate_proj.weight"] = MLXArray.zeros([32, 64])
            weights["\(prefix).up_proj.weight"] = MLXArray.zeros([32, 64])
            weights["\(prefix).down_proj.weight"] = MLXArray.zeros([64, 32])
        }
        return weights
    }

    func testPerExpertWeightsStackIntoSwitchMLP() {
        let out = SharpAIQwen35MoeTextExperts.stack(
            perExpertWeights(), layers: 2, numExperts: 4)

        XCTAssertEqual(out["model.layers.0.mlp.switch_mlp.gate_proj.weight"]?.shape, [4, 32, 64])
        XCTAssertEqual(out["model.layers.0.mlp.switch_mlp.up_proj.weight"]?.shape, [4, 32, 64])
        XCTAssertEqual(out["model.layers.0.mlp.switch_mlp.down_proj.weight"]?.shape, [4, 64, 32])
        XCTAssertFalse(out.keys.contains { $0.contains(".experts.") })
    }

    func testFusedGateUpIsSplitIntoSwitchMLP() {
        var weights = [String: MLXArray]()
        weights["model.layers.1.mlp.experts.gate_up_proj"] = MLXArray.zeros([4, 64, 64])
        weights["model.layers.1.mlp.experts.down_proj"] = MLXArray.zeros([4, 64, 32])

        let out = SharpAIQwen35MoeTextExperts.stack(weights, layers: 2, numExperts: 4)

        XCTAssertEqual(out["model.layers.1.mlp.switch_mlp.gate_proj.weight"]?.shape, [4, 32, 64])
        XCTAssertEqual(out["model.layers.1.mlp.switch_mlp.up_proj.weight"]?.shape, [4, 32, 64])
        XCTAssertEqual(out["model.layers.1.mlp.switch_mlp.down_proj.weight"]?.shape, [4, 64, 32])
        XCTAssertNil(out["model.layers.1.mlp.experts.gate_up_proj"])
        XCTAssertNil(out["model.layers.1.mlp.experts.down_proj"])
    }

    func testIncompleteExpertSetIsLeftUnchanged() {
        var weights = perExpertWeights(numExperts: 3)
        weights.removeValue(forKey: "model.layers.0.mlp.experts.2.up_proj.weight")

        let out = SharpAIQwen35MoeTextExperts.stack(weights, layers: 1, numExperts: 4)

        XCTAssertNil(out["model.layers.0.mlp.switch_mlp.gate_proj.weight"])
        XCTAssertNotNil(out["model.layers.0.mlp.experts.0.gate_proj.weight"])
    }

    func testNonMoEKeysPassThrough() {
        var weights = perExpertWeights()
        weights["model.embed_tokens.weight"] = MLXArray.zeros([100, 64])
        weights["model.layers.0.input_layernorm.weight"] = MLXArray.zeros([64])

        let out = SharpAIQwen35MoeTextExperts.stack(weights, layers: 1, numExperts: 4)

        XCTAssertEqual(out["model.embed_tokens.weight"]?.shape, [100, 64])
        XCTAssertEqual(out["model.layers.0.input_layernorm.weight"]?.shape, [64])
    }

    func testModelSanitizeStacksExpertsBeforeKeyRemapping() throws {
        let config = try JSONDecoder.json5().decode(
            SharpAIQwen35MoeTextConfiguration.self,
            from: Data(SharpAIQwen35MoeTextTestFixture.tinyJSON().utf8))
        let model = SharpAIQwen35MoeTextModel(config.textConfiguration)

        let out = model.sanitize(weights: perExpertWeights(layer: 0, numExperts: 4))

        XCTAssertEqual(out["model.layers.0.mlp.switch_mlp.gate_proj.weight"]?.shape, [4, 32, 64])
        XCTAssertFalse(out.keys.contains { $0.contains(".experts.") })
    }

    private func mtpPerExpertWeights(prefix: String, numExperts: Int = 4) -> [String: MLXArray] {
        var weights = [String: MLXArray]()
        for expert in 0 ..< numExperts {
            let name = "\(prefix).experts.\(expert)"
            weights["\(name).gate_proj.weight"] = MLXArray.zeros([32, 64])
            weights["\(name).up_proj.weight"] = MLXArray.zeros([32, 64])
            weights["\(name).down_proj.weight"] = MLXArray.zeros([64, 32])
        }
        return weights
    }

    /// UNVERIFIED layout: MTP keys follow the wrapped Qwen3.5 MoE Step 4 prefixes minus
    /// `language_model.`. The stacking itself is checked here on synthetic zero tensors.
    func testMTPExpertStackingIntoSwitchMLP() {
        let out = SharpAIQwen35MoeTextExperts.stack(
            mtpPerExpertWeights(prefix: "mtp.0.layers.0.mlp"),
            layers: 0, numExperts: 4, mtpLayers: 1)

        XCTAssertEqual(
            out["mtp.0.layers.0.mlp.switch_mlp.gate_proj.weight"]?.shape, [4, 32, 64])
        XCTAssertEqual(
            out["mtp.0.layers.0.mlp.switch_mlp.up_proj.weight"]?.shape, [4, 32, 64])
        XCTAssertEqual(
            out["mtp.0.layers.0.mlp.switch_mlp.down_proj.weight"]?.shape, [4, 64, 32])
        XCTAssertFalse(out.keys.contains { $0.contains(".experts.") })
    }

    func testMTPFusedGateUpIsSplitIntoSwitchMLP() {
        var weights = [String: MLXArray]()
        weights["mtp.layers.1.mlp.experts.gate_up_proj"] = MLXArray.zeros([4, 64, 64])
        weights["mtp.layers.1.mlp.experts.down_proj"] = MLXArray.zeros([4, 64, 32])

        let out = SharpAIQwen35MoeTextExperts.stack(
            weights, layers: 0, numExperts: 4, mtpLayers: 2)

        XCTAssertEqual(out["mtp.layers.1.mlp.switch_mlp.gate_proj.weight"]?.shape, [4, 32, 64])
        XCTAssertEqual(out["mtp.layers.1.mlp.switch_mlp.up_proj.weight"]?.shape, [4, 32, 64])
        XCTAssertEqual(out["mtp.layers.1.mlp.switch_mlp.down_proj.weight"]?.shape, [4, 64, 32])
        XCTAssertNil(out["mtp.layers.1.mlp.experts.gate_up_proj"])
    }

    func testMTPExpertsPassThroughWhenNoMTPLayers() {
        let weights = mtpPerExpertWeights(prefix: "mtp.0.layers.0.mlp")
        let out = SharpAIQwen35MoeTextExperts.stack(weights, layers: 0, numExperts: 4)

        XCTAssertEqual(out.count, weights.count)
        XCTAssertNil(out["mtp.0.layers.0.mlp.switch_mlp.gate_proj.weight"])
    }
}
