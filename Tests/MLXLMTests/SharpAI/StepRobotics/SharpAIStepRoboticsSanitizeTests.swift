// SharpAI fork overlay tests for the StepVL key remap. Not part of upstream.
//
// Offline checks only: zero tensors with the checkpoint shapes, no weights, no network.

import Foundation
import MLX
import XCTest

@testable import MLXVLM

final class SharpAIStepRoboticsSanitizeTests: XCTestCase {

    private func sanitized() -> [String: MLXArray] {
        let weights: [String: MLXArray] = [
            "model.layers.0.self_attn.q_proj.weight": MLXArray.zeros([16, 32]),
            "model.layers.0.self_attn.k_proj.weight": MLXArray.zeros([8, 32]),
            "model.layers.0.self_attn.v_proj.weight": MLXArray.zeros([8, 32]),
            "model.layers.0.self_attn.wq.weight": MLXArray.zeros([16, 16]),
            "model.layers.0.self_attn.rotary_emb.inv_freq": MLXArray.zeros([4]),
            "model.layers.1.moe.gate.weight": MLXArray.zeros([4, 32]),
            "model.layers.1.moe.gate_proj.weight": MLXArray.zeros([4, 16, 32]),
            "model.layers.1.moe.up_proj.weight": MLXArray.zeros([4, 16, 32]),
            "model.layers.1.moe.down_proj.weight": MLXArray.zeros([4, 32, 16]),
            "model.layers.1.share_expert.gate_proj.weight": MLXArray.zeros([16, 32]),
            "model.norm.weight": MLXArray.zeros([32]),
            "lm_head.weight": MLXArray.zeros([64, 32]),
            "vision_model.conv1.weight": MLXArray.zeros([16, 3, 14, 14]),
            "vision_model.vit_downsampler1.weight": MLXArray.zeros([32, 16, 3, 3]),
            "vision_model.positional_embedding": MLXArray.zeros([4, 16]),
            "vision_model.transformer.resblocks.0.attn.in_proj_weight": MLXArray.zeros([48, 16]),
            "vision_model.transformer.resblocks.0.attn.out_proj.weight": MLXArray.zeros([16, 16]),
            "vision_model.transformer.resblocks.0.mlp.c_fc.weight": MLXArray.zeros([32, 16]),
            "vision_model.transformer.resblocks.0.mlp.c_proj.bias": MLXArray.zeros([16]),
            "vision_model.transformer.resblocks.0.ls_1.gamma": MLXArray.zeros([16]),
            "vision_model.freqs_cache": MLXArray.zeros([1]),
            "vit_large_projector.weight": MLXArray.zeros([32, 64]),
        ]
        return SharpAIStepRoboticsSanitize.sanitize(weights: weights)
    }

    func testTextKeysMoveUnderLanguageModel() {
        let out = sanitized()

        XCTAssertNotNil(out["language_model.model.norm.weight"])
        XCTAssertNotNil(out["language_model.model.layers.1.moe.gate.weight"])
        XCTAssertNotNil(out["language_model.model.layers.1.share_expert.gate_proj.weight"])
        XCTAssertNotNil(out["language_model.lm_head.weight"])
        XCTAssertNil(out["model.norm.weight"])
        XCTAssertNil(out["lm_head.weight"])
    }

    func testTextQKVAreFusedInOrder() throws {
        let out = sanitized()

        let qkv = try XCTUnwrap(out["language_model.model.layers.0.self_attn.qkv_proj.weight"])
        XCTAssertEqual(qkv.shape, [32, 32])
        XCTAssertNil(out["language_model.model.layers.0.self_attn.q_proj.weight"])
        XCTAssertNil(out["language_model.model.layers.0.self_attn.k_proj.weight"])
        XCTAssertNil(out["language_model.model.layers.0.self_attn.v_proj.weight"])
        // The `wq` projection keeps its own name.
        XCTAssertNotNil(out["language_model.model.layers.0.self_attn.wq.weight"])
    }

    func testRoutedExpertsUseSwitchMLPPaths() throws {
        let out = sanitized()

        let gateProj = try XCTUnwrap(
            out["language_model.model.layers.1.moe.switch_mlp.gate_proj.weight"])
        XCTAssertEqual(gateProj.shape, [4, 16, 32])
        XCTAssertNotNil(out["language_model.model.layers.1.moe.switch_mlp.up_proj.weight"])
        XCTAssertNotNil(out["language_model.model.layers.1.moe.switch_mlp.down_proj.weight"])
        XCTAssertNil(out["language_model.model.layers.1.moe.gate_proj.weight"])
    }

    func testVisionKeysStayTopLevelAndRenameEncoderModules() throws {
        let out = sanitized()

        XCTAssertNotNil(out["vit_large_projector.weight"])
        XCTAssertNotNil(out["vision_model.positional_embedding"])
        XCTAssertNotNil(
            out["vision_model.transformer.resblocks.0.attn.qkv_proj.weight"])
        XCTAssertNotNil(out["vision_model.transformer.resblocks.0.attn.out_proj.weight"])
        XCTAssertNotNil(out["vision_model.transformer.resblocks.0.mlp.fc1.weight"])
        XCTAssertNotNil(out["vision_model.transformer.resblocks.0.mlp.fc2.bias"])
        XCTAssertNotNil(out["vision_model.transformer.resblocks.0.ls_1.gamma"])
        XCTAssertNil(out["vision_model.transformer.resblocks.0.attn.in_proj_weight"])
        XCTAssertNil(out["vision_model.transformer.resblocks.0.mlp.c_fc.weight"])
    }

    func testConvWeightsMoveToMLXLayout() throws {
        let out = sanitized()

        // PyTorch [out, in, kH, kW] becomes MLX [out, kH, kW, in].
        let conv1 = try XCTUnwrap(out["vision_model.conv1.weight"])
        XCTAssertEqual(conv1.shape, [16, 14, 14, 3])
        let down1 = try XCTUnwrap(out["vision_model.vit_downsampler1.weight"])
        XCTAssertEqual(down1.shape, [32, 3, 3, 16])
    }

    func testRotaryAndFreqsTablesAreDropped() {
        let out = sanitized()

        XCTAssertFalse(out.keys.contains { $0.contains("rotary_emb") })
        XCTAssertFalse(out.keys.contains { $0.hasSuffix("freqs_cache") })
    }

    func testMappedNameIsIdempotentForAlreadyMappedKeys() {
        XCTAssertEqual(
            SharpAIStepRoboticsSanitize.mappedName("language_model.model.norm.weight"),
            "language_model.model.norm.weight")
        XCTAssertEqual(
            SharpAIStepRoboticsSanitize.mappedName("vit_large_projector.weight"),
            "vit_large_projector.weight")
    }
}
