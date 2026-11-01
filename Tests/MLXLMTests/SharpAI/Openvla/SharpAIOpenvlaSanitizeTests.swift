// SharpAI fork overlay tests for OpenVLA checkpoint key mapping. Not part of upstream.
//
// Offline checks only: synthetic key names and small zero arrays, no weights.

import Foundation
import MLX
import XCTest

@testable import MLXVLM

final class SharpAIOpenvlaSanitizeTests: XCTestCase {

    func testMapsCheckpointKeysToModulePaths() {
        let placeholder = MLXArray.zeros([2, 2])
        let weights: [String: MLXArray] = [
            "vision_backbone.featurizer.blocks.0.attn.qkv.weight": placeholder,
            "vision_backbone.featurizer.blocks.0.ls1.gamma": placeholder,
            "vision_backbone.fused_featurizer.blocks.0.mlp.fc1.weight": placeholder,
            "vision_backbone.featurizer.norm.weight": placeholder,
            "vision_backbone.fused_featurizer.attn_pool.q.weight": placeholder,
            "vision_backbone.fused_featurizer.head.weight": placeholder,
            "projector.fc1.weight": placeholder,
            "model.language_model.model.layers.0.self_attn.q_proj.weight": placeholder,
            "llm_backbone.llm.model.embed_tokens.weight": placeholder,
            "llm_backbone.llm.lm_head.weight": placeholder,
            "model.rotary_emb.inv_freq": placeholder,
        ]

        let sanitized = SharpAIOpenvlaSanitize.sanitize(weights: weights)

        XCTAssertEqual(
            Set(sanitized.keys),
            [
                "vision_backbone.featurizer.blocks.0.attn.qkv.weight",
                "vision_backbone.featurizer.blocks.0.ls1.gamma",
                "vision_backbone.fused_featurizer.blocks.0.mlp.fc1.weight",
                "projector.fc1.weight",
                "language_model.model.layers.0.self_attn.q_proj.weight",
                "language_model.model.embed_tokens.weight",
                "language_model.lm_head.weight",
            ])
    }

    func testPatchEmbedWeightMovesToMLXLayout() {
        // PyTorch layout [out, in, kH, kW] with kH != kW is not MLX layout, so it is transposed.
        let weight = MLXArray.zeros([16, 3, 14, 14])
        let sanitized = SharpAIOpenvlaSanitize.sanitize(weights: [
            "vision_backbone.featurizer.patch_embed.proj.weight": weight
        ])

        XCTAssertEqual(
            sanitized["vision_backbone.featurizer.patch_embed.proj.weight"]?.shape,
            [16, 14, 14, 3])
    }

    func testPatchEmbedWeightInMLXLayoutIsKept() {
        // [out, kH, kW, in] with out >= kH == kW already matches MLX layout.
        let weight = MLXArray.zeros([16, 14, 14, 3])
        let sanitized = SharpAIOpenvlaSanitize.sanitize(weights: [
            "vision_backbone.fused_featurizer.patch_embed.proj.weight": weight
        ])

        XCTAssertEqual(
            sanitized["vision_backbone.fused_featurizer.patch_embed.proj.weight"]?.shape,
            [16, 14, 14, 3])
    }

    func testUnusedFinalNormsAreDropped() {
        let placeholder = MLXArray.zeros([2])
        let sanitized = SharpAIOpenvlaSanitize.sanitize(weights: [
            "vision_backbone.featurizer.norm.weight": placeholder,
            "vision_backbone.featurizer.norm.bias": placeholder,
            "vision_backbone.fused_featurizer.norm.weight": placeholder,
        ])

        XCTAssertTrue(sanitized.isEmpty)
    }
}
