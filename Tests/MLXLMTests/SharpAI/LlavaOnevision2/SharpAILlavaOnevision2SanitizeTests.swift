// SharpAI fork overlay tests for llava_onevision2 key sanitization. Not part of upstream.

import Foundation
import MLX
import MLXVLM
import XCTest

final class SharpAILlavaOnevision2SanitizeTests: XCTestCase {

    private func scalar() -> MLXArray {
        MLXArray(Float(1.0))
    }

    func testMapsModelLanguageModelPrefixToQwen3Names() throws {
        let weights: [String: MLXArray] = [
            "model.language_model.embed_tokens.weight": scalar(),
            "model.language_model.layers.0.self_attn.q_proj.weight": scalar(),
            "model.language_model.norm.weight": scalar(),
        ]
        let sanitized = try SharpAILlavaOnevision2Sanitizer.sanitize(weights)

        XCTAssertEqual(
            Set(sanitized.keys),
            [
                "model.embed_tokens.weight",
                "model.layers.0.self_attn.q_proj.weight",
                "model.norm.weight",
            ])
    }

    func testMapsLanguageModelPrefixAndKeepsLMHead() throws {
        let weights: [String: MLXArray] = [
            "language_model.model.norm.weight": scalar(),
            "language_model.lm_head.weight": scalar(),
        ]
        let sanitized = try SharpAILlavaOnevision2Sanitizer.sanitize(weights)

        XCTAssertEqual(Set(sanitized.keys), ["model.norm.weight", "lm_head.weight"])
    }

    func testDropsVisionAndProjectorWeights() throws {
        let weights: [String: MLXArray] = [
            "model.vision_tower.encoder.layers.0.weight": scalar(),
            "vision_tower.patch_embed.weight": scalar(),
            "model.multi_modal_projector.linear.weight": scalar(),
            "multi_modal_projector.linear.weight": scalar(),
            "model.language_model.norm.weight": scalar(),
        ]
        let sanitized = try SharpAILlavaOnevision2Sanitizer.sanitize(weights)

        XCTAssertEqual(Set(sanitized.keys), ["model.norm.weight"])
    }

    func testDropsRotaryInvFreq() throws {
        let weights: [String: MLXArray] = [
            "model.language_model.layers.0.self_attn.rotary_emb.inv_freq": scalar(),
            "model.language_model.norm.weight": scalar(),
        ]
        let sanitized = try SharpAILlavaOnevision2Sanitizer.sanitize(weights)

        XCTAssertEqual(Set(sanitized.keys), ["model.norm.weight"])
    }

    func testThrowsOnDuplicateKeyAfterRemap() {
        let weights: [String: MLXArray] = [
            "model.language_model.norm.weight": scalar(),
            "language_model.model.norm.weight": scalar(),
        ]

        XCTAssertThrowsError(try SharpAILlavaOnevision2Sanitizer.sanitize(weights))
    }

    func testFlattensPatchConvAndRenamesMergerLinears() throws {
        let weights: [String: MLXArray] = [
            "model.visual.embeddings.patch_embedding.weight": MLXArray.zeros([8, 3, 14, 14]),
            "model.visual.merger.mlp.0.weight": scalar(),
            "model.visual.merger.mlp.2.bias": scalar(),
        ]
        let sanitized = try SharpAILlavaOnevision2Sanitizer.sanitize(weights)

        XCTAssertEqual(
            Set(sanitized.keys),
            [
                "model.visual.embeddings.patch_embedding.weight",
                "model.visual.merger.fc1.weight",
                "model.visual.merger.fc2.bias",
            ])
        XCTAssertEqual(
            sanitized["model.visual.embeddings.patch_embedding.weight"]?.shape, [8, 3 * 14 * 14])
    }

    func testEmptyInputGivesEmptyOutput() throws {
        let sanitized = try SharpAILlavaOnevision2Sanitizer.sanitize([:])

        XCTAssertTrue(sanitized.isEmpty)
    }
}
