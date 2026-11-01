// SharpAI fork overlay tests for cosmos3_edge checkpoint key sanitization. Not part of upstream.

import Foundation
import MLX
import MLXVLM
import XCTest

final class SharpAICosmos3EdgeSanitizeTests: XCTestCase {

    private func vector(_ value: Float) -> MLXArray {
        MLXArray([value, value + 1])
    }

    func testDropsRotaryBuffers() {
        let weights: [String: MLXArray] = [
            "model.layers.0.self_attn.rotary_emb.inv_freq": vector(0),
            "model.layers.0.self_attn.q_proj.weight": vector(1),
        ]

        let result = SharpAICosmos3EdgeSanitizer.sanitize(weights)

        XCTAssertNil(result["model.layers.0.self_attn.rotary_emb.inv_freq"])
        XCTAssertNotNil(result["model.layers.0.self_attn.q_proj.weight"])
    }

    func testKeepsHFKeyNamesUnchanged() {
        let names = [
            "language_model.backbone.embeddings.weight",
            "language_model.lm_head.weight",
            "vision_tower.patch_embed.proj.weight",
        ]
        var weights: [String: MLXArray] = [:]
        for name in names {
            weights[name] = vector(2)
        }

        let result = SharpAICosmos3EdgeSanitizer.sanitize(weights)

        XCTAssertEqual(Set(result.keys), Set(names))
    }

    func testEmptyInputStaysEmpty() {
        XCTAssertTrue(SharpAICosmos3EdgeSanitizer.sanitize([:]).isEmpty)
    }

    // Key names are taken from the real index of nvidia/Cosmos3-Edge (vision_encoder/model.safetensors)
    // and the prefix targets from the vLLM hf_to_vllm_mapper for cosmos3_edge.
    func testPatchMergerKeyRemap() {
        let weights: [String: MLXArray] = [
            "model.projector.linear_fc1.weight": vector(3),
            "model.projector.linear_fc1.bias": vector(4),
            "model.projector.linear_fc2.weight": vector(5),
            "model.projector.norm.weight": vector(6),
            "model.visual.embeddings.patch_embedding.weight": vector(7),
            "model.visual.encoder.layers.0.self_attn.q_proj.weight": vector(8),
            "model.visual.post_layernorm.weight": vector(9),
        ]

        let result = SharpAICosmos3EdgeSanitizer.sanitize(weights)

        // vLLM maps model.visual.* to visual.encoder.*, so the vision encoder's own `encoder`
        // attribute gives visual.encoder.encoder.layers.*.
        XCTAssertEqual(
            Set(result.keys),
            Set([
                "visual.projector.linear_fc1.weight",
                "visual.projector.linear_fc1.bias",
                "visual.projector.linear_fc2.weight",
                "visual.projector.norm.weight",
                "visual.encoder.embeddings.patch_embedding.weight",
                "visual.encoder.encoder.layers.0.self_attn.q_proj.weight",
                "visual.encoder.post_layernorm.weight",
            ]))
        XCTAssertNil(result["model.projector.norm.weight"])
        XCTAssertNil(result["model.visual.post_layernorm.weight"])
    }

    func testRemapLeavesUnanchoredPrefixAlone() {
        // The prefixes are anchored at the start, so a nested match is not renamed.
        let names = [
            "language_model.model.projector.norm.weight",
            "wrapper.model.visual.post_layernorm.weight",
        ]
        var weights: [String: MLXArray] = [:]
        for name in names {
            weights[name] = vector(10)
        }

        let result = SharpAICosmos3EdgeSanitizer.sanitize(weights)

        XCTAssertEqual(Set(result.keys), Set(names))
    }
}
