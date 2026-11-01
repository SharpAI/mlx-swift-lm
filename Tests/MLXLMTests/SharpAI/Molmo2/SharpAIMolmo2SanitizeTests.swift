// SharpAI fork overlay tests for molmo2 key sanitization. Not part of upstream.
// UNVERIFIED: the vision-side key prefixes are assumed; no real molmo2 checkpoint index was read.

import Foundation
import MLX
import MLXVLM
import XCTest

final class SharpAIMolmo2SanitizeTests: XCTestCase {

    private func scalar() -> MLXArray {
        MLXArray(Float(1.0))
    }

    func testLanguageKeysPassThroughUnchanged() {
        let weights: [String: MLXArray] = [
            "model.transformer.wte.weight": scalar(),
            "model.transformer.blocks.0.self_attn.q_proj.weight": scalar(),
            "lm_head.weight": scalar(),
        ]
        let sanitized = SharpAIMolmo2Sanitizer.sanitize(weights)

        XCTAssertEqual(Set(sanitized.keys), Set(weights.keys))
    }

    func testDropsVisionAndProjectorWeights() {
        let weights: [String: MLXArray] = [
            "model.vision_backbone.image_vit.encoder.layers.0.weight": scalar(),
            "vision_backbone.image_vit.patch_embed.weight": scalar(),
            "model.image_pooling_2d.wq.weight": scalar(),
            "image_projector.linear.weight": scalar(),
            "lm_head.weight": scalar(),
        ]
        let sanitized = SharpAIMolmo2Sanitizer.sanitize(weights)

        XCTAssertEqual(Set(sanitized.keys), ["lm_head.weight"])
    }

    func testDropsRotaryInvFreq() {
        let weights: [String: MLXArray] = [
            "model.transformer.blocks.0.self_attn.rotary_emb.inv_freq": scalar(),
            "lm_head.weight": scalar(),
        ]
        let sanitized = SharpAIMolmo2Sanitizer.sanitize(weights)

        XCTAssertEqual(Set(sanitized.keys), ["lm_head.weight"])
    }
}
