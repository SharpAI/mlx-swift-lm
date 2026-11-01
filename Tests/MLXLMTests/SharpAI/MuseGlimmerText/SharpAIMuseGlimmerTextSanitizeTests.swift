// SharpAI fork overlay tests for muse_glimmer_text checkpoint key mapping. Not part of upstream.

import Foundation
import MLX
import MLXLMCommon
import XCTest

@testable import MLXLLM

final class SharpAIMuseGlimmerTextSanitizeTests: XCTestCase {

    private func weights(_ keys: [String]) -> [String: MLXArray] {
        var result = [String: MLXArray]()
        for key in keys {
            result[key] = MLXArray.zeros([1])
        }
        return result
    }

    func testTextLayoutKeysPassThrough() {
        let keys = [
            "model.embed_tokens.weight",
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.gate_proj.weight",
            "model.layers.0.self_attn.qk_norm.weight",
            "model.layers.0.input_layernorm.weight",
            "model.norm.weight",
            "lm_head.weight",
        ]
        let sanitized = SharpAIMuseGlimmerTextSanitize.sanitize(weights(keys))

        XCTAssertEqual(Set(sanitized.keys), Set(keys))
    }

    func testVLMLanguageModelPrefixIsRemapped() {
        XCTAssertEqual(
            SharpAIMuseGlimmerTextSanitize.remapKey(
                "model.language_model.layers.3.mlp.up_proj.weight"),
            "model.layers.3.mlp.up_proj.weight")
        XCTAssertEqual(
            SharpAIMuseGlimmerTextSanitize.remapKey("language_model.model.norm.weight"),
            "model.norm.weight")
        XCTAssertEqual(
            SharpAIMuseGlimmerTextSanitize.remapKey("language_model.lm_head.weight"),
            "lm_head.weight")
    }

    func testRotaryInvFreqIsDropped() {
        let sanitized = SharpAIMuseGlimmerTextSanitize.sanitize(
            weights([
                "model.layers.0.self_attn.rotary_emb.inv_freq",
                "model.layers.0.self_attn.q_proj.weight",
            ]))

        XCTAssertEqual(Set(sanitized.keys), ["model.layers.0.self_attn.q_proj.weight"])
    }

    func testVisionKeysAreDropped() {
        let sanitized = SharpAIMuseGlimmerTextSanitize.sanitize(
            weights([
                "model.vision_tower.patch_embedder.weight",
                "model.vision_adapter.proj.weight",
                "model.vision_projection.weight",
                "model.perception_emb_norm.weight",
                "model.embed_tokens.weight",
            ]))

        XCTAssertEqual(Set(sanitized.keys), ["model.embed_tokens.weight"])
    }

    func testModelSanitizeMatchesOverlayFunction() throws {
        let json = SharpAIMuseGlimmerTextTestFixtures.tinyJSON
        let model = SharpAIMuseGlimmerTextModel(
            try SharpAIMuseGlimmerTextTestFixtures.configuration(json))

        let sanitized = model.sanitize(
            weights: weights([
                "model.language_model.embed_tokens.weight",
                "language_model.lm_head.weight",
                "model.layers.0.self_attn.rotary_emb.inv_freq",
            ]))

        XCTAssertEqual(
            Set(sanitized.keys), ["model.embed_tokens.weight", "lm_head.weight"])
    }
}
