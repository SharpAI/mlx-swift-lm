// SharpAI fork overlay tests for the isaac key remap. Not part of upstream.
// Offline only: zero-valued tensors stand in for weights.

import Foundation
import MLX
import XCTest

@testable import MLXVLM

final class SharpAIIsaacSanitizeTests: XCTestCase {

    private func sanitizedKeys(_ keys: [String]) throws -> Set<String> {
        var weights = [String: MLXArray]()
        for key in keys {
            weights[key] = MLXArray.zeros([1])
        }
        return Set(try SharpAIIsaacSanitizer.sanitize(weights).keys)
    }

    func testVisionPrefixesMapToVisionTowerNamespace() throws {
        XCTAssertEqual(
            try sanitizedKeys([
                "model.vision_tower.embeddings.patch_embedding.weight",
                "vision_model.post_layernorm.weight",
                "vision_tower.encoder.layers.0.mlp.fc1.weight",
            ]),
            [
                "vision_tower.embeddings.patch_embedding.weight",
                "vision_tower.post_layernorm.weight",
                "vision_tower.encoder.layers.0.mlp.fc1.weight",
            ])
    }

    func testProjectorAndTextPrefixesMapToModelNamespaces() throws {
        XCTAssertEqual(
            try sanitizedKeys([
                "model.multimodal_projector.linear_1.weight",
                "model.text_model.layers.0.self_attn.q_norm.weight",
                "model.embed_tokens.weight",
            ]),
            [
                "multimodal_projector.linear_1.weight",
                "text_model.layers.0.self_attn.q_norm.weight",
                "text_model.embed_tokens.weight",
            ])
    }

    func testPositionIdsAndRotaryBuffersAreDropped() throws {
        XCTAssertEqual(
            try sanitizedKeys([
                "model.vision_tower.embeddings.position_ids",
                "model.rotary_emb.inv_freq",
                "model.vision_tower.embeddings.position_embedding",
            ]),
            ["vision_tower.embeddings.position_embedding"])
    }

    func testLanguageKeysPassThrough() throws {
        let keys = [
            "language_model.model.embed_tokens.weight",
            "lm_head.weight",
        ]
        XCTAssertEqual(try sanitizedKeys(keys), Set(keys))
    }

    func testTwoKeysMappingToOneNameThrow() {
        var weights = [String: MLXArray]()
        weights["model.vision_tower.post_layernorm.weight"] = MLXArray.zeros([1])
        weights["vision_tower.post_layernorm.weight"] = MLXArray.zeros([1])
        XCTAssertThrowsError(try SharpAIIsaacSanitizer.sanitize(weights)) { error in
            XCTAssertEqual(
                (error as? SharpAIIsaacError)?.errorDescription,
                SharpAIIsaacError.duplicateWeightKey("vision_tower.post_layernorm.weight")
                    .errorDescription)
        }
    }
}
