// SharpAI fork overlay tests for the ColPali key remap. Not part of upstream.
//
// Offline checks only: zero-valued tensors stand in for weights, so no checkpoint is read.

import Foundation
import MLX
import XCTest

@testable import MLXVLM

final class SharpAIColpaliSanitizeTests: XCTestCase {

    private func sanitizedKeys(_ keys: [String]) -> Set<String> {
        var weights = [String: MLXArray]()
        for key in keys {
            weights[key] = MLXArray.zeros([1])
        }
        return Set(SharpAIColpaliSanitize.sanitize(weights: weights).keys)
    }

    func testVLMWrappedKeysKeepTheirPaths() {
        let keys = [
            "vlm.vision_tower.vision_model.encoder.layers.0.mlp.fc1.weight",
            "vlm.language_model.model.layers.0.self_attn.q_proj.weight",
            "vlm.language_model.model.norm.weight",
            "vlm.multi_modal_projector.linear.weight",
            "embedding_proj_layer.weight",
            "embedding_proj_layer.bias",
        ]
        XCTAssertEqual(sanitizedKeys(keys), Set(keys))
    }

    func testModelPrefixIsReplacedByVLMPrefix() {
        XCTAssertEqual(
            sanitizedKeys(["model.vision_tower.vision_model.post_layernorm.weight"]),
            ["vlm.vision_tower.vision_model.post_layernorm.weight"])
    }

    func testMissingModelSegmentInsideLanguageModelIsAdded() {
        XCTAssertEqual(
            sanitizedKeys(["vlm.language_model.layers.1.mlp.up_proj.weight"]),
            ["vlm.language_model.model.layers.1.mlp.up_proj.weight"])
        XCTAssertEqual(
            sanitizedKeys(["model.language_model.norm.weight"]),
            ["vlm.language_model.model.norm.weight"])
    }

    func testBareBackboneKeysGetVLMPrefix() {
        XCTAssertEqual(
            sanitizedKeys(["multi_modal_projector.linear.bias"]),
            ["vlm.multi_modal_projector.linear.bias"])
    }

    func testCustomTextProjectionMapsToEmbeddingProjLayer() {
        XCTAssertEqual(
            sanitizedKeys(["custom_text_proj.weight", "custom_text_proj.bias"]),
            ["embedding_proj_layer.weight", "embedding_proj_layer.bias"])
        XCTAssertEqual(
            sanitizedKeys(["model.custom_text_proj.weight"]),
            ["embedding_proj_layer.weight"])
    }

    func testLMHeadAndRotaryCacheAreDropped() {
        let keys = [
            "vlm.language_model.lm_head.weight",
            "vlm.language_model.model.layers.0.self_attn.rotary_emb.inv_freq",
            "vlm.language_model.model.norm.weight",
        ]
        XCTAssertEqual(sanitizedKeys(keys), ["vlm.language_model.model.norm.weight"])
    }

    func testVisionPositionIdsAreDropped() {
        let keys = [
            "vlm.vision_tower.vision_model.embeddings.position_ids",
            "vlm.vision_tower.vision_model.embeddings.position_embedding.weight",
        ]
        XCTAssertEqual(
            sanitizedKeys(keys),
            ["vlm.vision_tower.vision_model.embeddings.position_embedding.weight"])
    }

    func testPatchEmbeddingIsMovedToMLXLayout() {
        var weights = [String: MLXArray]()
        weights["vlm.vision_tower.vision_model.embeddings.patch_embedding.weight"] =
            MLXArray.zeros([16, 3, 14, 14])

        let result = SharpAIColpaliSanitize.sanitize(weights: weights)
        XCTAssertEqual(
            result["vlm.vision_tower.vision_model.embeddings.patch_embedding.weight"]?.shape,
            [16, 14, 14, 3])
    }
}
