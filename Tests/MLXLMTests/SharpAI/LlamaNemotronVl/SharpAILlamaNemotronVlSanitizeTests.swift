// SharpAI fork overlay tests for the LlamaNemotronVL key remap. Not part of upstream.
//
// Offline checks only: zero-valued tensors stand in for weights, so no checkpoint is read.

import Foundation
import MLX
import XCTest

@testable import MLXVLM

final class SharpAILlamaNemotronVlSanitizeTests: XCTestCase {

    private func sanitizedKeys(_ keys: [String]) -> Set<String> {
        var weights = [String: MLXArray]()
        for key in keys {
            weights[key] = MLXArray.zeros([1])
        }
        return Set(SharpAILlamaNemotronVlSanitize.sanitize(weights: weights).keys)
    }

    func testVisionAndLanguageKeysKeepTheirPaths() {
        let keys = [
            "vision_model.embeddings.position_embedding.weight",
            "vision_model.encoder.layers.0.self_attn.q_proj.weight",
            "vision_model.encoder.layers.0.mlp.fc1.bias",
            "vision_model.post_layernorm.weight",
            "language_model.embed_tokens.weight",
            "language_model.layers.0.self_attn.q_proj.weight",
            "language_model.layers.15.mlp.down_proj.weight",
            "language_model.norm.weight",
        ]
        XCTAssertEqual(sanitizedKeys(keys), Set(keys))
    }

    func testStandardLlamaModelSegmentIsDropped() {
        XCTAssertEqual(
            sanitizedKeys(["language_model.model.layers.1.mlp.up_proj.weight"]),
            ["language_model.layers.1.mlp.up_proj.weight"])
        XCTAssertEqual(
            sanitizedKeys(["language_model.model.norm.weight"]),
            ["language_model.norm.weight"])
    }

    func testProjectorIndicesMapToNamedLayers() {
        XCTAssertEqual(
            sanitizedKeys(["mlp1.0.weight", "mlp1.0.bias", "mlp1.1.weight", "mlp1.3.weight", "mlp1.3.bias"]),
            [
                "mlp1.norm.weight", "mlp1.norm.bias", "mlp1.fc1.weight",
                "mlp1.fc2.weight", "mlp1.fc2.bias",
            ])
    }

    func testBuffersAndHeadsAreDropped() {
        XCTAssertEqual(
            sanitizedKeys([
                "vision_model.embeddings.position_ids",
                "norm_mean",
                "norm_std",
                "language_model.rotary_emb.inv_freq",
                "lm_head.weight",
                "language_model.lm_head.weight",
            ]),
            [])
    }

    func testTorchPatchEmbeddingIsMovedToMLXLayout() {
        let torchLayout = MLXArray.zeros([16, 3, 14, 14])
        let mlxLayout = MLXArray.zeros([16, 14, 14, 3])
        let weights = [
            "vision_model.embeddings.patch_embedding.weight": torchLayout,
            "vision_model.embeddings.position_embedding.weight": MLXArray.zeros([4, 16]),
        ]

        let sanitized = SharpAILlamaNemotronVlSanitize.sanitize(weights: weights)

        XCTAssertEqual(
            sanitized["vision_model.embeddings.patch_embedding.weight"]?.shape, [16, 14, 14, 3])
        XCTAssertEqual(
            sanitized["vision_model.embeddings.position_embedding.weight"]?.shape, [4, 16])
        XCTAssertFalse(SharpAILlamaNemotronVlSanitize.isTorchConvLayout(mlxLayout))
    }
}
