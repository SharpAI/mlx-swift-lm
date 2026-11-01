import Foundation
import MLX
import MLXNN
import XCTest

@testable import MLXVLM

final class SharpAIQwen3VlNemotronEmbedSanitizeTests: XCTestCase {

    private func sanitizedKeys(_ keys: [String], tied: Bool = true) throws -> Set<String> {
        let json =
            tied
            ? SharpAIQwen3VlNemotronEmbedTestFixtures.tinyConfigJSON
            : SharpAIQwen3VlNemotronEmbedTestFixtures.tinyConfigJSON.replacingOccurrences(
                of: "\"tie_word_embeddings\": true", with: "\"tie_word_embeddings\": false")
        let config = try JSONDecoder().decode(
            SharpAIQwen3VlNemotronEmbedConfiguration.self, from: Data(json.utf8))
        var weights: [String: MLXArray] = [:]
        for key in keys {
            weights[key] = MLXArray(Float(0))
        }
        return Set(SharpAIQwen3VlNemotronEmbedSanitize.sanitize(weights: weights, config: config).keys)
    }

    func testQwen3VLStyleKeysMapToModulePaths() throws {
        let result = try sanitizedKeys([
            "model.visual.blocks.0.attn.qkv.weight",
            "model.language_model.layers.0.mlp.gate_proj.weight",
            "model.language_model.embed_tokens.weight",
            "embedding_proj_layer.weight",
            "embedding_proj_layer.bias",
        ])

        XCTAssertEqual(
            result,
            [
                "vision_tower.blocks.0.attn.qkv.weight",
                "language_model.model.layers.0.mlp.gate_proj.weight",
                "language_model.model.embed_tokens.weight",
                "embedding_proj_layer.weight",
                "embedding_proj_layer.bias",
            ])
    }

    func testTiedEmbeddingsDropLmHead() throws {
        let result = try sanitizedKeys([
            "lm_head.weight",
            "language_model.lm_head.weight",
        ])

        XCTAssertTrue(result.isEmpty, "tied embeddings must not keep an lm_head")
    }

    func testUntiedEmbeddingsKeepLmHead() throws {
        let result = try sanitizedKeys(["lm_head.weight"], tied: false)

        XCTAssertEqual(result, ["language_model.lm_head.weight"])
    }

    func testPositionIdsAreDropped() throws {
        let result = try sanitizedKeys([
            "model.visual.pos_ids.position_ids",
            "embedding_proj_layer.weight",
        ])

        XCTAssertEqual(result, ["embedding_proj_layer.weight"])
    }

    /// Key names copied from the real checkpoint index of nvidia/nemotron-colembed-vl-8b-v2
    /// (model.safetensors.index.json, top-level tie_word_embeddings = false). The backbone keys
    /// must map onto module paths. The reference modeling file (modeling_qwen3_vl_nemotron_embed.py)
    /// has no projection head, and the checkpoint has no `embedding_proj_layer` key.
    func testRealCheckpointKeyLayout() throws {
        let realKeys = [
            "model.language_model.embed_tokens.weight",
            "model.language_model.layers.0.self_attn.q_proj.weight",
            "model.visual.blocks.0.attn.qkv.weight",
            "lm_head.weight",
        ]
        let result = try sanitizedKeys(realKeys, tied: false)

        XCTAssertTrue(result.contains("language_model.model.embed_tokens.weight"))
        XCTAssertTrue(result.contains("language_model.model.layers.0.self_attn.q_proj.weight"))
        XCTAssertTrue(result.contains("vision_tower.blocks.0.attn.qkv.weight"))
        XCTAssertTrue(result.contains("language_model.lm_head.weight"))

        // The model must expose no embedding head, because no real checkpoint key feeds one.
        // The embedding path is the backbone itself, per-token and bidirectional, with no projection.
        let config = try JSONDecoder().decode(
            SharpAIQwen3VlNemotronEmbedConfiguration.self,
            from: Data(SharpAIQwen3VlNemotronEmbedTestFixtures.tinyConfigJSON.utf8))
        let modelKeys = Set(
            SharpAIQwen3VlNemotronEmbedModel(config).parameters().flattened().map { $0.0 })
        XCTAssertFalse(modelKeys.contains("embedding_proj_layer.weight"))
        XCTAssertFalse(modelKeys.contains { $0.hasPrefix("embedding_proj_layer") })
        XCTAssertTrue(modelKeys.contains("language_model.model.embed_tokens.weight"))
    }
}
