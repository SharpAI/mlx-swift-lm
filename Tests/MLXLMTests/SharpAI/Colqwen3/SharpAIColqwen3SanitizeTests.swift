import Foundation
import MLX
import XCTest

@testable import MLXVLM

final class SharpAIColqwen3SanitizeTests: XCTestCase {

    private func sanitizedKeys(_ keys: [String], tied: Bool = true) throws -> Set<String> {
        let json = tied
            ? SharpAIColqwen3TestFixtures.tinyConfigJSON
            : SharpAIColqwen3TestFixtures.tinyConfigJSON.replacingOccurrences(
                of: "\"tie_word_embeddings\": true", with: "\"tie_word_embeddings\": false")
        let config = try JSONDecoder().decode(
            SharpAIColqwen3Configuration.self, from: Data(json.utf8))
        var weights: [String: MLXArray] = [:]
        for key in keys {
            weights[key] = MLXArray(Float(0))
        }
        return Set(SharpAIColqwen3Sanitize.sanitize(weights: weights, config: config).keys)
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

    /// UNVERIFIED: the "vlm." wrapper prefix comes from colqwen2 layout, not from a ColQwen3 checkpoint.
    func testVlmWrapperPrefixIsStripped() throws {
        let result = try sanitizedKeys([
            "vlm.model.language_model.norm.weight",
            "embedding_proj_layer.weight",
        ])

        XCTAssertEqual(
            result,
            [
                "language_model.model.norm.weight",
                "embedding_proj_layer.weight",
            ])
    }
}
