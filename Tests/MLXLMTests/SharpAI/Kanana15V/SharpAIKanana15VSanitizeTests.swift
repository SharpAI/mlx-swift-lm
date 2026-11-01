import Foundation
import MLX
import XCTest

@testable import MLXVLM

final class SharpAIKanana15VSanitizeTests: XCTestCase {

    private func checkpointWeights() -> [String: MLXArray] {
        [
            // PyTorch Conv3d layout: [out, in, T, H, W].
            "vision_model.patch_embed.proj.weight": MLXArray.zeros([8, 3, 2, 2, 2]),
            "vision_model.blocks.0.attn.qkv.weight": MLXArray.zeros([24, 8]),
            "vision_model.position_ids": MLXArray.zeros([1, 4]),
            "language_model.model.embed_tokens.weight": MLXArray.zeros([32, 16]),
            "language_model.model.layers.0.self_attn.rotary_emb.inv_freq": MLXArray.zeros([4]),
            "language_model.model.layers.0.self_attn.q_proj.weight": MLXArray.zeros([16, 16]),
            "language_model.lm_head.weight": MLXArray.zeros([32, 16]),
        ]
    }

    func testPatchEmbedIsMovedToMLXLayout() throws {
        let config = try SharpAIKanana15VTestFixtures.decode()
        let result = try SharpAIKanana15VSanitizer.sanitize(
            checkpointWeights(), configuration: config)

        let patch = try XCTUnwrap(result["vision_model.patch_embed.proj.weight"])
        XCTAssertEqual(patch.shape, [8, 2, 2, 2, 3])
    }

    func testDropsPositionIdsAndRotaryInvFreq() throws {
        let config = try SharpAIKanana15VTestFixtures.decode()
        let result = try SharpAIKanana15VSanitizer.sanitize(
            checkpointWeights(), configuration: config)

        XCTAssertNil(result["vision_model.position_ids"])
        XCTAssertNil(result["language_model.model.layers.0.self_attn.rotary_emb.inv_freq"])
    }

    func testKeepsTextAndVisionKeysUnchanged() throws {
        let config = try SharpAIKanana15VTestFixtures.decode()
        let result = try SharpAIKanana15VSanitizer.sanitize(
            checkpointWeights(), configuration: config)

        XCTAssertNotNil(result["vision_model.blocks.0.attn.qkv.weight"])
        XCTAssertNotNil(result["language_model.model.embed_tokens.weight"])
        XCTAssertNotNil(result["language_model.model.layers.0.self_attn.q_proj.weight"])
    }

    func testKeepsLMHeadWhenEmbeddingsAreUntied() throws {
        let config = try SharpAIKanana15VTestFixtures.decode()
        let result = try SharpAIKanana15VSanitizer.sanitize(
            checkpointWeights(), configuration: config)

        XCTAssertNotNil(result["language_model.lm_head.weight"])
    }

    func testDropsLMHeadWhenEmbeddingsAreTied() throws {
        let json = SharpAIKanana15VTestFixtures.tinyJSON
            .replacingOccurrences(
                of: "\"tie_word_embeddings\": false", with: "\"tie_word_embeddings\": true")
        let config = try SharpAIKanana15VTestFixtures.decode(json)
        let result = try SharpAIKanana15VSanitizer.sanitize(
            checkpointWeights(), configuration: config)

        XCTAssertNil(result["language_model.lm_head.weight"])
    }
}
