import Foundation
import MLX
import XCTest

@testable import MLXVLM

final class SharpAIEagle25VlSanitizeTests: XCTestCase {

    private func checkpointWeights() -> [String: MLXArray] {
        [
            "vision_model.vision_model.embeddings.patch_embedding.weight": MLXArray.zeros([8, 3, 14, 14]),
            "vision_model.vision_model.embeddings.patch_embedding.bias": MLXArray.zeros([8]),
            "vision_model.vision_model.embeddings.position_embedding.weight": MLXArray.zeros([4, 8]),
            "vision_model.vision_model.embeddings.position_ids": MLXArray.zeros([1, 4]),
            "vision_model.vision_model.encoder.layers.0.self_attn.q_proj.weight": MLXArray.zeros([8, 8]),
            "vision_model.vision_model.encoder.layers.1.self_attn.q_proj.weight": MLXArray.zeros([8, 8]),
            "vision_model.vision_model.post_layernorm.weight": MLXArray.zeros([8]),
            "vision_model.vision_model.head.probe": MLXArray.zeros([1, 8]),
            "mlp1.0.weight": MLXArray.zeros([32]),
            "mlp1.0.bias": MLXArray.zeros([32]),
            "mlp1.1.weight": MLXArray.zeros([16, 32]),
            "mlp1.3.weight": MLXArray.zeros([16, 16]),
            "language_model.model.embed_tokens.weight": MLXArray.zeros([32, 16]),
            "language_model.model.layers.0.self_attn.rotary_emb.inv_freq": MLXArray.zeros([2]),
            "language_model.lm_head.weight": MLXArray.zeros([32, 16]),
        ]
    }

    func testRemapsProjectorIndicesAndKeepsTextKeys() throws {
        let config = try SharpAIEagle25VlTestFixtures.decode()
        let result = try SharpAIEagle25VlSanitizer.sanitize(
            checkpointWeights(), configuration: config)

        XCTAssertNotNil(result["mlp1.norm.weight"])
        XCTAssertNotNil(result["mlp1.norm.bias"])
        XCTAssertNotNil(result["mlp1.linear_1.weight"])
        XCTAssertNotNil(result["mlp1.linear_2.weight"])
        XCTAssertNil(result["mlp1.0.weight"])
        XCTAssertNil(result["mlp1.1.weight"])
        XCTAssertNil(result["mlp1.3.weight"])

        XCTAssertNotNil(result["language_model.model.embed_tokens.weight"])
        XCTAssertNotNil(result["language_model.lm_head.weight"])
    }

    func testDropsUnusedKeys() throws {
        let config = try SharpAIEagle25VlTestFixtures.decode()
        let result = try SharpAIEagle25VlSanitizer.sanitize(
            checkpointWeights(), configuration: config)

        XCTAssertNil(result["vision_model.vision_model.embeddings.position_ids"])
        XCTAssertNil(result["vision_model.vision_model.head.probe"])
        XCTAssertNil(result["language_model.model.layers.0.self_attn.rotary_emb.inv_freq"])
    }

    func testPatchEmbeddingIsChannelsLast() throws {
        let config = try SharpAIEagle25VlTestFixtures.decode()
        let result = try SharpAIEagle25VlSanitizer.sanitize(
            checkpointWeights(), configuration: config)

        XCTAssertEqual(
            result["vision_model.vision_model.embeddings.patch_embedding.weight"]?.shape,
            [8, 14, 14, 3])
    }

    func testKeepsEncoderLayersAndPostNormWhenAllLayersApply() throws {
        let config = try SharpAIEagle25VlTestFixtures.decode()
        let result = try SharpAIEagle25VlSanitizer.sanitize(
            checkpointWeights(), configuration: config)

        XCTAssertNotNil(result["vision_model.vision_model.encoder.layers.1.self_attn.q_proj.weight"])
        XCTAssertNotNil(result["vision_model.vision_model.post_layernorm.weight"])
    }

    func testTruncatedVisionDropsExtraLayersAndPostNorm() throws {
        let json = SharpAIEagle25VlTestFixtures.tinyJSON
            .replacingOccurrences(of: "\"select_layer\": -1,", with: "\"select_layer\": 0,")
        let config = try SharpAIEagle25VlTestFixtures.decode(json)
        let result = try SharpAIEagle25VlSanitizer.sanitize(
            checkpointWeights(), configuration: config)

        XCTAssertNil(result["vision_model.vision_model.encoder.layers.1.self_attn.q_proj.weight"])
        XCTAssertNotNil(result["vision_model.vision_model.encoder.layers.0.self_attn.q_proj.weight"])
        XCTAssertNil(result["vision_model.vision_model.post_layernorm.weight"])
    }

    func testTiedEmbeddingsDropLMHead() throws {
        let json = SharpAIEagle25VlTestFixtures.tinyJSON
            .replacingOccurrences(
                of: "\"tie_word_embeddings\": false", with: "\"tie_word_embeddings\": true")
        let config = try SharpAIEagle25VlTestFixtures.decode(json)
        let result = try SharpAIEagle25VlSanitizer.sanitize(
            checkpointWeights(), configuration: config)

        XCTAssertNil(result["language_model.lm_head.weight"])
        XCTAssertNotNil(result["language_model.model.embed_tokens.weight"])
    }

    func testDuplicateAfterRemapThrows() throws {
        let config = try SharpAIEagle25VlTestFixtures.decode()
        var weights = checkpointWeights()
        weights["mlp1.norm.weight"] = MLXArray.zeros([32])

        XCTAssertThrowsError(
            try SharpAIEagle25VlSanitizer.sanitize(weights, configuration: config))
    }
}
