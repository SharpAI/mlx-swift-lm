import Foundation
import MLX
import XCTest

@testable import MLXVLM

final class SharpAIPhi4SiglipSanitizeTests: XCTestCase {

    private let vision = "model.vision_tower.vision_tower.vision_model."

    private func checkpointWeights() -> [String: MLXArray] {
        [
            "\(vision)embeddings.patch_embedding.weight": MLXArray.zeros([8, 12]),
            "\(vision)embeddings.patch_embedding.bias": MLXArray.zeros([8]),
            "\(vision)embeddings.position_embedding.weight": MLXArray.zeros([4, 8]),
            "\(vision)embeddings.position_ids": MLXArray.zeros([1, 4]),
            "\(vision)encoder.layers.0.self_attn.q_proj.weight": MLXArray.zeros([8, 8]),
            "\(vision)encoder.layers.1.self_attn.q_proj.weight": MLXArray.zeros([8, 8]),
            "\(vision)post_layernorm.weight": MLXArray.zeros([8]),
            "\(vision)head.probe": MLXArray.zeros([1, 8]),
            "model.mm_projector.0.weight": MLXArray.zeros([16, 8]),
            "model.mm_projector.0.bias": MLXArray.zeros([16]),
            "model.mm_projector.2.weight": MLXArray.zeros([16, 16]),
            "model.embed_tokens.weight": MLXArray.zeros([32, 16]),
            "model.layers.0.self_attn.qkv_proj.weight": MLXArray.zeros([64, 16]),
            "model.layers.0.self_attn.o_proj.weight": MLXArray.zeros([16, 16]),
            "model.layers.0.self_attn.rotary_emb.inv_freq": MLXArray.zeros([2]),
            "model.layers.0.mlp.gate_up_proj.weight": MLXArray.zeros([64, 16]),
            "model.layers.0.mlp.down_proj.weight": MLXArray.zeros([16, 32]),
            "model.norm.weight": MLXArray.zeros([16]),
            "lm_head.weight": MLXArray.zeros([32, 16]),
        ]
    }

    func testKeepsModuleKeysUnchanged() throws {
        let config = try SharpAIPhi4SiglipTestFixtures.decode()
        let result = try SharpAIPhi4SiglipSanitizer.sanitize(
            checkpointWeights(), configuration: config)

        XCTAssertNotNil(result["model.mm_projector.0.weight"])
        XCTAssertNotNil(result["model.mm_projector.0.bias"])
        XCTAssertNotNil(result["model.mm_projector.2.weight"])
        XCTAssertNotNil(result["model.embed_tokens.weight"])
        XCTAssertNotNil(result["model.layers.0.self_attn.qkv_proj.weight"])
        XCTAssertNotNil(result["model.layers.0.self_attn.o_proj.weight"])
        XCTAssertNotNil(result["model.layers.0.mlp.gate_up_proj.weight"])
        XCTAssertNotNil(result["model.layers.0.mlp.down_proj.weight"])
        XCTAssertNotNil(result["model.norm.weight"])
        XCTAssertNotNil(result["lm_head.weight"])
        XCTAssertNotNil(result["\(vision)encoder.layers.1.self_attn.q_proj.weight"])
        XCTAssertNotNil(result["\(vision)post_layernorm.weight"])
    }

    func testDropsUnusedKeys() throws {
        let config = try SharpAIPhi4SiglipTestFixtures.decode()
        let result = try SharpAIPhi4SiglipSanitizer.sanitize(
            checkpointWeights(), configuration: config)

        XCTAssertNil(result["\(vision)embeddings.position_ids"])
        XCTAssertNil(result["\(vision)head.probe"])
        XCTAssertNil(result["model.layers.0.self_attn.rotary_emb.inv_freq"])
    }

    func testDropsLmHeadWhenEmbeddingsAreTied() throws {
        let json = SharpAIPhi4SiglipTestFixtures.tinyJSON
            .replacingOccurrences(
                of: "\"tie_word_embeddings\": false,", with: "\"tie_word_embeddings\": true,")
        let config = try SharpAIPhi4SiglipTestFixtures.decode(json)
        let result = try SharpAIPhi4SiglipSanitizer.sanitize(
            checkpointWeights(), configuration: config)

        XCTAssertNil(result["lm_head.weight"])
        XCTAssertNotNil(result["model.embed_tokens.weight"])
    }

    func testConvPatchWeightBecomesFlatLinearWeight() throws {
        let config = try SharpAIPhi4SiglipTestFixtures.decode()
        var weights = checkpointWeights()
        weights["\(vision)embeddings.patch_embedding.weight"] = MLXArray.zeros([8, 3, 2, 2])

        let result = try SharpAIPhi4SiglipSanitizer.sanitize(weights, configuration: config)

        XCTAssertEqual(
            result["\(vision)embeddings.patch_embedding.weight"]?.shape, [8, 12])
    }

    func testLinearPatchWeightIsUnchanged() throws {
        let config = try SharpAIPhi4SiglipTestFixtures.decode()
        let result = try SharpAIPhi4SiglipSanitizer.sanitize(
            checkpointWeights(), configuration: config)

        XCTAssertEqual(
            result["\(vision)embeddings.patch_embedding.weight"]?.shape, [8, 12])
    }
}
