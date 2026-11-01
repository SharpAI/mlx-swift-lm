// SharpAI fork overlay tests for qianfan_ocr checkpoint key sanitization. Not part of upstream.
// Shapes only. No weights are read.

import Foundation
import MLX
@testable import MLXVLM
import XCTest

final class SharpAIQianfanOcrSanitizeTests: XCTestCase {

    private func tinyConfig() throws -> SharpAIQianfanOcrConfiguration {
        try SharpAIQianfanOcrTestConfig.decode(SharpAIQianfanOcrTestConfig.tiny())
    }

    func testFusedQKVWeightSplitsIntoThreeProjections() throws {
        // Tiny hidden size is 16, so the stacked qkv weight is [48, 16].
        let weights: [String: MLXArray] = [
            "vision_model.encoder.layers.0.attn.qkv.weight": MLXArray.zeros([48, 16])
        ]

        let result = SharpAIQianfanOcrSanitize.sanitize(weights: weights, config: try tinyConfig())

        XCTAssertEqual(result["vision_tower.layers.0.attention.q_proj.weight"]?.shape, [16, 16])
        XCTAssertEqual(result["vision_tower.layers.0.attention.k_proj.weight"]?.shape, [16, 16])
        XCTAssertEqual(result["vision_tower.layers.0.attention.v_proj.weight"]?.shape, [16, 16])
        XCTAssertFalse(result.keys.contains { $0.contains("attn.qkv") })
        XCTAssertEqual(result.count, 3)
    }

    func testFusedQKVBiasSplitsIntoThreeVectors() throws {
        let weights: [String: MLXArray] = [
            "vision_model.encoder.layers.1.attn.qkv.bias": MLXArray.zeros([48])
        ]

        let result = SharpAIQianfanOcrSanitize.sanitize(weights: weights, config: try tinyConfig())

        XCTAssertEqual(result["vision_tower.layers.1.attention.q_proj.bias"]?.shape, [16])
        XCTAssertEqual(result["vision_tower.layers.1.attention.k_proj.bias"]?.shape, [16])
        XCTAssertEqual(result["vision_tower.layers.1.attention.v_proj.bias"]?.shape, [16])
    }

    func testPatchConvWeightMovesToChannelLast() throws {
        // PyTorch [out, in, kH, kW] = [16, 3, 14, 14] becomes MLX [out, kH, kW, in].
        let weights: [String: MLXArray] = [
            "vision_model.embeddings.patch_embedding.weight": MLXArray.zeros([16, 3, 14, 14])
        ]

        let result = SharpAIQianfanOcrSanitize.sanitize(weights: weights, config: try tinyConfig())

        XCTAssertEqual(
            result["vision_tower.embeddings.patch_embeddings.projection.weight"]?.shape,
            [16, 14, 14, 3])
    }

    func testVisionNamesMapToModulePaths() {
        let expected: [String: String] = [
            "vision_model.embeddings.class_embedding":
                "vision_tower.embeddings.cls_token",
            "vision_model.embeddings.position_embedding":
                "vision_tower.embeddings.position_embeddings",
            "vision_model.embeddings.patch_embedding.bias":
                "vision_tower.embeddings.patch_embeddings.projection.bias",
            "vision_model.encoder.layers.1.attn.proj.weight":
                "vision_tower.layers.1.attention.projection_layer.weight",
            "vision_model.encoder.layers.1.norm1.bias":
                "vision_tower.layers.1.layernorm_before.bias",
            "vision_model.encoder.layers.1.norm2.weight":
                "vision_tower.layers.1.layernorm_after.weight",
            "vision_model.encoder.layers.1.ls1": "vision_tower.layers.1.lambda_1",
            "vision_model.encoder.layers.1.ls2": "vision_tower.layers.1.lambda_2",
            "vision_model.encoder.layers.1.mlp.fc1.weight":
                "vision_tower.layers.1.mlp.fc1.weight",
            "model.vision_tower.layers.1.mlp.fc2.weight":
                "vision_tower.layers.1.mlp.fc2.weight",
        ]
        for (checkpoint, module) in expected {
            XCTAssertEqual(
                SharpAIQianfanOcrSanitize.checkpointName(checkpoint), module, checkpoint)
        }
    }

    func testProjectorAndLanguageNamesMapToModulePaths() {
        XCTAssertEqual(
            SharpAIQianfanOcrSanitize.checkpointName("mlp1.0.weight"),
            "multi_modal_projector.layer_norm.weight")
        XCTAssertEqual(
            SharpAIQianfanOcrSanitize.checkpointName("mlp1.1.bias"),
            "multi_modal_projector.linear_1.bias")
        XCTAssertEqual(
            SharpAIQianfanOcrSanitize.checkpointName("mlp1.3.weight"),
            "multi_modal_projector.linear_2.weight")
        XCTAssertEqual(
            SharpAIQianfanOcrSanitize.checkpointName("language_model.model.layers.0.self_attn.q_norm.weight"),
            "language_model.model.layers.0.self_attn.q_norm.weight")
        XCTAssertEqual(
            SharpAIQianfanOcrSanitize.checkpointName("language_model.lm_head.weight"),
            "language_model.lm_head.weight")
        XCTAssertEqual(
            SharpAIQianfanOcrSanitize.checkpointName("lm_head.weight"),
            "language_model.lm_head.weight")
    }

    func testRotaryBuffersAreDropped() throws {
        let weights: [String: MLXArray] = [
            "language_model.model.layers.0.self_attn.rotary_emb.inv_freq": MLXArray.zeros([4]),
            "language_model.model.embed_tokens.weight": MLXArray.zeros([40, 16]),
        ]

        let result = SharpAIQianfanOcrSanitize.sanitize(weights: weights, config: try tinyConfig())

        XCTAssertEqual(result.count, 1)
        XCTAssertNotNil(result["language_model.model.embed_tokens.weight"])
    }

    func testTiedEmbeddingsDropLMHead() throws {
        var config = try tinyConfig()
        config = SharpAIQianfanOcrConfiguration(
            visionConfiguration: config.visionConfiguration,
            textConfiguration: SharpAIQianfanOcrTextConfiguration(
                hiddenSize: 16, hiddenLayers: 2, intermediateSize: 32, attentionHeads: 2,
                kvHeads: 1, headDim: 8, vocabularySize: 40, tieWordEmbeddings: true),
            imageTokenId: 30)
        let weights: [String: MLXArray] = [
            "language_model.lm_head.weight": MLXArray.zeros([40, 16]),
            "language_model.model.embed_tokens.weight": MLXArray.zeros([40, 16]),
        ]

        let result = SharpAIQianfanOcrSanitize.sanitize(weights: weights, config: config)

        XCTAssertNil(result["language_model.lm_head.weight"])
        XCTAssertNotNil(result["language_model.model.embed_tokens.weight"])
    }
}
