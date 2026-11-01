// SharpAI fork overlay tests for ModernVBERT key remapping. Not part of upstream.
//
// Offline checks only: key names and one patch-conv shape. No weights.

import Foundation
import MLX
import XCTest

@testable import MLXVLM

final class SharpAIModernvbertSanitizeTests: XCTestCase {

    func testColModernVBertPrefixIsKeptAsIs() {
        XCTAssertEqual(
            SharpAIModernvbertSanitize.mappedName("model.text_model.embeddings.tok_embeddings.weight"),
            "model.text_model.embeddings.tok_embeddings.weight")
        XCTAssertEqual(
            SharpAIModernvbertSanitize.mappedName("model.connector.modality_projection.proj.weight"),
            "model.connector.modality_projection.proj.weight")
    }

    func testPlainModernVBertKeysGainModelPrefix() {
        XCTAssertEqual(
            SharpAIModernvbertSanitize.mappedName("text_model.layers.1.attn.Wqkv.weight"),
            "model.text_model.layers.1.attn.Wqkv.weight")
        XCTAssertEqual(
            SharpAIModernvbertSanitize.mappedName("connector.modality_projection.proj.weight"),
            "model.connector.modality_projection.proj.weight")
    }

    func testDoubledSiglipWrapperCollapses() {
        XCTAssertEqual(
            SharpAIModernvbertSanitize.mappedName(
                "model.vision_model.vision_model.encoder.layers.0.self_attn.q_proj.weight"),
            "model.vision_model.encoder.layers.0.self_attn.q_proj.weight")
        XCTAssertEqual(
            SharpAIModernvbertSanitize.mappedName(
                "vision_model.vision_model.post_layernorm.weight"),
            "model.vision_model.post_layernorm.weight")
    }

    func testEmbeddingProjectionIsKept() {
        XCTAssertEqual(
            SharpAIModernvbertSanitize.mappedName("embedding_proj_layer.weight"),
            "embedding_proj_layer.weight")
    }

    func testCustomTextProjectionMapsToEmbeddingProjection() {
        XCTAssertEqual(
            SharpAIModernvbertSanitize.mappedName("custom_text_proj.weight"),
            "embedding_proj_layer.weight")
    }

    func testMaskedLMHeadAndRotaryCachesAreDropped() {
        XCTAssertNil(SharpAIModernvbertSanitize.mappedName("lm_head.weight"))
        XCTAssertNil(SharpAIModernvbertSanitize.mappedName("decoder.weight"))
        XCTAssertNil(
            SharpAIModernvbertSanitize.mappedName("model.text_model.rotary_emb.inv_freq"))
        XCTAssertNil(
            SharpAIModernvbertSanitize.mappedName(
                "model.vision_model.embeddings.position_ids"))
    }

    func testSiglipPoolingHeadIsDropped() {
        XCTAssertNil(
            SharpAIModernvbertSanitize.mappedName("model.vision_model.head.probe"))
        XCTAssertNil(
            SharpAIModernvbertSanitize.mappedName("vision_model.vision_model.head.probe"))
    }

    func testUnknownKeysPassThroughUnchanged() {
        XCTAssertEqual(SharpAIModernvbertSanitize.mappedName("something.else.weight"), "something.else.weight")
    }

    func testSanitizeDropsAndRenamesWholeDictionary() {
        let weights: [String: MLXArray] = [
            "text_model.final_norm.weight": MLXArray.ones([16]),
            "lm_head.weight": MLXArray.zeros([64, 16]),
            "embedding_proj_layer.weight": MLXArray.ones([16, 16]),
        ]

        let sanitized = SharpAIModernvbertSanitize.sanitize(weights: weights)

        XCTAssertEqual(Set(sanitized.keys), ["model.text_model.final_norm.weight", "embedding_proj_layer.weight"])
    }

    func testPatchEmbeddingIsMovedToMLXLayout() {
        // PyTorch NCHW layout: [out 8, in 3, kH 8, kW 8].
        let weights: [String: MLXArray] = [
            "vision_model.embeddings.patch_embedding.weight": MLXArray.zeros([8, 3, 8, 8])
        ]

        let sanitized = SharpAIModernvbertSanitize.sanitize(weights: weights)

        XCTAssertEqual(sanitized["model.vision_model.embeddings.patch_embedding.weight"]?.shape, [8, 8, 8, 3])
    }

    func testPatchEmbeddingAlreadyInMLXLayoutIsUnchanged() {
        // MLX layout: [out 8, kH 8, kW 8, in 3].
        let weights: [String: MLXArray] = [
            "model.vision_model.embeddings.patch_embedding.weight": MLXArray.zeros([8, 8, 8, 3])
        ]

        let sanitized = SharpAIModernvbertSanitize.sanitize(weights: weights)

        XCTAssertEqual(sanitized["model.vision_model.embeddings.patch_embedding.weight"]?.shape, [8, 8, 8, 3])
    }
}
