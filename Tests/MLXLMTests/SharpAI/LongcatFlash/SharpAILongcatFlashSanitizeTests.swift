// SharpAI fork overlay tests for longcat_flash checkpoint key mapping. Not part of upstream.

import Foundation
import MLX
import XCTest

@testable import MLXLLM

final class SharpAILongcatFlashSanitizeTests: XCTestCase {

    func testRenamesBaseEmbeddingToNgramWordEmbeddings() throws {
        let model = try SharpAILongcatFlashTestFixtures.model()
        let out = model.sanitize(weights: [
            "model.embed_tokens.weight": MLXArray.zeros([32, 16])
        ])

        XCTAssertNil(out["model.embed_tokens.weight"])
        XCTAssertEqual(
            out["model.ngram_embeddings.word_embeddings.weight"]?.shape, [32, 16])
    }

    func testStacksPerExpertWeightsIntoSwitchMLP() throws {
        let model = try SharpAILongcatFlashTestFixtures.model(numLayers: 1)
        var weights: [String: MLXArray] = [:]
        for e in 0 ..< 4 {
            weights["model.layers.0.mlp.experts.\(e).gate_proj.weight"] = MLXArray.zeros([8, 16])
        }
        let out = model.sanitize(weights: weights)

        XCTAssertEqual(out["model.layers.0.mlp.switch_mlp.gate_proj.weight"]?.shape, [4, 8, 16])
        XCTAssertNil(out["model.layers.0.mlp.experts.0.gate_proj.weight"])
        XCTAssertNil(out["model.layers.0.mlp.experts.3.gate_proj.weight"])
    }

    func testLeavesPartialExpertSetUntouched() throws {
        let model = try SharpAILongcatFlashTestFixtures.model(numLayers: 1)
        let weights: [String: MLXArray] = [
            "model.layers.0.mlp.experts.0.up_proj.weight": MLXArray.zeros([8, 16]),
            "model.layers.0.mlp.experts.1.up_proj.weight": MLXArray.zeros([8, 16]),
        ]
        let out = model.sanitize(weights: weights)

        XCTAssertNil(out["model.layers.0.mlp.switch_mlp.up_proj.weight"])
        XCTAssertNotNil(out["model.layers.0.mlp.experts.0.up_proj.weight"])
    }

    func testKeepsKvBProjNameAndDropsMultiTokenPredictionKeys() throws {
        let model = try SharpAILongcatFlashTestFixtures.model(numLayers: 1)
        let out = model.sanitize(weights: [
            "model.layers.0.self_attn.0.kv_b_proj.weight": MLXArray.zeros([64, 8]),
            "model.mtp.layers.0.eh_proj.weight": MLXArray.zeros([16, 16]),
        ])

        XCTAssertNotNil(out["model.layers.0.self_attn.0.kv_b_proj.weight"])
        XCTAssertNil(out["model.mtp.layers.0.eh_proj.weight"])
    }

    func testKeepsRouterAndNormKeysUnchanged() throws {
        let model = try SharpAILongcatFlashTestFixtures.model(numLayers: 1)
        let out = model.sanitize(weights: [
            "model.layers.0.mlp.router.classifier.weight": MLXArray.zeros([6, 16]),
            "model.layers.0.mlp.router.e_score_correction_bias": MLXArray.zeros([6]),
            "model.norm.weight": MLXArray.zeros([16]),
        ])

        XCTAssertNotNil(out["model.layers.0.mlp.router.classifier.weight"])
        XCTAssertNotNil(out["model.layers.0.mlp.router.e_score_correction_bias"])
        XCTAssertNotNil(out["model.norm.weight"])
    }
}
