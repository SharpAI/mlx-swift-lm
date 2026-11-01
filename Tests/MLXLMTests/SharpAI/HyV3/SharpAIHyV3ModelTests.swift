// SharpAI fork overlay tests for the hy_v3 model. Not part of upstream.
//
// Offline checks only: a tiny synthetic config, random weights, no network.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import XCTest

@testable import MLXLLM

final class SharpAIHyV3ModelTests: XCTestCase {

    private func tinyModel(tied: Bool = false) throws -> SharpAIHyV3Model {
        let config = try JSONDecoder.json5().decode(
            SharpAIHyV3Configuration.self,
            from: Data(SharpAIHyV3TestFixture.tinyJSON(tied: tied).utf8))
        return SharpAIHyV3Model(config)
    }

    func testForwardPassProducesLogitsShape() throws {
        let model = try tinyModel()
        let inputs = MLXArray([0, 1, 2, 3, 4] as [Int32]).reshaped(1, 5)
        let logits = model(inputs, cache: nil)
        eval(logits)
        XCTAssertEqual(logits.shape, [1, 5, 100])
    }

    func testForwardPassWithTiedEmbeddingsProducesLogitsShape() throws {
        let model = try tinyModel(tied: true)
        let inputs = MLXArray([0, 1, 2] as [Int32]).reshaped(1, 3)
        let logits = model(inputs, cache: nil)
        eval(logits)
        XCTAssertEqual(logits.shape, [1, 3, 100])
    }

    func testDenseLayerUsesMLPAndSparseLayersUseRouter() throws {
        let model = try tinyModel()
        let keys = Set(model.parameters().flattened().map(\.0))

        XCTAssertTrue(keys.contains("model.layers.0.mlp.gate_proj.weight"))
        XCTAssertFalse(keys.contains("model.layers.0.mlp.gate.weight"))
        XCTAssertTrue(keys.contains("model.layers.1.mlp.gate.weight"))
        // Verified against HYV3MoE.__init__: the expert-selection bias is a buffer on the MoE
        // block, not on the router gate.
        XCTAssertTrue(keys.contains("model.layers.1.mlp.e_score_correction_bias"))
        XCTAssertFalse(keys.contains("model.layers.1.mlp.gate.e_score_correction_bias"))
        XCTAssertTrue(keys.contains("model.layers.1.mlp.switch_mlp.gate_proj.weight"))
        XCTAssertTrue(keys.contains("model.layers.1.mlp.shared_experts.gate_proj.weight"))
    }

    func testAttentionCarriesQkNormWhenEnabled() throws {
        let model = try tinyModel()
        let keys = Set(model.parameters().flattened().map(\.0))

        XCTAssertTrue(keys.contains("model.layers.0.self_attn.q_norm.weight"))
        XCTAssertTrue(keys.contains("model.layers.0.self_attn.k_norm.weight"))
        XCTAssertFalse(keys.contains("model.layers.0.self_attn.q_proj.bias"))
    }

    /// The tiny fixture has num_hidden_layers 3 and num_nextn_predict_layers 1, so the MTP
    /// layer is at index 3. Its modules follow vLLM hy_v3_mtp.py: enorm, hnorm, eh_proj
    /// (2H to H, no bias), final_layernorm, and a decoder block. embed_tokens and lm_head are
    /// shared with the main model, so the MTP layer has no copies of them.
    func testMTPHeadIsBuiltAtLayerIndexNumHiddenLayers() throws {
        let model = try tinyModel()
        let params = model.parameters().flattened()
        let keys = Set(params.map(\.0))

        XCTAssertTrue(keys.contains("model.layers.3.enorm.weight"))
        XCTAssertTrue(keys.contains("model.layers.3.hnorm.weight"))
        XCTAssertTrue(keys.contains("model.layers.3.eh_proj.weight"))
        XCTAssertTrue(keys.contains("model.layers.3.final_layernorm.weight"))
        XCTAssertTrue(keys.contains("model.layers.3.self_attn.q_proj.weight"))
        XCTAssertTrue(keys.contains("model.layers.3.mlp.switch_mlp.gate_proj.weight"))
        XCTAssertFalse(keys.contains { $0.hasPrefix("model.layers.2.enorm") })
        XCTAssertFalse(keys.contains { $0.hasPrefix("model.layers.3.embed_tokens") })
        XCTAssertFalse(keys.contains { $0.hasPrefix("model.layers.3.shared_head") })

        // eh_proj maps concat(embedding, hidden) of width 2H to H: weight is [H, 2H] = [32, 64].
        let ehProj = params.first { $0.0 == "model.layers.3.eh_proj.weight" }?.1
        XCTAssertEqual(ehProj?.shape, [32, 64])
    }

    func testMTPLogitsHaveVocabularyShape() throws {
        let model = try tinyModel()
        let nextTokens = MLXArray([1, 2, 3] as [Int32]).reshaped(1, 3)
        let hidden = zeros([1, 3, 32])
        let logits = model.mtpLogits(nextTokens: nextTokens, hiddenStates: hidden)
        eval(logits)
        XCTAssertEqual(logits.shape, [1, 3, 100])
    }

    func testMTPLogitsWithTiedEmbeddingsHaveVocabularyShape() throws {
        let model = try tinyModel(tied: true)
        let nextTokens = MLXArray([4, 5] as [Int32]).reshaped(1, 2)
        let hidden = zeros([1, 2, 32])
        let logits = model.mtpLogits(nextTokens: nextTokens, hiddenStates: hidden)
        eval(logits)
        XCTAssertEqual(logits.shape, [1, 2, 100])
    }
}
