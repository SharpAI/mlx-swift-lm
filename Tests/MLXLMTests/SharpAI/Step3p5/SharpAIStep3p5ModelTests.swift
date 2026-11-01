// SharpAI fork overlay tests for the step3p5 model. Not part of upstream.
//
// Offline checks only: a tiny synthetic config, random weights, no network.
// UNVERIFIED: written without compiling. Run it before relying on it.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import XCTest

@testable import MLXLLM

final class SharpAIStep3p5ModelTests: XCTestCase {

    private func tinyModel(_ overrides: [String: Any] = [:]) throws -> SharpAIStep3p5Model {
        let config = try JSONDecoder.json5().decode(
            SharpAIStep3p5Configuration.self,
            from: Data(SharpAIStep3p5TestFixture.json(overrides).utf8))
        return SharpAIStep3p5Model(config)
    }

    private func parameterShapes(_ model: SharpAIStep3p5Model) -> [String: [Int]] {
        Dictionary(
            uniqueKeysWithValues: model.parameters().flattened().map { ($0.0, $0.1.shape) })
    }

    func testForwardPassProducesLogitsShape() throws {
        let model = try tinyModel()
        let inputs = MLXArray([0, 1, 2, 3, 4] as [Int32]).reshaped(1, 5)
        let logits = model(inputs, cache: nil)
        eval(logits)
        XCTAssertEqual(logits.shape, [1, 5, 100])
    }

    func testForwardPassWithTiedEmbeddingsProducesLogitsShape() throws {
        let model = try tinyModel(["tie_word_embeddings": true])
        let inputs = MLXArray([0, 1, 2] as [Int32]).reshaped(1, 3)
        let logits = model(inputs, cache: nil)
        eval(logits)
        XCTAssertEqual(logits.shape, [1, 3, 100])
    }

    func testForwardPassWithCachesRunsSlidingAndFullLayers() throws {
        let model = try tinyModel()
        let cache = try model.newCache(parameters: nil)
        XCTAssertEqual(cache.count, 3)

        // Longer than the window of 4, so the sliding layer uses its window mask.
        let inputs = MLXArray([0, 1, 2, 3, 4, 5] as [Int32]).reshaped(1, 6)
        let logits = model(inputs, cache: cache)
        eval(logits)
        XCTAssertEqual(logits.shape, [1, 6, 100])
    }

    func testPerLayerHeadCountsSetQueryAndGateShapes() throws {
        let shapes = parameterShapes(try tinyModel())

        // Layer 0 has 4 heads, layer 1 has 6 heads, and head_dim is 8.
        XCTAssertEqual(shapes["model.layers.0.self_attn.q_proj.weight"], [32, 32])
        XCTAssertEqual(shapes["model.layers.1.self_attn.q_proj.weight"], [48, 32])
        XCTAssertEqual(shapes["model.layers.0.self_attn.g_proj.weight"], [4, 32])
        XCTAssertEqual(shapes["model.layers.1.self_attn.g_proj.weight"], [6, 32])
    }

    func testDenseLayerUsesMLPAndMoELayerUsesRouter() throws {
        let shapes = parameterShapes(try tinyModel())

        XCTAssertEqual(shapes["model.layers.0.mlp.gate_proj.weight"], [48, 32])
        XCTAssertNil(shapes["model.layers.0.moe.gate.weight"])
        XCTAssertNil(shapes["model.layers.1.mlp.gate_proj.weight"])
        XCTAssertEqual(shapes["model.layers.1.moe.gate.weight"], [4, 32])
        XCTAssertEqual(shapes["model.layers.1.moe.router_bias"], [4])
        XCTAssertEqual(shapes["model.layers.1.moe.switch_mlp.gate_proj.weight"], [4, 16, 32])
        XCTAssertEqual(shapes["model.layers.1.moe.share_expert.gate_proj.weight"], [16, 32])
    }

    func testNormsAndQKNormHaveHeadDimOrHiddenShape() throws {
        let shapes = parameterShapes(try tinyModel())

        XCTAssertEqual(shapes["model.layers.0.input_layernorm.weight"], [32])
        XCTAssertEqual(shapes["model.layers.0.self_attn.q_norm.weight"], [8])
        XCTAssertEqual(shapes["model.layers.0.self_attn.k_norm.weight"], [8])
        XCTAssertEqual(shapes["model.norm.weight"], [32])
    }

    func testAttentionHasNoBiasByDefault() throws {
        let shapes = parameterShapes(try tinyModel())

        XCTAssertNil(shapes["model.layers.0.self_attn.q_proj.bias"])
    }

    func testGateProjectionOmittedWhenHeadWiseGateIsOff() throws {
        let shapes = parameterShapes(try tinyModel(["use_head_wise_attn_gate": false]))

        XCTAssertNil(shapes["model.layers.0.self_attn.g_proj.weight"])
        XCTAssertNil(shapes["model.layers.1.self_attn.g_proj.weight"])
    }

    func testTiedEmbeddingsHaveNoLMHeadParameter() throws {
        let shapes = parameterShapes(try tinyModel(["tie_word_embeddings": true]))

        XCTAssertNil(shapes["lm_head.weight"])
    }

    /// The Step3p5MTP draft head is not built. Its checkpoint keys are dropped by sanitize.
    func testMTPDraftHeadIsNotBuiltOrLoaded() throws {
        let config = try JSONDecoder.json5().decode(
            SharpAIStep3p5Configuration.self,
            from: Data(SharpAIStep3p5TestFixture.json().utf8))
        let weights: [String: MLXArray] = [
            "model.layers.3.eh_proj.weight": zeros([32, 64]),
            "model.layers.3.mtp_block.self_attn.q_proj.weight": zeros([32, 32]),
        ]

        let sanitized = SharpAIStep3p5Sanitizer.sanitize(weights, configuration: config)

        XCTAssertTrue(sanitized.isEmpty)
        XCTAssertFalse(parameterShapes(try tinyModel()).keys.contains { $0.contains(".mtp_") })
        XCTAssertFalse(parameterShapes(try tinyModel()).keys.contains { $0.contains(".eh_proj.") })
    }
}
