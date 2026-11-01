// SharpAI fork overlay tests for the laguna model. Not part of upstream.
//
// Offline checks only: a tiny synthetic config, random weights, no network.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import XCTest

@testable import MLXLLM

final class SharpAILagunaModelTests: XCTestCase {

    private func tinyModel(tied: Bool = false) throws -> SharpAILagunaModel {
        let config = try JSONDecoder.json5().decode(
            SharpAILagunaConfiguration.self,
            from: Data(SharpAILagunaTestFixture.tinyJSON(tied: tied).utf8))
        return SharpAILagunaModel(config)
    }

    private func parameterShapes(_ model: SharpAILagunaModel) -> [String: [Int]] {
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
        let model = try tinyModel(tied: true)
        let inputs = MLXArray([0, 1, 2] as [Int32]).reshaped(1, 3)
        let logits = model(inputs, cache: nil)
        eval(logits)
        XCTAssertEqual(logits.shape, [1, 3, 100])
    }

    func testForwardPassWithCachesRunsSlidingAndFullLayers() throws {
        let model = try tinyModel()
        let cache = try model.newCache(parameters: nil)
        XCTAssertEqual(cache.count, 2)

        // Longer than the window of 4, so the sliding layer uses its window mask.
        let inputs = MLXArray([0, 1, 2, 3, 4, 5] as [Int32]).reshaped(1, 6)
        let logits = model(inputs, cache: cache)
        eval(logits)
        XCTAssertEqual(logits.shape, [1, 6, 100])
    }

    func testPerLayerHeadCountsSetGateAndQueryShapes() throws {
        let shapes = parameterShapes(try tinyModel())

        // Layer 0 has 4 heads, layer 1 has 6 heads, and head_dim is 8.
        XCTAssertEqual(shapes["model.layers.0.self_attn.q_proj.weight"], [32, 32])
        XCTAssertEqual(shapes["model.layers.1.self_attn.q_proj.weight"], [48, 32])
        XCTAssertEqual(shapes["model.layers.0.self_attn.g_proj.weight"], [4, 32])
        XCTAssertEqual(shapes["model.layers.1.self_attn.g_proj.weight"], [6, 32])
    }

    func testDenseLayerUsesMLPAndSparseLayerUsesRouter() throws {
        let shapes = parameterShapes(try tinyModel())

        XCTAssertNotNil(shapes["model.layers.0.mlp.gate_proj.weight"])
        XCTAssertNil(shapes["model.layers.0.mlp.gate.weight"])
        XCTAssertEqual(shapes["model.layers.1.mlp.gate.weight"], [4, 32])
        XCTAssertEqual(shapes["model.layers.1.mlp.gate.e_score_correction_bias"], [4])
        XCTAssertEqual(shapes["model.layers.1.mlp.switch_mlp.gate_proj.weight"], [4, 16, 32])
        XCTAssertEqual(shapes["model.layers.1.mlp.shared_expert.gate_proj.weight"], [16, 32])
    }

    func testAttentionHasQkNormAndNoBias() throws {
        let shapes = parameterShapes(try tinyModel())

        XCTAssertNotNil(shapes["model.layers.0.self_attn.q_norm.weight"])
        XCTAssertNotNil(shapes["model.layers.0.self_attn.k_norm.weight"])
        XCTAssertNil(shapes["model.layers.0.self_attn.q_proj.bias"])
    }

    func testUngatedConfigOmitsGateProjection() throws {
        let config = try JSONDecoder.json5().decode(
            SharpAILagunaConfiguration.self,
            from: Data(SharpAILagunaTestFixture.tinyJSON(gating: "false").utf8))
        let shapes = parameterShapes(SharpAILagunaModel(config))

        XCTAssertNil(shapes["model.layers.0.self_attn.g_proj.weight"])
        XCTAssertNil(shapes["model.layers.1.self_attn.g_proj.weight"])
    }

    /// The DFlash drafter (DFlashLagunaForCausalLM, a speculative drafter for Laguna-XS.2) is
    /// not implemented. The overlay must reject its config at validation, before any weight is
    /// built. The architecture name is from docs/research/model-coverage.csv in this repo.
    func testDFlashDrafterIsNotImplemented() throws {
        let config = try JSONDecoder.json5().decode(
            SharpAILagunaConfiguration.self,
            from: Data(SharpAILagunaTestFixture.dflashJSON.utf8))

        XCTAssertThrowsError(try config.validateModelConfiguration()) { error in
            guard case SharpAILagunaError.unsupported(let message) = error else {
                XCTFail("unexpected error: \(error)")
                return
            }
            XCTAssertTrue(message.contains("DFlashLagunaForCausalLM"))
        }
    }
}
