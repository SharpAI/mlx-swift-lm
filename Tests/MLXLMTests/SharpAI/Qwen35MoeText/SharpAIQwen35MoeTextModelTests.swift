// SharpAI fork overlay tests for the qwen3_5_moe_text model and creator. Not part of upstream.
//
// Offline checks only: a tiny synthetic config, random weights, no network.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import XCTest

@testable import MLXLLM

final class SharpAIQwen35MoeTextModelTests: XCTestCase {

    private func tinyModel() throws -> SharpAIQwen35MoeTextModel {
        let config = try JSONDecoder.json5().decode(
            SharpAIQwen35MoeTextConfiguration.self,
            from: Data(SharpAIQwen35MoeTextTestFixture.tinyJSON().utf8))
        return SharpAIQwen35MoeTextModel(config.textConfiguration)
    }

    func testForwardPassProducesLogitsShape() throws {
        let model = try tinyModel()
        let inputs = MLXArray([0, 1, 2, 3, 4] as [Int32]).reshaped(1, 5)
        let logits = model(inputs, cache: nil)
        eval(logits)
        XCTAssertEqual(logits.shape, [1, 5, 100])
    }

    func testParametersUseSwitchMLPLayout() throws {
        let model = try tinyModel()
        let shapes = Dictionary(
            uniqueKeysWithValues: model.parameters().flattened().map { ($0.0, $0.1.shape) })

        XCTAssertEqual(
            shapes["model.layers.0.mlp.switch_mlp.gate_proj.weight"], [4, 32, 64])
        XCTAssertEqual(
            shapes["model.layers.0.mlp.switch_mlp.down_proj.weight"], [4, 64, 32])
        XCTAssertEqual(shapes["model.layers.0.mlp.gate.weight"], [4, 64])
    }

    func testCreatorBuildsSharpAIModel() throws {
        guard let creator = SharpAIQwen35MoeTextOverlay.creators["qwen3_5_moe_text"] else {
            XCTFail("qwen3_5_moe_text creator missing")
            return
        }
        let model = try creator(Data(SharpAIQwen35MoeTextTestFixture.tinyJSON().utf8))
        XCTAssertTrue(model is SharpAIQwen35MoeTextModel)
    }

    func testCreatorRejectsDenseConfiguration() throws {
        guard let creator = SharpAIQwen35MoeTextOverlay.creators["qwen3_5_moe_text"] else {
            XCTFail("qwen3_5_moe_text creator missing")
            return
        }
        XCTAssertThrowsError(
            try creator(Data(SharpAIQwen35MoeTextTestFixture.tinyJSON(numExperts: 0).utf8)))
    }

    /// The MTP drafter registry has no entry for `qwen3_5_moe_text`. Upstream registers only
    /// `qwen3_5_text`, `qwen3_8_text`, `qwen3_5`, `qwen3_5_moe`, and the `_mtp` variants
    /// (see `Qwen35TextMTPRegistration`). Registering this type is a separate decision.
    func testMTPDrafterIsNotRegistered() async throws {
        // `ModelTypeRegistry` is an actor, so `createModel` must be awaited. XCTAssertThrowsError
        // takes a synchronous autoclosure, so the throw is caught explicitly instead.
        do {
            _ = try await MTPDrafterTypeRegistry.shared.createModel(
                configuration: Data(SharpAIQwen35MoeTextTestFixture.tinyJSON().utf8),
                modelType: "qwen3_5_moe_text")
            XCTFail("expected createModel to throw for qwen3_5_moe_text")
        } catch {
            guard case ModelFactoryError.unsupportedModelType(let type) = error else {
                XCTFail("expected unsupportedModelType, got \(error)")
                return
            }
            XCTAssertEqual(type, "qwen3_5_moe_text")
        }
    }
}
