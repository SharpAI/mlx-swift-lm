// SharpAI fork overlay tests for the qwen4_exp creator. Not part of upstream.

import Foundation
import MLX
import MLXLMCommon
import MLXVLM
import XCTest

final class SharpAIQwen4ExpRegistrationTests: XCTestCase {

    private func tinyModel(_ json: String = SharpAIQwen4ExpTinyFixtures.configJSON) throws
        -> any LanguageModel
    {
        guard let creator = SharpAIQwen4ExpOverlay.creators["qwen4_exp"] else {
            throw XCTSkip("qwen4_exp creator missing")
        }
        return try creator(Data(json.utf8))
    }

    func testCreatorIsRegisteredForModelType() {
        XCTAssertNotNil(SharpAIQwen4ExpOverlay.creators["qwen4_exp"])
    }

    /// The released config decodes and validates. The model is not built here: it is about 35B
    /// parameters, and the tiny config covers construction and execution.
    func testReleasedConfigDecodesAndValidates() throws {
        let configuration = try JSONDecoder.json5().decode(
            SharpAIQwen4ExpConfiguration.self, from: Data(SharpAIQwen4ExpFixtures.configJSON.utf8))
        try configuration.validateModelConfiguration()
        XCTAssertEqual(configuration.textConfig.numHiddenLayers, 48)
        XCTAssertEqual(configuration.textConfig.hcCount, 4)
    }

    func testTinyConfigBuildsModel() throws {
        _ = try tinyModel()
    }

    func testWrongModelTypeFailsBeforeConstruction() throws {
        let json = SharpAIQwen4ExpTinyFixtures.configJSON.replacingOccurrences(
            of: "\"model_type\": \"qwen4_exp\",", with: "\"model_type\": \"llama\",")
        XCTAssertThrowsError(try tinyModel(json))
    }

    func testTinyForwardProducesFiniteLogits() throws {
        let model = try tinyModel()
        let tokens = MLXArray([Int32(5), 9, 2, 7, 3, 11]).reshaped(1, 6)

        let logits = model.callAsFunction(tokens, cache: nil)

        XCTAssertEqual(logits.shape, [1, 6, 64])
        XCTAssertTrue(logits.asArray(Float.self).allSatisfy { $0.isFinite })
    }

    /// Prefill followed by one-token decode steps must reproduce the full-sequence logits. This
    /// exercises the GatedDeltaNet conv and recurrent state, the PLE short-conv state, the n-gram
    /// context (the sequence contains EOS), and the indexer key cache.
    func testIncrementalDecodeMatchesFullForward() throws {
        let model = try tinyModel()
        let tokens = MLXArray([Int32(5), 9, 2, 7, 3, 11, 4, 2]).reshaped(1, 8)
        let full = model.callAsFunction(tokens, cache: nil)

        let cache = try model.newCache(parameters: nil)
        _ = model.callAsFunction(tokens[0..., ..<5], cache: cache)
        for position in 5 ..< 8 {
            let step = model.callAsFunction(
                tokens[0..., position ..< (position + 1)], cache: cache)
            let expected = full[0..., position ..< (position + 1), 0...]
            let difference = abs(step - expected).max().item(Float.self)
            XCTAssertLessThan(difference, 1e-3, "decode position \(position)")
        }
    }
}
