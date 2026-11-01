// SharpAI fork overlay tests for step1 registration. Not part of upstream.
// The architecture follows stepfun-ai/Step-Audio-EditX modeling_step1.py, so the creator
// builds the text model and these tests run a tiny forward pass through it.

import Foundation
import MLX
import XCTest

@testable import MLXVLM

final class SharpAIStep1RegistrationTests: XCTestCase {

    private let validJSON = """
        {
            "model_type": "step1",
            "vocab_size": 64,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 4
        }
        """

    private func makeModel(json: String) throws -> SharpAIStep1Model {
        let creator = try XCTUnwrap(SharpAIStep1Overlay.creators["step1"], "step1 creator missing")
        let made = try creator(Data(json.utf8))
        return try XCTUnwrap(made as? SharpAIStep1Model, "step1 creator did not build a model")
    }

    private func assertClose(
        _ a: MLXArray, _ b: MLXArray, tolerance: Float = 1e-4,
        file: StaticString = #filePath, line: UInt = #line
    ) {
        XCTAssertEqual(a.shape, b.shape, file: file, line: line)
        let x = a.asArray(Float.self)
        let y = b.asArray(Float.self)
        XCTAssertEqual(x.count, y.count, file: file, line: line)
        for (u, v) in zip(x, y) {
            XCTAssertEqual(u, v, accuracy: tolerance, file: file, line: line)
        }
    }

    func testRegistersStep1Key() {
        XCTAssertNotNil(SharpAIStep1Overlay.creators["step1"])
    }

    func testCreatorBuildsTextModelFromValidConfiguration() throws {
        let model = try makeModel(json: validJSON)

        XCTAssertEqual(model.kvHeads, [4])
        XCTAssertEqual(model.model.layers.count, 1)
    }

    func testCreatorRejectsInvalidConfiguration() {
        guard let creator = SharpAIStep1Overlay.creators["step1"] else {
            return XCTFail("step1 creator missing")
        }
        let invalidJSON = validJSON.replacingOccurrences(
            of: "\"hidden_size\": 16", with: "\"hidden_size\": 18")

        XCTAssertThrowsError(try creator(Data(invalidJSON.utf8))) { error in
            guard case SharpAIStep1Error.invalidConfiguration = error else {
                return XCTFail("expected invalidConfiguration, got \(error)")
            }
        }
    }

    func testForwardPassTinyShape() throws {
        let model = try makeModel(json: validJSON)
        let tokens = MLXArray([Int32(1), 5, 9]).reshaped([1, 3])

        let logits = model(tokens, cache: nil)

        XCTAssertEqual(logits.shape, [1, 3, 64])
        XCTAssertTrue(logits.sum().item(Float.self).isFinite)
    }

    func testForwardIsCausal() throws {
        let model = try makeModel(json: validJSON)

        // Logits at positions 0 and 1 must not depend on the token at position 2.
        let a = model(MLXArray([Int32(1), 5, 9]).reshaped([1, 3]), cache: nil)
        let b = model(MLXArray([Int32(1), 5, 20]).reshaped([1, 3]), cache: nil)
        assertClose(a[0..., ..<2], b[0..., ..<2])

        // The last position does depend on it, so the comparison above is not trivial.
        let lastA = a[0..., 2].asArray(Float.self)
        let lastB = b[0..., 2].asArray(Float.self)
        XCTAssertNotEqual(lastA, lastB)
    }

    func testCachedDecodeMatchesFullForward() throws {
        let model = try makeModel(json: validJSON)

        let full = model(MLXArray([Int32(1), 5, 9, 2]).reshaped([1, 4]), cache: nil)

        let cache = try model.newCache(parameters: nil)
        _ = model(MLXArray([Int32(1), 5, 9]).reshaped([1, 3]), cache: cache)
        let step = model(MLXArray([Int32(2)]).reshaped([1, 1]), cache: cache)

        assertClose(step[0..., 0], full[0..., 3])
    }
}
