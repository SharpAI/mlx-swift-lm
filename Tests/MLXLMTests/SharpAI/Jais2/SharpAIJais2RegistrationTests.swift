// SharpAI fork overlay tests for jais2 registration. Not part of upstream.

import Foundation
import MLX
import MLXLMCommon
import XCTest

@testable import MLXLLM

final class SharpAIJais2RegistrationTests: XCTestCase {

    private let validJSON = """
        {
            "model_type": "jais2",
            "vocab_size": 64,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 4
        }
        """

    func testRegistersJais2Key() {
        XCTAssertNotNil(SharpAIJais2Overlay.creators["jais2"])
    }

    func testCreatorBuildsModelForValidConfiguration() throws {
        guard let creator = SharpAIJais2Overlay.creators["jais2"] else {
            return XCTFail("jais2 creator missing")
        }

        let model = try creator(Data(validJSON.utf8))
        XCTAssertTrue(model is SharpAIJais2Model)
    }

    func testCreatorRejectsUnsupportedActivation() {
        guard let creator = SharpAIJais2Overlay.creators["jais2"] else {
            return XCTFail("jais2 creator missing")
        }
        let geluJSON = validJSON.replacingOccurrences(
            of: "\"model_type\": \"jais2\"",
            with: "\"model_type\": \"jais2\", \"hidden_act\": \"gelu\"")

        XCTAssertThrowsError(try creator(Data(geluJSON.utf8))) { error in
            guard case SharpAIJais2Error.unsupported = error else {
                return XCTFail("expected unsupported, got \(error)")
            }
        }
    }

    func testCreatorRejectsInvalidConfigurationBeforeUnsupported() {
        guard let creator = SharpAIJais2Overlay.creators["jais2"] else {
            return XCTFail("jais2 creator missing")
        }
        let invalidJSON = validJSON.replacingOccurrences(of: "\"hidden_size\": 16", with: "\"hidden_size\": 18")

        XCTAssertThrowsError(try creator(Data(invalidJSON.utf8))) { error in
            guard case SharpAIJais2Error.invalidConfiguration = error else {
                return XCTFail("expected invalidConfiguration, got \(error)")
            }
        }
    }

    func testForwardPassProducesVocabularyLogits() throws {
        let configuration = try JSONDecoder.json5().decode(
            SharpAIJais2Configuration.self, from: Data(validJSON.utf8))
        let model = SharpAIJais2Model(configuration)

        let tokens = MLXArray([Int32(1), 2, 3]).reshaped(1, 3)
        let logits = model(tokens, cache: nil)

        XCTAssertEqual(logits.shape, [1, 3, 64])
        XCTAssertTrue(logits.asArray(Float.self).allSatisfy { $0.isFinite })
    }

    func testForwardPassWithTiedEmbeddingsHasNoSeparateHead() throws {
        let tiedJSON = validJSON.replacingOccurrences(
            of: "\"model_type\": \"jais2\"",
            with: "\"model_type\": \"jais2\", \"tie_word_embeddings\": true")
        let configuration = try JSONDecoder.json5().decode(
            SharpAIJais2Configuration.self, from: Data(tiedJSON.utf8))
        let model = SharpAIJais2Model(configuration)

        XCTAssertNil(model.lmHead)

        let logits = model(MLXArray([Int32(5), 6]).reshaped(1, 2), cache: nil)
        XCTAssertEqual(logits.shape, [1, 2, 64])
        XCTAssertTrue(logits.asArray(Float.self).allSatisfy { $0.isFinite })
    }
}
