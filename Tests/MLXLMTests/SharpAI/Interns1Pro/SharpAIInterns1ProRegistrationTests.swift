// SharpAI fork overlay tests for the interns1_pro creator. Not part of upstream.

import Foundation
import MLX
import MLXLMCommon
import MLXVLM
import XCTest

final class SharpAIInterns1ProRegistrationTests: XCTestCase {

    /// Names the model type only. The creator needs text_config, so this must be rejected.
    private let validJSON = """
        {
            "model_type": "interns1_pro",
            "architectures": ["InternS1ProForConditionalGeneration"]
        }
        """

    func testCreatorIsRegisteredForModelType() {
        XCTAssertNotNil(SharpAIInterns1ProOverlay.creators["interns1_pro"])
    }

    func testConfigWithoutTextConfigIsRejected() throws {
        guard let creator = SharpAIInterns1ProOverlay.creators["interns1_pro"] else {
            XCTFail("interns1_pro creator missing")
            return
        }

        XCTAssertThrowsError(try creator(Data(validJSON.utf8))) { error in
            guard case ModelFactoryError.invalidConfiguration = error else {
                XCTFail("unexpected error: \(error)")
                return
            }
        }
    }

    func testWrongModelTypeFailsBeforeConstruction() throws {
        guard let creator = SharpAIInterns1ProOverlay.creators["interns1_pro"] else {
            XCTFail("interns1_pro creator missing")
            return
        }
        let json = """
            { "model_type": "llama" }
            """

        XCTAssertThrowsError(try creator(Data(json.utf8))) { error in
            guard case ModelFactoryError.invalidConfiguration = error else {
                XCTFail("unexpected error: \(error)")
                return
            }
        }
    }

    /// Builds the tiny FoPE model through the creator and runs a forward pass. The logits must
    /// have shape [batch, seq, vocab] and be finite.
    func testTinyForwardPassProducesModel() throws {
        guard let creator = SharpAIInterns1ProOverlay.creators["interns1_pro"] else {
            XCTFail("interns1_pro creator missing")
            return
        }
        let model = try creator(Data(SharpAIInterns1ProTinyFixture.fope.utf8))

        let tokens = MLXArray([Int32(1), 2, 3]).reshaped(1, 3)
        let logits = model(tokens, cache: nil)

        XCTAssertEqual(logits.shape, [1, 3, SharpAIInterns1ProTinyFixture.vocabularySize])
        XCTAssertTrue(logits.asArray(Float.self).allSatisfy(\.isFinite))
    }

    func testTinyForwardPassWithoutFoPE() throws {
        guard let creator = SharpAIInterns1ProOverlay.creators["interns1_pro"] else {
            XCTFail("interns1_pro creator missing")
            return
        }
        let model = try creator(Data(SharpAIInterns1ProTinyFixture.plain.utf8))

        let logits = model(MLXArray([Int32(4), 5]).reshaped(1, 2), cache: nil)

        XCTAssertEqual(logits.shape, [1, 2, SharpAIInterns1ProTinyFixture.vocabularySize])
        XCTAssertTrue(logits.asArray(Float.self).allSatisfy(\.isFinite))
    }
}
