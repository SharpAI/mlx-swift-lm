// SharpAI fork overlay tests for the deepseek_v41 creator. Not part of upstream.

import Foundation
import MLX
import MLXLMCommon
import MLXVLM
import XCTest

final class SharpAIDeepseekV41RegistrationTests: XCTestCase {

    private let validJSON = """
        {
            "model_type": "deepseek_v41",
            "architectures": ["DeepseekV41ForCausalLM"]
        }
        """

    func testCreatorIsRegisteredForModelType() {
        XCTAssertNotNil(SharpAIDeepseekV41Overlay.creators["deepseek_v41"])
    }

    /// A config with no text_config has no text model to build, so the creator refuses it.
    func testConfigWithoutTextConfigIsRejected() throws {
        guard let creator = SharpAIDeepseekV41Overlay.creators["deepseek_v41"] else {
            XCTFail("deepseek_v41 creator missing")
            return
        }

        XCTAssertThrowsError(try creator(Data(validJSON.utf8))) { error in
            guard case SharpAIDeepseekV41Error.missingTextConfig = error else {
                XCTFail("unexpected error: \(error)")
                return
            }
        }
    }

    func testWrongModelTypeFailsBeforeConstruction() throws {
        guard let creator = SharpAIDeepseekV41Overlay.creators["deepseek_v41"] else {
            XCTFail("deepseek_v41 creator missing")
            return
        }
        let json = """
            { "model_type": "llama" }
            """

        XCTAssertThrowsError(try creator(Data(json.utf8))) { error in
            guard case ModelFactoryError.invalidConfiguration = error else {
                XCTFail("config validation should fail first, got \(error)")
                return
            }
        }
    }

    /// Engram layers are rejected, as mlx-lm's ModelArgs.__post_init__ does.
    func testEngramLayersAreRejected() throws {
        guard let creator = SharpAIDeepseekV41Overlay.creators["deepseek_v41"] else {
            XCTFail("deepseek_v41 creator missing")
            return
        }
        let json = """
            {
                "model_type": "deepseek_v41",
                "text_config": { "model_type": "deepseek_v41_text", "engram_layer_ids": [1, 14] }
            }
            """

        XCTAssertThrowsError(try creator(Data(json.utf8))) { error in
            guard case SharpAIDeepseekV41Error.engramNotSupported(let layers) = error else {
                XCTFail("unexpected error: \(error)")
                return
            }
            XCTAssertEqual(layers, [1, 14])
        }
    }

    /// The creator builds the text model from the tiny fixture, and its forward pass gives logits
    /// over the vocabulary.
    func testTinyForwardPassProducesModel() throws {
        guard let creator = SharpAIDeepseekV41Overlay.creators["deepseek_v41"] else {
            XCTFail("deepseek_v41 creator missing")
            return
        }
        let created = try creator(Data(DeepseekV41TinyFixture.json.utf8))
        guard let model = created as? SharpAIDeepseekV41Model else {
            XCTFail("creator returned \(type(of: created)), not SharpAIDeepseekV41Model")
            return
        }

        let tokens = MLXArray([Int32(1), 2, 3]).reshaped(1, 3)
        let logits = model(tokens, cache: model.newCache(parameters: nil))

        XCTAssertEqual(logits.shape, [1, 3, DeepseekV41TinyFixture.vocabSize])
    }
}
