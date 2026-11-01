// SharpAI fork overlay tests for the mistral_large4 creator. Not part of upstream.
// The creator refuses to build a model until the forward pass exists, so the forward-pass
// test is a known-issue skip.
// UNVERIFIED: written without compiling. Run it before relying on it.

import Foundation
import MLXLMCommon
import MLXVLM
import XCTest

final class SharpAIMistralLarge4RegistrationTests: XCTestCase {

    private let validJSON = """
        {
          "model_type": "mistral_large4",
          "architectures": ["MistralLarge4ForConditionalGeneration"],
          "text_config": {"hidden_size": 32, "num_hidden_layers": 2},
          "vision_config": {"hidden_size": 16}
        }
        """

    func testCreatorIsRegisteredForModelType() {
        XCTAssertNotNil(SharpAIMistralLarge4Overlay.creators["mistral_large4"])
    }

    func testValidConfigFailsUntilForwardIsImplemented() throws {
        guard let creator = SharpAIMistralLarge4Overlay.creators["mistral_large4"] else {
            XCTFail("mistral_large4 creator missing")
            return
        }

        XCTAssertThrowsError(try creator(Data(validJSON.utf8))) { error in
            guard case SharpAIMistralLarge4Error.forwardNotImplemented(let type) = error else {
                XCTFail("unexpected error: \(error)")
                return
            }
            XCTAssertEqual(type, "mistral_large4")
        }
    }

    func testInvalidConfigFailsBeforeConstruction() throws {
        guard let creator = SharpAIMistralLarge4Overlay.creators["mistral_large4"] else {
            XCTFail("mistral_large4 creator missing")
            return
        }
        let json = """
            { "model_type": "mistral_large4" }
            """

        XCTAssertThrowsError(try creator(Data(json.utf8))) { error in
            if case SharpAIMistralLarge4Error.forwardNotImplemented = error {
                XCTFail("config validation should fail first")
            }
        }
    }

    /// KNOWN GAP: the tiny forward pass does not exist yet, so the creator still throws.
    /// Reference: transformers `src/transformers/models/mistral_large4/` (modular_mistral_large4.py
    /// subclasses GlmMoeDsa attention, with a DSA indexer, KV sharing, SSMax scaling, RoPE/NoPE
    /// alternation, grouped top-k MoE and a Pixtral vision tower). Only WebFetch summaries and one
    /// partial quote were obtainable, not the full verbatim source. The checkpoint config.json and
    /// model.safetensors.index.json return 404 at mistralai/Mistral-Large-4-1T-A52B/resolve/main.
    /// The assertion below flips to a hard failure once a model is constructed from the tiny config.
    func testTinyForwardPassSkipped() throws {
        guard let creator = SharpAIMistralLarge4Overlay.creators["mistral_large4"] else {
            XCTFail("mistral_large4 creator missing")
            return
        }

        try XCTExpectFailure(
            "KNOWN GAP: mistral_large4 forward pass not ported; needs the verbatim "
                + "transformers mistral_large4 modeling sources (GlmMoeDsa base, indexer, MoE "
                + "router, Pixtral tower) and a published checkpoint config (resolve/main 404)"
        ) {
            XCTAssertNoThrow(try creator(Data(validJSON.utf8)))
        }
    }
}
