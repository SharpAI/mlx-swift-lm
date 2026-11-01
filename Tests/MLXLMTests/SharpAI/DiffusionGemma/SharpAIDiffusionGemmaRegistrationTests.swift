// SharpAI fork overlay tests for the diffusion_gemma creator. Not part of upstream.

import Foundation
import MLXLMCommon
import MLXVLM
import XCTest

final class SharpAIDiffusionGemmaRegistrationTests: XCTestCase {

    private let validJSON = """
        {
            "model_type": "diffusion_gemma",
            "text_config": {
                "vocab_size": 1024,
                "hidden_size": 256,
                "num_hidden_layers": 2,
                "num_attention_heads": 8,
                "num_key_value_heads": 4
            },
            "vision_config": {
                "model_type": "vision"
            }
        }
        """

    func testCreatorIsRegisteredForModelType() {
        XCTAssertNotNil(SharpAIDiffusionGemmaOverlay.creators["diffusion_gemma"])
    }

    func testValidConfigFailsUntilForwardIsImplemented() {
        guard let creator = SharpAIDiffusionGemmaOverlay.creators["diffusion_gemma"] else {
            XCTFail("diffusion_gemma creator missing")
            return
        }

        XCTAssertThrowsError(try creator(Data(validJSON.utf8))) { error in
            guard case SharpAIDiffusionGemmaError.forwardNotImplemented(let type) = error else {
                XCTFail("unexpected error: \(error)")
                return
            }
            XCTAssertEqual(type, "diffusion_gemma")
        }
    }

    func testInvalidConfigFailsBeforeConstruction() {
        guard let creator = SharpAIDiffusionGemmaOverlay.creators["diffusion_gemma"] else {
            XCTFail("diffusion_gemma creator missing")
            return
        }

        XCTAssertThrowsError(try creator(Data("{ \"hidden_size\": 64 }".utf8))) { error in
            if case SharpAIDiffusionGemmaError.forwardNotImplemented = error {
                XCTFail("config validation should fail first")
            }
        }
    }

    func testTinyForwardPassAndDenoiseSkipped() {
        // Reference sources exist upstream: transformers modeling_diffusion_gemma.py (encoder,
        // decoder, MoE, self-conditioning) and generation_diffusion_gemma.py (EntropyBoundSampler,
        // temperature schedule, StableAndConfidentStoppingCriteria). The Swift port is not written.
        XCTExpectFailure(
            "KNOWN GAP: diffusion_gemma model class and denoise engine are not ported to Swift. "
                + "Reference found and read: transformers modeling_diffusion_gemma.py and "
                + "generation_diffusion_gemma.py (raw.githubusercontent.com, main). Not yet written: "
                + "decoder layers with append_to_cache and block-diffusion masks, 128-expert MoE router, "
                + "self-conditioning, vision tower (reuse Gemma4 vision), and EntropyBoundSampler. "
                + "The creator still throws forwardNotImplemented for a valid config. "
                + "Flips to a hard failure once construction succeeds."
        ) {
            guard let creator = SharpAIDiffusionGemmaOverlay.creators["diffusion_gemma"] else {
                XCTFail("diffusion_gemma creator missing")
                return
            }
            do {
                _ = try creator(Data(validJSON.utf8))
            } catch {
                XCTFail("tiny diffusion_gemma config failed to construct: \(error)")
            }
        }
    }
}
