// SharpAI fork overlay tests for the "unknown" creator. Not part of upstream.
// Offline only. Checks the dispatch by architecture. No model is loaded.
// Written without compiling. Run it before relying on it.

import Foundation
import MLXLMCommon
import XCTest

@testable import MLXLLM

final class SharpAIUnknownRegistrationTests: XCTestCase {

    private let eagleJSON = """
        {
            "architectures": ["EagleCohereForCausalLM"],
            "hidden_size": 32,
            "num_hidden_layers": 2,
            "intermediate_size": 64,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "layer_norm_eps": 1e-05,
            "vocab_size": 100,
            "logit_scale": 1.0
        }
        """

    private let panguJSON = """
        {
            "architectures": ["PanguProMoEV2ForCausalLM"],
            "hidden_size": 64,
            "num_hidden_layers": 4,
            "num_attention_heads": 8,
            "vocab_size": 500,
            "n_routed_experts": 16,
            "num_experts_per_tok": 4,
            "moe_intermediate_size": 32
        }
        """

    private func creator() throws -> (Data) throws -> any LanguageModel {
        try XCTUnwrap(SharpAIUnknownOverlay.creators["unknown"])
    }

    func testRegistersUnknownKey() {
        XCTAssertNotNil(SharpAIUnknownOverlay.creators["unknown"])
    }

    func testEagleArchitectureIsDraftOnly() throws {
        let make = try creator()

        XCTAssertThrowsError(try make(Data(eagleJSON.utf8))) { error in
            guard case SharpAIUnknownError.draftOnly = error else {
                return XCTFail("expected draftOnly, got \(error)")
            }
        }
    }

    func testPanguArchitectureReportsForwardNotImplemented() throws {
        let make = try creator()

        XCTAssertThrowsError(try make(Data(panguJSON.utf8))) { error in
            guard case SharpAIUnknownError.forwardNotImplemented = error else {
                return XCTFail("expected forwardNotImplemented, got \(error)")
            }
        }
    }

    func testUnknownArchitectureIsRejected() throws {
        let make = try creator()
        let text = """
            { "architectures": ["SomeOtherForCausalLM"], "hidden_size": 8 }
            """

        XCTAssertThrowsError(try make(Data(text.utf8))) { error in
            guard case SharpAIUnknownError.unsupportedArchitecture = error else {
                return XCTFail("expected unsupportedArchitecture, got \(error)")
            }
        }
    }

    func testInvalidEagleConfigFailsBeforeDraftError() throws {
        let make = try creator()
        let text = eagleJSON.replacingOccurrences(
            of: "\"num_key_value_heads\": 2", with: "\"num_key_value_heads\": 3")

        XCTAssertThrowsError(try make(Data(text.utf8))) { error in
            guard case SharpAIUnknownError.invalidConfiguration = error else {
                return XCTFail("expected invalidConfiguration, got \(error)")
            }
        }
    }
}
