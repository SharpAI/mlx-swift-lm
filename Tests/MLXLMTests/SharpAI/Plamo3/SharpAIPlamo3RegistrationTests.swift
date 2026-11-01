// SharpAI fork overlay tests for the plamo3 creator. Not part of upstream.
// The creator builds SharpAIPlamo3Model; the forward pass runs with random weights and no checkpoint.
// The tokenizer is not part of this test.

import Foundation
import MLX
import MLXLMCommon
import XCTest

@testable import MLXLLM

final class SharpAIPlamo3RegistrationTests: XCTestCase {

    private let tinyJSON = """
        {
          "architectures": ["Plamo3ForCausalLM"],
          "model_type": "plamo3",
          "hidden_size": 64,
          "intermediate_size": 128,
          "num_hidden_layers": 2,
          "num_attention_heads": 8,
          "num_key_value_heads": 2,
          "vocab_size": 1000
        }
        """

    func testCreatorIsRegisteredForModelType() {
        XCTAssertNotNil(SharpAIPlamo3Overlay.creators["plamo3"])
    }

    func testWrongArchitectureFailsAtValidation() throws {
        guard let creator = SharpAIPlamo3Overlay.creators["plamo3"] else {
            XCTFail("plamo3 creator missing")
            return
        }
        let json = tinyJSON.replacingOccurrences(
            of: "Plamo3ForCausalLM", with: "LlamaForCausalLM")

        XCTAssertThrowsError(try creator(Data(json.utf8))) { error in
            guard let typed = error as? SharpAIPlamo3Error,
                case .invalidConfiguration = typed
            else {
                XCTFail("expected invalidConfiguration, got \(error)")
                return
            }
        }
    }

    func testValidConfigBuildsPlamo3Model() throws {
        guard let creator = SharpAIPlamo3Overlay.creators["plamo3"] else {
            XCTFail("plamo3 creator missing")
            return
        }

        let model = try creator(Data(tinyJSON.utf8))

        XCTAssertTrue(model is SharpAIPlamo3Model)
    }

    func testForwardPassWithRandomWeightsRuns() throws {
        guard let creator = SharpAIPlamo3Overlay.creators["plamo3"] else {
            XCTFail("plamo3 creator missing")
            return
        }
        guard let model = try creator(Data(tinyJSON.utf8)) as? SharpAIPlamo3Model else {
            XCTFail("creator did not return SharpAIPlamo3Model")
            return
        }
        let tokens = MLXArray([Int32(1), 2, 3]).reshaped(1, 3)

        let logits = model(tokens, cache: nil)

        XCTAssertEqual(logits.shape, [1, 3, 1000])
    }
}
