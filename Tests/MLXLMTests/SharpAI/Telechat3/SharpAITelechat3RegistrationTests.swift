// SharpAI fork overlay tests for the telechat3 creator. Not part of upstream.
// UNVERIFIED: written without compiling. Run it before relying on it.

import Foundation
import MLX
import MLXLMCommon
import XCTest

@testable import MLXLLM

final class SharpAITelechat3RegistrationTests: XCTestCase {

    private let validJSON = """
        {
          "architectures": ["TeleChat3ForCausalLM"],
          "model_type": "telechat3",
          "hidden_size": 8,
          "intermediate_size": 16,
          "num_hidden_layers": 1,
          "num_attention_heads": 2,
          "num_key_value_heads": 1,
          "vocab_size": 32,
          "rms_norm_eps": 1e-05,
          "rope_scaling": {
            "type": "telechat3-yarn",
            "factor": 4.0,
            "original_max_position_embeddings": 16
          }
        }
        """

    func testCreatorIsRegisteredForModelType() {
        XCTAssertNotNil(SharpAITelechat3Overlay.creators["telechat3"])
    }

    func testCreatorBuildsTelechat3Model() throws {
        let creator = try XCTUnwrap(SharpAITelechat3Overlay.creators["telechat3"])
        let model = try creator(Data(validJSON.utf8))

        XCTAssertTrue(model is SharpAITelechat3Model)
        XCTAssertTrue(model is LlamaModel)
    }

    func testTelechat3YarnUsesYarnRoPE() throws {
        let model = try XCTUnwrap(
            SharpAITelechat3Overlay.creators["telechat3"]?(Data(validJSON.utf8))
                as? SharpAITelechat3Model)

        let rope = model.model.layers[0].attention.rope
        XCTAssertTrue(rope is YarnRoPE)
    }

    func testInvalidConfigThrows() throws {
        let creator = try XCTUnwrap(SharpAITelechat3Overlay.creators["telechat3"])
        let json = """
            { "model_type": "telechat3", "hidden_size": 8 }
            """

        XCTAssertThrowsError(try creator(Data(json.utf8)))
    }

    func testWrongArchitectureThrows() throws {
        let creator = try XCTUnwrap(SharpAITelechat3Overlay.creators["telechat3"])
        let json = validJSON.replacingOccurrences(
            of: "TeleChat3ForCausalLM", with: "LlamaForCausalLM")

        XCTAssertThrowsError(try creator(Data(json.utf8)))
    }

    func testTinyForwardPassWithRandomWeights() throws {
        let creator = try XCTUnwrap(SharpAITelechat3Overlay.creators["telechat3"])
        let model = try XCTUnwrap(creator(Data(validJSON.utf8)) as? SharpAITelechat3Model)

        let tokens = MLXArray([1, 2, 3] as [Int32]).reshaped(1, 3)
        let logits = model(tokens, cache: nil)
        eval(logits)

        XCTAssertEqual(logits.shape, [1, 3, 32])
    }

    /// Reference: Tele-AI/TeleChat3-36B-Thinking modeling_telechat3.py, `get_mscale`
    /// returns `0.07 * mscale * ln(scale) + 1` and the config has no mscale fields, so the
    /// attention factor is `get_mscale(4.0)`. YarnRoPE is linear, so it scales the norm of its
    /// output by exactly `_mscale`. The rope is applied to the model's own layer 0.
    func testYarnMscaleCoefficientMatchesTelechat3Reference() throws {
        let model = try XCTUnwrap(
            SharpAITelechat3Overlay.creators["telechat3"]?(Data(validJSON.utf8))
                as? SharpAITelechat3Model)
        let yarn = try XCTUnwrap(
            model.model.layers[0].attention.rope as? YarnRoPE)

        let x = MLXArray((1 ... 12).map { Float($0) }).reshaped(1, 1, 3, 4)
        let y = yarn(x, offset: 0)
        eval(y)

        let inputNorm = sqrt((x * x).sum().item(Float.self))
        let outputNorm = sqrt((y * y).sum().item(Float.self))
        let ratio = outputNorm / inputNorm
        let expected = Float(1.0 + 0.07 * log(4.0))

        // The shared YaRN slope 0.1 would give about 1.1386, which this accuracy rejects.
        XCTAssertEqual(ratio, expected, accuracy: 1e-4)
    }
}
