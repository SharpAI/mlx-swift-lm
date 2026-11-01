// SharpAI fork overlay tests for the pangu_ultra_moe creator. Not part of upstream.
// The tiny config uses the real key names from config.json of openPangu-Ultra-MoE-718B-V1.1, with
// small dimensions. The forward tests run the model and check the output shape, that it is
// finite, and that it is causal. The reference Python was not run, so no output is compared.
// UNVERIFIED: written without compiling. Run it before relying on it.

import Foundation
import MLX
import MLXLMCommon
import XCTest

@testable import MLXLLM

final class SharpAIPanguUltraMoeRegistrationTests: XCTestCase {

    private let vocabSize = 50

    /// Two layers: layer 0 is dense (num_dense_layers 1), layer 1 is MoE with a shared expert.
    private let validJSON = """
        {
          "model_type": "pangu_ultra_moe",
          "architectures": ["PanguUltraMoEForCausalLM"],
          "hidden_size": 32,
          "intermediate_size": 48,
          "moe_intermediate_size": 8,
          "num_hidden_layers": 2,
          "num_dense_layers": 1,
          "num_attention_heads": 4,
          "num_key_value_heads": 4,
          "vocab_size": 50,
          "num_routed_experts": 4,
          "num_shared_experts": 1,
          "num_experts_per_tok": 2,
          "routed_scaling_factor": 2.5,
          "attention_q_lora_dim": 16,
          "attention_kv_lora_dim": 8,
          "attention_qk_dim": 8,
          "attention_qk_rope_dim": 4,
          "attention_v_dim": 8,
          "sandwich_norm": true,
          "rope_theta": 10000,
          "max_position_embeddings": 64
        }
        """

    private func creator() throws -> (Data) throws -> any LanguageModel {
        try XCTUnwrap(SharpAIPanguUltraMoeOverlay.creators["pangu_ultra_moe"])
    }

    private func makeTinyModel() throws -> SharpAIPanguUltraMoeModel {
        let built = try creator()(Data(validJSON.utf8))
        return try XCTUnwrap(built as? SharpAIPanguUltraMoeModel)
    }

    func testCreatorIsRegisteredForModelType() {
        XCTAssertNotNil(SharpAIPanguUltraMoeOverlay.creators["pangu_ultra_moe"])
    }

    func testValidConfigBuildsModel() throws {
        let built = try creator()(Data(validJSON.utf8))
        XCTAssertTrue(built is SharpAIPanguUltraMoeModel)
    }

    func testInvalidConfigFailsBeforeConstruction() throws {
        let json = """
            { "model_type": "pangu_ultra_moe" }
            """

        // Missing required keys fail while decoding, so no model is built.
        XCTAssertThrowsError(try creator()(Data(json.utf8)))
    }

    func testConfigMissingForwardDimensionFailsBeforeConstruction() throws {
        let json = validJSON.replacingOccurrences(of: "\"attention_v_dim\": 8,", with: "")

        XCTAssertThrowsError(try creator()(Data(json.utf8))) { error in
            guard case SharpAIPanguUltraMoeError.invalidConfiguration(let detail) = error else {
                XCTFail("expected invalidConfiguration, got \(error)")
                return
            }
            XCTAssertTrue(detail.contains("attention_v_dim"), detail)
        }
    }

    func testTinyForwardPass() throws {
        let model = try makeTinyModel()

        let logits = model(MLXArray([Int32(1), 7, 3], [1, 3]), cache: nil)

        XCTAssertEqual(logits.shape, [1, 3, vocabSize])
        let values = logits.asArray(Float.self)
        XCTAssertTrue(values.allSatisfy { $0.isFinite }, "logits must be finite")
    }

    func testForwardIsCausal() throws {
        let model = try makeTinyModel()

        // Logits for the first two tokens must not change when a third token is appended.
        let short = model(MLXArray([Int32(1), 7], [1, 2]), cache: nil).asArray(Float.self)
        let long = model(MLXArray([Int32(1), 7, 3], [1, 3]), cache: nil)
        let prefix = long[0..., ..<2, 0...].asArray(Float.self)

        XCTAssertEqual(short.count, prefix.count)
        for (a, b) in zip(short, prefix) {
            XCTAssertEqual(a, b, accuracy: 1e-3)
        }
    }
}
