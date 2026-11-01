// SharpAI fork overlay tests for pangu_ultra_moe checkpoint key sanitization. Not part of upstream.
// Expert keys keep their HF names (model.layers.N.mlp.experts.E.*), which the model modules use,
// so nothing is stacked. The MTP drop follows model.safetensors.index.json of openPangu-Ultra-MoE-718B-V1.1.
// Written without compiling. Run it before relying on it.

import Foundation
import MLX
import MLXLMCommon
import XCTest

@testable import MLXLLM

final class SharpAIPanguUltraMoeSanitizeTests: XCTestCase {

    private func makeConfig(tie: Bool) throws -> SharpAIPanguUltraMoeConfiguration {
        let json = """
            {
              "model_type": "pangu_ultra_moe",
              "hidden_size": 8,
              "num_hidden_layers": 2,
              "num_attention_heads": 2,
              "vocab_size": 16,
              "tie_word_embeddings": \(tie)
            }
            """
        return try JSONDecoder.json5().decode(
            SharpAIPanguUltraMoeConfiguration.self, from: Data(json.utf8))
    }

    private func vector(_ value: Float) -> MLXArray {
        MLXArray([value, value + 1])
    }

    func testDropsRotaryBuffers() throws {
        let config = try makeConfig(tie: false)
        let weights: [String: MLXArray] = [
            "model.layers.0.self_attn.rotary_emb.inv_freq": vector(0),
            "model.layers.0.self_attn.q_proj.weight": vector(1),
        ]

        let result = SharpAIPanguUltraMoeSanitizer.sanitize(weights, configuration: config)

        XCTAssertNil(result["model.layers.0.self_attn.rotary_emb.inv_freq"])
        XCTAssertNotNil(result["model.layers.0.self_attn.q_proj.weight"])
    }

    func testDropsLMHeadWhenTied() throws {
        let config = try makeConfig(tie: true)
        let weights: [String: MLXArray] = [
            "lm_head.weight": vector(2),
            "model.embed_tokens.weight": vector(3),
        ]

        let result = SharpAIPanguUltraMoeSanitizer.sanitize(weights, configuration: config)

        XCTAssertNil(result["lm_head.weight"])
        XCTAssertNotNil(result["model.embed_tokens.weight"])
    }

    func testKeepsLMHeadWhenUntied() throws {
        let config = try makeConfig(tie: false)
        let weights: [String: MLXArray] = ["lm_head.weight": vector(2)]

        let result = SharpAIPanguUltraMoeSanitizer.sanitize(weights, configuration: config)

        XCTAssertNotNil(result["lm_head.weight"])
    }

    func testDropsMultiTokenPredictionLayers() throws {
        // makeConfig uses num_hidden_layers 2, so layer index 2 is the MTP layer.
        let config = try makeConfig(tie: false)
        let weights: [String: MLXArray] = [
            "model.layers.2.eh_proj.weight": vector(5),
            "model.layers.2.mlp.experts.0.gate_proj.weight": vector(6),
            "model.layers.1.mlp.gate.weight": vector(7),
        ]

        let result = SharpAIPanguUltraMoeSanitizer.sanitize(weights, configuration: config)

        XCTAssertNil(result["model.layers.2.eh_proj.weight"])
        XCTAssertNil(result["model.layers.2.mlp.experts.0.gate_proj.weight"])
        XCTAssertNotNil(result["model.layers.1.mlp.gate.weight"])
    }

    func testKeepsHFNamesForExpertWeights() throws {
        let config = try makeConfig(tie: false)
        let name = "model.layers.1.mlp.experts.0.gate_proj.weight"
        let weights: [String: MLXArray] = [name: vector(4)]

        let result = SharpAIPanguUltraMoeSanitizer.sanitize(weights, configuration: config)

        XCTAssertNotNil(result[name])
    }
}
