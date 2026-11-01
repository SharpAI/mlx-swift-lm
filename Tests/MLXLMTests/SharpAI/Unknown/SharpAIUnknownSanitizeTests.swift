// SharpAI fork overlay tests for "unknown" checkpoint key sanitization. Not part of upstream.
// Offline only. UNVERIFIED: key prefixes and the layer offset come from vLLM, not a checkpoint.
// Written without compiling. Run it before relying on it.

import Foundation
import MLX
import MLXLMCommon
import XCTest

@testable import MLXLLM

final class SharpAIUnknownSanitizeTests: XCTestCase {

    private func eagleConfig(tie: Bool) throws -> SharpAIEagleCohereConfiguration {
        let json = """
            {
                "architectures": ["EagleCohereForCausalLM"],
                "hidden_size": 8,
                "num_hidden_layers": 2,
                "intermediate_size": 16,
                "num_attention_heads": 2,
                "num_key_value_heads": 2,
                "vocab_size": 16,
                "layer_norm_eps": 1e-05,
                "logit_scale": 1.0,
                "tie_word_embeddings": \(tie)
            }
            """
        return try JSONDecoder.json5().decode(
            SharpAIEagleCohereConfiguration.self, from: Data(json.utf8))
    }

    private func panguConfig(tie: Bool) throws -> SharpAIPanguProMoEV2Configuration {
        let json = """
            {
                "architectures": ["PanguProMoEV2ForCausalLM"],
                "hidden_size": 8,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "vocab_size": 16,
                "n_routed_experts": 4,
                "num_experts_per_tok": 2,
                "moe_intermediate_size": 4,
                "tie_word_embeddings": \(tie)
            }
            """
        return try JSONDecoder.json5().decode(
            SharpAIPanguProMoEV2Configuration.self, from: Data(json.utf8))
    }

    private func vector(_ value: Float) -> MLXArray {
        MLXArray([value, value + 1])
    }

    func testEagleStripsModelPrefix() throws {
        let config = try eagleConfig(tie: false)
        let result = SharpAIUnknownSanitizer.eagleCohere(
            ["model.fc.weight": vector(1), "model.norm.weight": vector(2)],
            configuration: config)

        XCTAssertNotNil(result["fc.weight"])
        XCTAssertNotNil(result["norm.weight"])
        XCTAssertNil(result["model.fc.weight"])
    }

    func testEagleShiftsLayerIndexByOffset() throws {
        let config = try eagleConfig(tie: false)
        let result = SharpAIUnknownSanitizer.eagleCohere(
            [
                "model.layers.5.mlp.up_proj.weight": vector(3),
                "model.layers.6.mlp.up_proj.weight": vector(4),
            ],
            configuration: config, layerOffset: 5)

        XCTAssertNotNil(result["layers.0.mlp.up_proj.weight"])
        XCTAssertNotNil(result["layers.1.mlp.up_proj.weight"])
        XCTAssertNil(result["layers.5.mlp.up_proj.weight"])
    }

    func testEagleDropsLayersOutsideDraftDepth() throws {
        let config = try eagleConfig(tie: false)
        let result = SharpAIUnknownSanitizer.eagleCohere(
            ["model.layers.9.mlp.up_proj.weight": vector(5)],
            configuration: config, layerOffset: 0)

        XCTAssertTrue(result.isEmpty)
    }

    func testEagleDropsLMHeadAndRotaryBuffers() throws {
        let config = try eagleConfig(tie: false)
        let result = SharpAIUnknownSanitizer.eagleCohere(
            [
                "lm_head.weight": vector(6),
                "model.layers.0.self_attn.rotary_emb.inv_freq": vector(7),
                "model.fc.bias": vector(8),
            ],
            configuration: config)

        XCTAssertNil(result["lm_head.weight"])
        XCTAssertNil(result["layers.0.self_attn.rotary_emb.inv_freq"])
        XCTAssertNotNil(result["fc.bias"])
    }

    func testEagleDropsEmbeddingWhenTied() throws {
        let tied = SharpAIUnknownSanitizer.eagleCohere(
            ["model.embed_tokens.weight": vector(9)], configuration: try eagleConfig(tie: true))
        let untied = SharpAIUnknownSanitizer.eagleCohere(
            ["model.embed_tokens.weight": vector(9)], configuration: try eagleConfig(tie: false))

        XCTAssertNil(tied["embed_tokens.weight"])
        XCTAssertNotNil(untied["embed_tokens.weight"])
    }

    func testPanguDropsRotaryBuffers() throws {
        let config = try panguConfig(tie: false)
        let result = SharpAIUnknownSanitizer.panguProMoEV2(
            [
                "model.layers.0.self_attn.rotary_emb.inv_freq": vector(0),
                "model.layers.0.self_attn.q_proj.weight": vector(1),
            ],
            configuration: config)

        XCTAssertNil(result["model.layers.0.self_attn.rotary_emb.inv_freq"])
        XCTAssertNotNil(result["model.layers.0.self_attn.q_proj.weight"])
    }

    func testPanguDropsLMHeadWhenTied() throws {
        let tied = SharpAIUnknownSanitizer.panguProMoEV2(
            ["lm_head.weight": vector(2)], configuration: try panguConfig(tie: true))
        let untied = SharpAIUnknownSanitizer.panguProMoEV2(
            ["lm_head.weight": vector(2)], configuration: try panguConfig(tie: false))

        XCTAssertNil(tied["lm_head.weight"])
        XCTAssertNotNil(untied["lm_head.weight"])
    }

    func testPanguKeepsHFNamesForExpertAndGateWeights() throws {
        let config = try panguConfig(tie: false)
        let names = [
            "model.layers.1.mlp.experts.0.gate_proj.weight",
            "model.layers.1.mlp.gate.weight",
            "model.layers.1.mlp.gate.e_score_correction_bias",
        ]
        let weights = Dictionary(uniqueKeysWithValues: names.map { ($0, vector(3)) })
        let result = SharpAIUnknownSanitizer.panguProMoEV2(weights, configuration: config)

        for name in names {
            XCTAssertNotNil(result[name], name)
        }
    }
}
