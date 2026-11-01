// SharpAI fork overlay tests for flint checkpoint key sanitization. Not part of upstream.

import Foundation
import MLX
import MLXLMCommon
import XCTest

@testable import MLXLLM

final class SharpAIFlintSanitizeTests: XCTestCase {

    private func makeConfiguration() throws -> SharpAIFlintConfiguration {
        try JSONDecoder.json5().decode(
            SharpAIFlintConfiguration.self,
            from: Data(
                #"""
                {"num_hidden_layers": 2, "mlp_only_layers": [0], "num_experts": 4,
                 "num_experts_per_tok": 2, "hidden_size": 32, "moe_intermediate_size": 16}
                """#.utf8))
    }

    func testStacksExpertsIntoSwitchMLP() throws {
        let config = try makeConfiguration()
        var weights: [String: MLXArray] = [:]
        for expert in 0 ..< 4 {
            for proj in ["gate_proj", "up_proj", "down_proj"] {
                weights["model.layers.1.mlp.experts.\(expert).\(proj).weight"] = zeros([16, 32])
            }
        }

        let sanitized = SharpAIFlintSanitizer.sanitize(weights, configuration: config)

        XCTAssertNil(sanitized["model.layers.1.mlp.experts.0.gate_proj.weight"])
        XCTAssertEqual(
            sanitized["model.layers.1.mlp.switch_mlp.gate_proj.weight"]?.shape, [4, 16, 32])
        XCTAssertEqual(
            sanitized["model.layers.1.mlp.switch_mlp.down_proj.weight"]?.shape, [4, 16, 32])
        XCTAssertEqual(sanitized.count, 3)
    }

    func testKeepsHFNamesForSinksRouterAndSharedExpert() throws {
        let config = try makeConfiguration()
        let names = [
            "model.layers.1.self_attn.softmax_offset",
            "model.layers.1.mlp.gate.weight",
            "model.layers.1.mlp.shared_expert.gate_proj.weight",
            "model.layers.0.mlp.gate_proj.weight",
            "model.layers.0.self_attn.q_proj.weight",
            "model.embed_tokens.weight",
            "lm_head.weight",
        ]
        var weights: [String: MLXArray] = [:]
        for name in names {
            weights[name] = zeros([2])
        }

        let sanitized = SharpAIFlintSanitizer.sanitize(weights, configuration: config)

        XCTAssertEqual(Set(sanitized.keys), Set(names))
    }

    func testDropsRotaryInvFreq() throws {
        let config = try makeConfiguration()
        let weights: [String: MLXArray] = [
            "model.layers.0.self_attn.rotary_emb.inv_freq": zeros([4]),
            "model.embed_tokens.weight": zeros([2]),
        ]

        let sanitized = SharpAIFlintSanitizer.sanitize(weights, configuration: config)

        XCTAssertNil(sanitized["model.layers.0.self_attn.rotary_emb.inv_freq"])
        XCTAssertNotNil(sanitized["model.embed_tokens.weight"])
    }

    func testIncompleteExpertSetIsLeftUnchanged() throws {
        let config = try makeConfiguration()
        var weights: [String: MLXArray] = [:]
        // Experts 0..2 only; expert 3 is missing, so nothing is stacked.
        for expert in 0 ..< 3 {
            weights["model.layers.1.mlp.experts.\(expert).down_proj.weight"] = zeros([16, 32])
        }

        let sanitized = SharpAIFlintSanitizer.sanitize(weights, configuration: config)

        XCTAssertNotNil(sanitized["model.layers.1.mlp.experts.0.down_proj.weight"])
        XCTAssertNil(sanitized["model.layers.1.mlp.switch_mlp.down_proj.weight"])
    }
}
