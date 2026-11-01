// SharpAI fork overlay tests for interns2_mobius checkpoint key sanitization. Not part of upstream.

import Foundation
import MLX
import MLXLLM
import MLXNN
import MLXVLM
import XCTest

final class SharpAIInterns2MobiusSanitizeTests: XCTestCase {

    private func vector(_ value: Float) -> MLXArray {
        MLXArray([value, value + 1])
    }

    func testDropsRotaryBuffers() {
        let weights: [String: MLXArray] = [
            "model.layers.0.self_attn.rotary_emb.inv_freq": vector(0),
            "model.layers.0.self_attn.q_proj.weight": vector(1),
        ]

        let result = SharpAIInterns2MobiusSanitizer.sanitize(weights)

        XCTAssertNil(result["model.layers.0.self_attn.rotary_emb.inv_freq"])
        XCTAssertNotNil(result["model.layers.0.self_attn.q_proj.weight"])
    }

    func testKeepsHFKeyNamesUnchanged() {
        let names = [
            "language_model.model.embed_tokens.weight",
            "language_model.lm_head.weight",
            "vision_model.embeddings.patch_embedding.weight",
        ]
        var weights: [String: MLXArray] = [:]
        for name in names {
            weights[name] = vector(2)
        }

        let result = SharpAIInterns2MobiusSanitizer.sanitize(weights)

        XCTAssertEqual(Set(result.keys), Set(names))
    }

    func testEmptyInputStaysEmpty() {
        XCTAssertTrue(SharpAIInterns2MobiusSanitizer.sanitize([:]).isEmpty)
    }

    func testLayerIndexReadsFirstLayersSegment() {
        XCTAssertEqual(
            SharpAIInterns2MobiusSanitizer.layerIndex(of: "model.layers.12.mlp.experts.gate"), 12)
        XCTAssertEqual(
            SharpAIInterns2MobiusSanitizer.layerIndex(of: "language_model.model.layers.3.x"), 3)
    }

    func testLayerIndexIsNilWithoutLayersSegment() {
        XCTAssertNil(SharpAIInterns2MobiusSanitizer.layerIndex(of: "lm_head.weight"))
        XCTAssertNil(SharpAIInterns2MobiusSanitizer.layerIndex(of: "model.layers.x.weight"))
    }

    func testSharedBankKeysGroupedByLayer() {
        let keys = [
            "model.layers.0.shared_experts.up_proj.weight",
            "model.layers.0.shared_experts.down_proj.weight",
            "model.layers.1.shared_experts.up_proj.weight",
            "model.layers.1.mlp.experts.up_proj.weight",
            "lm_head.weight",
        ]

        let grouped = SharpAIInterns2MobiusSanitizer.sharedBankKeysByLayer(keys)

        XCTAssertEqual(
            grouped[0],
            [
                "model.layers.0.shared_experts.down_proj.weight",
                "model.layers.0.shared_experts.up_proj.weight",
            ])
        XCTAssertEqual(grouped[1], ["model.layers.1.shared_experts.up_proj.weight"])
        XCTAssertEqual(grouped.count, 2)
    }

    // Source: HF modeling_interns2_mobius.py selects bank `layer_idx % num_blocks` (num_blocks
    // defaults to 4). The checkpoint already stores banks as `meta_mlp.<block>`, so sanitize keeps them.
    func testSharedBankRemapKnownIssue() {
        XCTAssertEqual(SharpAIInterns2MobiusSanitizer.bankIndex(forLayer: 0, numBlocks: 4), 0)
        XCTAssertEqual(SharpAIInterns2MobiusSanitizer.bankIndex(forLayer: 3, numBlocks: 4), 3)
        XCTAssertEqual(SharpAIInterns2MobiusSanitizer.bankIndex(forLayer: 4, numBlocks: 4), 0)
        XCTAssertEqual(SharpAIInterns2MobiusSanitizer.bankIndex(forLayer: 39, numBlocks: 4), 3)

        let bankKey = "model.language_model.meta_mlp.2.experts.gate_up_proj"
        let result = SharpAIInterns2MobiusSanitizer.sanitize([bankKey: vector(3)])
        XCTAssertNotNil(result[bankKey])
    }

    /// Model-level sanitize, from the real checkpoint names: language_model prefix, fused expert split,
    /// conv1d layout, and dropped mtp / visual / rotary keys.
    func testModelSanitizeMapsCheckpointNames() throws {
        let json = """
            {
                "model_type": "interns2_mobius",
                "text_config": {
                    "hidden_size": 32, "num_hidden_layers": 4,
                    "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"],
                    "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 16,
                    "linear_num_key_heads": 2, "linear_num_value_heads": 4,
                    "linear_key_head_dim": 8, "linear_value_head_dim": 8,
                    "num_experts": 4, "num_experts_per_tok": 2,
                    "moe_intermediate_size": 8, "shared_expert_intermediate_size": 8,
                    "num_blocks": 2, "vocab_size": 64
                }
            }
            """
        let config = try JSONDecoder.json5().decode(
            SharpAIInterns2MobiusConfiguration.self, from: Data(json.utf8))
        let model = SharpAIInterns2MobiusModel(config.textConfiguration)

        let weights: [String: MLXArray] = [
            "model.language_model.embed_tokens.weight": vector(0),
            "model.language_model.meta_mlp.1.experts.gate_up_proj": MLXArray.zeros([4, 16, 32]),
            "model.language_model.meta_mlp.1.experts.down_proj": MLXArray.zeros([4, 32, 8]),
            "model.language_model.meta_mlp.1.gate.weight": MLXArray.zeros([4, 32]),
            "model.language_model.layers.0.linear_attn.conv1d.weight": MLXArray.zeros([32, 1, 4]),
            "model.language_model.layers.0.linear_attn.A_log": vector(1),
            "mtp.fc.weight": vector(2),
            "model.visual.blocks.0.attn.qkv.weight": vector(3),
            "lm_head.weight": vector(4),
        ]

        let result = model.sanitize(weights: weights)

        XCTAssertNotNil(result["model.embed_tokens.weight"])
        XCTAssertEqual(result["model.meta_mlp.1.switch_mlp.gate_proj.weight"]?.shape, [4, 8, 32])
        XCTAssertEqual(result["model.meta_mlp.1.switch_mlp.up_proj.weight"]?.shape, [4, 8, 32])
        XCTAssertEqual(result["model.meta_mlp.1.switch_mlp.down_proj.weight"]?.shape, [4, 32, 8])
        XCTAssertNotNil(result["model.meta_mlp.1.gate.weight"])
        XCTAssertEqual(result["model.layers.0.linear_attn.conv1d.weight"]?.shape, [32, 4, 1])
        XCTAssertNotNil(result["model.layers.0.linear_attn.A_log"])
        XCTAssertNotNil(result["lm_head.weight"])
        XCTAssertTrue(result.keys.allSatisfy { !$0.hasPrefix("mtp.") && !$0.contains("visual") })
        XCTAssertNil(result["model.language_model.embed_tokens.weight"])
    }

    func testTiedEmbeddingsDropLMHeadAndHaveNoHeadModule() throws {
        let json = """
            {
                "model_type": "interns2_mobius",
                "text_config": {
                    "hidden_size": 32, "num_hidden_layers": 4, "tie_word_embeddings": true,
                    "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"],
                    "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 16,
                    "linear_num_key_heads": 2, "linear_num_value_heads": 4,
                    "linear_key_head_dim": 8, "linear_value_head_dim": 8,
                    "num_experts": 4, "num_experts_per_tok": 2,
                    "moe_intermediate_size": 8, "shared_expert_intermediate_size": 8,
                    "num_blocks": 2, "vocab_size": 64
                }
            }
            """
        let config = try JSONDecoder.json5().decode(
            SharpAIInterns2MobiusConfiguration.self, from: Data(json.utf8))
        let model = SharpAIInterns2MobiusModel(config.textConfiguration)

        let result = model.sanitize(weights: [
            "model.language_model.embed_tokens.weight": vector(0),
            "lm_head.weight": vector(4),
        ])

        XCTAssertNil(result["lm_head.weight"])
        XCTAssertNotNil(result["model.embed_tokens.weight"])
        XCTAssertFalse(model.parameters().flattened().contains { $0.0.hasPrefix("lm_head") })
    }
}
