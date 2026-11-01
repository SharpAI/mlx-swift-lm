// SharpAI fork overlay tests for the minicpmv4_6 key mapping. Not part of upstream.

import Foundation
import MLX
import MLXLMCommon
@testable import MLXVLM
import XCTest

final class SharpAIMinicpmv46SanitizeTests: XCTestCase {

    func testRemapsVisionTowerAndMergers() {
        XCTAssertEqual(
            SharpAIMinicpmv46.remapKey("model.vision_tower.embeddings.patch_embedding.weight"),
            "vpm.embeddings.patch_embedding.weight")
        XCTAssertEqual(
            SharpAIMinicpmv46.remapKey("model.vision_tower.encoder.layers.0.mlp.fc1.weight"),
            "vpm.encoder.layers.0.mlp.fc1.weight")
        XCTAssertEqual(
            SharpAIMinicpmv46.remapKey("model.vision_tower.vit_merger.pre_norm.weight"),
            "vit_merger.pre_norm.weight")
        XCTAssertEqual(
            SharpAIMinicpmv46.remapKey("model.merger.mlp.0.linear_1.weight"),
            "merger.mlp.0.linear_1.weight")
    }

    func testRemapsLegacyVisionPrefixes() {
        XCTAssertEqual(
            SharpAIMinicpmv46.remapKey("model.vpm.post_layernorm.weight"),
            "vpm.post_layernorm.weight")
        XCTAssertEqual(
            SharpAIMinicpmv46.remapKey("model.vit_merger.linear_2.weight"),
            "vit_merger.linear_2.weight")
    }

    func testRemapsLanguageModelAndLMHead() {
        XCTAssertEqual(
            SharpAIMinicpmv46.remapKey("model.language_model.layers.3.mlp.gate_proj.weight"),
            "language_model.model.layers.3.mlp.gate_proj.weight")
        XCTAssertEqual(
            SharpAIMinicpmv46.remapKey("lm_head.weight"),
            "language_model.lm_head.weight")
    }

    func testDropsMTPAndRotaryKeys() {
        XCTAssertNil(SharpAIMinicpmv46.remapKey("mtp.0.fc.weight"))
        XCTAssertNil(
            SharpAIMinicpmv46.remapKey(
                "model.language_model.layers.0.self_attn.rotary_emb.inv_freq"))
        XCTAssertNil(
            SharpAIMinicpmv46.remapKey("model.vision_tower.embeddings.position_ids"))
    }

    func testSanitizeDropsTiedLMHeadAndAppliesRenames() throws {
        let config = try JSONDecoder.json5().decode(
            SharpAIMinicpmv46Configuration.self, from: Data(Self.tinyJSON.utf8))
        let model = try SharpAIMinicpmv46(config)

        let weights: [String: MLXArray] = [
            "model.vision_tower.embeddings.patch_embedding.weight": MLXArray.zeros([16, 3, 14, 14]),
            "model.language_model.embed_tokens.weight": MLXArray.zeros([128, 32]),
            "lm_head.weight": MLXArray.zeros([128, 32]),
            "mtp.0.fc.weight": MLXArray.zeros([4, 4]),
        ]
        let sanitized = model.sanitize(weights: weights)

        XCTAssertNotNil(sanitized["vpm.embeddings.patch_embedding.weight"])
        XCTAssertNotNil(sanitized["language_model.model.embed_tokens.weight"])
        XCTAssertNil(sanitized["language_model.lm_head.weight"])
        XCTAssertNil(sanitized["mtp.0.fc.weight"])
    }

    func testPatchEmbeddingIsChannelsLast() throws {
        let config = try JSONDecoder.json5().decode(
            SharpAIMinicpmv46Configuration.self, from: Data(Self.tinyJSON.utf8))
        let model = try SharpAIMinicpmv46(config)

        // HF layout (out, in, kH, kW) is moved to MLX layout (out, kH, kW, in).
        let hfWeight = MLXArray.zeros([16, 3, 14, 14])
        let sanitized = model.sanitize(weights: [
            "model.vision_tower.embeddings.patch_embedding.weight": hfWeight
        ])
        let moved = sanitized["vpm.embeddings.patch_embedding.weight"]
        XCTAssertEqual(moved?.shape, [16, 14, 14, 3])
    }

    static let tinyJSON = """
        {
            "model_type": "minicpmv4_6",
            "insert_layer_id": 1,
            "tie_word_embeddings": true,
            "text_config": {
                "model_type": "qwen3_5_text",
                "hidden_size": 32,
                "num_hidden_layers": 4,
                "intermediate_size": 64,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 16,
                "linear_num_key_heads": 2,
                "linear_num_value_heads": 2,
                "linear_key_head_dim": 16,
                "linear_value_head_dim": 16,
                "linear_conv_kernel_dim": 4,
                "vocab_size": 128,
                "tie_word_embeddings": true,
                "full_attention_interval": 4,
                "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"]
            },
            "vision_config": {
                "hidden_size": 16,
                "intermediate_size": 32,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "num_channels": 3,
                "image_size": 56,
                "patch_size": 14,
                "layer_norm_eps": 1e-6
            }
        }
        """
}
