// SharpAI fork overlay tests for cosmos3_omni checkpoint key sanitization. Not part of upstream.

import Foundation
import MLX
import MLXVLM
import XCTest

final class SharpAICosmos3OmniSanitizeTests: XCTestCase {

    private func vector(_ value: Float) -> MLXArray {
        MLXArray([value, value + 1])
    }

    func testRejectsGenerationExpertWeights() {
        let weights: [String: MLXArray] = [
            "model.language_model.layers.0.self_attn.q_proj_moe_gen.weight": vector(0)
        ]

        XCTAssertThrowsError(try SharpAICosmos3OmniSanitizer.sanitize(weights)) { error in
            guard case SharpAICosmos3OmniError.generationWeightNotSupported(let key) = error else {
                XCTFail("unexpected error: \(error)")
                return
            }
            XCTAssertEqual(key, "model.language_model.layers.0.self_attn.q_proj_moe_gen.weight")
        }
    }

    func testDropsPositionIds() throws {
        let weights: [String: MLXArray] = [
            "model.visual.position_ids": vector(0),
            "model.visual.blocks.0.attn.qkv.weight": vector(1),
        ]

        let result = try SharpAICosmos3OmniSanitizer.sanitize(weights)

        XCTAssertEqual(Set(result.keys), ["vision_tower.blocks.0.attn.qkv.weight"])
    }

    func testRemapsUnderstandingPrefixes() throws {
        let weights: [String: MLXArray] = [
            "model.visual.blocks.0.attn.qkv.weight": vector(1),
            "model.language_model.embed_tokens.weight": vector(2),
            "lm_head.weight": vector(3),
        ]

        let result = try SharpAICosmos3OmniSanitizer.sanitize(weights)

        XCTAssertEqual(
            Set(result.keys),
            [
                "vision_tower.blocks.0.attn.qkv.weight",
                "language_model.model.embed_tokens.weight",
                "language_model.lm_head.weight",
            ])
    }

    func testTiedEmbeddingsDropLMHead() throws {
        let weights: [String: MLXArray] = [
            "model.language_model.embed_tokens.weight": vector(1),
            "lm_head.weight": vector(2),
        ]

        let result = try SharpAICosmos3OmniSanitizer.sanitize(weights, tieWordEmbeddings: true)

        XCTAssertNil(result["language_model.lm_head.weight"])
        XCTAssertNotNil(result["language_model.model.embed_tokens.weight"])
    }

    func testAttentionNamesMapToQwen3VLLayout() throws {
        let weights: [String: MLXArray] = [
            "language_model.model.layers.0.self_attn.to_q.weight": vector(0),
            "language_model.model.layers.0.self_attn.to_k.weight": vector(1),
            "language_model.model.layers.0.self_attn.to_v.weight": vector(2),
            "language_model.model.layers.0.self_attn.to_out.weight": vector(3),
            "language_model.model.layers.0.self_attn.norm_q.weight": vector(4),
            "language_model.model.layers.0.self_attn.norm_k.weight": vector(5),
            "language_model.model.layers.0.self_attn.add_q_proj.weight": vector(6),
        ]

        let result = try SharpAICosmos3OmniSanitizer.qwen3VLAttentionNames(weights)

        XCTAssertEqual(
            Set(result.keys),
            [
                "language_model.model.layers.0.self_attn.q_proj.weight",
                "language_model.model.layers.0.self_attn.k_proj.weight",
                "language_model.model.layers.0.self_attn.v_proj.weight",
                "language_model.model.layers.0.self_attn.o_proj.weight",
                "language_model.model.layers.0.self_attn.q_norm.weight",
                "language_model.model.layers.0.self_attn.k_norm.weight",
                "language_model.model.layers.0.self_attn.add_q_proj.weight",
            ])
        XCTAssertEqual(
            result["language_model.model.layers.0.self_attn.q_proj.weight"]?.asArray(Float.self),
            [0, 1])
    }

    func testPatchEmbedTransposedWhenChannelsAreLast() throws {
        let weights: [String: MLXArray] = [
            "model.visual.patch_embed.proj.weight": MLXArray.zeros([2, 2, 2, 2, 4])
        ]

        let result = try SharpAICosmos3OmniSanitizer.sanitize(weights, visionInChannels: 3)

        XCTAssertEqual(result["vision_tower.patch_embed.proj.weight"]?.shape, [2, 2, 2, 4, 2])
    }

    func testDuplicateRemappedKeyThrows() {
        let weights: [String: MLXArray] = [
            "model.language_model.norm.weight": vector(1),
            "language_model.model.norm.weight": vector(2),
        ]

        XCTAssertThrowsError(try SharpAICosmos3OmniSanitizer.sanitize(weights)) { error in
            guard case SharpAICosmos3OmniError.duplicateWeightKey(let key) = error else {
                XCTFail("unexpected error: \(error)")
                return
            }
            XCTAssertEqual(key, "language_model.model.norm.weight")
        }
    }

    func testEmptyInputStaysEmpty() throws {
        XCTAssertTrue(try SharpAICosmos3OmniSanitizer.sanitize([:]).isEmpty)
    }

    /// Understanding-branch keys of nvidia/Cosmos3-Nano, root `model.safetensors.index.json`
    /// (1165 keys). The 181 `*_moe_gen` keys and 18 diffusion-side keys (`action_*`, `audio_*`,
    /// `proj_in`/`proj_out`, `time_embedder.*`) are excluded. Counts: blocks 27x12, deepstack
    /// mergers 3x6, merger 6, patch_embed 2, pos_embed 1, layers 36x17, embed/lm_head/norm 3 = 966.
    private func cosmos3NanoUnderstandingKeys() -> [String] {
        let blockSuffixes = [
            "attn.proj.bias", "attn.proj.weight", "attn.qkv.bias", "attn.qkv.weight",
            "mlp.linear_fc1.bias", "mlp.linear_fc1.weight", "mlp.linear_fc2.bias",
            "mlp.linear_fc2.weight", "norm1.bias", "norm1.weight", "norm2.bias", "norm2.weight",
        ]
        let mergerSuffixes = [
            "linear_fc1.bias", "linear_fc1.weight", "linear_fc2.bias", "linear_fc2.weight",
            "norm.bias", "norm.weight",
        ]
        let layerSuffixes = [
            "input_layernorm.weight", "post_attention_layernorm.weight",
            "mlp.down_proj.weight", "mlp.gate_proj.weight", "mlp.up_proj.weight",
            "self_attn.add_k_proj.weight", "self_attn.add_q_proj.weight",
            "self_attn.add_v_proj.weight", "self_attn.norm_added_k.weight",
            "self_attn.norm_added_q.weight", "self_attn.norm_k.weight",
            "self_attn.norm_q.weight", "self_attn.to_add_out.weight", "self_attn.to_k.weight",
            "self_attn.to_out.weight", "self_attn.to_q.weight", "self_attn.to_v.weight",
        ]

        var keys = [String]()
        for i in 0..<27 { keys += blockSuffixes.map { "blocks.\(i).\($0)" } }
        for i in 0..<3 { keys += mergerSuffixes.map { "deepstack_merger_list.\(i).\($0)" } }
        keys += mergerSuffixes.map { "merger.\($0)" }
        keys += ["patch_embed.proj.bias", "patch_embed.proj.weight", "pos_embed.weight"]
        for i in 0..<36 { keys += layerSuffixes.map { "layers.\(i).\($0)" } }
        keys += ["embed_tokens.weight", "lm_head.weight", "norm.weight"]
        return keys
    }

    func testCheckpointKeyLayoutMapsRealUnderstandingKeys() throws {
        let keys = cosmos3NanoUnderstandingKeys()
        XCTAssertEqual(keys.count, 966)

        var weights = [String: MLXArray]()
        for key in keys { weights[key] = vector(0) }

        let result = try SharpAICosmos3OmniSanitizer.sanitize(weights)

        XCTAssertEqual(result.count, 966)
        XCTAssertTrue(
            result.keys.allSatisfy {
                $0.hasPrefix("vision_tower.") || $0.hasPrefix("language_model.")
            })
        XCTAssertNotNil(result["vision_tower.blocks.26.norm2.weight"])
        XCTAssertNotNil(result["vision_tower.deepstack_merger_list.2.linear_fc2.weight"])
        XCTAssertNotNil(result["vision_tower.merger.norm.bias"])
        XCTAssertNotNil(result["vision_tower.pos_embed.weight"])
        XCTAssertNotNil(result["vision_tower.patch_embed.proj.weight"])
        XCTAssertNotNil(result["language_model.model.embed_tokens.weight"])
        XCTAssertNotNil(result["language_model.model.norm.weight"])
        XCTAssertNotNil(result["language_model.lm_head.weight"])
        XCTAssertNotNil(result["language_model.model.layers.35.self_attn.to_q.weight"])
    }

    func testRejectsRealMoeGenKey() {
        // Real key from nvidia/Cosmos3-Nano index: layers.0.mlp_moe_gen.down_proj.weight
        let key = "layers.0.mlp_moe_gen.down_proj.weight"
        XCTAssertThrowsError(try SharpAICosmos3OmniSanitizer.sanitize([key: vector(0)])) { error in
            guard case SharpAICosmos3OmniError.generationWeightNotSupported(let rejected) = error
            else {
                XCTFail("unexpected error: \(error)")
                return
            }
            XCTAssertEqual(rejected, key)
        }
    }
}
