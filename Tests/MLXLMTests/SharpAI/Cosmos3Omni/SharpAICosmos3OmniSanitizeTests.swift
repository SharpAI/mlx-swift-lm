// SharpAI fork overlay tests for cosmos3_omni checkpoint key sanitization. Not part of upstream.

import Foundation
import MLX
@testable import MLXVLM
import XCTest

final class SharpAICosmos3OmniSanitizeTests: XCTestCase {

    private func vector(_ value: Float) -> MLXArray {
        MLXArray([value, value + 1])
    }

    func testMapsGenerationTowerKeysToContract() throws {
        let weights: [String: MLXArray] = [
            "layers.3.self_attn.add_q_proj.weight": vector(0),
            "layers.3.self_attn.add_k_proj.weight": vector(1),
            "layers.3.self_attn.add_v_proj.weight": vector(2),
            "layers.3.self_attn.to_add_out.weight": vector(3),
            "layers.3.self_attn.norm_added_q.weight": vector(4),
            "layers.3.self_attn.norm_added_k.weight": vector(5),
            "layers.3.mlp_moe_gen.gate_proj.weight": vector(6),
            "layers.3.mlp_moe_gen.up_proj.weight": vector(7),
            "layers.3.mlp_moe_gen.down_proj.weight": vector(8),
            "layers.3.input_layernorm_moe_gen.weight": vector(9),
            "layers.3.post_attention_layernorm_moe_gen.weight": vector(10),
            "norm_moe_gen.weight": vector(11),
        ]

        let result = try SharpAICosmos3OmniSanitizer.sanitize(weights)

        XCTAssertEqual(
            Set(result.keys),
            [
                "gen.layers.3.self_attn.q_proj.weight",
                "gen.layers.3.self_attn.k_proj.weight",
                "gen.layers.3.self_attn.v_proj.weight",
                "gen.layers.3.self_attn.o_proj.weight",
                "gen.layers.3.self_attn.q_norm.weight",
                "gen.layers.3.self_attn.k_norm.weight",
                "gen.layers.3.mlp.gate_proj.weight",
                "gen.layers.3.mlp.up_proj.weight",
                "gen.layers.3.mlp.down_proj.weight",
                "gen.layers.3.input_layernorm.weight",
                "gen.layers.3.post_attention_layernorm.weight",
                "gen.norm.weight",
            ])
        // Values follow their key: add_q_proj -> q_proj, to_add_out -> o_proj.
        XCTAssertEqual(result["gen.layers.3.self_attn.q_proj.weight"]?.asArray(Float.self), [0, 1])
        XCTAssertEqual(result["gen.layers.3.self_attn.o_proj.weight"]?.asArray(Float.self), [3, 4])
        XCTAssertEqual(result["gen.norm.weight"]?.asArray(Float.self), [11, 12])
    }

    func testGenerationAttentionKeysSurviveAttentionRename() throws {
        let sanitized = try SharpAICosmos3OmniSanitizer.sanitize([
            "layers.0.self_attn.to_q.weight": vector(0),
            "layers.0.self_attn.add_q_proj.weight": vector(1),
        ])

        let result = try SharpAICosmos3OmniSanitizer.qwen3VLAttentionNames(sanitized)

        XCTAssertEqual(
            Set(result.keys),
            [
                "language_model.model.layers.0.self_attn.q_proj.weight",
                "gen.layers.0.self_attn.q_proj.weight",
            ])
    }

    func testMapsAllHeadKeysUnderGenHeads() throws {
        var weights = [String: MLXArray]()
        for key in SharpAICosmos3OmniSanitizer.generationHeadKeys { weights[key] = vector(0) }
        XCTAssertEqual(weights.count, 18)

        let result = try SharpAICosmos3OmniSanitizer.sanitize(weights)

        XCTAssertEqual(Set(result.keys), Set(weights.keys.map { "gen_heads." + $0 }))
    }

    func testPartitionSeparatesGenerationFromUnderstanding() throws {
        let result = try SharpAICosmos3OmniSanitizer.sanitize([
            "embed_tokens.weight": vector(0),
            "norm_moe_gen.weight": vector(1),
            "proj_in.weight": vector(2),
        ])

        let parts = SharpAICosmos3OmniSanitizer.partitionGeneration(result)

        XCTAssertEqual(Set(parts.understanding.keys), ["language_model.model.embed_tokens.weight"])
        XCTAssertEqual(Set(parts.generation.keys), ["gen.norm.weight", "gen_heads.proj_in.weight"])
    }

    func testUnrecognizedGenerationKeyThrows() {
        for key in [
            "layers.0.self_attn.add_q_proj.bias",
            "layers.0.mlp_moe_gen.router.weight",
            "layers.0.self_attn.q_proj_moe_gen.weight",
            "layers.x.mlp_moe_gen.up_proj.weight",
            "action_proj_in.extra.weight",
            "audio_proj_out.fc.weight",
        ] {
            XCTAssertThrowsError(try SharpAICosmos3OmniSanitizer.sanitize([key: vector(0)]), key) {
                error in
                guard case SharpAICosmos3OmniError.unrecognizedWeightKey(let rejected) = error else {
                    XCTFail("unexpected error: \(error)")
                    return
                }
                XCTAssertEqual(rejected, key)
            }
        }
    }

    func testUnrecognizedUnderstandingKeyThrows() {
        let weights: [String: MLXArray] = ["mystery.weight": vector(0)]
        XCTAssertThrowsError(try SharpAICosmos3OmniSanitizer.sanitize(weights)) { error in
            guard case SharpAICosmos3OmniError.unrecognizedWeightKey(let key) = error else {
                XCTFail("unexpected error: \(error)")
                return
            }
            XCTAssertEqual(key, "mystery.weight")
        }
    }

    func testDuplicateGenerationKeyThrows() {
        let weights: [String: MLXArray] = [
            "norm_moe_gen.weight": vector(0),
            "model.language_model.norm_moe_gen.weight": vector(1),
        ]

        XCTAssertThrowsError(try SharpAICosmos3OmniSanitizer.sanitize(weights)) { error in
            guard case SharpAICosmos3OmniError.duplicateWeightKey(let key) = error else {
                XCTFail("unexpected error: \(error)")
                return
            }
            XCTAssertEqual(key, "gen.norm.weight")
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

    /// Synthetic key list that follows the verified counts of the nvidia/Cosmos3-Nano root
    /// `model.safetensors.index.json` (1165 keys): vision 351 (blocks 27x12, deepstack mergers 3x6,
    /// merger 6, patch_embed 2, pos_embed 1), understanding 399 (36 layers x 11, embed/lm_head/norm),
    /// generation tower 397 (36 layers x 6 attention + 3 mlp + 2 norms, plus norm_moe_gen), and
    /// 18 head keys.
    private func cosmos3NanoKeys() -> (
        vision: [String], understanding: [String], generation: [String], heads: [String]
    ) {
        let blockSuffixes = [
            "attn.proj.bias", "attn.proj.weight", "attn.qkv.bias", "attn.qkv.weight",
            "mlp.linear_fc1.bias", "mlp.linear_fc1.weight", "mlp.linear_fc2.bias",
            "mlp.linear_fc2.weight", "norm1.bias", "norm1.weight", "norm2.bias", "norm2.weight",
        ]
        let mergerSuffixes = [
            "linear_fc1.bias", "linear_fc1.weight", "linear_fc2.bias", "linear_fc2.weight",
            "norm.bias", "norm.weight",
        ]
        let undLayerSuffixes = [
            "input_layernorm.weight", "post_attention_layernorm.weight",
            "mlp.down_proj.weight", "mlp.gate_proj.weight", "mlp.up_proj.weight",
            "self_attn.norm_k.weight", "self_attn.norm_q.weight", "self_attn.to_k.weight",
            "self_attn.to_out.weight", "self_attn.to_q.weight", "self_attn.to_v.weight",
        ]
        let genLayerSuffixes = [
            "self_attn.add_k_proj.weight", "self_attn.add_q_proj.weight",
            "self_attn.add_v_proj.weight", "self_attn.norm_added_k.weight",
            "self_attn.norm_added_q.weight", "self_attn.to_add_out.weight",
            "mlp_moe_gen.gate_proj.weight", "mlp_moe_gen.up_proj.weight",
            "mlp_moe_gen.down_proj.weight", "input_layernorm_moe_gen.weight",
            "post_attention_layernorm_moe_gen.weight",
        ]

        var vision = [String]()
        for i in 0..<27 { vision += blockSuffixes.map { "blocks.\(i).\($0)" } }
        for i in 0..<3 { vision += mergerSuffixes.map { "deepstack_merger_list.\(i).\($0)" } }
        vision += mergerSuffixes.map { "merger.\($0)" }
        vision += ["patch_embed.proj.bias", "patch_embed.proj.weight", "pos_embed.weight"]

        var understanding = [String]()
        for i in 0..<36 { understanding += undLayerSuffixes.map { "layers.\(i).\($0)" } }
        understanding += ["embed_tokens.weight", "lm_head.weight", "norm.weight"]

        var generation = [String]()
        for i in 0..<36 { generation += genLayerSuffixes.map { "layers.\(i).\($0)" } }
        generation += ["norm_moe_gen.weight"]

        let heads = [
            "proj_in.weight", "proj_in.bias", "proj_out.weight", "proj_out.bias",
            "time_embedder.linear_1.weight", "time_embedder.linear_1.bias",
            "time_embedder.linear_2.weight", "time_embedder.linear_2.bias",
            "audio_modality_embed", "audio_proj_in.weight", "audio_proj_in.bias",
            "audio_proj_out.weight", "audio_proj_out.bias",
            "action_modality_embed", "action_proj_in.fc.weight", "action_proj_in.bias.weight",
            "action_proj_out.fc.weight", "action_proj_out.bias.weight",
        ]
        return (vision, understanding, generation, heads)
    }

    func testCheckpointKeyLayoutMapsEveryKeyToContract() throws {
        let keys = cosmos3NanoKeys()
        XCTAssertEqual(keys.vision.count, 351)
        XCTAssertEqual(keys.understanding.count, 399)
        XCTAssertEqual(keys.generation.count, 397)
        XCTAssertEqual(keys.heads.count, 18)

        let all = keys.vision + keys.understanding + keys.generation + keys.heads
        XCTAssertEqual(all.count, 1165)
        XCTAssertEqual(Set(all).count, 1165)

        var weights = [String: MLXArray]()
        for (index, key) in all.enumerated() { weights[key] = vector(Float(index)) }

        // The full pipeline the wrapper runs: sanitize, then the Qwen3-VL attention names.
        let sanitized = try SharpAICosmos3OmniSanitizer.sanitize(weights)
        let result = try SharpAICosmos3OmniSanitizer.qwen3VLAttentionNames(sanitized)

        // Nothing dropped, nothing duplicated.
        XCTAssertEqual(sanitized.count, 1165)
        XCTAssertEqual(result.count, 1165)

        // Expected output built independently of the rename tables.
        var expected = Set<String>()
        expected.formUnion(keys.vision.map { "vision_tower." + $0 })
        for i in 0..<36 {
            for name in ["q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm"] {
                expected.insert("language_model.model.layers.\(i).self_attn.\(name).weight")
                expected.insert("gen.layers.\(i).self_attn.\(name).weight")
            }
            for name in ["gate_proj", "up_proj", "down_proj"] {
                expected.insert("language_model.model.layers.\(i).mlp.\(name).weight")
                expected.insert("gen.layers.\(i).mlp.\(name).weight")
            }
            for name in ["input_layernorm", "post_attention_layernorm"] {
                expected.insert("language_model.model.layers.\(i).\(name).weight")
                expected.insert("gen.layers.\(i).\(name).weight")
            }
        }
        expected.formUnion([
            "language_model.model.embed_tokens.weight", "language_model.model.norm.weight",
            "language_model.lm_head.weight", "gen.norm.weight",
        ])
        expected.formUnion(keys.heads.map { "gen_heads." + $0 })
        XCTAssertEqual(expected.count, 1165)
        XCTAssertEqual(Set(result.keys), expected)

        let parts = SharpAICosmos3OmniSanitizer.partitionGeneration(result)
        XCTAssertEqual(parts.understanding.count, 750)
        XCTAssertEqual(parts.generation.count, 415)
        XCTAssertEqual(parts.generation.keys.filter { $0.hasPrefix("gen.") }.count, 397)
        XCTAssertEqual(parts.generation.keys.filter { $0.hasPrefix("gen_heads.") }.count, 18)

        // Spot check that values follow their keys through both renames.
        func value(_ key: String) -> Float? { result[key]?.asArray(Float.self).first }
        func original(_ key: String) -> Float? { all.firstIndex(of: key).map { Float($0) } }
        XCTAssertEqual(
            value("gen.layers.35.self_attn.o_proj.weight"),
            original("layers.35.self_attn.to_add_out.weight"))
        XCTAssertEqual(
            value("language_model.model.layers.35.self_attn.o_proj.weight"),
            original("layers.35.self_attn.to_out.weight"))
        XCTAssertEqual(value("gen.norm.weight"), original("norm_moe_gen.weight"))
        XCTAssertEqual(
            value("gen_heads.action_proj_out.bias.weight"),
            original("action_proj_out.bias.weight"))
    }
}
