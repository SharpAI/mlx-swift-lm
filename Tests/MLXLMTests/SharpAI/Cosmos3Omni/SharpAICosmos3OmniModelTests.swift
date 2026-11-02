// SharpAI fork overlay tests for the full cosmos3_omni wrapper model. Not part of upstream.
//
// Air-gapped: a synthetic safetensors file with the REAL checkpoint key names (diffusers layout,
// shapes of a shrunken config), no downloads, no MLXRandom.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import XCTest

@testable import MLXLMCommon
@testable import MLXVLM

final class SharpAICosmos3OmniModelTests: XCTestCase {

    /// Shrunken config: hidden 32, intermediate 32, 2 layers, 4 heads, 2 KV heads, head_dim 8, vocab 64,
    /// latent_channel 2 x patch 2 (patch_latent_dim 8), sound_dim 4, max_action_dim 4, 3 embodiment domains.
    private let configJSON = #"""
{"model_type":"cosmos3_omni","architectures":["Cosmos3OmniForConditionalGeneration"],"tie_word_embeddings":false,"image_token_id":60,"video_token_id":61,"vision_start_token_id":62,"vision_end_token_id":63,"model":{"config":{"sound_gen":true,"action_gen":true,"latent_channel":2,"latent_patch_size":2,"max_action_dim":4,"sound_dim":4,"num_embodiment_domains":3}},"text_config":{"model_type":"qwen3_vl_text","hidden_size":32,"intermediate_size":32,"num_hidden_layers":2,"num_attention_heads":4,"num_key_value_heads":2,"head_dim":8,"max_position_embeddings":512,"rms_norm_eps":1e-06,"vocab_size":64,"rope_parameters":{"mrope_interleaved":true,"mrope_section":[2,1,1],"rope_theta":10000,"rope_type":"default"}},"vision_config":{"deepstack_visual_indexes":[0,1],"depth":2,"hidden_size":32,"in_channels":3,"intermediate_size":32,"model_type":"qwen3_vl_vision","num_heads":4,"num_position_embeddings":16,"out_hidden_size":32,"patch_size":16,"spatial_merge_size":1,"temporal_patch_size":2}}
"""#

    /// Understanding and vision keys of the hf-tiny-v2 checkpoint (vocab shrunk to 64), same
    /// shapes as the Registration tests use.
    private let understandingEntries: [(String, [Int])] = [
            ("blocks.0.attn.proj.bias", [32]),
            ("blocks.0.attn.proj.weight", [32, 32]),
            ("blocks.0.attn.qkv.bias", [96]),
            ("blocks.0.attn.qkv.weight", [96, 32]),
            ("blocks.0.mlp.linear_fc1.bias", [32]),
            ("blocks.0.mlp.linear_fc1.weight", [32, 32]),
            ("blocks.0.mlp.linear_fc2.bias", [32]),
            ("blocks.0.mlp.linear_fc2.weight", [32, 32]),
            ("blocks.0.norm1.bias", [32]),
            ("blocks.0.norm1.weight", [32]),
            ("blocks.0.norm2.bias", [32]),
            ("blocks.0.norm2.weight", [32]),
            ("blocks.1.attn.proj.bias", [32]),
            ("blocks.1.attn.proj.weight", [32, 32]),
            ("blocks.1.attn.qkv.bias", [96]),
            ("blocks.1.attn.qkv.weight", [96, 32]),
            ("blocks.1.mlp.linear_fc1.bias", [32]),
            ("blocks.1.mlp.linear_fc1.weight", [32, 32]),
            ("blocks.1.mlp.linear_fc2.bias", [32]),
            ("blocks.1.mlp.linear_fc2.weight", [32, 32]),
            ("blocks.1.norm1.bias", [32]),
            ("blocks.1.norm1.weight", [32]),
            ("blocks.1.norm2.bias", [32]),
            ("blocks.1.norm2.weight", [32]),
            ("deepstack_merger_list.0.linear_fc1.bias", [32]),
            ("deepstack_merger_list.0.linear_fc1.weight", [32, 32]),
            ("deepstack_merger_list.0.linear_fc2.bias", [32]),
            ("deepstack_merger_list.0.linear_fc2.weight", [32, 32]),
            ("deepstack_merger_list.0.norm.bias", [32]),
            ("deepstack_merger_list.0.norm.weight", [32]),
            ("deepstack_merger_list.1.linear_fc1.bias", [32]),
            ("deepstack_merger_list.1.linear_fc1.weight", [32, 32]),
            ("deepstack_merger_list.1.linear_fc2.bias", [32]),
            ("deepstack_merger_list.1.linear_fc2.weight", [32, 32]),
            ("deepstack_merger_list.1.norm.bias", [32]),
            ("deepstack_merger_list.1.norm.weight", [32]),
            ("embed_tokens.weight", [64, 32]),
            ("layers.0.input_layernorm.weight", [32]),
            ("layers.0.mlp.down_proj.weight", [32, 32]),
            ("layers.0.mlp.gate_proj.weight", [32, 32]),
            ("layers.0.mlp.up_proj.weight", [32, 32]),
            ("layers.0.post_attention_layernorm.weight", [32]),
            ("layers.0.self_attn.norm_k.weight", [8]),
            ("layers.0.self_attn.norm_q.weight", [8]),
            ("layers.0.self_attn.to_k.weight", [16, 32]),
            ("layers.0.self_attn.to_out.weight", [32, 32]),
            ("layers.0.self_attn.to_q.weight", [32, 32]),
            ("layers.0.self_attn.to_v.weight", [16, 32]),
            ("layers.1.input_layernorm.weight", [32]),
            ("layers.1.mlp.down_proj.weight", [32, 32]),
            ("layers.1.mlp.gate_proj.weight", [32, 32]),
            ("layers.1.mlp.up_proj.weight", [32, 32]),
            ("layers.1.post_attention_layernorm.weight", [32]),
            ("layers.1.self_attn.norm_k.weight", [8]),
            ("layers.1.self_attn.norm_q.weight", [8]),
            ("layers.1.self_attn.to_k.weight", [16, 32]),
            ("layers.1.self_attn.to_out.weight", [32, 32]),
            ("layers.1.self_attn.to_q.weight", [32, 32]),
            ("layers.1.self_attn.to_v.weight", [16, 32]),
            ("lm_head.weight", [64, 32]),
            ("merger.linear_fc1.bias", [32]),
            ("merger.linear_fc1.weight", [32, 32]),
            ("merger.linear_fc2.bias", [32]),
            ("merger.linear_fc2.weight", [32, 32]),
            ("merger.norm.bias", [32]),
            ("merger.norm.weight", [32]),
            ("norm.weight", [32]),
            ("patch_embed.proj.bias", [32]),
            ("patch_embed.proj.weight", [32, 3, 2, 16, 16]),
            ("pos_embed.weight", [16, 32]),
    ]

    /// Generation tower and head keys with the checkpoint names of nvidia/Cosmos3-Nano, shrunken.
    private func generationEntries() -> [(String, [Int])] {
        var out = [(String, [Int])]()
        for n in 0 ..< 2 {
            let p = "layers.\(n)."
            out += [
                (p + "self_attn.add_q_proj.weight", [32, 32]),
                (p + "self_attn.add_k_proj.weight", [16, 32]),
                (p + "self_attn.add_v_proj.weight", [16, 32]),
                (p + "self_attn.to_add_out.weight", [32, 32]),
                (p + "self_attn.norm_added_q.weight", [8]),
                (p + "self_attn.norm_added_k.weight", [8]),
                (p + "mlp_moe_gen.gate_proj.weight", [32, 32]),
                (p + "mlp_moe_gen.up_proj.weight", [32, 32]),
                (p + "mlp_moe_gen.down_proj.weight", [32, 32]),
                (p + "input_layernorm_moe_gen.weight", [32]),
                (p + "post_attention_layernorm_moe_gen.weight", [32]),
            ]
        }
        out += [
            ("norm_moe_gen.weight", [32]),
            ("proj_in.weight", [32, 8]), ("proj_in.bias", [32]),
            ("proj_out.weight", [8, 32]), ("proj_out.bias", [8]),
            ("time_embedder.linear_1.weight", [32, 256]), ("time_embedder.linear_1.bias", [32]),
            ("time_embedder.linear_2.weight", [32, 32]), ("time_embedder.linear_2.bias", [32]),
            ("audio_modality_embed", [32]),
            ("audio_proj_in.weight", [32, 4]), ("audio_proj_in.bias", [32]),
            ("audio_proj_out.weight", [4, 32]), ("audio_proj_out.bias", [4]),
            ("action_modality_embed", [32]),
            ("action_proj_in.fc.weight", [3, 4 * 32]), ("action_proj_in.bias.weight", [3, 32]),
            ("action_proj_out.fc.weight", [3, 32 * 4]), ("action_proj_out.bias.weight", [3, 4]),
        ]
        return out
    }

    private func makeModel() throws -> SharpAICosmos3OmniModel {
        guard let creator = SharpAICosmos3OmniOverlay.creators["cosmos3_omni"],
            let model = try creator(Data(configJSON.utf8)) as? SharpAICosmos3OmniModel
        else {
            XCTFail("cosmos3_omni creator did not return the wrapper model")
            throw SharpAICosmos3OmniError.forwardNotImplemented("cosmos3_omni")
        }
        return model
    }

    private func checkpointArrays() -> [String: MLXArray] {
        var arrays = [String: MLXArray]()
        for (name, shape) in understandingEntries + generationEntries() {
            arrays[name] = SharpAIAirgap.fill(name: name, shape: shape, dtype: .float32)
        }
        return arrays
    }

    func testSanitizedKeysEqualModelParameters() throws {
        let model = try makeModel()
        let checkpoint = checkpointArrays()
        let sanitized = try model.sanitize(weights: checkpoint)
        let parameters = Dictionary(uniqueKeysWithValues: model.parameters().flattened())

        XCTAssertEqual(sanitized.count, checkpoint.count, "no key may be dropped")
        XCTAssertEqual(Set(sanitized.keys).subtracting(parameters.keys).sorted(), [])
        XCTAssertEqual(Set(parameters.keys).subtracting(sanitized.keys).sorted(), [])
        for (key, value) in sanitized {
            XCTAssertEqual(value.shape, parameters[key]?.shape, key)
        }
        XCTAssertTrue(sanitized.keys.contains("und.language_model.model.layers.0.self_attn.q_proj.weight"))
        XCTAssertTrue(sanitized.keys.contains("gen.layers.1.self_attn.k_norm.weight"))
        XCTAssertTrue(sanitized.keys.contains("gen_heads.action_proj_in.bias.weight"))
    }

    func testLoadsCheckpointAndRunsBothForwards() async throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("sharpai-cosmos3-omni-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        try Data(configJSON.utf8).write(to: dir.appendingPathComponent("config.json"))
        try save(arrays: checkpointArrays(), url: dir.appendingPathComponent("model.safetensors"))

        let model = try makeModel()
        try await loadWeights(modelDirectory: dir, model: model)
        eval(model)

        // Understanding forward.
        let tokens = MLXArray([1, 2, 3, 4, 5] as [Int32]).reshaped(1, 5)
        let logits = model(LMInput.Text(tokens: tokens), cache: nil, state: nil).logits
        XCTAssertEqual(logits.shape, [1, 5, 64])
        XCTAssertTrue(all(isFinite(logits)).item(Bool.self))
        XCTAssertGreaterThan(abs(logits).max().item(Float.self), 0)

        // Generation forward: 4 latent patches of patch_latent_dim 8.
        func velocity(_ scale: Float) -> SharpAICosmos3OmniVelocityOutput {
            let ramp = MLXArray(0 ..< 32).asType(.float32).reshaped(1, 4, 8)
            let patches = sin(ramp * MLXArray(Float(0.3))) * MLXArray(scale)
            let positions = broadcast(
                MLXArray(0 ..< 4).asType(.int32).reshaped(1, 1, 4), to: [3, 1, 4])
            return model.generateVelocity(
                inputIds: tokens, latentPatches: patches, timesteps: MLXArray([500] as [Float]),
                generationPositions: positions)
        }
        let first = velocity(1)
        let second = velocity(3)
        eval(first.velocity, first.understandingHidden, second.velocity, second.understandingHidden)

        XCTAssertEqual(first.velocity.shape, [1, 4, 8])
        XCTAssertEqual(first.generationHidden.shape, [1, 4, 32])
        XCTAssertEqual(first.understandingHidden.shape, [1, 5, 32])
        XCTAssertTrue(all(isFinite(first.velocity)).item(Bool.self))
        XCTAssertGreaterThan(abs(first.velocity).max().item(Float.self), 0)

        // The understanding side does not see generation inputs; the generation side does.
        XCTAssertTrue(
            allClose(first.understandingHidden, second.understandingHidden, rtol: 1e-5, atol: 1e-6)
                .item(Bool.self))
        XCTAssertFalse(
            allClose(first.velocity, second.velocity, rtol: 1e-4, atol: 1e-5).item(Bool.self))

        // The plain understanding forward is unchanged by generation calls.
        let again = model(LMInput.Text(tokens: tokens), cache: nil, state: nil).logits
        XCTAssertTrue(allClose(logits, again, rtol: 1e-5, atol: 1e-6).item(Bool.self))
    }

    func testUnknownCheckpointKeyIsRejected() throws {
        let model = try makeModel()
        var checkpoint = checkpointArrays()
        checkpoint["layers.0.self_attn.q_proj_moe_gen.weight"] = MLXArray.zeros([32, 32])
        XCTAssertThrowsError(try model.sanitize(weights: checkpoint))
    }
}
