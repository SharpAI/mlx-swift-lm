// SharpAI fork overlay tests for the cosmos3_omni creator. Not part of upstream.

import Foundation
import MLX
import MLXLMCommon
import MLXVLM
import XCTest

final class SharpAICosmos3OmniRegistrationTests: XCTestCase {

    private let validJSON = """
        {
            "architectures": ["Cosmos3ForConditionalGeneration"],
            "model_type": "cosmos3_omni",
            "text_config": {
                "model_type": "qwen3_vl_text",
                "hidden_size": 5120,
                "intermediate_size": 25600,
                "num_hidden_layers": 64,
                "num_attention_heads": 64,
                "num_key_value_heads": 8,
                "head_dim": 128,
                "max_position_embeddings": 262144,
                "vocab_size": 151936
            },
            "vision_config": {
                "model_type": "qwen3_vl",
                "depth": 27,
                "hidden_size": 1152,
                "intermediate_size": 4304,
                "out_hidden_size": 5120,
                "num_heads": 16,
                "patch_size": 16,
                "spatial_merge_size": 2,
                "temporal_patch_size": 2,
                "num_position_embeddings": 2304
            }
        }
        """

    func testCreatorIsRegisteredForModelType() {
        XCTAssertNotNil(SharpAICosmos3OmniOverlay.creators["cosmos3_omni"])
    }

    /// The 8B-class fixture builds the wrapper model (lazy weights, nothing is evaluated).
    func testValidConfigBuildsUnderstandingModel() throws {
        guard let creator = SharpAICosmos3OmniOverlay.creators["cosmos3_omni"] else {
            XCTFail("cosmos3_omni creator missing")
            return
        }

        let model = try creator(Data(validJSON.utf8))
        guard let qwen = model as? SharpAICosmos3OmniModel else {
            XCTFail("expected SharpAICosmos3OmniModel, got \(type(of: model))")
            return
        }
        XCTAssertEqual(qwen.vocabularySize, 151_936)
    }

    func testWrongModelTypeFailsBeforeConstruction() throws {
        guard let creator = SharpAICosmos3OmniOverlay.creators["cosmos3_omni"] else {
            XCTFail("cosmos3_omni creator missing")
            return
        }

        XCTAssertThrowsError(try creator(Data(#"{ "model_type": "llama" }"#.utf8))) { error in
            if case SharpAICosmos3OmniError.forwardNotImplemented = error {
                XCTFail("config validation should fail first")
            }
        }
    }


    /// Real config.json of hf-tiny-v2/tiny-random-Cosmos3OmniForConditionalGeneration (transformers
    /// 5.16.0.dev0, float32, 2 layers, hidden 32).
    private let tinyJSON = """
        {
          "architectures": ["Cosmos3OmniForConditionalGeneration"],
          "dtype": "float32",
          "image_token_id": 3,
          "model_type": "cosmos3_omni",
          "pad_token_id": 151643,
          "text_config": {
            "attention_bias": false,
            "attention_dropout": 0.0,
            "bos_token_id": 1,
            "eos_token_id": 151645,
            "head_dim": 8,
            "hidden_act": "silu",
            "hidden_size": 32,
            "initializer_range": 0.02,
            "intermediate_size": 32,
            "max_position_embeddings": 512,
            "model_type": "qwen3_vl_text",
            "num_attention_heads": 4,
            "num_hidden_layers": 2,
            "num_key_value_heads": 2,
            "pad_token_id": 151643,
            "rms_norm_eps": 1e-06,
            "rope_parameters": {
              "mrope_interleaved": true,
              "mrope_section": [16, 8, 8],
              "rope_theta": 10000,
              "rope_type": "default"
            },
            "use_cache": true,
            "vocab_size": 151669
          },
          "tie_word_embeddings": false,
          "transformers_version": "5.16.0.dev0",
          "video_token_id": 4,
          "vision_config": {
            "deepstack_visual_indexes": [0, 1],
            "depth": 2,
            "hidden_act": "silu",
            "hidden_size": 32,
            "in_channels": 3,
            "initializer_range": 0.02,
            "intermediate_size": 32,
            "model_type": "qwen3_vl_vision",
            "num_heads": 4,
            "num_position_embeddings": 16,
            "out_hidden_size": 32,
            "patch_size": 16,
            "spatial_merge_size": 1,
            "temporal_patch_size": 2
          },
          "vision_end_token_id": 6,
          "vision_start_token_id": 5
        }
        """

    /// Tensor names and shapes of the tiny checkpoint `model.safetensors` (safetensors header read
    /// from https://huggingface.co/hf-tiny-v2/tiny-random-Cosmos3OmniForConditionalGeneration/resolve/main/model.safetensors).
    /// Shapes are as stored (patch embed is channels-first [out, in, T, H, W]).
    private let tinyCheckpointEntries: [(String, [Int])] = [
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
            ("embed_tokens.weight", [151669, 32]),
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
            ("lm_head.weight", [151669, 32]),
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

    /// Tiny forward pass through the understanding model built from the reference config.
    func testTinyForwardPass() throws {
        guard let creator = SharpAICosmos3OmniOverlay.creators["cosmos3_omni"] else {
            XCTFail("cosmos3_omni creator missing")
            return
        }

        let model = try creator(Data(tinyJSON.utf8))
        guard let qwen = model as? SharpAICosmos3OmniModel else {
            XCTFail("expected SharpAICosmos3OmniModel, got \(type(of: model))")
            return
        }

        let tokens = MLXArray([1, 2, 3, 4] as [Int32]).reshaped(1, 4)
        let logits = qwen(LMInput.Text(tokens: tokens), cache: nil, state: nil).logits

        XCTAssertEqual(logits.shape, [1, 4, 151_669])
        let total = logits.sum().item(Float.self)
        XCTAssertFalse(total.isNaN)
        XCTAssertFalse(total.isInfinite)
    }

    /// The checkpoint names, after the sanitize and attention renames, are exactly the parameters
    /// of the `Qwen3VL` built from the same config, with equal shapes.
    func testTinyCheckpointLayoutMatchesQwen3VLParameters() throws {
        var weights = [String: MLXArray]()
        for (key, shape) in tinyCheckpointEntries {
            weights[key] = MLXArray.zeros(shape)
        }

        let sanitized = try SharpAICosmos3OmniSanitizer.sanitize(weights)
        let mapped = try SharpAICosmos3OmniSanitizer.qwen3VLAttentionNames(sanitized)

        let config = try SharpAICosmos3OmniConfiguration.qwen3VLConfiguration(
            from: Data(tinyJSON.utf8))
        let parameters = Dictionary(
            uniqueKeysWithValues: Qwen3VL(config).parameters().flattened())

        XCTAssertEqual(mapped.count, tinyCheckpointEntries.count)
        XCTAssertEqual(Set(mapped.keys), Set(parameters.keys))
        for (key, value) in mapped {
            XCTAssertEqual(value.shape, parameters[key]?.shape, key)
        }
    }

    /// The tiny reference config fails the generation tower checks (mrope_section) but the heads must
    /// still get the right sizes: hidden 32, patch_latent_dim 192, and no audio / action heads.
    func testTinyReferenceConfigBuildsHeadsWithCorrectShapes() throws {
        guard let creator = SharpAICosmos3OmniOverlay.creators["cosmos3_omni"] else {
            XCTFail("cosmos3_omni creator missing")
            return
        }
        let model = try creator(Data(tinyJSON.utf8))
        guard let cosmos = model as? SharpAICosmos3OmniModel else {
            XCTFail("expected SharpAICosmos3OmniModel")
            return
        }
        let shapes = Dictionary(
            uniqueKeysWithValues: cosmos.parameters().flattened().map { ($0.0, $0.1.shape) })
        XCTAssertEqual(shapes["gen_heads.proj_in.weight"], [32, 192])
        XCTAssertEqual(shapes["gen_heads.proj_out.weight"], [192, 32])
        XCTAssertEqual(shapes["gen_heads.time_embedder.linear_1.weight"], [32, 256])
        XCTAssertFalse(shapes.keys.contains { $0.hasPrefix("gen_heads.audio_") })
        XCTAssertFalse(shapes.keys.contains { $0.hasPrefix("gen_heads.action_") })
    }

    /// `sound_gen` / `action_gen` in the config create exactly the matching head parameters.
    func testGenerationFlagsSelectHeadParameters() throws {
        guard let creator = SharpAICosmos3OmniOverlay.creators["cosmos3_omni"] else {
            XCTFail("cosmos3_omni creator missing")
            return
        }
        func headKeys(_ extra: String) throws -> Set<String> {
            var json = tinyJSON
            let marker = "\"model_type\": \"cosmos3_omni\","
            XCTAssertTrue(json.contains(marker))
            // Flags live in `model.config` of the root config, as in nvidia/Cosmos3-Nano.
            json = json.replacingOccurrences(
                of: marker, with: marker + #" "model": {"config": {"# + extra + #"}},"#)
            let model = try creator(Data(json.utf8))
            guard let cosmos = model as? SharpAICosmos3OmniModel else {
                XCTFail("expected SharpAICosmos3OmniModel")
                return []
            }
            return Set(cosmos.parameters().flattened().map { $0.0 }.filter { $0.hasPrefix("gen_heads.") })
        }
        XCTAssertEqual(try headKeys(#""latent_channel": 48"#).count, 8)
        let all = try headKeys(
            #""sound_gen": true, "sound_dim": 4, "action_gen": true, "max_action_dim": 5"#)
        XCTAssertEqual(all.count, 18)
        let soundOnly = try headKeys(#""sound_gen": true, "sound_dim": 4"#)
        XCTAssertEqual(soundOnly.count, 13)
        XCTAssertFalse(soundOnly.contains { $0.hasPrefix("gen_heads.action_") })
        // An enabled head with no size fails at creation.
        XCTAssertThrowsError(try headKeys(#""sound_gen": true"#))
    }
}
