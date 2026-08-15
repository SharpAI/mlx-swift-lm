import Foundation
import MLX
import MLXLMCommon
import MLXNN
import MLXVLM
import Testing

/// LFM2VL.swift, alongside Gemma4 (#49) and Qwen3VL (#50), was one of the largest
/// zero-coverage files in the model layer.
///
/// Its hybrid mechanism differs from Qwen35's fixed-period one: layers are attention
/// only at explicit indices, `full_attn_idxs` (or derived from `layer_types`), and a
/// short causal convolution (`LFM2ShortConv`) everywhere else — a sparser, more
/// arbitrary pattern than a modulus. Setting `full_attn_idxs: [1]` over 2 layers makes
/// layer 0 the conv block and layer 1 the attention block, so both branches of
/// `LFM2DecoderLayer` build and run, and `newCache` — which reads `fullAttnIdxs`
/// independently to build `MambaCache` versus `KVCacheSimple` per layer — is exercised
/// against the same config that built the model, not a default.
///
/// The vision half carries the same silent constraint seen in Qwen3VL's tower:
/// `num_patches` must be a perfect square, since a grid side is derived via `sqrt`.
///
/// These construct from a tiny configuration and run a forward pass. They establish
/// shape and finiteness, not numerical correctness — the weights are random.
extension MLXTestingSuite {
    @Suite
    struct VLMLFM2VLTests {

        private func tinyConfigData() -> Data {
            let json = """
            {
                "model_type": "lfm2_vl",
                "image_token_id": 5,
                "text_config": {
                    "model_type": "lfm2",
                    "hidden_size": 64,
                    "num_hidden_layers": 2,
                    "num_attention_heads": 4,
                    "num_key_value_heads": 2,
                    "vocab_size": 128,
                    "norm_eps": 1e-5,
                    "conv_L_cache": 3,
                    "full_attn_idxs": [1],
                    "rope_theta": 1000000.0
                },
                "vision_config": {
                    "model_type": "siglip2",
                    "hidden_size": 32,
                    "intermediate_size": 64,
                    "num_hidden_layers": 1,
                    "num_attention_heads": 4,
                    "num_channels": 3,
                    "image_size": 64,
                    "patch_size": 16,
                    "num_patches": 16,
                    "layer_norm_eps": 1e-6
                }
            }
            """
            return json.data(using: .utf8)!
        }

        @Test("LFM2VL configuration decodes explicit full-attention indices")
        func testConfigurationDecoding() throws {
            let config = try JSONDecoder().decode(
                LFM2VLConfiguration.self, from: tinyConfigData())

            #expect(config.textConfiguration.hiddenSize == 64)
            #expect(config.textConfiguration.fullAttnIdxs == [1])
            #expect(config.visionConfiguration.hiddenSize == 32)
            #expect(config.textConfiguration.vocabularySize == 128)
        }

        @Test("LFM2VL instantiates with both a conv layer and an attention layer")
        func testInstantiation() throws {
            let config = try JSONDecoder().decode(
                LFM2VLConfiguration.self, from: tinyConfigData())
            let model = LFM2VL(config)
            #expect(model.vocabularySize == 128)
        }

        @Test("LFM2VL text-only forward pass has the right shape")
        func testTextOnlyForwardPass() throws {
            let config = try JSONDecoder().decode(
                LFM2VLConfiguration.self, from: tinyConfigData())
            let model = LFM2VL(config)

            // newCache reads fullAttnIdxs itself, independently of how the layers were
            // built, so exercising it here checks the two stay in agreement rather than
            // assuming they do.
            let cache = model.newCache(parameters: nil)
            let input = MLXArray(0 ..< 6).reshaped(1, 6)
            let result = model(input, cache: cache)

            #expect(result.shape == [1, 6, 128])
            let sum = result.sum().item(Float.self)
            #expect(!sum.isNaN)
            #expect(!sum.isInfinite)
        }

        @Test("LFM2VL forward pass is deterministic")
        func testDeterminism() throws {
            let config = try JSONDecoder().decode(
                LFM2VLConfiguration.self, from: tinyConfigData())
            let model = LFM2VL(config)

            let input = MLXArray(0 ..< 5).reshaped(1, 5)
            let a = model(input, cache: model.newCache(parameters: nil))
            let b = model(input, cache: model.newCache(parameters: nil))
            #expect(allClose(a, b).item(Bool.self))
        }
    }
}
