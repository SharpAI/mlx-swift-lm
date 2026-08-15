import Foundation
import MLX
import MLXLMCommon
import MLXNN
import MLXVLM
import Testing

/// Qwen35.swift, alongside Qwen3VL.swift and LFM2VL.swift, was one of the largest
/// zero-coverage files in the model layer after #49/#147 covered Gemma4.
///
/// Unlike the other two, its language model is a **hybrid**: layers alternate between
/// full self-attention and a linear-attention block (`GatedDeltaNet`) on a fixed
/// period, `full_attention_interval`. `DecoderLayer.isLinear` is computed as
/// `(layerIdx + 1) % fullAttentionInterval != 0`, so with the interval set to the
/// layer count, exactly one layer is linear and the interval boundary itself is
/// exercised — a config that never turns the linear path on would leave the whole
/// `GatedDeltaNet` module dead, which is the actual risk in a two-branch layer like
/// this.
///
/// It also carries MoE fields (`num_experts`, `decoder_sparse_step`, ...), left at
/// their dense defaults here — the routed-MoE path is a separate shape, not this one.
///
/// These construct from a tiny configuration and run a forward pass through the real
/// per-layer cache (`newCache`, not `nil`), which is what selects `MambaCache` for the
/// linear layer and a standard KV cache for the attention layer. They establish shape
/// and finiteness, not numerical correctness — the weights are random.
extension MLXTestingSuite {
    @Suite
    struct VLMQwen35Tests {

        /// `full_attention_interval: 2` over 2 layers makes layer 0 linear
        /// ((0+1)%2=1≠0) and layer 1 full attention ((1+1)%2=0), so both branches of
        /// `DecoderLayer` build and run. Vision config reuses Qwen3VL's shape — Qwen35
        /// literally reuses `Qwen3VLConfiguration.VisionConfiguration` — including its
        /// two silent constraints: `num_position_embeddings` must be a perfect square,
        /// and `hidden_size` must divide evenly by `num_heads`.
        private func tinyConfigData() -> Data {
            let json = """
            {
                "model_type": "qwen3_5",
                "image_token_id": 3,
                "video_token_id": 7,
                "vision_start_token_id": 8,
                "vision_end_token_id": 9,
                "text_config": {
                    "model_type": "qwen3_5",
                    "hidden_size": 64,
                    "num_hidden_layers": 2,
                    "intermediate_size": 128,
                    "num_attention_heads": 4,
                    "num_key_value_heads": 2,
                    "full_attention_interval": 2,
                    "linear_num_value_heads": 4,
                    "linear_num_key_heads": 2,
                    "linear_key_head_dim": 8,
                    "linear_value_head_dim": 8,
                    "linear_conv_kernel_dim": 4,
                    "rms_norm_eps": 1e-6,
                    "vocab_size": 128,
                    "rope_theta": 100000.0,
                    "partial_rotary_factor": 0.25,
                    "max_position_embeddings": 512,
                    "tie_word_embeddings": true,
                    "num_experts": 0
                },
                "vision_config": {
                    "model_type": "qwen3_vl_vision",
                    "depth": 1,
                    "hidden_size": 32,
                    "intermediate_size": 64,
                    "out_hidden_size": 64,
                    "num_heads": 4,
                    "patch_size": 16,
                    "spatial_merge_size": 2,
                    "temporal_patch_size": 1,
                    "num_position_embeddings": 16,
                    "in_channels": 3
                }
            }
            """
            return json.data(using: .utf8)!
        }

        @Test("Qwen35 configuration decodes the hybrid text half and shared vision half")
        func testConfigurationDecoding() throws {
            let config = try JSONDecoder().decode(
                Qwen35Configuration.self, from: tinyConfigData())

            #expect(config.textConfiguration.hiddenSize == 64)
            #expect(config.textConfiguration.fullAttentionInterval == 2)
            #expect(config.visionConfiguration.hiddenSize == 32)
            #expect(config.vocabSize == 128)
        }

        @Test("Qwen35 instantiates with both a linear and a full-attention layer")
        func testInstantiation() throws {
            let config = try JSONDecoder().decode(
                Qwen35Configuration.self, from: tinyConfigData())
            let model = Qwen35(config)
            #expect(model.vocabularySize == 128)
        }

        @Test("Qwen35 text-only forward pass has the right shape")
        func testTextOnlyForwardPass() throws {
            let config = try JSONDecoder().decode(
                Qwen35Configuration.self, from: tinyConfigData())
            let model = Qwen35(config)

            // The real per-layer cache, not nil: this is what routes the linear layer
            // to a MambaCache and the attention layer to a standard one, which is the
            // model-specific behaviour this test exists to exercise.
            let cache = model.newCache(parameters: nil)
            let input = MLXArray(0 ..< 6).reshaped(1, 6)
            let result = model(input, cache: cache)

            #expect(result.shape == [1, 6, 128])
            let sum = result.sum().item(Float.self)
            #expect(!sum.isNaN)
            #expect(!sum.isInfinite)
        }

        @Test("Qwen35 forward pass is deterministic")
        func testDeterminism() throws {
            let config = try JSONDecoder().decode(
                Qwen35Configuration.self, from: tinyConfigData())
            let model = Qwen35(config)

            let input = MLXArray(0 ..< 5).reshaped(1, 5)
            let a = model(input, cache: model.newCache(parameters: nil))
            let b = model(input, cache: model.newCache(parameters: nil))
            #expect(allClose(a, b).item(Bool.self))
        }
    }
}
