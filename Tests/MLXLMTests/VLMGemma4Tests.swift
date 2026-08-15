import Foundation
import MLX
import MLXLMCommon
import MLXNN
import MLXVLM
import Testing

/// The first unit tests for anything under MLXVLM.
///
/// Measured coverage put the whole VLM half of the model layer at 0.3% over 13,304
/// lines, with `MLXVLM/Models/Gemma4.swift` — 1,771 lines — the largest file no test
/// executed. That file is where SharpAI/mlx-swift-lm#45's changes landed, which is
/// what issue #128 flagged and what this starts to close.
///
/// Vision models are exercised end to end by the `vision` and `omni` CI jobs, so the
/// gap is not that they never run. It is that a shape or dtype fault inside one
/// surfaces as a wrong answer about an image rather than a failing assertion, which
/// is a slow and ambiguous way to find out.
///
/// These construct from tiny configurations and run a forward pass. They establish
/// shape and finiteness, not numerical correctness — the weights are random.
extension MLXTestingSuite {
    @Suite
    struct VLMGemma4Tests {

        /// Mirrors the shape of a real gemma-4 VLM config: a text half and a vision
        /// half, each with its own model_type, under one wrapper.
        private func tinyConfigData() -> Data {
            let json = """
            {
                "model_type": "gemma4",
                "image_token_id": 3,
                "audio_token_id": 4,
                "boi_token_id": 5,
                "eoi_token_id": 6,
                "vision_soft_tokens_per_image": 4,
                "text_config": {
                    "model_type": "gemma4_text",
                    "hidden_size": 64,
                    "num_hidden_layers": 2,
                    "intermediate_size": 128,
                    "num_attention_heads": 4,
                    "num_key_value_heads": 2,
                    "head_dim": 16,
                    "global_head_dim": 16,
                    "rms_norm_eps": 1e-6,
                    "vocab_size": 128,
                    "vocab_size_per_layer_input": 16,
                    "num_kv_shared_layers": 0,
                    "hidden_size_per_layer_input": 32,
                    "sliding_window": 128,
                    "sliding_window_pattern": 1,
                    "max_position_embeddings": 512,
                    "rope_traditional": false,
                    "rope_theta": 10000.0,
                    "use_double_wide_mlp": false,
                    "tie_word_embeddings": true,
                    "final_logit_softcapping": 30.0,
                    "enable_moe_block": false,
                    "attention_k_eq_v": false
                },
                "vision_config": {
                    "model_type": "gemma4_vision",
                    "num_hidden_layers": 2,
                    "hidden_size": 32,
                    "intermediate_size": 64,
                    "num_attention_heads": 4,
                    "num_key_value_heads": 4,
                    "head_dim": 8,
                    "patch_size": 16,
                    "rms_norm_eps": 1e-6,
                    "default_output_length": 4,
                    "position_embedding_size": 16,
                    "pooling_kernel_size": 2,
                    "use_clipped_linears": false,
                    "standardize": false,
                    "rope_parameters": {"rope_theta": 10000.0}
                }
            }
            """
            return json.data(using: .utf8)!
        }

        @Test("VLM Gemma4 configuration decodes both halves")
        func testConfigurationDecoding() throws {
            let config = try JSONDecoder().decode(
                Gemma4Configuration.self, from: tinyConfigData())

            // The wrapper derives these from the text half when absent, which is the
            // behaviour a checkpoint without top-level copies depends on.
            #expect(config.vocabularySize == 128)
            #expect(config.hiddenSize == 64)
            #expect(config.imageTokenId == 3)
        }

        @Test("VLM Gemma4 instantiates from a tiny configuration")
        func testInstantiation() throws {
            let config = try JSONDecoder().decode(
                Gemma4Configuration.self, from: tinyConfigData())
            let model = Gemma4(config)
            #expect(model.vocabularySize == 128)
        }

        @Test("VLM Gemma4 text-only forward pass has the right shape")
        func testTextOnlyForwardPass() throws {
            let config = try JSONDecoder().decode(
                Gemma4Configuration.self, from: tinyConfigData())
            let model = Gemma4(config)

            // No image: a VLM still has to answer a text-only prompt, and that path is
            // what every chat request without an attachment takes.
            let input = LMInput(tokens: MLXArray(0 ..< 6))
            let result = model(input.text.tokens[.newAxis], cache: nil)

            #expect(result.shape == [1, 6, 128])
            let sum = result.sum().item(Float.self)
            #expect(!sum.isNaN)
            #expect(!sum.isInfinite)
        }

        @Test("VLM Gemma4 forward pass is deterministic")
        func testDeterminism() throws {
            let config = try JSONDecoder().decode(
                Gemma4Configuration.self, from: tinyConfigData())
            let model = Gemma4(config)

            let input = MLXArray(0 ..< 5).reshaped(1, 5)
            let a = model(input, cache: nil)
            let b = model(input, cache: nil)
            #expect(allClose(a, b).item(Bool.self))
        }
    }
}
