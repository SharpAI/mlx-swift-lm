import Foundation
import MLX
import MLXLMCommon
import MLXNN
import MLXVLM
import Testing

/// The second zero-coverage VLM architecture, after Gemma4 (#49).
///
/// Qwen3VL.swift is 1770 lines and was, at the time this was written, the largest
/// untested file in the model layer — Gemma4's slot after #49. Unlike Gemma4, its
/// text and vision configurations decode as independent nested structs
/// (`Qwen3VLConfiguration.TextConfiguration` / `.VisionConfiguration`) rather than
/// one flat shape, so a config mistake here looks different: a field misplaced at
/// the wrong nesting level rather than a missing top-level key.
///
/// These construct from tiny configurations and run a forward pass. They establish
/// shape and finiteness, not numerical correctness — the weights are random.
extension MLXTestingSuite {
    @Suite
    struct VLMQwen3VLTests {

        /// Vision-tower sizing has two constraints a hand-written config can violate
        /// silently: `numPositionEmbeddings` must be a perfect square (the model derives
        /// a grid side via sqrt), and `hiddenSize` must divide evenly by `numHeads`.
        private func tinyConfigData() -> Data {
            let json = """
            {
                "model_type": "qwen3_vl",
                "image_token_id": 3,
                "video_token_id": 7,
                "vision_start_token_id": 8,
                "vision_end_token_id": 9,
                "vision_token_id": 10,
                "text_config": {
                    "model_type": "qwen3",
                    "hidden_size": 64,
                    "intermediate_size": 128,
                    "num_hidden_layers": 2,
                    "num_attention_heads": 4,
                    "num_key_value_heads": 2,
                    "head_dim": 16,
                    "rope_theta": 1000000.0,
                    "max_position_embeddings": 512,
                    "rms_norm_eps": 1e-6,
                    "tie_word_embeddings": true,
                    "vocab_size": 128
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

        @Test("Qwen3VL configuration decodes text and vision as independent structs")
        func testConfigurationDecoding() throws {
            let config = try JSONDecoder().decode(
                Qwen3VLConfiguration.self, from: tinyConfigData())

            #expect(config.textConfiguration.hiddenSize == 64)
            #expect(config.visionConfiguration.hiddenSize == 32)
            // Derived from text_config.vocab_size when no top-level override is present.
            #expect(config.vocabSize == 128)
            #expect(config.imageTokenId == 3)
        }

        @Test("Qwen3VL instantiates from a tiny configuration")
        func testInstantiation() throws {
            let config = try JSONDecoder().decode(
                Qwen3VLConfiguration.self, from: tinyConfigData())
            let model = Qwen3VL(config)
            #expect(model.vocabularySize == 128)
        }

        @Test("Qwen3VL text-only forward pass has the right shape")
        func testTextOnlyForwardPass() throws {
            let config = try JSONDecoder().decode(
                Qwen3VLConfiguration.self, from: tinyConfigData())
            let model = Qwen3VL(config)

            // No image: the vision tower is built at init but must not be required on
            // the plain-text path, which is what every attachment-free chat request
            // takes.
            let input = MLXArray(0 ..< 6).reshaped(1, 6)
            let result = model(input, cache: nil)

            #expect(result.shape == [1, 6, 128])
            let sum = result.sum().item(Float.self)
            #expect(!sum.isNaN)
            #expect(!sum.isInfinite)
        }

        @Test("Qwen3VL forward pass is deterministic")
        func testDeterminism() throws {
            let config = try JSONDecoder().decode(
                Qwen3VLConfiguration.self, from: tinyConfigData())
            let model = Qwen3VL(config)

            let input = MLXArray(0 ..< 5).reshaped(1, 5)
            let a = model(input, cache: nil)
            let b = model(input, cache: nil)
            #expect(allClose(a, b).item(Bool.self))
        }
    }
}
