import Foundation
import MLX
import MLXLMCommon
import MLXNN
import MLXVLM
import Testing

/// Qwen25VL.swift (844 lines) was the second-largest zero-coverage file remaining
/// after GlmOcr, in the same follow-on to #147-#149.
///
/// A fourth distinct config pattern: here it is the **text** configuration, not just
/// the base one, that overlays the top-level container — `Qwen25VLConfiguration.init`
/// decodes `VisionConfiguration` from the nested `vision_config` key, then decodes
/// both `TextConfiguration` and `BaseConfiguration` straight from the same top-level
/// decoder. Several fields (`hidden_size`, `num_attention_heads`,
/// `num_hidden_layers`, ...) are declared in both structs under the same key, so one
/// top-level JSON field populates two parallel properties rather than one being
/// derived from the other, which is how Gemma4/GlmOcr handle the equivalent overlap.
///
/// The vision tower separately carries a windowed/full-attention split
/// (`fullatt_block_indexes`), but that only engages on the image-forward path, which
/// these tests do not exercise — same scope as the other VLM tests this session:
/// construction plus a text-only forward pass.
///
/// One field has no default and will crash construction if omitted:
/// `rope_scaling.mrope_section` (`Attention.init` `fatalError`s without it). Its
/// values are doubled and cumulatively summed to produce split points along the head
/// dimension, so `sum(mrope_section) * 2` must not exceed `head_dim` — a config with
/// too large a section list fails at a slice, not at decode time.
extension MLXTestingSuite {
    @Suite
    struct VLMQwen25VLTests {

        private func tinyConfigData() -> Data {
            let json = """
            {
                "model_type": "qwen2_5_vl",
                "vocab_size": 128,
                "hidden_size": 64,
                "num_hidden_layers": 2,
                "intermediate_size": 128,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "sliding_window": 32768,
                "use_sliding_window": false,
                "max_window_layers": 2,
                "image_token_id": 6,
                "video_token_id": 7,
                "vision_start_token_id": 8,
                "vision_end_token_id": 9,
                "vision_token_id": 10,
                "tie_word_embeddings": true,
                "rope_theta": 1000000.0,
                "rope_scaling": {
                    "mrope_section": [2, 1, 1],
                    "type": "mrope"
                },
                "vision_config": {
                    "depth": 1,
                    "hidden_size": 32,
                    "intermediate_size": 64,
                    "out_hidden_size": 32,
                    "num_heads": 4,
                    "patch_size": 16,
                    "spatial_patch_size": 16,
                    "spatial_merge_size": 2,
                    "temporal_patch_size": 1,
                    "window_size": 16,
                    "fullatt_block_indexes": [0],
                    "tokens_per_second": 2
                }
            }
            """
            return json.data(using: .utf8)!
        }

        @Test("Qwen25VL text and base configuration both overlay the top-level container")
        func testConfigurationDecoding() throws {
            let config = try JSONDecoder().decode(
                Qwen25VLConfiguration.self, from: tinyConfigData())

            #expect(config.textConfiguration.hiddenSize == 64)
            // The same top-level hidden_size populates baseConfiguration too — not
            // derived from textConfiguration, decoded independently from it.
            #expect(config.baseConfiguration.hiddenSize == 64)
            #expect(config.visionConfiguration.hiddenSize == 32)
            #expect(config.baseConfiguration.vocabularySize == 128)
        }

        @Test("Qwen25VL instantiates from a tiny configuration")
        func testInstantiation() throws {
            let config = try JSONDecoder().decode(
                Qwen25VLConfiguration.self, from: tinyConfigData())
            let model = Qwen25VL(config)
            #expect(model.vocabularySize == 128)
        }

        @Test("Qwen25VL text-only forward pass has the right shape")
        func testTextOnlyForwardPass() throws {
            let config = try JSONDecoder().decode(
                Qwen25VLConfiguration.self, from: tinyConfigData())
            let model = Qwen25VL(config)

            let input = MLXArray(0 ..< 6).reshaped(1, 6)
            let result = model(input, cache: nil)

            #expect(result.shape == [1, 6, 128])
            let sum = result.sum().item(Float.self)
            #expect(!sum.isNaN)
            #expect(!sum.isInfinite)
        }

        @Test("Qwen25VL forward pass is deterministic")
        func testDeterminism() throws {
            let config = try JSONDecoder().decode(
                Qwen25VLConfiguration.self, from: tinyConfigData())
            let model = Qwen25VL(config)

            let input = MLXArray(0 ..< 5).reshaped(1, 5)
            let a = model(input, cache: nil)
            let b = model(input, cache: nil)
            #expect(allClose(a, b).item(Bool.self))
        }
    }
}
