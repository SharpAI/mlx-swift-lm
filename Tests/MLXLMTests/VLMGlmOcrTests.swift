import Foundation
import MLX
import MLXLMCommon
import MLXNN
import MLXVLM
import Testing

/// GlmOcr.swift (938 lines) was the largest zero-coverage file remaining in the model
/// layer after Gemma4/Qwen3VL/Qwen35/LFM2VL (#147-#149).
///
/// Its config shape is a third distinct pattern from the others: `BaseConfiguration`
/// (`model_type`, `image_token_id`, `vocab_size`, ...) is decoded from the **same
/// top-level container** as `text_config`/`vision_config`, i.e. those fields sit
/// alongside the nested configs rather than inside either one —
/// `GlmOcrConfiguration.init` decodes `TextConfiguration` and `VisionConfiguration`
/// from their keyed children, then re-runs `BaseConfiguration(from: decoder)` against
/// the *same* decoder to pick up the flat fields.
///
/// That split has a real trap worth naming rather than working around: `GlmOcr`'s
/// reported `vocabularySize` reads `config.baseConfiguration.vocabularySize` (a
/// top-level field, defaulting to 59392), while the language model's actual output
/// width is `text_config.vocab_size`. A checkpoint that sets one and not the other has
/// a model whose declared vocabulary size does not match its logits' last dimension.
/// The tiny config below sets both to the same value on purpose — this establishes the
/// ordinary path, not the mismatch.
extension MLXTestingSuite {
    @Suite
    struct VLMGlmOcrTests {

        private func tinyConfigData() -> Data {
            let json = """
            {
                "model_type": "glm_ocr",
                "vocab_size": 128,
                "image_token_id": 6,
                "video_token_id": 7,
                "image_start_token_id": 8,
                "text_config": {
                    "hidden_size": 64,
                    "num_hidden_layers": 2,
                    "intermediate_size": 128,
                    "num_attention_heads": 4,
                    "num_key_value_heads": 2,
                    "head_dim": 16,
                    "vocab_size": 128,
                    "rope_parameters": {
                        "mrope_section": [3, 3, 2],
                        "rope_theta": 10000.0,
                        "partial_rotary_factor": 1.0
                    },
                    "rms_norm_eps": 1e-5,
                    "tie_word_embeddings": true
                },
                "vision_config": {
                    "depth": 1,
                    "hidden_size": 32,
                    "intermediate_size": 64,
                    "num_heads": 4,
                    "patch_size": 16,
                    "out_hidden_size": 32,
                    "spatial_merge_size": 2,
                    "temporal_patch_size": 1,
                    "in_channels": 3,
                    "rms_norm_eps": 1e-5
                }
            }
            """
            return json.data(using: .utf8)!
        }

        @Test("GlmOcr configuration overlays base fields onto the same top-level decoder")
        func testConfigurationDecoding() throws {
            let config = try JSONDecoder().decode(
                GlmOcrConfiguration.self, from: tinyConfigData())

            #expect(config.textConfiguration.hiddenSize == 64)
            #expect(config.visionConfiguration.hiddenSize == 32)
            // baseConfiguration is decoded from the same top-level container as
            // text_config/vision_config, not nested inside either.
            #expect(config.baseConfiguration.vocabularySize == 128)
            #expect(config.baseConfiguration.imageTokenId == 6)
        }

        @Test("GlmOcr instantiates from a tiny configuration")
        func testInstantiation() throws {
            let config = try JSONDecoder().decode(
                GlmOcrConfiguration.self, from: tinyConfigData())
            let model = GlmOcr(config)
            #expect(model.vocabularySize == 128)
        }

        @Test("GlmOcr text-only forward pass has the right shape")
        func testTextOnlyForwardPass() throws {
            let config = try JSONDecoder().decode(
                GlmOcrConfiguration.self, from: tinyConfigData())
            let model = GlmOcr(config)

            let input = MLXArray(0 ..< 6).reshaped(1, 6)
            let result = model(input, cache: nil)

            // The output width is text_config.vocab_size, which is why the tiny config
            // sets it to match the top-level vocab_size used for model.vocabularySize
            // above — see the type doc for what happens when a checkpoint does not.
            #expect(result.shape == [1, 6, 128])
            let sum = result.sum().item(Float.self)
            #expect(!sum.isNaN)
            #expect(!sum.isInfinite)
        }

        @Test("GlmOcr forward pass is deterministic")
        func testDeterminism() throws {
            let config = try JSONDecoder().decode(
                GlmOcrConfiguration.self, from: tinyConfigData())
            let model = GlmOcr(config)

            let input = MLXArray(0 ..< 5).reshaped(1, 5)
            let a = model(input, cache: nil)
            let b = model(input, cache: nil)
            #expect(allClose(a, b).item(Bool.self))
        }
    }
}
