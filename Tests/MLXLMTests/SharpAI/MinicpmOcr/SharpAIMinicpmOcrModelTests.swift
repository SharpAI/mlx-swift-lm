// SharpAI fork overlay tests for the minicpm_ocr model class. Not part of upstream.
// Random weights, tiny shapes, no network. Shapes and key mapping only; values are not checked
// against the reference, because no checkpoint is on disk.

import Foundation
import MLX
import MLXLMCommon
@testable import MLXVLM
import XCTest

final class SharpAIMinicpmOcrModelTests: XCTestCase {

    /// Tiny config in the layout of openbmb/MiniCPM-OCR-1.0 config.json.
    static func tinyJSON(downsampleMode: String = "4x") -> String {
        """
        {
            "model_type": "minicpm_ocr",
            "insert_layer_id": 1,
            "downsample_mode": "\(downsampleMode)",
            "tie_word_embeddings": true,
            "image_token_id": 100,
            "video_token_id": 101,
            "text_config": {
                "model_type": "qwen3_5_text",
                "hidden_size": 32,
                "num_hidden_layers": 4,
                "intermediate_size": 64,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 16,
                "linear_num_key_heads": 2,
                "linear_num_value_heads": 2,
                "linear_key_head_dim": 16,
                "linear_value_head_dim": 16,
                "linear_conv_kernel_dim": 4,
                "vocab_size": 128,
                "tie_word_embeddings": true,
                "full_attention_interval": 4,
                "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"]
            },
            "vision_config": {
                "hidden_size": 16,
                "intermediate_size": 32,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "num_channels": 3,
                "image_size": 56,
                "patch_size": 14,
                "layer_norm_eps": 1e-6
            }
        }
        """
    }

    private func makeModel(downsampleMode: String = "4x") throws -> SharpAIMinicpmOcr {
        let config = try JSONDecoder.json5().decode(
            SharpAIMinicpmOcrConfiguration.self,
            from: Data(Self.tinyJSON(downsampleMode: downsampleMode).utf8))
        return try SharpAIMinicpmOcr(config)
    }

    func testRegistrationKeysMinicpmOcr() {
        XCTAssertNotNil(SharpAIMinicpmOcrOverlay.creators["minicpm_ocr"])
    }

    func testCreatorBuildsModelFromCompleteConfig() throws {
        let creator = try XCTUnwrap(SharpAIMinicpmOcrOverlay.creators["minicpm_ocr"])
        let model = try creator(Data(Self.tinyJSON().utf8))

        XCTAssertTrue(model is SharpAIMinicpmOcr)
    }

    func testCreatorRejectsMismatchedModelType() throws {
        let creator = try XCTUnwrap(SharpAIMinicpmOcrOverlay.creators["minicpm_ocr"])
        let json = #"{"model_type": "minicpm"}"#

        XCTAssertThrowsError(try creator(Data(json.utf8)))
    }

    func testForwardPassWithRandomWeights() throws {
        let model = try makeModel()

        // 56px / patch 14 = 4x4 patches. The 4x path has no window merger, and the MLP merge
        // gives 2x2 visual tokens of the text width (32).
        let features = try model.encodeSlices(MLXRandom.normal([1, 56, 56, 3]))
        XCTAssertEqual(features.shape, [4, 32])

        // Text path through the Qwen35 backbone, vocabulary logits per position.
        let tokens = MLXArray([Int32(1), Int32(2), Int32(3)]).reshaped(1, 3)
        let output = model.callAsFunction(
            LMInput.Text(tokens: tokens), cache: nil, state: nil)
        XCTAssertEqual(output.logits.shape, [1, 3, 128])
    }

    func testWindowMergerPathProducesOneTokenPerSlice() throws {
        let model = try makeModel(downsampleMode: "16x")

        // 56px / 14 = 4x4 patches. The window merger gives 2x2, the MLP merge gives 1x1.
        let features = try model.encodeSlices(MLXRandom.normal([2, 56, 56, 3]))
        XCTAssertEqual(features.shape, [2, 32])
    }

    func testRejectsGridThatDoesNotDivideByMergeFactor() throws {
        let model = try makeModel()

        // 42px / 14 = 3 patches per side, which is not divisible by 2.
        XCTAssertThrowsError(try model.encodeSlice(MLXRandom.normal([42, 42, 3])))
    }

    func testRemapKeyMatchesReferenceCheckpointLayout() {
        XCTAssertEqual(
            SharpAIMinicpmOcr.remapKey("model.vision_tower.embeddings.patch_embedding.weight"),
            "vpm.embeddings.patch_embedding.weight")
        XCTAssertEqual(
            SharpAIMinicpmOcr.remapKey("model.vision_tower.vit_merger.pre_norm.weight"),
            "vit_merger.pre_norm.weight")
        XCTAssertEqual(
            SharpAIMinicpmOcr.remapKey("model.merger.mlp.0.linear_1.weight"),
            "merger.mlp.0.linear_1.weight")
        XCTAssertEqual(
            SharpAIMinicpmOcr.remapKey("model.language_model.layers.3.mlp.gate_proj.weight"),
            "language_model.model.layers.3.mlp.gate_proj.weight")
        XCTAssertEqual(
            SharpAIMinicpmOcr.remapKey("lm_head.weight"),
            "language_model.lm_head.weight")
        XCTAssertNil(SharpAIMinicpmOcr.remapKey("model.vision_tower.embeddings.position_ids"))
    }

    func testSanitizeDropsTiedLMHeadAndMovesPatchEmbedding() throws {
        let model = try makeModel()

        // HF patch embedding is (out, in, kH, kW). MLX wants (out, kH, kW, in).
        let sanitized = model.sanitize(weights: [
            "model.vision_tower.embeddings.patch_embedding.weight": MLXArray.zeros([16, 3, 14, 14]),
            "model.language_model.embed_tokens.weight": MLXArray.zeros([128, 32]),
            "lm_head.weight": MLXArray.zeros([128, 32]),
        ])

        XCTAssertEqual(
            sanitized["vpm.embeddings.patch_embedding.weight"]?.shape, [16, 14, 14, 3])
        XCTAssertNotNil(sanitized["language_model.model.embed_tokens.weight"])
        XCTAssertNil(sanitized["language_model.lm_head.weight"])
    }
}
