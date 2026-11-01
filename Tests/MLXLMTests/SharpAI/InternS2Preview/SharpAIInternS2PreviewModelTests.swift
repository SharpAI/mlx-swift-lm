// SharpAI fork overlay tests for the intern_s2_preview model class. Not part of upstream.
//
// Tiny random-weight model, no downloads. The text dims follow the Qwen35 prefill tests, which run
// in CI, so the forward pass uses shapes already exercised upstream.

import Foundation
import MLX
import MLXLMCommon
import XCTest

@testable import MLXVLM

final class SharpAIInternS2PreviewModelTests: XCTestCase {

    private let tinyJSON = """
        {
            "model_type": "intern_s2_preview",
            "vocab_size": 512,
            "image_token_id": 500, "video_token_id": 501,
            "vision_start_token_id": 502, "vision_end_token_id": 503,
            "text_config": {
                "model_type": "qwen3_5_moe_text", "hidden_size": 64,
                "num_hidden_layers": 2, "intermediate_size": 128,
                "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 32,
                "vocab_size": 512, "full_attention_interval": 2,
                "tie_word_embeddings": false, "linear_num_value_heads": 2,
                "linear_num_key_heads": 1, "linear_key_head_dim": 32,
                "linear_value_head_dim": 32, "linear_conv_kernel_dim": 4,
                "num_experts": 0, "num_experts_per_tok": 0,
                "max_position_embeddings": 4096,
                "rope_parameters": {"mrope_section": [8,4,4], "rope_theta": 100000,
                    "partial_rotary_factor": 1.0, "type": "default"}
            },
            "vision_config": {"model_type": "qwen3_vl", "depth": 1,
                "hidden_size": 32, "intermediate_size": 64, "out_hidden_size": 64,
                "num_heads": 2, "patch_size": 16, "spatial_merge_size": 2,
                "temporal_patch_size": 2, "num_position_embeddings": 64}
        }
        """

    private func makeTinyModel() throws -> SharpAIInternS2PreviewModel {
        let configuration = try JSONDecoder.json5().decode(
            Qwen35Configuration.self, from: Data(tinyJSON.utf8))
        return SharpAIInternS2PreviewModel(configuration)
    }

    func testForwardPassReturnsVocabSizedLogits() throws {
        let model = try makeTinyModel()
        let tokens = MLXArray((0 ..< 6).map { ($0 * 13 + 7) % 480 }).reshaped(1, 6)

        let output = model.callAsFunction(
            LMInput.Text(tokens: tokens), cache: nil, state: nil)
        eval(output.logits)

        XCTAssertEqual(output.logits.dim(-1), 512)
        XCTAssertEqual(output.logits.dim(0), 1)
    }

    // Time-series keys must be dropped before Qwen35 remaps bare `model.*` keys.
    func testSanitizeDropsTimeSeriesAndKeepsLanguageKeys() throws {
        let model = try makeTinyModel()
        let dummy = MLXArray.zeros([1, 1])
        let weights: [String: MLXArray] = [
            "model.time_series.encoder.weight": dummy,
            "time_series.head.weight": dummy,
            "model.layers.0.mlp.up_proj.weight": dummy,
            "model.embed_tokens.weight": dummy,
        ]

        let sanitized = try model.sanitize(weights: weights)

        for key in sanitized.keys {
            XCTAssertFalse(key.contains("time_series"), "time_series key leaked: \(key)")
        }
        XCTAssertNotNil(sanitized["language_model.model.layers.0.mlp.up_proj.weight"])
        XCTAssertNotNil(sanitized["language_model.model.embed_tokens.weight"])
    }

    // The load path goes through prepareCheckpoint, which renames `model.*` before sanitize.
    func testPrepareCheckpointDropsTimeSeries() throws {
        let model = try makeTinyModel()
        let dummy = MLXArray.zeros([1, 1])
        let checkpoint = ModelCheckpoint(weights: [
            "model.time_series.encoder.weight": dummy,
            "time_series.head.weight": dummy,
            "model.embed_tokens.weight": dummy,
        ])

        let prepared = try model.prepareCheckpoint(checkpoint)

        for key in prepared.weights.keys {
            XCTAssertFalse(key.contains("time_series"), "time_series key leaked: \(key)")
        }
        XCTAssertNotNil(prepared.weights["language_model.model.embed_tokens.weight"])
    }

    func testMetadataSanitizeAlsoDropsTimeSeries() throws {
        let model = try makeTinyModel()
        let dummy = MLXArray.zeros([1, 1])
        let weights: [String: MLXArray] = [
            "model.time_series.encoder.weight": dummy,
            "model.embed_tokens.weight": dummy,
        ]

        let sanitized = try model.sanitize(weights: weights, metadata: [:])

        for key in sanitized.keys {
            XCTAssertFalse(key.contains("time_series"), "time_series key leaked: \(key)")
        }
    }
}
