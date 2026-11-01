// SharpAI fork overlay tests for vibevoice_asr key remapping and conv layout. Not part of upstream.
// No network and no real weights: tensors are zeros with HF (PyTorch) shapes.

import Foundation
import MLX
import MLXLMCommon
import MLXVLM
import XCTest

final class SharpAIVibevoiceAsrSanitizeTests: XCTestCase {

    static let tinyJSON = """
        {
          "model_type": "vibevoice_asr",
          "audio_token_id": 9,
          "acoustic_tokenizer_encoder_config": {
            "hidden_size": 4,
            "num_filters": 2,
            "kernel_size": 3,
            "depths": [1, 1, 1, 1, 1, 1, 1]
          },
          "semantic_tokenizer_encoder_config": {
            "hidden_size": 3,
            "num_filters": 2,
            "kernel_size": 3,
            "depths": [1, 1, 1, 1, 1, 1, 1]
          },
          "text_config": {
            "hidden_size": 8,
            "num_hidden_layers": 1,
            "intermediate_size": 16,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "vocab_size": 16,
            "max_position_embeddings": 64,
            "rope_theta": 10000.0,
            "tie_word_embeddings": false
          }
        }
        """

    private func tinyConfiguration(tied: Bool = false) throws -> SharpAIVibevoiceAsrConfiguration {
        var config = try JSONDecoder.json5().decode(
            SharpAIVibevoiceAsrConfiguration.self, from: Data(Self.tinyJSON.utf8))
        config.text.tieWordEmbeddings = tied
        return config
    }

    func testModelPrefixIsStrippedForTopLevelModules() throws {
        let config = try tinyConfiguration()
        let weights: [String: MLXArray] = [
            "model.multi_modal_projector.acoustic_linear_1.weight": MLXArray.zeros([8, 4]),
            "language_model.model.embed_tokens.weight": MLXArray.zeros([16, 8]),
        ]
        let out = try SharpAIVibevoiceAsrSanitizer.sanitize(weights, configuration: config)

        XCTAssertNotNil(out["multi_modal_projector.acoustic_linear_1.weight"])
        XCTAssertNil(out["model.multi_modal_projector.acoustic_linear_1.weight"])
        XCTAssertNotNil(out["language_model.model.embed_tokens.weight"])
    }

    func testTokenizerConvWeightsAreTransposedToMLXLayout() throws {
        let config = try tinyConfiguration()
        let weights: [String: MLXArray] = [
            // PyTorch: (out, in / groups, kernel)
            "acoustic_tokenizer_encoder.stem.conv.conv.weight": MLXArray.zeros([2, 1, 3]),
            // Depthwise mixer: (channels, 1, kernel)
            "acoustic_tokenizer_encoder.stem.stage.0.mixer.conv.weight": MLXArray.zeros([2, 1, 3]),
            "semantic_tokenizer_encoder.head.conv.weight": MLXArray.zeros([3, 16, 3]),
        ]
        let out = try SharpAIVibevoiceAsrSanitizer.sanitize(weights, configuration: config)

        XCTAssertEqual(out["acoustic_tokenizer_encoder.stem.conv.conv.weight"]!.shape, [2, 3, 1])
        XCTAssertEqual(
            out["acoustic_tokenizer_encoder.stem.stage.0.mixer.conv.weight"]!.shape, [2, 3, 1])
        XCTAssertEqual(out["semantic_tokenizer_encoder.head.conv.weight"]!.shape, [3, 3, 16])
    }

    func testBiasAndLinearWeightsKeepTheirShape() throws {
        let config = try tinyConfiguration()
        let weights: [String: MLXArray] = [
            "acoustic_tokenizer_encoder.stem.conv.conv.bias": MLXArray.zeros([2]),
            "language_model.model.layers.0.mlp.gate_proj.weight": MLXArray.zeros([16, 8]),
        ]
        let out = try SharpAIVibevoiceAsrSanitizer.sanitize(weights, configuration: config)

        XCTAssertEqual(out["acoustic_tokenizer_encoder.stem.conv.conv.bias"]!.shape, [2])
        XCTAssertEqual(
            out["language_model.model.layers.0.mlp.gate_proj.weight"]!.shape, [16, 8])
    }

    func testRotaryBuffersAreDropped() throws {
        let config = try tinyConfiguration()
        let weights: [String: MLXArray] = [
            "language_model.model.layers.0.self_attn.rotary_emb.inv_freq": MLXArray.zeros([4])
        ]
        let out = try SharpAIVibevoiceAsrSanitizer.sanitize(weights, configuration: config)
        XCTAssertTrue(out.isEmpty)
    }

    func testOutputHeadIsDroppedWhenEmbeddingsAreTied() throws {
        let weights: [String: MLXArray] = [
            "language_model.lm_head.weight": MLXArray.zeros([16, 8])
        ]

        let untied = try SharpAIVibevoiceAsrSanitizer.sanitize(
            weights, configuration: try tinyConfiguration(tied: false))
        XCTAssertNotNil(untied["language_model.lm_head.weight"])

        let tied = try SharpAIVibevoiceAsrSanitizer.sanitize(
            weights, configuration: try tinyConfiguration(tied: true))
        XCTAssertNil(tied["language_model.lm_head.weight"])
    }
}
