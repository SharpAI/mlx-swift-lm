// SharpAI fork overlay tests for moss_transcribe_diarize checkpoint key sanitization.

import Foundation
import MLX
@testable import MLXVLM
import XCTest

final class SharpAIMossTranscribeDiarizeSanitizeTests: XCTestCase {

    private func config() throws -> SharpAIMossTranscribeDiarizeConfiguration {
        try SharpAIMossTranscribeDiarizeTestConfig.decode(
            SharpAIMossTranscribeDiarizeTestConfig.tiny())
    }

    func testConvWeightsMoveToMLXLayout() throws {
        let weights: [String: MLXArray] = [
            // PyTorch [out, in, kernel]: conv1 in = 4 mel bins, conv2 in = 8 hidden.
            "audio_tower.conv1.weight": MLXArray.zeros([8, 4, 3]),
            "audio_tower.conv2.weight": MLXArray.zeros([8, 8, 3]),
        ]

        let result = SharpAIMossTranscribeDiarizeSanitize.sanitize(
            weights: weights, config: try config())

        XCTAssertEqual(result["audio_tower.conv1.weight"]?.shape, [8, 3, 4])
        XCTAssertEqual(result["audio_tower.conv2.weight"]?.shape, [8, 3, 8])
    }

    func testRemapsTransformersPrefixesAndDropsRotaryBuffers() throws {
        let weights: [String: MLXArray] = [
            "model.language_model.embed_tokens.weight": MLXArray.zeros([32, 8]),
            "model.language_model.layers.0.self_attn.rotary_emb.inv_freq": MLXArray.zeros([2]),
            "model.language_model.layers.0.self_attn.q_norm.weight": MLXArray.zeros([4]),
            "model.audio_tower.layer_norm.weight": MLXArray.zeros([8]),
            "model.multi_modal_projector.linear_1.weight": MLXArray.zeros([8, 16]),
            "lm_head.weight": MLXArray.zeros([32, 8]),
        ]

        let result = SharpAIMossTranscribeDiarizeSanitize.sanitize(
            weights: weights, config: try config())

        XCTAssertNotNil(result["language_model.model.embed_tokens.weight"])
        XCTAssertNotNil(result["language_model.model.layers.0.self_attn.q_norm.weight"])
        XCTAssertNotNil(result["audio_tower.layer_norm.weight"])
        XCTAssertNotNil(result["multi_modal_projector.linear_1.weight"])
        XCTAssertNil(result["model.language_model.embed_tokens.weight"])
        XCTAssertNil(result["lm_head.weight"])
        XCTAssertFalse(result.keys.contains { $0.contains("rotary_emb.inv_freq") })
        // The tiny config ties embeddings, so `lm_head` is dropped. Four keys remain.
        XCTAssertEqual(result.count, 4)
    }

    func testEncoderPrefixMapsIntoAudioTower() {
        XCTAssertEqual(
            SharpAIMossTranscribeDiarizeSanitize.checkpointName(
                "model.encoder.layers.0.self_attn.k_proj.weight"),
            "audio_tower.layers.0.self_attn.k_proj.weight")
    }

    func testBareDecoderPrefixMapsIntoLanguageModel() {
        XCTAssertEqual(
            SharpAIMossTranscribeDiarizeSanitize.checkpointName(
                "model.layers.3.mlp.up_proj.weight"),
            "language_model.model.layers.3.mlp.up_proj.weight")
    }

    func testVLMLayoutKeysPassThroughUnchanged() throws {
        let weights: [String: MLXArray] = [
            "audio_tower.layers.0.self_attn.k_proj.weight": MLXArray.zeros([8, 8]),
            "language_model.model.layers.0.mlp.up_proj.weight": MLXArray.zeros([16, 8]),
            "language_model.lm_head.weight": MLXArray.zeros([32, 8]),
        ]

        let result = SharpAIMossTranscribeDiarizeSanitize.sanitize(
            weights: weights, config: try config())

        // The tiny config ties embeddings, so sanitize drops `language_model.lm_head.weight`
        // (filterLMHeadWeights). Every other key passes through unchanged.
        let expectedKeys = Set(weights.keys).subtracting(["language_model.lm_head.weight"])
        XCTAssertEqual(Set(result.keys), expectedKeys)
        XCTAssertNil(result["language_model.lm_head.weight"])
        XCTAssertEqual(result["audio_tower.layers.0.self_attn.k_proj.weight"]?.shape, [8, 8])
    }
}
