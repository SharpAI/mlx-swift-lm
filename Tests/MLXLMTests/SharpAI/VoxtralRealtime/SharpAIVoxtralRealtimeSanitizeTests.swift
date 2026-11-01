// SharpAI fork overlay tests for voxtral_realtime key mapping and conv layout. No network.

import Foundation
import MLX
import XCTest

@testable import MLXVLM

final class SharpAIVoxtralRealtimeSanitizeTests: XCTestCase {

    private func configuration() throws -> SharpAIVoxtralRealtimeConfiguration {
        try SharpAIVoxtralRealtimeTestConfig.decodeTiny()
    }

    func testLeadingModelPrefixIsRemoved() throws {
        let weights: [String: MLXArray] = [
            "model.language_model.model.embed_tokens.weight": MLXRandom.normal([32, 8])
        ]
        let out = try SharpAIVoxtralRealtimeSanitizer.sanitize(
            weights, configuration: try configuration())

        XCTAssertNotNil(out["language_model.model.embed_tokens.weight"])
        XCTAssertNil(out["model.language_model.model.embed_tokens.weight"])
    }

    func testUnprefixedKeysAreKept() throws {
        let weights: [String: MLXArray] = [
            "language_model.model.embed_tokens.weight": MLXRandom.normal([32, 8]),
            "audio_tower.layers.0.mlp.gate_proj.weight": MLXRandom.normal([16, 8]),
        ]
        let out = try SharpAIVoxtralRealtimeSanitizer.sanitize(
            weights, configuration: try configuration())

        XCTAssertNotNil(out["audio_tower.layers.0.mlp.gate_proj.weight"])
    }

    func testConv1dWeightsMoveToMLXLayout() throws {
        // PyTorch (out=8, in=4, k=3) becomes MLX (out=8, k=3, in=4).
        let weights: [String: MLXArray] = [
            "language_model.model.embed_tokens.weight": MLXRandom.normal([32, 8]),
            "audio_tower.conv1.weight": MLXRandom.normal([8, 4, 3]),
            "audio_tower.conv1.bias": MLXRandom.normal([8]),
        ]
        let out = try SharpAIVoxtralRealtimeSanitizer.sanitize(
            weights, configuration: try configuration())

        XCTAssertEqual(out["audio_tower.conv1.weight"]?.shape, [8, 3, 4])
        XCTAssertEqual(out["audio_tower.conv1.bias"]?.shape, [8])
    }

    func testConv1dWeightsAlreadyInMLXLayoutAreNotTransposed() throws {
        // MLX layout (out=8, k=3, in=4) must pass through unchanged.
        let weights: [String: MLXArray] = [
            "language_model.model.embed_tokens.weight": MLXRandom.normal([32, 8]),
            "audio_tower.conv1.weight": MLXRandom.normal([8, 3, 4]),
        ]
        let once = try SharpAIVoxtralRealtimeSanitizer.sanitize(
            weights, configuration: try configuration())
        XCTAssertEqual(once["audio_tower.conv1.weight"]?.shape, [8, 3, 4])

        let twice = try SharpAIVoxtralRealtimeSanitizer.sanitize(
            once, configuration: try configuration())
        XCTAssertEqual(twice["audio_tower.conv1.weight"]?.shape, [8, 3, 4])
    }

    func testTiedHeadAndBuffersAreDropped() throws {
        let weights: [String: MLXArray] = [
            "language_model.model.embed_tokens.weight": MLXRandom.normal([32, 8]),
            "lm_head.weight": MLXRandom.normal([32, 8]),
            "audio_tower.rotary_emb.inv_freq": MLXRandom.normal([2]),
            "audio_tower.conv1.num_batches_tracked": MLXArray(Int32(0)),
        ]
        let out = try SharpAIVoxtralRealtimeSanitizer.sanitize(
            weights, configuration: try configuration())

        XCTAssertNil(out["lm_head.weight"])
        XCTAssertNil(out["audio_tower.rotary_emb.inv_freq"])
        XCTAssertNil(out["audio_tower.conv1.num_batches_tracked"])
        XCTAssertEqual(out.count, 1)
    }

    func testMissingEmbeddingThrows() throws {
        let weights: [String: MLXArray] = [
            "audio_tower.conv1.bias": MLXRandom.normal([8])
        ]
        XCTAssertThrowsError(
            try SharpAIVoxtralRealtimeSanitizer.sanitize(
                weights, configuration: try configuration()))
    }
}
