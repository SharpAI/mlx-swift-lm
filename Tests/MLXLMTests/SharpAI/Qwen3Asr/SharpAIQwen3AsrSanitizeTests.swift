// SharpAI fork overlay tests for qwen3_asr key remapping and conv layout. Not part of upstream.
// No network and no real weights: tensors are zeros with PyTorch shapes.
// UNVERIFIED: written without compiling. Run it before relying on it.

import Foundation
import MLX
import MLXLMCommon
import MLXVLM
import XCTest

final class SharpAIQwen3AsrSanitizeTests: XCTestCase {

    private func configuration(tied: Bool, aligner: Bool = false) throws
        -> SharpAIQwen3AsrConfiguration
    {
        var config = try JSONDecoder.json5().decode(
            SharpAIQwen3AsrConfiguration.self,
            from: Data(SharpAIQwen3AsrConfigurationTests.tinyJSON.utf8))
        config.tieWordEmbeddings = tied
        config.forcedAligner = aligner
        return config
    }

    func testThinkerPrefixIsStrippedAndConvIsTransposed() throws {
        let config = try configuration(tied: true)
        let weights: [String: MLXArray] = [
            "thinker.audio_tower.conv2d1.weight": MLXArray.zeros([8, 1, 3, 3]),
            "thinker.audio_tower.conv2d1.bias": MLXArray.zeros([8]),
            "thinker.audio_tower.proj1.weight": MLXArray.zeros([16, 16]),
            "thinker.model.embed_tokens.weight": MLXArray.zeros([64, 32]),
            "thinker.model.layers.0.self_attn.q_proj.weight": MLXArray.zeros([32, 32]),
        ]

        let out = try SharpAIQwen3AsrSanitizer.sanitize(weights, configuration: config)

        XCTAssertEqual(out["audio_tower.conv2d1.weight"]?.shape, [8, 3, 3, 1])
        XCTAssertEqual(out["audio_tower.conv2d1.bias"]?.shape, [8])
        XCTAssertEqual(out["audio_tower.proj1.weight"]?.shape, [16, 16])
        XCTAssertEqual(out["model.embed_tokens.weight"]?.shape, [64, 32])
        XCTAssertEqual(out["model.layers.0.self_attn.q_proj.weight"]?.shape, [32, 32])
        XCTAssertNil(out["thinker.model.embed_tokens.weight"])
    }

    func testConvAlreadyInMlxLayoutIsNotTransposedAgain() throws {
        let config = try configuration(tied: true)
        let channels = config.audioConfiguration.downsampleHiddenSize
        let weights: [String: MLXArray] = [
            // MLX layout (out, kH, kW, in): the last dim is the input channel count.
            "thinker.audio_tower.conv2d1.weight": MLXArray.zeros([channels, 3, 3, 1]),
            "thinker.audio_tower.conv2d2.weight": MLXArray.zeros([channels, 3, 3, channels]),
        ]

        let out = try SharpAIQwen3AsrSanitizer.sanitize(weights, configuration: config)

        XCTAssertEqual(out["audio_tower.conv2d1.weight"]?.shape, [channels, 3, 3, 1])
        XCTAssertEqual(out["audio_tower.conv2d2.weight"]?.shape, [channels, 3, 3, channels])
    }

    func testTiedAsrDropsLmHead() throws {
        let config = try configuration(tied: true)
        let weights: [String: MLXArray] = [
            "thinker.lm_head.weight": MLXArray.zeros([64, 32]),
            "thinker.model.norm.weight": MLXArray.zeros([32]),
        ]

        let out = try SharpAIQwen3AsrSanitizer.sanitize(weights, configuration: config)

        XCTAssertNil(out["lm_head.weight"])
        XCTAssertNotNil(out["model.norm.weight"])
    }

    func testUntiedAsrKeepsLmHead() throws {
        let config = try configuration(tied: false)
        let weights: [String: MLXArray] = [
            "thinker.lm_head.weight": MLXArray.zeros([64, 32])
        ]

        let out = try SharpAIQwen3AsrSanitizer.sanitize(weights, configuration: config)

        XCTAssertEqual(out["lm_head.weight"]?.shape, [64, 32])
    }

    func testTransformersLayoutIsMapped() throws {
        let config = try configuration(tied: false)
        let weights: [String: MLXArray] = [
            "model.audio_tower.conv_out.weight": MLXArray.zeros([16, 64]),
            "model.multi_modal_projector.linear_1.weight": MLXArray.zeros([16, 16]),
            "model.multi_modal_projector.linear_2.weight": MLXArray.zeros([32, 16]),
            "model.language_model.norm.weight": MLXArray.zeros([32]),
            "model.language_model.layers.0.mlp.up_proj.weight": MLXArray.zeros([64, 32]),
        ]

        let out = try SharpAIQwen3AsrSanitizer.sanitize(weights, configuration: config)

        XCTAssertNotNil(out["audio_tower.conv_out.weight"])
        XCTAssertNotNil(out["audio_tower.proj1.weight"])
        XCTAssertNotNil(out["audio_tower.proj2.weight"])
        XCTAssertNotNil(out["model.norm.weight"])
        XCTAssertNotNil(out["model.layers.0.mlp.up_proj.weight"])
        XCTAssertNil(out["model.multi_modal_projector.linear_1.weight"])
    }

    func testPositionalBuffersAreDropped() throws {
        let config = try configuration(tied: true)
        let weights: [String: MLXArray] = [
            "thinker.audio_tower.positional_embedding": MLXArray.zeros([64, 16])
        ]

        let out = try SharpAIQwen3AsrSanitizer.sanitize(weights, configuration: config)

        XCTAssertTrue(out.isEmpty)
    }

    func testAlignerHeadIsRenamedToClassifier() throws {
        let config = try configuration(tied: false, aligner: true)
        let weights: [String: MLXArray] = [
            "thinker.lm_head.weight": MLXArray.zeros([5000, 32])
        ]

        let out = try SharpAIQwen3AsrSanitizer.sanitize(weights, configuration: config)

        XCTAssertEqual(out["classifier.weight"]?.shape, [5000, 32])
        XCTAssertNil(out["lm_head.weight"])
    }
}
