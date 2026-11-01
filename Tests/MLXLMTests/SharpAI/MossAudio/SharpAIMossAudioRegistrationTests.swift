// SharpAI fork overlay tests for the moss_audio creator. Not part of upstream.
// The forward pass runs on a tiny random-init model. Its audio encoder, adapter, DeepStack mergers
// and Qwen3 backbone have the same structure as the 4B checkpoint, with small widths.
// UNVERIFIED: written without compiling. Run it before relying on it. The checkpoint layout and
// the stem flatten order are not checked against real weights.

import Foundation
import MLX
import MLXLMCommon
import MLXVLM
import XCTest

final class SharpAIMossAudioRegistrationTests: XCTestCase {

    private let tinyJSON = """
        {
          "architectures": ["MossAudioModel"],
          "model_type": "moss_audio",
          "adapter_hidden_size": 16,
          "audio_config": {
            "d_model": 32,
            "output_dim": 32,
            "num_mel_bins": 128,
            "encoder_layers": 3,
            "encoder_attention_heads": 4,
            "encoder_ffn_dim": 64,
            "downsample_hidden_size": 8,
            "max_source_positions": 1500,
            "n_window": 200,
            "deepstack_encoder_layer_indexes": [0, 1, 2]
          },
          "language_config": {
            "model_type": "qwen3",
            "hidden_size": 64,
            "num_hidden_layers": 2,
            "intermediate_size": 128,
            "num_attention_heads": 4,
            "rms_norm_eps": 0.000001,
            "vocab_size": 100,
            "num_key_value_heads": 2,
            "head_dim": 16
          }
        }
        """

    func testCreatorIsKeyedByModelType() {
        XCTAssertNotNil(SharpAIMossAudioOverlay.creators["moss_audio"])
    }

    func testWrongModelTypeFailsAtValidation() throws {
        let creator = try XCTUnwrap(SharpAIMossAudioOverlay.creators["moss_audio"])
        let json = tinyJSON.replacingOccurrences(
            of: "\"model_type\": \"moss_audio\"", with: "\"model_type\": \"qwen3\"")

        XCTAssertThrowsError(try creator(Data(json.utf8))) { error in
            guard let typed = error as? SharpAIMossAudioError,
                case .invalidConfiguration = typed
            else {
                XCTFail("expected invalidConfiguration, got \(error)")
                return
            }
        }
    }

    func testValidConfigBuildsModel() throws {
        let creator = try XCTUnwrap(SharpAIMossAudioOverlay.creators["moss_audio"])

        let model = try creator(Data(tinyJSON.utf8))
        XCTAssertTrue(model is SharpAIMossAudio)
    }

    /// Text-only prefill, then audio spliced into the placeholder positions.
    func testTinyForwardPass() throws {
        let creator = try XCTUnwrap(SharpAIMossAudioOverlay.creators["moss_audio"])
        let model = try XCTUnwrap(try creator(Data(tinyJSON.utf8)) as? SharpAIMossAudio)

        // Text only: [1, 5] tokens give [1, 5, vocab] logits.
        let textTokens = MLXArray([1, 2, 3, 4, 5] as [Int32]).expandedDimensions(axis: 0)
        let textLogits = model(textTokens, cache: nil)
        XCTAssertEqual(textLogits.shape, [1, 5, 100])

        // 480 mel frames: a 400-frame chunk (50 tokens) and an 80-frame chunk (10 tokens).
        let frames = 480
        let expectedAudioTokens = sharpAIMossAudioTokenCount(frames: frames, chunkFrames: 400)
        XCTAssertEqual(expectedAudioTokens, 60)

        let features = MLXArray(
            (0 ..< (128 * frames)).map { Float(sin(Double($0) * 0.01)) }, [1, 128, frames])

        // Placeholder id 5 is in range for the tiny vocabulary of 100.
        var ids: [Int32] = [1, 2]
        ids += Array(repeating: Int32(5), count: expectedAudioTokens)
        ids += [3]
        let tokens = MLXArray(ids).expandedDimensions(axis: 0)

        let logits = try model.multimodalLogits(
            tokens: tokens, audioFeatures: features, audioTokenId: 5)
        XCTAssertEqual(logits.shape, [1, ids.count, 100])
        XCTAssertTrue(logits.asArray(Float.self).allSatisfy { $0.isFinite })

        // One placeholder too few must be rejected, not silently padded.
        var short = ids
        short.removeLast(2)
        short.append(3)
        XCTAssertThrowsError(
            try model.multimodalLogits(
                tokens: MLXArray(short).expandedDimensions(axis: 0),
                audioFeatures: features, audioTokenId: 5)
        ) { error in
            guard let typed = error as? SharpAIMossAudioError,
                case .audioTokenMismatch = typed
            else {
                XCTFail("expected audioTokenMismatch, got \(error)")
                return
            }
        }
    }
}
