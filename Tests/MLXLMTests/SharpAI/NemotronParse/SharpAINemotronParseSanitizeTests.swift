// SharpAI fork overlay tests for nemotron_parse weight sanitize. Not part of upstream.

import MLX
import MLXVLM
import XCTest

final class SharpAINemotronParseSanitizeTests: XCTestCase {

    func testSanitizePassesKeysThroughUnchanged() {
        let weights: [String: MLXArray] = [
            "decoder.model.embed_tokens.weight": MLXArray.zeros([4, 2]),
            "encoder.model.patch_embed.weight": MLXArray.zeros([2, 2]),
        ]

        let result = SharpAINemotronParseSanitize.sanitize(weights)

        XCTAssertEqual(Set(result.keys), Set(weights.keys))
    }

    /// Key names and shapes as written in the nvidia/NVIDIA-Nemotron-Parse-v1.2 safetensors header.
    func testRealCheckpointKeysAreRemapped() {
        let normMean = "encoder.model_encoder.radio_model.input_conditioner.norm_mean"
        let normStd = "encoder.model_encoder.radio_model.input_conditioner.norm_std"
        let weights: [String: MLXArray] = [
            normMean: MLXArray.zeros([3, 1, 1]),
            normStd: MLXArray.zeros([3, 1, 1]),
            "encoder.conv1.weight": MLXArray.zeros([4, 3, 1]),
            "encoder.conv2.weight": MLXArray.zeros([4, 3, 1, 4]),
            "decoder.layers.0.encoder_attn.q_proj.weight": MLXArray.zeros([4, 2]),
            "lm_head.weight": MLXArray.zeros([8, 4]),
        ]

        let result = SharpAINemotronParseSanitize.sanitize(weights)

        XCTAssertEqual(
            Set(result.keys),
            Set(weights.keys).subtracting([normMean, normStd]))
        XCTAssertEqual(result["encoder.conv1.weight"]?.shape, [4, 1, 3])
        XCTAssertEqual(result["encoder.conv2.weight"]?.shape, [4, 1, 4, 3])
        XCTAssertEqual(result["decoder.layers.0.encoder_attn.q_proj.weight"]?.shape, [4, 2])
        XCTAssertEqual(result["lm_head.weight"]?.shape, [8, 4])
    }

    /// Non-parameter buffers and the unused decoder position table are dropped.
    func testNonParameterBuffersAreDropped() {
        let summaryIdxs = "encoder.model_encoder.radio_model.summary_idxs"
        let weights: [String: MLXArray] = [
            summaryIdxs: MLXArray([Int32(0), 1, 2]),
            "decoder.embed_positions.weight": MLXArray.zeros([20002, 4]),
            "decoder.embed_tokens.weight": MLXArray.zeros([8, 4]),
        ]

        let result = SharpAINemotronParseSanitize.sanitize(weights)

        XCTAssertEqual(Set(result.keys), ["decoder.embed_tokens.weight"])
    }
}
