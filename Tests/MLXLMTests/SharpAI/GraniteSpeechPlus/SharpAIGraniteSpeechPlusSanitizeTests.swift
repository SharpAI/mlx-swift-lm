// SharpAI fork overlay tests for granite_speech_plus key sanitization. Not part of upstream.
// UNVERIFIED: this file was written without compiling. Run it before relying on it.

import MLX
import XCTest

@testable import MLXVLM

final class SharpAIGraniteSpeechPlusSanitizeTests: XCTestCase {

    func testDropsRotaryInvFreqAndKeepsOtherKeysUnchanged() {
        let weights: [String: MLXArray] = [
            "language_model.model.embed_tokens.weight": MLXArray.zeros([64, 32]),
            "language_model.model.layers.0.self_attn.rotary_emb.inv_freq": MLXArray.zeros([8]),
        ]

        let sanitized = SharpAIGraniteSpeechPlusSanitizer.sanitize(weights)

        XCTAssertEqual(Set(sanitized.keys), ["language_model.model.embed_tokens.weight"])
        XCTAssertEqual(sanitized["language_model.model.embed_tokens.weight"]?.shape, [64, 32])
    }

    func testKeepsWeightsWithoutRotaryBuffers() {
        let weights: [String: MLXArray] = [
            "language_model.lm_head.weight": MLXArray.zeros([64, 32])
        ]

        let sanitized = SharpAIGraniteSpeechPlusSanitizer.sanitize(weights)

        XCTAssertEqual(Set(sanitized.keys), ["language_model.lm_head.weight"])
    }

    func testRemapsHfPrefixesAndDropsComputedBuffers() {
        let weights: [String: MLXArray] = [
            "model.language_model.embed_tokens.weight": MLXArray.zeros([64, 32]),
            "lm_head.weight": MLXArray.zeros([64, 32]),
            "model.projector.linear.weight": MLXArray.zeros([32, 16]),
            "model.encoder.layers.0.conv.up_conv.weight": MLXArray.zeros([8, 4, 1]),
            "model.encoder.attention_dists": MLXArray.zeros([4, 4]),
        ]

        let sanitized = SharpAIGraniteSpeechPlusSanitizer.sanitize(weights)

        XCTAssertEqual(
            Set(sanitized.keys),
            [
                "language_model.model.embed_tokens.weight",
                "language_model.lm_head.weight",
                "projector.linear.weight",
                "encoder.layers.0.conv.up_conv.weight",
            ])
    }

    func testCheckpointLayoutMapsToModuleTree() {
        // Key shapes from the released index: the text stack sits under language_model.model.model.
        let weights: [String: MLXArray] = [
            "language_model.model.model.embed_tokens.weight": MLXArray.zeros([64, 32]),
            "language_model.model.model.layers.0.self_attn.q_proj.weight": MLXArray.zeros([32, 32]),
            "language_model.model.model.norm.weight": MLXArray.zeros([32]),
            "projector.qformer.encoder.layer.0.attention.attention.key.bias": MLXArray.zeros([16]),
            // PyTorch depthwise weight is (channels, 1, kernel).
            "encoder.layers.0.conv.depth_conv.conv.weight": MLXArray.zeros([32, 1, 3]),
        ]

        let sanitized = SharpAIGraniteSpeechPlusSanitizer.sanitize(weights)

        XCTAssertEqual(
            Set(sanitized.keys),
            [
                "language_model.model.embed_tokens.weight",
                "language_model.model.layers.0.self_attn.q_proj.weight",
                "language_model.model.norm.weight",
                "projector.qformer.encoder.layer.0.attention.attention.key.bias",
                "encoder.layers.0.conv.depth_conv.conv.weight",
            ])
        XCTAssertEqual(
            sanitized["encoder.layers.0.conv.depth_conv.conv.weight"]?.shape, [32, 3, 1])
    }

    func testConvWeightsMoveToMlxLayout() {
        // PyTorch (out, in / groups, k) becomes MLX (out, k, in / groups).
        let weights: [String: MLXArray] = [
            "model.encoder.layers.0.conv.up_conv.weight": MLXArray.zeros([8, 4, 1])
        ]

        let sanitized = SharpAIGraniteSpeechPlusSanitizer.sanitize(weights)

        XCTAssertEqual(
            sanitized["encoder.layers.0.conv.up_conv.weight"]?.shape, [8, 1, 4])
    }
}
