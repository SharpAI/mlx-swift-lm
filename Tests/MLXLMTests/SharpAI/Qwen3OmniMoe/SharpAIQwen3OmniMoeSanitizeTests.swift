// SharpAI fork overlay tests for qwen3_omni_moe key sanitization. Random tensors, no checkpoint.

import Foundation
import MLX
import MLXLMCommon
@testable import MLXVLM
import XCTest

final class SharpAIQwen3OmniMoeSanitizeTests: XCTestCase {

    private func config() throws -> SharpAIQwen3OmniMoeConfiguration {
        try JSONDecoder.json5().decode(
            SharpAIQwen3OmniMoeConfiguration.self,
            from: Data(SharpAIQwen3OmniMoeTestConfig.tiny().utf8))
    }

    func testThinkerPrefixIsStrippedAndTalkerIsDropped() {
        XCTAssertEqual(
            SharpAIQwen3OmniMoeSanitize.checkpointName("thinker.model.norm.weight"),
            "language_model.model.norm.weight")
        XCTAssertEqual(
            SharpAIQwen3OmniMoeSanitize.checkpointName("thinker.lm_head.weight"),
            "language_model.lm_head.weight")
        XCTAssertEqual(
            SharpAIQwen3OmniMoeSanitize.checkpointName("thinker.audio_tower.ln_post.weight"),
            "audio_tower.ln_post.weight")
        XCTAssertNil(
            SharpAIQwen3OmniMoeSanitize.checkpointName("talker.model.layers.0.weight"))
        XCTAssertNil(SharpAIQwen3OmniMoeSanitize.checkpointName("code2wav.conv.weight"))
    }

    func testVisualKeysMapToVisionTowerAndDeepstackMergers() {
        XCTAssertEqual(
            SharpAIQwen3OmniMoeSanitize.checkpointName("thinker.visual.blocks.0.attn.qkv.weight"),
            "vision_tower.blocks.0.attn.qkv.weight")
        XCTAssertEqual(
            SharpAIQwen3OmniMoeSanitize.checkpointName(
                "thinker.visual.merger_list.1.linear_fc1.weight"),
            "vision_tower.deepstack_merger_list.1.linear_fc1.weight")
    }

    func testExpertsStackIntoSwitchMLPAndConvWeightsTranspose() throws {
        let config = try config()
        let experts = config.textConfiguration.numExperts
        var weights: [String: MLXArray] = [
            "thinker.model.embed_tokens.weight": MLXRandom.normal([64, 32]),
            "thinker.lm_head.weight": MLXRandom.normal([64, 32]),
            "thinker.audio_tower.conv2d1.weight": MLXRandom.normal([4, 1, 3, 3]),
            "thinker.audio_tower.conv2d2.weight": MLXRandom.normal([4, 4, 3, 3]),
            "talker.model.layers.0.self_attn.q_proj.weight": MLXRandom.normal([8, 8]),
        ]
        for expert in 0 ..< experts {
            weights["thinker.model.layers.0.mlp.experts.\(expert).gate_proj.weight"] =
                MLXRandom.normal([16, 32])
            weights["thinker.model.layers.0.mlp.experts.\(expert).up_proj.weight"] =
                MLXRandom.normal([16, 32])
            weights["thinker.model.layers.0.mlp.experts.\(expert).down_proj.weight"] =
                MLXRandom.normal([32, 16])
        }

        let result = SharpAIQwen3OmniMoeSanitize.sanitize(weights: weights, config: config)

        XCTAssertEqual(result["language_model.model.embed_tokens.weight"]?.shape, [64, 32])
        XCTAssertEqual(result["language_model.lm_head.weight"]?.shape, [64, 32])
        XCTAssertEqual(
            result["language_model.model.layers.0.mlp.switch_mlp.gate_proj.weight"]?.shape,
            [experts, 16, 32])
        XCTAssertEqual(
            result["language_model.model.layers.0.mlp.switch_mlp.down_proj.weight"]?.shape,
            [experts, 32, 16])
        XCTAssertNil(result["language_model.model.layers.0.mlp.experts.0.gate_proj.weight"])
        XCTAssertEqual(result["audio_tower.conv2d1.weight"]?.shape, [4, 3, 3, 1])
        XCTAssertEqual(result["audio_tower.conv2d2.weight"]?.shape, [4, 3, 3, 4])
        XCTAssertFalse(result.keys.contains { $0.hasPrefix("talker.") })
    }

    func testTiedEmbeddingsDropLMHead() throws {
        let json = SharpAIQwen3OmniMoeTestConfig.tiny()
            .replacingOccurrences(
                of: "\"tie_word_embeddings\": false", with: "\"tie_word_embeddings\": true")
        let config = try JSONDecoder.json5().decode(
            SharpAIQwen3OmniMoeConfiguration.self, from: Data(json.utf8))
        let weights: [String: MLXArray] = [
            "thinker.model.embed_tokens.weight": MLXRandom.normal([64, 32]),
            "thinker.lm_head.weight": MLXRandom.normal([64, 32]),
        ]

        let result = SharpAIQwen3OmniMoeSanitize.sanitize(weights: weights, config: config)

        XCTAssertNil(result["language_model.lm_head.weight"])
        XCTAssertNotNil(result["language_model.model.embed_tokens.weight"])
    }
}
