// SharpAI fork overlay tests for funaudiochat checkpoint key sanitization. Not part of upstream.

import Foundation
import MLX
import MLXVLM
import XCTest

final class SharpAIFunaudiochatSanitizeTests: XCTestCase {

    func testDropsRotaryBuffersAndKeepsOtherKeys() {
        let weights: [String: MLXArray] = [
            "model.layers.0.self_attn.rotary_emb.inv_freq": MLXArray.zeros([2]),
            "model.layers.0.self_attn.q_proj.weight": MLXArray.zeros([8, 8]),
        ]

        let result = SharpAIFunaudiochatSanitize.sanitize(weights: weights)

        XCTAssertEqual(result.count, 1)
        XCTAssertNotNil(result["model.layers.0.self_attn.q_proj.weight"])
        XCTAssertNil(result["model.layers.0.self_attn.rotary_emb.inv_freq"])
    }

    /// vLLM funaudiochat.py: `orig_to_new_prefix={"audio_invert_tower.": None}`, so these keys are dropped.
    func testDropsAudioInvertTowerKeys() {
        let weights: [String: MLXArray] = [
            "audio_invert_tower.crq_transformer.layers.0.self_attn.q_proj.weight": MLXArray.zeros([8, 8]),
            "audio_invert_tower.lm_head.weight": MLXArray.zeros([8, 8]),
            "continuous_audio_tower.layers.0.self_attn.q_proj.weight": MLXArray.zeros([8, 8]),
        ]

        let result = SharpAIFunaudiochatSanitize.sanitize(weights: weights)

        XCTAssertEqual(result.count, 1)
        XCTAssertNotNil(result["continuous_audio_tower.layers.0.self_attn.q_proj.weight"])
        XCTAssertNil(result["audio_invert_tower.lm_head.weight"])
    }
}
