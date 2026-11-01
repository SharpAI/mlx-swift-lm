// SharpAI fork overlay tests for moss_audio checkpoint key sanitization. Not part of upstream.
// UNVERIFIED: key names are placeholders until a MOSS-Audio checkpoint is checked.

import MLX
import MLXVLM
import XCTest

final class SharpAIMossAudioSanitizeTests: XCTestCase {

    func testDropsRotaryBuffersAndKeepsOtherKeys() {
        let weights: [String: MLXArray] = [
            "model.layers.0.self_attn.rotary_emb.inv_freq": MLXArray.zeros([2]),
            "model.layers.0.self_attn.q_proj.weight": MLXArray.zeros([8, 8]),
        ]

        let result = SharpAIMossAudioSanitize.sanitize(weights: weights)

        XCTAssertEqual(result.count, 1)
        XCTAssertEqual(result["model.layers.0.self_attn.q_proj.weight"]?.shape, [8, 8])
        XCTAssertNil(result["model.layers.0.self_attn.rotary_emb.inv_freq"])
    }

    func testEmptyInputGivesEmptyOutput() {
        XCTAssertTrue(SharpAIMossAudioSanitize.sanitize(weights: [:]).isEmpty)
    }
}
