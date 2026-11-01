// SharpAI fork overlay tests for step5 sanitize (key cleanup). Not part of upstream.
// Key names come from the official Step-5-Preview-BF16 index. Expert tensors are already
// stacked as [experts, out, in] under moe.{gate,up,down}_proj.weight.

import Foundation
import MLX
import MLXLMCommon
import XCTest

@testable import MLXLLM

final class SharpAIStep5SanitizeTests: XCTestCase {

    func testDropsRotaryInvFreqAndKeepsOtherKeys() {
        let sanitized = SharpAIStep5Sanitize.sanitize(
            weights: [
                "model.layers.0.self_attn.rotary_emb.inv_freq": MLXArray.zeros([2]),
                "model.layers.0.self_attn.q_proj.weight": MLXArray.zeros([4, 4]),
                "model.embed_tokens.weight": MLXArray.zeros([8, 4]),
            ])

        XCTAssertNil(sanitized["model.layers.0.self_attn.rotary_emb.inv_freq"])
        XCTAssertNotNil(sanitized["model.layers.0.self_attn.q_proj.weight"])
        XCTAssertNotNil(sanitized["model.embed_tokens.weight"])
    }

    func testPassesThroughLMHeadUnchanged() {
        // The untied lm_head is a real checkpoint tensor and is kept as is.
        let sanitized = SharpAIStep5Sanitize.sanitize(
            weights: ["lm_head.weight": MLXArray.zeros([8, 4])])

        XCTAssertEqual(sanitized["lm_head.weight"]?.shape, [8, 4])
    }

    func testRenamesStackedExpertsAndDropsMTPAndVisionKeys() throws {
        let config = try JSONDecoder.json5().decode(
            SharpAIStep5Configuration.self,
            from: Data(
                """
                {
                    "vocab_size": 8,
                    "hidden_size": 4,
                    "intermediate_size": 8,
                    "num_hidden_layers": 2,
                    "num_attention_heads": 2,
                    "layer_types": ["sliding_attention", "sliding_attention", "sliding_attention"],
                    "moe_layer_list": [1]
                }
                """.utf8))
        let sanitized = SharpAIStep5Sanitize.sanitize(
            weights: [
                "model.layers.1.moe.gate_proj.weight": MLXArray.zeros([4, 3, 4]),
                "model.layers.1.moe.router_bias": MLXArray.zeros([4]),
                "model.layers.1.moe.gate.weight": MLXArray.zeros([4, 4]),
                "model.layers.1.share_expert.up_proj.weight": MLXArray.zeros([3, 4]),
                // MTP layer (index >= num_hidden_layers) is dropped.
                "model.layers.2.eh_proj.weight": MLXArray.zeros([4, 8]),
                // Vision and projector keys are dropped for text-only load.
                "vision_model.conv1.weight": MLXArray.zeros([2, 2]),
                "vit_large_projector.weight": MLXArray.zeros([2, 2]),
            ],
            configuration: config)

        XCTAssertNotNil(sanitized["model.layers.1.moe.switch_mlp.gate_proj.weight"])
        XCTAssertNil(sanitized["model.layers.1.moe.gate_proj.weight"])
        XCTAssertNotNil(sanitized["model.layers.1.moe.router_bias"])
        XCTAssertNotNil(sanitized["model.layers.1.moe.gate.weight"])
        XCTAssertNotNil(sanitized["model.layers.1.share_expert.up_proj.weight"])
        XCTAssertNil(sanitized["model.layers.2.eh_proj.weight"])
        XCTAssertNil(sanitized["vision_model.conv1.weight"])
        XCTAssertNil(sanitized["vit_large_projector.weight"])
    }
}
