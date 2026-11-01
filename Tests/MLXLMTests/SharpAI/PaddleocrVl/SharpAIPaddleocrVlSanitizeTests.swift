// SharpAI fork overlay tests for paddleocr_vl weight key mapping. Not part of upstream.
// Zero-valued tensors only, no network. Keys follow the HF layout as recalled, not checked
// against a checkpoint.
// UNVERIFIED: this file has not been compiled or run.

import Foundation
import MLX
import MLXLMCommon
@testable import MLXVLM
import XCTest

final class SharpAIPaddleocrVlSanitizeTests: XCTestCase {

    func testRemapsVisionProjectorAndLanguageKeys() {
        XCTAssertEqual(
            SharpAIPaddleocrVl.remapKey(
                "visual.vision_model.embeddings.patch_embedding.weight", textLayerCount: 2),
            "vision_tower.embeddings.patch_embedding.weight")
        XCTAssertEqual(
            SharpAIPaddleocrVl.remapKey("mlp_AR.linear_1.weight", textLayerCount: 2),
            "projector.linear_1.weight")
        XCTAssertEqual(
            SharpAIPaddleocrVl.remapKey("model.embed_tokens.weight", textLayerCount: 2),
            "language_model.model.embed_tokens.weight")
        XCTAssertEqual(
            SharpAIPaddleocrVl.remapKey("lm_head.weight", textLayerCount: 2),
            "language_model.lm_head.weight")
    }

    func testDropsPackingPositionEmbeddingAndHead() {
        XCTAssertNil(
            SharpAIPaddleocrVl.remapKey(
                "visual.vision_model.embeddings.packing_position_embedding.weight",
                textLayerCount: 2))
        XCTAssertNil(
            SharpAIPaddleocrVl.remapKey(
                "visual.vision_model.head.probe", textLayerCount: 2))
    }

    func testDropsMTPPositionAndExtraLayerKeys() {
        XCTAssertNil(SharpAIPaddleocrVl.remapKey("mtp.layers.0.weight", textLayerCount: 2))
        XCTAssertNil(SharpAIPaddleocrVl.remapKey("model.rotary_emb.inv_freq", textLayerCount: 2))
        XCTAssertNil(
            SharpAIPaddleocrVl.remapKey(
                "visual.vision_model.embeddings.position_ids", textLayerCount: 2))
        XCTAssertNil(
            SharpAIPaddleocrVl.remapKey(
                "model.layers.2.mlp.gate_proj.weight", textLayerCount: 2))
        XCTAssertEqual(
            SharpAIPaddleocrVl.remapKey(
                "model.layers.1.mlp.gate_proj.weight", textLayerCount: 2),
            "language_model.model.layers.1.mlp.gate_proj.weight")
    }

    func testSanitizeDropsTiedLMHeadAndChannelsLastPatchWeight() throws {
        let config = try JSONDecoder.json5().decode(
            SharpAIPaddleocrVlConfiguration.self,
            from: Data(SharpAIPaddleocrVlConfigurationTests.tinyJSON.utf8))
        let model = SharpAIPaddleocrVl(config)

        let sanitized = model.sanitize(weights: [
            "lm_head.weight": MLXArray.zeros([128, 16]),
            "model.embed_tokens.weight": MLXArray.zeros([128, 16]),
            "visual.vision_model.embeddings.patch_embedding.weight": MLXArray.zeros([16, 3, 14, 14]),
        ])

        XCTAssertNil(sanitized["language_model.lm_head.weight"])
        XCTAssertNotNil(sanitized["language_model.model.embed_tokens.weight"])
        XCTAssertEqual(
            sanitized["vision_tower.embeddings.patch_embedding.weight"]?.shape,
            [16, 14, 14, 3])
    }
}
