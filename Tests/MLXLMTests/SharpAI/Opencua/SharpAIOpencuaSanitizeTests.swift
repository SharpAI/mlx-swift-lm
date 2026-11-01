import Foundation
import MLX
import MLXLMCommon
import XCTest

@testable import MLXVLM

final class SharpAIOpencuaSanitizeTests: XCTestCase {

    func testRenamesVisualAndLanguagePrefixes() {
        XCTAssertEqual(
            SharpAIOpencuaSanitizer.renameKey("model.visual.blocks.0.attn.qkv.weight"),
            "vision_tower.blocks.0.attn.qkv.weight")
        XCTAssertEqual(
            SharpAIOpencuaSanitizer.renameKey("visual.merger.mlp.0.weight"),
            "vision_tower.merger.mlp.0.weight")
        XCTAssertEqual(
            SharpAIOpencuaSanitizer.renameKey("model.language_model.layers.0.mlp.up_proj.weight"),
            "model.layers.0.mlp.up_proj.weight")
    }

    func testLeavesAlreadyMappedKeysUnchanged() {
        for key in [
            "vision_tower.blocks.0.attn.qkv.weight",
            "model.layers.0.mlp.up_proj.weight",
            "lm_head.weight",
            "language_model.model.embed_tokens.weight",
        ] {
            XCTAssertEqual(SharpAIOpencuaSanitizer.renameKey(key), key)
        }
    }

    func testModelSanitizeProducesQwen25VLLanguageKeys() throws {
        let model = SharpAIOpencuaModel(
            try SharpAIOpencuaTestFixtures.decode().qwen25VLConfiguration(
                from: Data(SharpAIOpencuaTestFixtures.tinyJSON.utf8)))
        let weights: [String: MLXArray] = [
            "model.layers.0.self_attn.q_proj.weight": MLXArray.zeros([64, 64]),
            "model.embed_tokens.weight": MLXArray.zeros([128, 64]),
        ]

        let result = model.sanitize(weights: weights)

        XCTAssertNotNil(result["language_model.model.layers.0.self_attn.q_proj.weight"])
        XCTAssertNotNil(result["language_model.model.embed_tokens.weight"])
        XCTAssertNil(result["model.layers.0.self_attn.q_proj.weight"])
    }
}
