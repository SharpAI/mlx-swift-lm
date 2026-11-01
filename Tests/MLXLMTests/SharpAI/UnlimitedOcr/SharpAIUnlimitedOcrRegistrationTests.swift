// SharpAI fork overlay tests for the unlimited_ocr creator. Not part of upstream.

import Foundation
import MLX
import MLXLMCommon
import MLXVLM
import XCTest

final class SharpAIUnlimitedOcrRegistrationTests: XCTestCase {

    /// A text-only tiny config (no vision_config), so the model builds without the vision stack.
    private let validJSON = """
        {
            "model_type": "unlimited_ocr",
            "language_config": {
                "model_type": "deepseek_v2",
                "vocab_size": 1024,
                "hidden_size": 256,
                "intermediate_size": 512,
                "num_hidden_layers": 2,
                "num_attention_heads": 8,
                "num_key_value_heads": 8,
                "max_position_embeddings": 128,
                "use_mla": false,
                "kv_lora_rank": null
            }
        }
        """

    func testCreatorIsRegisteredForModelType() {
        XCTAssertNotNil(SharpAIUnlimitedOcrOverlay.creators["unlimited_ocr"])
    }

    func testValidConfigBuildsModelWithOneCachePerLayer() throws {
        guard let creator = SharpAIUnlimitedOcrOverlay.creators["unlimited_ocr"] else {
            XCTFail("unlimited_ocr creator missing")
            return
        }

        let model = try creator(Data(validJSON.utf8))
        let caches = try model.newCache(parameters: nil)
        XCTAssertEqual(caches.count, 2)
    }

    func testInvalidConfigFailsBeforeConstruction() throws {
        guard let creator = SharpAIUnlimitedOcrOverlay.creators["unlimited_ocr"] else {
            XCTFail("unlimited_ocr creator missing")
            return
        }

        XCTAssertThrowsError(try creator(Data("{ \"hidden_size\": 64 }".utf8))) { error in
            XCTAssertFalse(error is SharpAIUnlimitedOcrError)
        }
    }

    func testTinyForwardPass() throws {
        guard let creator = SharpAIUnlimitedOcrOverlay.creators["unlimited_ocr"] else {
            XCTFail("unlimited_ocr creator missing")
            return
        }

        let model = try creator(Data(validJSON.utf8))
        let tokens = MLXArray([Int32(1), 5, 9]).reshaped(1, 3)
        let logits = model(tokens, cache: nil)
        XCTAssertEqual(logits.shape, [1, 3, 1024])
    }
}
