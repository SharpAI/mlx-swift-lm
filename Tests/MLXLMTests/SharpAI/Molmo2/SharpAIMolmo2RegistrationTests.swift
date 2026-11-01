// SharpAI fork overlay tests for the molmo2 creator. Not part of upstream.

import Foundation
import MLX
import MLXLMCommon
@testable import MLXVLM
import XCTest

final class SharpAIMolmo2RegistrationTests: XCTestCase {

    /// Tiny text_config in the layout of Molmo2-O-7B: post-norm blocks, OLMo-style QK norm,
    /// YaRN on layer 1 only, grouped-query attention (4 query heads, 2 kv heads).
    private let tinyTextJSON = """
        {
            "hidden_size": 32,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "vocab_size": 64,
            "additional_vocab_size": 4,
            "qkv_bias": false,
            "num_hidden_layers": 2,
            "intermediate_size": 48,
            "hidden_act": "silu",
            "max_position_embeddings": 64,
            "rope_theta": 10000.0,
            "rope_scaling": {
                "rope_type": "yarn",
                "factor": 2.0,
                "original_max_position_embeddings": 16,
                "beta_fast": 32.0,
                "beta_slow": 1.0
            },
            "rope_scaling_layers": [1],
            "use_qk_norm": true,
            "qk_norm_type": "olmo",
            "layer_norm_eps": 1e-6,
            "norm_after": true
        }
        """

    private let validJSON = """
        {
            "model_type": "molmo2",
            "text_config": {
                "hidden_size": 256,
                "num_hidden_layers": 2,
                "num_attention_heads": 8,
                "num_key_value_heads": 2,
                "vocab_size": 1024
            },
            "vit_config": {}
        }
        """

    func testCreatorIsRegisteredForModelType() {
        XCTAssertNotNil(SharpAIMolmo2Overlay.creators["molmo2"])
    }

    func testValidConfigFailsUntilForwardIsImplemented() throws {
        guard let creator = SharpAIMolmo2Overlay.creators["molmo2"] else {
            XCTFail("molmo2 creator missing")
            return
        }

        XCTAssertThrowsError(try creator(Data(validJSON.utf8))) { error in
            guard case SharpAIMolmo2Error.forwardNotImplemented(let type) = error else {
                XCTFail("unexpected error: \(error)")
                return
            }
            XCTAssertEqual(type, "molmo2")
        }
    }

    func testInvalidConfigFailsBeforeConstruction() throws {
        guard let creator = SharpAIMolmo2Overlay.creators["molmo2"] else {
            XCTFail("molmo2 creator missing")
            return
        }

        XCTAssertThrowsError(try creator(Data("{ \"hidden_size\": 64 }".utf8))) { error in
            if case SharpAIMolmo2Error.forwardNotImplemented = error {
                XCTFail("config validation should fail first")
            }
        }
    }

    /// Text-stack forward pass with random weights. Checks shape, finiteness, causality (a prefix
    /// run reproduces the leading logits) and that a cached decode step reproduces the full run.
    /// Values are not compared with the reference: no Molmo2 weights are on disk.
    func testTinyTextForwardPass() throws {
        let args = try JSONDecoder().decode(
            SharpAIMolmo2TextArguments.self, from: Data(tinyTextJSON.utf8))
        let model = SharpAIMolmo2ForCausalLM(args)
        let ids = MLXArray([Int32(3), 7, 1, 9]).reshaped(1, 4)

        let logits = model(ids, cache: nil)
        XCTAssertEqual(logits.shape, [1, 4, 64])
        XCTAssertFalse(MLX.isNaN(logits).any().item(Bool.self))

        let prefix = model(ids[0..., 0 ..< 3], cache: nil)
        XCTAssertTrue(
            allClose(logits[0..., 0 ..< 3], prefix, rtol: 1e-4, atol: 1e-4).item(Bool.self))

        let caches: [KVCache] = (0 ..< args.layers).map { _ in KVCacheSimple() }
        _ = model(ids[0..., 0 ..< 3], cache: caches)
        let step = model(ids[0..., 3 ..< 4], cache: caches)
        XCTAssertTrue(
            allClose(logits[0..., 3 ..< 4], step, rtol: 1e-4, atol: 1e-4).item(Bool.self))
    }
}
