// SharpAI fork overlay tests for the openpangu_vl creator. Not part of upstream.

import Foundation
import MLX
import MLXLMCommon
import MLXVLM
import XCTest

final class SharpAIOpenpanguVlRegistrationTests: XCTestCase {

    private let validJSON = """
        {
            "model_type": "openpangu_vl",
            "hidden_size": 256,
            "intermediate_size": 512,
            "num_hidden_layers": 2,
            "num_attention_heads": 8,
            "num_key_value_heads": 8,
            "vocab_size": 1024,
            "rms_norm_eps": 1e-05,
            "rope_theta": 10000.0,
            "max_position_embeddings": 4096,
            "vision_config": {
                "depth": 2,
                "hidden_size": 64,
                "intermediate_size": 128,
                "num_heads": 4,
                "in_chans": 3,
                "out_hidden_size": 256,
                "patch_size": 14,
                "spatial_merge_size": 2,
                "window_size": 112,
                "fullatt_block_indexes": [1],
                "temporal_patch_size": 2
            }
        }
        """

    /// Tiny text decoder with the interleaved mRoPE layout of the real config. head_dim is 32,
    /// so mrope_section must sum to 16: [4, 6, 6].
    private let tinyJSON = """
        {
            "model_type": "openpangu_vl",
            "hidden_size": 256,
            "intermediate_size": 512,
            "num_hidden_layers": 2,
            "num_attention_heads": 8,
            "num_key_value_heads": 2,
            "bias": true,
            "vocab_size": 1024,
            "rms_norm_eps": 1e-05,
            "rope_theta": 10000.0,
            "max_position_embeddings": 4096,
            "rope_scaling": {
                "rope_type": "default",
                "mrope_section": [4, 6, 6],
                "mrope_interleaved": true
            },
            "vision_config": {
                "depth": 2,
                "hidden_size": 64,
                "intermediate_size": 128,
                "num_heads": 4,
                "in_chans": 3,
                "out_hidden_size": 256,
                "patch_size": 14,
                "spatial_merge_size": 2,
                "window_size": 112,
                "fullatt_block_indexes": [1],
                "temporal_patch_size": 2
            }
        }
        """

    private func makeTinyModel() throws -> any LanguageModel {
        guard let creator = SharpAIOpenpanguVlOverlay.creators["openpangu_vl"] else {
            throw SharpAIOpenpanguVlError.forwardNotImplemented("openpangu_vl")
        }
        return try creator(Data(tinyJSON.utf8))
    }

    func testCreatorIsRegisteredForModelType() {
        XCTAssertNotNil(SharpAIOpenpanguVlOverlay.creators["openpangu_vl"])
    }

    func testValidConfigBuildsTextDecoder() throws {
        guard let creator = SharpAIOpenpanguVlOverlay.creators["openpangu_vl"] else {
            XCTFail("openpangu_vl creator missing")
            return
        }

        // The ViT is not implemented, but the text decoder is, so the creator now builds a model.
        let model = try creator(Data(validJSON.utf8))
        guard let concrete = model as? SharpAIOpenpanguVlModel else {
            XCTFail("creator did not return SharpAIOpenpanguVlModel")
            return
        }
        XCTAssertEqual(concrete.vocabularySize, 1024)
    }

    func testInvalidConfigFailsBeforeConstruction() throws {
        guard let creator = SharpAIOpenpanguVlOverlay.creators["openpangu_vl"] else {
            XCTFail("openpangu_vl creator missing")
            return
        }

        XCTAssertThrowsError(try creator(Data("{ \"hidden_size\": 64 }".utf8))) { error in
            if case SharpAIOpenpanguVlError.forwardNotImplemented = error {
                XCTFail("config validation should fail first")
            }
        }
    }

    func testTinyForwardPass() throws {
        let model = try makeTinyModel()

        let tokens = MLXArray([Int32(1), 5, 9, 3, 7]).reshaped(1, 5)
        let logits = model(tokens, cache: nil)

        XCTAssertEqual(logits.shape, [1, 5, 1024])
        XCTAssertFalse(MLX.isNaN(logits).any().item(Bool.self), "NaN logits")
    }

    func testTinyForwardIsCausal() throws {
        let model = try makeTinyModel()

        // Changing only the last token must leave the logits of earlier positions unchanged.
        let a = model(MLXArray([Int32(1), 5, 9, 3, 7]).reshaped(1, 5), cache: nil)
        let b = model(MLXArray([Int32(1), 5, 9, 3, 2]).reshaped(1, 5), cache: nil)

        XCTAssertTrue(
            allClose(a[0..., ..<4], b[0..., ..<4], rtol: 1e-4, atol: 1e-4).item(Bool.self))
    }

    func testCachedDecodeMatchesFullPass() throws {
        let model = try makeTinyModel()

        // Prefill three tokens into a cache, then decode two more. The positions of the second
        // call must be anchored at the cache offset, so its logits match the full pass.
        let cache = try model.newCache(parameters: nil)
        _ = model(MLXArray([Int32(1), 5, 9]).reshaped(1, 3), cache: cache)
        let tail = model(MLXArray([Int32(3), 7]).reshaped(1, 2), cache: cache)

        let full = model(MLXArray([Int32(1), 5, 9, 3, 7]).reshaped(1, 5), cache: nil)

        XCTAssertTrue(
            allClose(tail, full[0..., 3 ..< 5], rtol: 1e-4, atol: 1e-4).item(Bool.self))
    }
}
