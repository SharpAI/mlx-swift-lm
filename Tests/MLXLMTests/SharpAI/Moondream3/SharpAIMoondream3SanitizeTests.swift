// SharpAI fork overlay: moondream3 sanitize tests. Not part of upstream.

import Foundation
import MLX
import XCTest

@testable import MLXVLM

final class SharpAIMoondream3SanitizeTests: XCTestCase {

    func testStripsModelPrefixAndKeepsStructuredKeys() throws {
        let weights: [String: MLXArray] = [
            "model.text.blocks.1.mlp.fc1.weight": MLXArray.zeros([4, 16, 16]),
            "model.vision.proj_mlp.fc1.weight": MLXArray.zeros([12, 16]),
            "model.vision.blocks.0.attn.qkv.weight": MLXArray.zeros([24, 8]),
        ]

        let result = try SharpAIMoondream3Sanitizer.sanitize(weights)

        XCTAssertNotNil(result["text.blocks.1.mlp.fc1.weight"])
        XCTAssertNotNil(result["vision.proj_mlp.fc1.weight"])
        XCTAssertNotNil(result["vision.blocks.0.attn.qkv.weight"])
        XCTAssertNil(result["model.text.blocks.1.mlp.fc1.weight"])
    }

    func testAddsWeightSuffixToBareEmbeddingTable() throws {
        let weights: [String: MLXArray] = [
            "model.text.wte": MLXArray.zeros([32, 16])
        ]

        let result = try SharpAIMoondream3Sanitizer.sanitize(weights)

        XCTAssertNotNil(result["text.wte.weight"])
        XCTAssertNil(result["text.wte"])
    }

    func testRejectsTwoKeysThatMapToSameName() throws {
        let weights: [String: MLXArray] = [
            "model.text.wte": MLXArray.zeros([32, 16]),
            "text.wte.weight": MLXArray.zeros([32, 16]),
        ]

        XCTAssertThrowsError(try SharpAIMoondream3Sanitizer.sanitize(weights))
    }

    func testDropsRegionKeysTheSwiftTreeDoesNotBuild() throws {
        // UNVERIFIED: key names are from the vLLM reference, not a real checkpoint.
        let weights: [String: MLXArray] = [
            "model.region.point.weight": MLXArray.zeros([4, 4]),
            "model.text.wte": MLXArray.zeros([32, 16]),
        ]

        let result = try SharpAIMoondream3Sanitizer.sanitize(weights)

        XCTAssertNil(result["region.point.weight"])
        XCTAssertNotNil(result["text.wte.weight"])
        XCTAssertEqual(result.count, 1)
    }

    func testRemapLeavesUnprefixedKeysAlone() {
        XCTAssertEqual(
            SharpAIMoondream3Sanitizer.remappedName("text.lm_head.bias"), "text.lm_head.bias")
    }
}
