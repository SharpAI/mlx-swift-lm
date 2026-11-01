// SharpAI fork overlay tests for k2_horizon model structure, forward pass and grouped norm.
// Not part of upstream. Random weights only, tiny shapes.

import Foundation
import MLX
import MLXLMCommon
import XCTest

@testable import MLXLLM

final class SharpAIK2HorizonModelTests: XCTestCase {

    func testForwardPassProducesLogitsShape() throws {
        let model = SharpAIK2HorizonModel(try SharpAIK2HorizonTestFixtures.configuration())
        let inputs = MLXArray([1, 2, 3] as [Int32]).reshaped(1, 3)

        let logits = model(inputs, cache: nil)
        eval(logits)

        XCTAssertEqual(logits.shape, [1, 3, 32])
    }

    func testForwardPassWithGroupedNormProducesLogitsShape() throws {
        let model = SharpAIK2HorizonModel(
            try SharpAIK2HorizonTestFixtures.configuration(
                extra: ", \"rms_norm_groups\": 2"))
        let inputs = MLXArray([1, 2, 3] as [Int32]).reshaped(1, 3)

        let logits = model(inputs, cache: nil)
        eval(logits)

        XCTAssertEqual(logits.shape, [1, 3, 32])
    }

    func testForwardPassWithUntiedHeadUsesLMHead() throws {
        let model = SharpAIK2HorizonModel(
            try SharpAIK2HorizonTestFixtures.configuration(
                extra: ", \"tie_word_embeddings\": false"))
        XCTAssertNotNil(model.lmHead)

        let inputs = MLXArray([1, 2] as [Int32]).reshaped(1, 2)
        let logits = model(inputs, cache: nil)
        eval(logits)

        XCTAssertEqual(logits.shape, [1, 2, 32])
    }

    func testTiedModelHasNoLMHead() throws {
        let model = SharpAIK2HorizonModel(try SharpAIK2HorizonTestFixtures.configuration())
        XCTAssertNil(model.lmHead)
    }

    func testForwardPassUpdatesCacheOncePerLayer() throws {
        let model = SharpAIK2HorizonModel(try SharpAIK2HorizonTestFixtures.configuration())
        let cache = try model.newCache(parameters: nil)
        let tokenCount = 5
        let inputs = MLXArray(Array(Int32(1) ... Int32(tokenCount))).reshaped(1, tokenCount)

        let logits = model(inputs, cache: cache)
        eval(logits)

        XCTAssertEqual(cache.count, 2)
        for (i, layerCache) in cache.enumerated() {
            XCTAssertEqual(
                layerCache.offset, tokenCount,
                "layer \(i) cache advanced by \(layerCache.offset) for \(tokenCount) tokens")
        }
    }

    func testGroupedNormNormalizesEachGroupIndependently() throws {
        let norm = SharpAIK2HorizonGroupedRMSNorm(dimensions: 4, groups: 2, eps: 0)
        let x = MLXArray([1, 2, 3, 4] as [Float]).reshaped(1, 4)

        let y = norm(x)
        eval(y)

        // Group 0 is [1, 2] with mean square 2.5. Group 1 is [3, 4] with mean square 12.5.
        let expected: [Float] = [
            1 / Float(2.5).squareRoot(),
            2 / Float(2.5).squareRoot(),
            3 / Float(12.5).squareRoot(),
            4 / Float(12.5).squareRoot(),
        ]
        let actual = y.reshaped(-1).asArray(Float.self)
        XCTAssertEqual(actual.count, expected.count)
        for (a, e) in zip(actual, expected) {
            XCTAssertEqual(a, e, accuracy: 1e-5)
        }
    }

    func testGroupedNormWithOneGroupIsFullWidthRMSNorm() throws {
        let norm = SharpAIK2HorizonGroupedRMSNorm(dimensions: 4, groups: 1, eps: 0)
        let x = MLXArray([1, 2, 3, 4] as [Float]).reshaped(1, 4)

        let y = norm(x)
        eval(y)

        // Mean square over all four values is 7.5.
        let scale = 1 / Float(7.5).squareRoot()
        let expected: [Float] = [1 * scale, 2 * scale, 3 * scale, 4 * scale]
        let actual = y.reshaped(-1).asArray(Float.self)
        for (a, e) in zip(actual, expected) {
            XCTAssertEqual(a, e, accuracy: 1e-5)
        }
    }
}
