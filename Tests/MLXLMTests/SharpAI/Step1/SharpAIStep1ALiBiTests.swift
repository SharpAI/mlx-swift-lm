// SharpAI fork overlay tests for the step1 sqrt ALiBi bias. Not part of upstream.
// Expected values come from build_alibi_cache in stepfun-ai/Step-Audio-EditX modeling_step1.py:
// slopes 2^(-8k/n), and bias -slope * sqrt(i - j) for j <= i, -inf for j > i.

import Foundation
import MLX
import XCTest

@testable import MLXVLM

final class SharpAIStep1ALiBiTests: XCTestCase {

    private func assertSlopes(
        _ actual: [Float], _ expected: [Float],
        file: StaticString = #filePath, line: UInt = #line
    ) {
        XCTAssertEqual(actual.count, expected.count, file: file, line: line)
        for (a, e) in zip(actual, expected) {
            XCTAssertEqual(a, e, accuracy: e * 1e-6, file: file, line: line)
        }
    }

    func testSlopesForPowerOfTwoHeads() {
        // n = 4, m0 = 2^(-8/4) = 0.25, so slope_k = 0.25^k.
        assertSlopes(
            SharpAIStep1ALiBi.slopes(heads: 4),
            [0.25, 0.0625, 0.015625, 0.00390625])
    }

    func testSlopesForNonPowerOfTwoHeads() {
        // heads = 6: n = 4 gives 0.25^k for k = 1...4. The two extra heads use
        // m1 = 2^(-4/4) = 0.5 raised to 1 and 3.
        assertSlopes(
            SharpAIStep1ALiBi.slopes(heads: 6),
            [0.25, 0.0625, 0.015625, 0.00390625, 0.5, 0.125])
    }

    func testBiasIsNegativeSqrtDistanceTimesSlopeAndMasksFuture() {
        // heads = 1: slope is 2^(-8) = 1/256. Shape [1, 1, 3, 3]; flat index is i * 3 + j.
        let bias = SharpAIStep1ALiBi.bias(queryLength: 3, keyLength: 3, heads: 1, dtype: .float32)
        XCTAssertEqual(bias.shape, [1, 1, 3, 3])
        let v = bias.asArray(Float.self)
        let slope: Float = 1.0 / 256

        // Diagonal: distance 0, so the bias is 0.
        XCTAssertEqual(v[0], 0, accuracy: 1e-7)
        XCTAssertEqual(v[4], 0, accuracy: 1e-7)
        XCTAssertEqual(v[8], 0, accuracy: 1e-7)

        // Below the diagonal: -slope * sqrt(i - j).
        XCTAssertEqual(v[3], -slope, accuracy: 1e-7)
        XCTAssertEqual(v[6], -Float(2).squareRoot() * slope, accuracy: 1e-7)
        XCTAssertEqual(v[7], -slope, accuracy: 1e-7)

        // Above the diagonal: masked to -inf.
        for flat in [1, 2, 5] {
            XCTAssertTrue(v[flat].isInfinite && v[flat] < 0, "index \(flat) should be -inf")
        }
    }

    func testDecodeStepUsesLastQueryRowOfKeyLengthBias() {
        // One query at position 2 against 3 keys, four heads. Head h uses slope_h, and the
        // flat index is h * 3 + j.
        let bias = SharpAIStep1ALiBi.bias(queryLength: 1, keyLength: 3, heads: 4, dtype: .float32)
        XCTAssertEqual(bias.shape, [1, 4, 1, 3])
        let v = bias.asArray(Float.self)
        let slopes = SharpAIStep1ALiBi.slopes(heads: 4)

        for (h, slope) in slopes.enumerated() {
            XCTAssertEqual(v[h * 3 + 0], -Float(2).squareRoot() * slope, accuracy: 1e-7)
            XCTAssertEqual(v[h * 3 + 1], -slope, accuracy: 1e-7)
            XCTAssertEqual(v[h * 3 + 2], 0, accuracy: 1e-7)
        }
    }
}
