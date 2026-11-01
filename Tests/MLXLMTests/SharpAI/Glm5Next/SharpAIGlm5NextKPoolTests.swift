// SharpAI fork overlay tests for the glm5_next kpool op. Not part of upstream.

import Foundation
import MLX
import MLXVLM
import XCTest

final class SharpAIGlm5NextKPoolTests: XCTestCase {

    private func assertClose(_ actual: [Float], _ expected: [Float], accuracy: Float = 1e-4) {
        XCTAssertEqual(actual.count, expected.count)
        for (a, e) in zip(actual, expected) {
            XCTAssertEqual(a, e, accuracy: accuracy)
        }
    }

    func testSylvesterHadamardOfFourMatchesNaturalOrder() {
        let h = SharpAIGlm5NextKPool.sylvesterHadamard(4).asArray(Float.self)
        assertClose(
            h,
            [
                1, 1, 1, 1,
                1, -1, 1, -1,
                1, 1, -1, -1,
                1, -1, -1, 1,
            ])
    }

    func testHadamardRotateOfOnesConcentratesInFirstEntry() {
        // H * 1 = n * e0, then scaled by 1 / sqrt(n): [sqrt(n), 0, ...].
        let rotated = SharpAIGlm5NextKPool.hadamardRotate(MLXArray.ones([4]))
        assertClose(rotated.asArray(Float.self), [2, 0, 0, 0])
    }

    func testHadamardRotatePreservesNorm() {
        let x = MLXArray([Float(3), 1, -2, 0.5], [4])
        let before = (x * x).sum().item(Float.self)
        let after = (SharpAIGlm5NextKPool.hadamardRotate(x) * SharpAIGlm5NextKPool.hadamardRotate(x))
            .sum().item(Float.self)
        XCTAssertEqual(after, before, accuracy: 1e-3)
    }

    func testCompressWithUniformScoresAveragesThenRotates() {
        // Uniform softmax over 3 tokens of ones gives ones, then the rotation gives [2, 0, 0, 0].
        let k = MLXArray.ones([3, 4])
        let score = MLXArray.zeros([3, 4])
        let ape = MLXArray.zeros([3, 4])
        let out = SharpAIGlm5NextKPool.compress(k: k, score: score, ape: ape)

        XCTAssertEqual(out.shape, [4])
        assertClose(out.asArray(Float.self), [2, 0, 0, 0])
    }
}
