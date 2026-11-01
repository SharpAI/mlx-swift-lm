// SharpAI fork overlay tests for alice_ai attention residuals. Not part of upstream.

import Foundation
import MLX
import MLXLLM
import XCTest

final class SharpAIAliceAiAttnResTests: XCTestCase {

    func testSingleSourceIsReturnedUnchanged() {
        let x = MLXArray([1, -2, 3] as [Float])
        let out = sharpAIAliceAiAttnRes(
            sources: [x], normWeight: MLXArray.ones([3]), qkWeight: MLXArray.zeros([3]),
            eps: 1e-6)
        SharpAIAliceAiTestSupport.assertClose(out.asArray(Float.self), [1, -2, 3])
    }

    func testZeroQueryGivesUniformAverage() {
        let a = MLXArray([1, 0] as [Float])
        let b = MLXArray([0, 1] as [Float])
        let out = sharpAIAliceAiAttnRes(
            sources: [a, b], normWeight: MLXArray.ones([2]), qkWeight: MLXArray.zeros([2]),
            eps: 1e-6)
        SharpAIAliceAiTestSupport.assertClose(out.asArray(Float.self), [0.5, 0.5])
    }

    /// a = [1, 0], b = [0, 1], query = [1, -1]: logits are +sqrt(2) and -sqrt(2).
    func testWeightsMatchHandComputedSoftmax() {
        let a = MLXArray([1, 0] as [Float])
        let b = MLXArray([0, 1] as [Float])
        let out = sharpAIAliceAiAttnRes(
            sources: [a, b], normWeight: MLXArray.ones([2]),
            qkWeight: MLXArray([1, -1] as [Float]), eps: 1e-6)

        let pA = Float(1 / (1 + exp(-2 * 2.0.squareRoot())))
        SharpAIAliceAiTestSupport.assertClose(out.asArray(Float.self), [pA, 1 - pA], tolerance: 1e-4)
    }

    func testNormWeightScalesTheLogits() {
        // Doubling norm * qk doubles both logits, which sharpens the softmax.
        let a = MLXArray([1, 0] as [Float])
        let b = MLXArray([0, 1] as [Float])
        let qk = MLXArray([1, -1] as [Float])
        let base = sharpAIAliceAiAttnRes(
            sources: [a, b], normWeight: MLXArray.ones([2]), qkWeight: qk, eps: 1e-6)
        let scaled = sharpAIAliceAiAttnRes(
            sources: [a, b], normWeight: MLXArray([2, 2] as [Float]), qkWeight: qk, eps: 1e-6)
        let baseA = base.asArray(Float.self)[0]
        let scaledA = scaled.asArray(Float.self)[0]
        XCTAssertGreaterThan(scaledA, baseA)
    }
}
