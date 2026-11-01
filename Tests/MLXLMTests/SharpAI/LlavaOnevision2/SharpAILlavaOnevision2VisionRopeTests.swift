// SharpAI fork overlay tests for the llava_onevision2 vision rotary embedding. Not part of upstream.

import Foundation
import MLX
import XCTest

@testable import MLXVLM

final class SharpAILlavaOnevision2VisionRopeTests: XCTestCase {

    func testRotateHalfSwapsHalvesWithSignFlip() {
        let x = MLXArray([Float(1), 2, 3, 4, 5, 6, 7, 8], [1, 1, 1, 8])
        let rotated = SharpAILlavaOnevision2VisionRope.rotateHalf(x)
        XCTAssertEqual(
            rotated.reshaped(-1).asArray(Float.self), [-5, -6, -7, -8, 1, 2, 3, 4])
    }

    func testRotationPreservesNormWithRepeatedTable() {
        let headDim = 32
        let (cos, sin) = SharpAILlavaOnevision2VisionRope.cosSin(
            positions: [(t: 1, h: 3, w: 5)], headDim: headDim, base: 10000)
        let values = (0 ..< headDim).map { Float($0) * 0.37 - 4.0 }
        let x = MLXArray(values, [1, 1, 1, headDim])
        let out =
            x * cos.reshaped(1, 1, 1, headDim)
            + SharpAILlavaOnevision2VisionRope.rotateHalf(x) * sin.reshaped(1, 1, 1, headDim)

        let before = (x * x).sum().item(Float.self)
        let after = (out * out).sum().item(Float.self)
        XCTAssertEqual(before, after, accuracy: 1e-2)
    }
}
