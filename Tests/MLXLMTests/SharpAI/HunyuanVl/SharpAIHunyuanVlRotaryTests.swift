// SharpAI fork overlay tests for hunyuan_vl rotary recomposition. Not part of upstream.

import Foundation
import MLX
import XCTest

@testable import MLXVLM

final class SharpAIHunyuanVlRotaryTests: XCTestCase {

    func testRotateHalfPartnersShareAxis() {
        // Four axes with distinct positions: any pairing of channel k with k + d/2 taken from
        // different axes shows up as cos/sin that differ between the two halves.
        let section = [2, 2, 2, 2]
        let headDim = 16
        let invFreq = SharpAIHunyuanVlRotary.inverseFrequencies(
            headDim: headDim, theta: 10000, alpha: nil)
        let positions = MLXArray([1, 2, 3, 4] as [Float]).reshaped(4, 1, 1)
        let (cosTable, sinTable) = SharpAIHunyuanVlRotary.cosSin(
            positions: positions, inverseFrequencies: invFreq, section: section)

        XCTAssertEqual(cosTable.shape, [1, 1, headDim])
        let half = headDim / 2
        XCTAssertTrue(
            allClose(cosTable[.ellipsis, 0 ..< half], cosTable[.ellipsis, half ..< headDim])
                .item(Bool.self))
        XCTAssertTrue(
            allClose(sinTable[.ellipsis, 0 ..< half], sinTable[.ellipsis, half ..< headDim])
                .item(Bool.self))
    }

    func testChunksComeFromAxisInOrder() {
        let section = [2, 2]
        let headDim = 8
        let invFreq = SharpAIHunyuanVlRotary.inverseFrequencies(
            headDim: headDim, theta: 10000, alpha: nil)
        let positions = MLXArray([3, 5] as [Float]).reshaped(2, 1, 1)
        let (cosTable, _) = SharpAIHunyuanVlRotary.cosSin(
            positions: positions, inverseFrequencies: invFreq, section: section)

        let values = cosTable.reshaped(-1).asArray(Float.self)
        let expected: [Float] = (0 ..< headDim).map { channel in
            let axis = (channel / 2) % 2
            let freq = invFreq[channel % (headDim / 2)]
            return cosf([3, 5][axis] * freq)
        }
        for (a, b) in zip(values, expected) {
            XCTAssertEqual(a, b, accuracy: 1e-5)
        }
    }
}
