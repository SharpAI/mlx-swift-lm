// SharpAI fork overlay tests for the UltraBERT text tower. Not part of upstream.
//
// Offline checks only: tiny random-init model, no weights, no network.

import Foundation
import MLX
import XCTest

@testable import MLXVLM

final class SharpAIUltravbertTextModelTests: XCTestCase {

    private func tinyModel() -> SharpAIUltravbertTextModel {
        SharpAIUltravbertTextModel(
            SharpAIUltravbertTextConfiguration(
                vocabularySize: 64, hiddenSize: 16, intermediateSize: 32, hiddenLayers: 6,
                attentionHeads: 2, slidingWindow: 2))
    }

    /// Source rule: layers with (index + 1) % 3 == 0 are full attention (0-based 2, 5, ...).
    func testLayerTypeCycleMatchesSource() {
        let types = (0 ..< 6).map(SharpAIUltravbertTextModel.isGlobalLayer)

        XCTAssertEqual(types, [false, false, true, false, false, true])
    }

    /// Padded positions must not change the hidden states of real tokens, in both the
    /// sliding and the full layers.
    func testPaddedTokensDoNotAffectRealTokens() {
        let model = tinyModel()
        let mask = MLXArray([true, true, true, false, false]).reshaped(1, 5)
        let idsA = MLXArray([Int32(1), Int32(2), Int32(3), Int32(4), Int32(5)]).reshaped(1, 5)
        let idsB = MLXArray([Int32(1), Int32(2), Int32(3), Int32(9), Int32(9)]).reshaped(1, 5)

        let outA = model(idsA, paddingMask: mask)[0, 0 ..< 3].asArray(Float.self)
        let outB = model(idsB, paddingMask: mask)[0, 0 ..< 3].asArray(Float.self)

        XCTAssertEqual(outA.count, outB.count)
        for (a, b) in zip(outA, outB) {
            XCTAssertEqual(a, b, accuracy: 1e-5)
        }
    }
}
