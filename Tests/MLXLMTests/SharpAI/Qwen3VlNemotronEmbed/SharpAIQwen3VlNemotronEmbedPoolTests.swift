import MLX
import XCTest

@testable import MLXVLM

/// Hand-computed pooling values. Hidden rows are [1,0], [0,1], [5,5].
final class SharpAIQwen3VlNemotronEmbedPoolTests: XCTestCase {

    private let hidden = MLXArray([Float(1), 0, 0, 1, 5, 5]).reshaped([1, 3, 2])

    func testLastTokenWithRightPadding() {
        let mask = MLXArray([Int32(1), 1, 0]).reshaped([1, 3])
        let out = SharpAIQwen3VlNemotronEmbedPool.pool(hidden, mask: mask, mode: .last)
        eval(out)

        XCTAssertEqual(out.shape, [1, 2])
        XCTAssertEqual(out.asArray(Float.self), [0, 1])
    }

    func testLastTokenWithLeftPadding() {
        let mask = MLXArray([Int32(0), 1, 1]).reshaped([1, 3])
        let out = SharpAIQwen3VlNemotronEmbedPool.pool(hidden, mask: mask, mode: .last)
        eval(out)

        XCTAssertEqual(out.asArray(Float.self), [5, 5])
    }

    func testLastTokenOfAllPaddingIsZero() {
        let mask = MLXArray([Int32(0), 0, 0]).reshaped([1, 3])
        let out = SharpAIQwen3VlNemotronEmbedPool.pool(hidden, mask: mask, mode: .last)
        eval(out)

        XCTAssertEqual(out.asArray(Float.self), [0, 0])
    }

    func testMeanOverRealTokensWithRightPadding() {
        let mask = MLXArray([Int32(1), 1, 0]).reshaped([1, 3])
        let out = SharpAIQwen3VlNemotronEmbedPool.pool(hidden, mask: mask, mode: .mean)
        eval(out)

        let values = out.asArray(Float.self)
        XCTAssertEqual(values[0], 0.5, accuracy: 1e-5)
        XCTAssertEqual(values[1], 0.5, accuracy: 1e-5)
    }

    func testMeanOverRealTokensWithLeftPadding() {
        let mask = MLXArray([Int32(0), 1, 1]).reshaped([1, 3])
        let out = SharpAIQwen3VlNemotronEmbedPool.pool(hidden, mask: mask, mode: .mean)
        eval(out)

        let values = out.asArray(Float.self)
        XCTAssertEqual(values[0], 2.5, accuracy: 1e-5)
        XCTAssertEqual(values[1], 3, accuracy: 1e-5)
    }
}
