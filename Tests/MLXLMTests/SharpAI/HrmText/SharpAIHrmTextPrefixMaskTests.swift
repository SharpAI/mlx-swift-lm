// SharpAI fork overlay tests for the generic prefix-LM mask. Not part of upstream.

import Foundation
import MLX
import MLXLLM
import XCTest

final class SharpAIHrmTextPrefixMaskTests: XCTestCase {

    private func values(_ mask: MLXArray) -> [Bool] {
        XCTAssertEqual(mask.dtype, .bool)
        return mask.asArray(Bool.self)
    }

    func testPrefixOfTwoInLengthFour() {
        let mask = SharpAIHrmTextPrefixMask.prefixLMMask(length: 4, prefixLength: 2)

        XCTAssertEqual(mask.shape, [4, 4])
        XCTAssertEqual(
            values(mask),
            [
                true, true, false, false,
                true, true, false, false,
                true, true, true, false,
                true, true, true, true,
            ])
    }

    func testZeroPrefixIsCausal() {
        let mask = SharpAIHrmTextPrefixMask.prefixLMMask(length: 3, prefixLength: 0)

        XCTAssertEqual(
            values(mask),
            [
                true, false, false,
                true, true, false,
                true, true, true,
            ])
    }

    func testFullPrefixIsAllTrue() {
        let mask = SharpAIHrmTextPrefixMask.prefixLMMask(length: 3, prefixLength: 3)

        XCTAssertEqual(values(mask), Array(repeating: true, count: 9))
    }

    func testPrefixLengthIsClamped() {
        let over = SharpAIHrmTextPrefixMask.prefixLMMask(length: 2, prefixLength: 10)
        let under = SharpAIHrmTextPrefixMask.prefixLMMask(length: 2, prefixLength: -1)

        XCTAssertEqual(values(over), Array(repeating: true, count: 4))
        XCTAssertEqual(values(under), [true, false, true, true])
    }

    func testPerStepCachesKnownIssueSkipped() {
        // Decode step after a 3-token prefill: query i sits at absolute position 3 + i and sees
        // keys 0...(3 + i). The prefix block is not applied once the cache is initialized
        // (HF: token_type_ids only count when the cache is not initialized), so the block ids
        // are ignored here and the rows are purely causal.
        let mask = SharpAIHrmTextPrefixMask.attentionMask(
            queryLength: 2, cacheOffset: 3, blockIds: [0, 0])

        XCTAssertEqual(mask.shape, [2, 5])
        XCTAssertEqual(
            values(mask),
            [
                true, true, true, true, false,
                true, true, true, true, true,
            ])
    }

    func testPrefillBlockMaskFollowsTokenTypeIds() {
        // token_type_ids == 1 marks the block (block id 0). Other positions are causal (-1).
        // Positions 0 and 2 are in the block; 1 and 3 are not.
        let mask = SharpAIHrmTextPrefixMask.attentionMask(
            queryLength: 4, cacheOffset: 0, blockIds: [0, -1, 0, -1])

        // Row 0 is in the block with position 2, so it sees key 2 even though 2 is in its future.
        XCTAssertEqual(
            values(mask),
            [
                true, false, true, false,
                true, true, false, false,
                true, true, true, false,
                true, true, true, true,
            ])
    }
}
