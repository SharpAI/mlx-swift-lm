import Foundation
import XCTest

@testable import MLXLLM

final class Qwen35MoeFlushGateTests: XCTestCase {

    private func gate(
        moe: Bool = true, streaming: Bool = true, tokens: Int, flushDecode: Bool = false
    ) -> Bool {
        Qwen35DecoderLayer.needsMoeFlush(
            isSparseMoE: moe, streamingExperts: streaming, tokenCount: tokens,
            flushDecode: flushDecode)
    }

    func testPrefillFlushesWhenStreamingMoE() {
        XCTAssertTrue(gate(tokens: 2))
        XCTAssertTrue(gate(tokens: 512))
    }

    func testSingleTokenDecodeSkipsFlush() {
        XCTAssertFalse(gate(tokens: 1))
    }

    func testDecodeFlushOverrideRestoresOldBehaviour() {
        XCTAssertTrue(gate(tokens: 1, flushDecode: true))
    }

    func testNeverFlushesWithoutStreamingOrWithDenseMLP() {
        XCTAssertFalse(gate(streaming: false, tokens: 512))
        XCTAssertFalse(gate(streaming: false, tokens: 1, flushDecode: true))
        XCTAssertFalse(gate(moe: false, tokens: 512))
        XCTAssertFalse(gate(moe: false, tokens: 1, flushDecode: true))
    }
}
