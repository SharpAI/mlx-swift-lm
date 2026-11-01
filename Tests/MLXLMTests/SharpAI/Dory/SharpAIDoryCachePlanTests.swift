// SharpAI fork overlay tests for the Dory per-loop KV-cache plan. Not part of upstream.

import Foundation
import XCTest

@testable import MLXLLM

final class SharpAIDoryCachePlanTests: XCTestCase {

    private func configuration(loops: Int) throws -> SharpAIDoryConfiguration {
        let json = SharpAIDoryTestFixture.json.replacingOccurrences(
            of: "\"n_recurrent_loops\": 2", with: "\"n_recurrent_loops\": \(loops)")
        return try JSONDecoder.json5().decode(
            SharpAIDoryConfiguration.self, from: Data(json.utf8))
    }

    func testSlotCountCountsRecurrentAttentionOncePerLoop() throws {
        let slots = try SharpAIDoryCachePlan.slots(for: configuration(loops: 2))

        // 8 input attention + 20 recurrent attention x 2 loops + 8 output attention.
        XCTAssertEqual(slots.count, 56)
    }

    func testOneLoopMatchesAttentionLayerCount() throws {
        let slots = try SharpAIDoryCachePlan.slots(for: configuration(loops: 1))

        // 36 attention sublayers in total, one cache each when there is a single loop.
        XCTAssertEqual(slots.count, 36)
    }

    func testStageCountsAndLoopIndices() throws {
        let slots = try SharpAIDoryCachePlan.slots(for: configuration(loops: 2))

        XCTAssertEqual(slots.filter { $0.stage == .input }.count, 8)
        XCTAssertEqual(slots.filter { $0.stage == .output }.count, 8)

        let recurrent = slots.filter { $0.stage == .recurrent }
        XCTAssertEqual(recurrent.count, 40)
        XCTAssertEqual(recurrent.filter { $0.loop == 0 }.count, 20)
        XCTAssertEqual(recurrent.filter { $0.loop == 1 }.count, 20)

        XCTAssertTrue(slots.filter { $0.stage != .recurrent }.allSatisfy { $0.loop == 0 })
    }

    func testKindCountsMatchFixtureLayout() throws {
        let slots = try SharpAIDoryCachePlan.slots(for: configuration(loops: 2))

        // Nine full-attention sublayers in the checkpoint. Five of them are recurrent, so they
        // gain an extra cache for the second loop: 9 + 5 = 14 full slots.
        XCTAssertEqual(slots.filter { $0.kind == .full }.count, 14)
        XCTAssertEqual(slots.filter { $0.kind == .sliding }.count, 42)
    }

    func testSublayersAreTheAttentionIndicesOnly() throws {
        let slots = try SharpAIDoryCachePlan.slots(for: configuration(loops: 2))

        for slot in slots {
            XCTAssertEqual(slot.sublayer % 2, 0, "mlp sublayers sit at odd indices")
        }
    }
}
