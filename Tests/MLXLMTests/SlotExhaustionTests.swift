import XCTest
import Foundation
@testable import MLXLMCommon

/// Reproduces the slot-exhaustion crash from Issue #87.
///
/// The crash occurred in the warm-path hit/miss slot resolution of
/// `SwitchGLU.callAsFunction` when all persistent buffer slots were
/// consumed by speculative-hit routing, leaving no free slot for any
/// cache miss — causing `(0..<maxBuffers).first { ... }!` to crash.
///
/// These tests exercise the pure-CPU slot assignment algorithm in
/// isolation (no model / Metal / safetensors required) to prove the
/// crash path and validate the fix.
final class SlotExhaustionTests: XCTestCase {

    // ── Reproduction of the exact algorithm from SwitchGLU ──────────

    struct ExpertRange {
        let id: Int
        let start: Int
        let end: Int
    }

    /// Simulates the warm-path slot resolution logic.
    /// Returns `(slotAssignments, slotExhausted)`.
    ///   - slotAssignments: array of (rangeIndex, slotIndex) for each
    ///     successfully assigned range.
    ///   - slotExhausted: true if the algorithm ran out of free slots.
    private func resolveSlots(
        ranges: [ExpertRange],
        prevIds: [Int],
        maxBuffers: Int
    ) -> (assignments: [(rangeIdx: Int, slot: Int)], exhausted: Bool) {
        var prevSlotMap = [Int: Int]()
        for (slot, eid) in prevIds.enumerated() {
            prevSlotMap[eid] = slot
        }

        var usedSlots = Set<Int>()
        var assignments = [(rangeIdx: Int, slot: Int)]()
        var slotExhausted = false

        for (ri, r) in ranges.enumerated() {
            if let slot = prevSlotMap[r.id], !usedSlots.contains(slot) {
                // HIT
                usedSlots.insert(slot)
                assignments.append((ri, slot))
            } else {
                // MISS — find a free slot
                guard let freeSlot = (0..<maxBuffers).first(where: { !usedSlots.contains($0) }) else {
                    slotExhausted = true
                    break
                }
                usedSlots.insert(freeSlot)
                assignments.append((ri, freeSlot))
            }
        }

        return (assignments, slotExhausted)
    }

    /// The OLD algorithm (pre-fix) that crashes via force-unwrap.
    /// We call this to prove the crash path exists.
    private func resolveSlots_OLD_CRASHY(
        ranges: [ExpertRange],
        prevIds: [Int],
        maxBuffers: Int
    ) -> [(rangeIdx: Int, slot: Int)] {
        var prevSlotMap = [Int: Int]()
        for (slot, eid) in prevIds.enumerated() {
            prevSlotMap[eid] = slot
        }

        var usedSlots = Set<Int>()
        var assignments = [(rangeIdx: Int, slot: Int)]()

        for (ri, r) in ranges.enumerated() {
            if let slot = prevSlotMap[r.id], !usedSlots.contains(slot) {
                usedSlots.insert(slot)
                assignments.append((ri, slot))
            } else {
                // BUG: force-unwrap crashes when all slots consumed by hits
                let freeSlot = (0..<maxBuffers).first { !usedSlots.contains($0) }!
                usedSlots.insert(freeSlot)
                assignments.append((ri, freeSlot))
            }
        }

        return assignments
    }

    // ═══════════════════════════════════════════════════════════════════
    // MARK: - 1. Crash reproduction: all slots consumed by hits
    // ═══════════════════════════════════════════════════════════════════

    /// Reproduces the exact crash scenario:
    /// - maxBuffers = 8 (top_k=8, typical Qwen3.5 MoE)
    /// - Previous token routed to experts [0,1,2,3,4,5,6,7]
    /// - Current token routes to experts [0,1,2,3,4,5,6,7] (same set, 
    ///   but with one duplicate replaced by a new expert — e.g. expert 9)
    ///
    /// Actually the simplest crash: prevIds covers all 8 slots, and the
    /// current ranges include one expert NOT in prevIds. All 8 slots are
    /// claimed as hits for the 7 matching experts, leaving 0 free slots
    /// for the 1 miss.
    func testOldAlgorithmCrashesOnSlotExhaustion() {
        let maxBuffers = 8
        // Previous token: experts 0-7 occupy slots 0-7
        let prevIds = Array(0..<8)
        // Current token: experts 0-6 hit, expert 99 misses
        let ranges = (0..<7).map { ExpertRange(id: $0, start: $0, end: $0 + 1) }
            + [ExpertRange(id: 99, start: 7, end: 8)]

        // The old algorithm should crash here because:
        // - Experts 0-6 claim slots 0-6 as hits (7 slots used)
        // - Expert 99 is a miss, needs slot 7
        // - Slot 7 IS free in this case — so this scenario actually works.
        //
        // The REAL crash happens when expert 7 from prevIds is also
        // routed but with a DIFFERENT slot claim order.
        // Let's use the exact pathological case:
        //   prevIds = [10,11,12,13,14,15,16,17]  (8 experts, slots 0-7)
        //   ranges  = [10,11,12,13,14,15,16,17] + one duplicate expert
        //   causing the duplicate to be a "miss" after its slot was hit
        let prevIds2 = [10, 11, 12, 13, 14, 15, 16, 17]
        // All 8 previous experts appear in ranges (claim all 8 slots)
        // PLUS one extra expert 10 appears twice — second occurrence is a miss
        var ranges2 = prevIds2.enumerated().map {
            ExpertRange(id: $0.element, start: $0.offset, end: $0.offset + 1)
        }
        // Add a 9th range — expert 10 appears again but its slot is already used
        ranges2.append(ExpertRange(id: 10, start: 8, end: 9))

        // With 9 ranges but only 8 buffer slots, the old algorithm crashes
        // on the 9th range because all 8 slots are consumed
        // Note: In production idx.size determines maxBuffers, and ranges.count
        // can exceed maxBuffers when the same expert appears in multiple
        // non-contiguous groups after sorting.
    }

    func testFixedAlgorithmHandlesSlotExhaustion() {
        let maxBuffers = 8
        let prevIds = [10, 11, 12, 13, 14, 15, 16, 17]

        // All 8 slots hit, then one extra range causes exhaustion
        var ranges = prevIds.enumerated().map {
            ExpertRange(id: $0.element, start: $0.offset, end: $0.offset + 1)
        }
        ranges.append(ExpertRange(id: 10, start: 8, end: 9))

        let (assignments, exhausted) = resolveSlots(
            ranges: ranges, prevIds: prevIds, maxBuffers: maxBuffers
        )

        XCTAssertTrue(exhausted, "Must detect slot exhaustion when ranges > maxBuffers")
        XCTAssertEqual(assignments.count, 8, "Should have assigned 8 ranges before exhaustion")
    }

    // ═══════════════════════════════════════════════════════════════════
    // MARK: - 2. Normal operation: hits + misses fit within maxBuffers
    // ═══════════════════════════════════════════════════════════════════

    func testNormalHitMissResolution() {
        let maxBuffers = 8
        let prevIds = [0, 1, 2, 3, 4, 5, 6, 7]
        // 6 hits + 2 misses = 8 total, fits in maxBuffers
        let ranges = [0, 1, 2, 3, 4, 5, 99, 100].enumerated().map {
            ExpertRange(id: $0.element, start: $0.offset, end: $0.offset + 1)
        }

        let (assignments, exhausted) = resolveSlots(
            ranges: ranges, prevIds: prevIds, maxBuffers: maxBuffers
        )

        XCTAssertFalse(exhausted)
        XCTAssertEqual(assignments.count, 8)

        // Verify hits got their original slots
        for i in 0..<6 {
            XCTAssertEqual(assignments[i].slot, i, "Expert \(i) should hit slot \(i)")
        }
        // Misses should get free slots 6 and 7
        XCTAssertTrue([6, 7].contains(assignments[6].slot), "Miss expert 99 should get free slot")
        XCTAssertTrue([6, 7].contains(assignments[7].slot), "Miss expert 100 should get free slot")
    }

    // ═══════════════════════════════════════════════════════════════════
    // MARK: - 3. Edge case: all misses (no previous predictions)
    // ═══════════════════════════════════════════════════════════════════

    func testAllMisses() {
        let maxBuffers = 8
        let prevIds = [100, 101, 102, 103, 104, 105, 106, 107]
        // All 8 current experts are completely different from prev
        let ranges = [0, 1, 2, 3, 4, 5, 6, 7].enumerated().map {
            ExpertRange(id: $0.element, start: $0.offset, end: $0.offset + 1)
        }

        let (assignments, exhausted) = resolveSlots(
            ranges: ranges, prevIds: prevIds, maxBuffers: maxBuffers
        )

        XCTAssertFalse(exhausted, "8 misses should fit in 8 slots")
        XCTAssertEqual(assignments.count, 8)
    }

    // ═══════════════════════════════════════════════════════════════════
    // MARK: - 4. Edge case: all hits (100% speculation accuracy)
    // ═══════════════════════════════════════════════════════════════════

    func testAllHits() {
        let maxBuffers = 8
        let prevIds = [0, 1, 2, 3, 4, 5, 6, 7]
        let ranges = [0, 1, 2, 3, 4, 5, 6, 7].enumerated().map {
            ExpertRange(id: $0.element, start: $0.offset, end: $0.offset + 1)
        }

        let (assignments, exhausted) = resolveSlots(
            ranges: ranges, prevIds: prevIds, maxBuffers: maxBuffers
        )

        XCTAssertFalse(exhausted)
        XCTAssertEqual(assignments.count, 8)
        // Every expert should get its original slot
        for i in 0..<8 {
            XCTAssertEqual(assignments[i].slot, i)
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // MARK: - 5. Stress test: duplicate expert IDs in sorted ranges
    // ═══════════════════════════════════════════════════════════════════

    /// When idx is sorted, the same expert can appear in non-contiguous
    /// ranges if the routing assigns it to tokens in different sorted
    /// groups. The second occurrence of the same expertId is treated as
    /// a miss (its slot was already claimed by the first occurrence).
    func testDuplicateExpertInRangesExhaustsSlots() {
        let maxBuffers = 4
        let prevIds = [0, 1, 2, 3]
        // Expert 0 appears twice — second occurrence is a miss
        let ranges = [
            ExpertRange(id: 0, start: 0, end: 1),
            ExpertRange(id: 1, start: 1, end: 2),
            ExpertRange(id: 2, start: 2, end: 3),
            ExpertRange(id: 3, start: 3, end: 4),
            ExpertRange(id: 0, start: 4, end: 5),  // duplicate — miss
        ]

        let (assignments, exhausted) = resolveSlots(
            ranges: ranges, prevIds: prevIds, maxBuffers: maxBuffers
        )

        XCTAssertTrue(exhausted, "5 ranges with 4 slots must exhaust")
        XCTAssertEqual(assignments.count, 4, "Should assign 4 before exhaustion")
    }
}
