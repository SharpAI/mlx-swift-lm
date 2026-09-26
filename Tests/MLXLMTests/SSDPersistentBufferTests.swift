import MLXLMCommon
import XCTest

final class SSDPersistentBufferTests: XCTestCase {
    /// Decode routes top_k = 8 slots, then an MTP verify step routes 16. The buffers
    /// must grow to 16 and stay there, or slot lookups for the 9th expert trap.
    func testBuffersRegrowForLargerCalls() {
        var allocated: Int? = nil
        var capacities = [Int]()
        for needed in [8, 16, 8, 32] {
            let plan = persistentBufferPlan(allocated: allocated, needed: needed)
            XCTAssertGreaterThanOrEqual(plan.capacity, needed)
            if allocated == nil || plan.regrow { allocated = plan.capacity }
            capacities.append(plan.capacity)
        }
        XCTAssertEqual(capacities, [8, 16, 16, 32])
    }
}
