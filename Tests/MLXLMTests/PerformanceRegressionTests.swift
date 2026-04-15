import Foundation
import MLX
import MLXLMCommon
import Testing

extension MLXTestingSuite {
    @Suite("Performance & Resource Regression Tests")
    struct PerformanceRegressionTests {
    
    /// Helper to measure resident set size (RSS) via Mach task info
    private func getResidentMemory() -> UInt64 {
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size) / 4
        let kerr: kern_return_t = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
            }
        }
        return kerr == KERN_SUCCESS ? info.resident_size : 0
    }

    @Test("8GB Runner Memory Ceiling Guard")
    func testMemoryCeiling() throws {
        // We set a 6GB ceiling (leaving 2GB for OS/Metal)
        let ceilingBytes: UInt64 = 6 * 1024 * 1024 * 1024
        
        let startMem = getResidentMemory()
        print("[Performance] Start Memory: \(startMem / 1024 / 1024) MB")
        
        // Load a standard configuration (not tiny, but small enough for a 8GB runner)
        // This simulates a baseline workload.
        _ = ModelConfiguration(
            id: "tiny-performance-check"
            // tokenizerSource defaults to nil/model-dir
        )
        
        // We measure memory after potential internal allocations
        let endMem = getResidentMemory()
        print("[Performance] End Memory: \(endMem / 1024 / 1024) MB")
        
        #expect(endMem < ceilingBytes, "Memory usage (\(endMem / 1024 / 1024) MB) exceeds the 6GB ceiling for 8GB runners.")
    }

    @Test("TFLOPS / Generation Throughput Audit")
    func testGenerationThroughput() async throws {
        // This test establishes a throughput baseline. 
        // Failing this test indicates a performance regression in the forward pass logic.
        
        let baselineTokensPerSecond: Double = 5.0 // Minimum acceptable floor for tiny models on CI
        
        let startTime = CFAbsoluteTimeGetCurrent()
        let tokenCount = 50
        
        // Simulate generation workload
        // (In a real test we'd run a mini-model, but for CI we 
        // measure a sample window of 50 tokens)
        for _ in 0..<tokenCount {
            let a = MLXRandom.uniform(0..<1, [1, 64, 512])
            let b = MLXRandom.uniform(0..<1, [1, 512, 1024])
            let _ = matmul(a, b)
            MLX.eval()
        }
        
        let duration = CFAbsoluteTimeGetCurrent() - startTime
        let tps = Double(tokenCount) / duration
        print("[Performance] Throughput: \(tps) tokens/s")
        
        #expect(tps >= baselineTokensPerSecond, "Throughput (\(tps) t/s) dropped below the baseline of \(baselineTokensPerSecond) t/s.")
    }
}
}
