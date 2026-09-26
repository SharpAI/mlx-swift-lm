import MLX
import MLXLMTestSupport
import XCTest

/// The numeric tests compare against float32 references, so float32 matmuls
/// must stay float32 on GPUs that would otherwise run them as TF32.
final class Float32PrecisionTests: XCTestCase {
    func testGPUFloat32MatmulMatchesCPU() {
        XCTAssertEqual(mlxlm_float32_matmul_pinned(), 1)

        MLXRandom.seed(0)
        // K < M, N so MLX uses the regular GEMM, which is the path that goes
        // to TF32 (a square 256^3 matmul takes the full-precision split-K path).
        let a = MLXRandom.normal([512, 256])
        let b = MLXRandom.normal([256, 512])
        let gpu = matmul(a, b, stream: .gpu)
        let cpu = matmul(a, b, stream: .cpu)

        // TF32 keeps 10 mantissa bits, which puts this near 1e-3.
        let error = (abs(gpu - cpu).max() / abs(cpu).max()).item(Float.self)
        XCTAssertLessThan(error, 1e-5, "relative error \(error)")
    }
}
