import Foundation
import MLX
import MLXLMCommon
import Testing

@Suite
struct CorruptSafetensorsTests {
    @Test
    func testDeadlock() throws {
        let path = "dummy_corrupt.safetensors"
        let dict: [String: MLXArray] = ["weight": MLXArray.zeros([256, 1024])]
        try MLX.save(arrays: dict, url: URL(fileURLWithPath: path))

        let fd = open(path, O_RDWR)
        ftruncate(fd, 100)
        close(fd)

        let dst = MLXArray.zeros([256, 1024])
        dst.eval()

        final class ThreadSafeError: @unchecked Sendable {
            let lock = NSLock()
            var error: Error?
            func catchError(_ block: () throws -> Void) {
                do {
                    try MLX.withError {
                        try block()
                    }
                } catch {
                    lock.withLock {
                        if self.error == nil { self.error = error }
                    }
                }
            }
        }

        let errState = ThreadSafeError()

        DispatchQueue.concurrentPerform(iterations: 16) { i in
            errState.catchError {
                MLXFast.preadIntoOffset(dst, safetensorsPath: path, tensorName: "weight", expertIndex: UInt32(i), dstOffset: i * 1024 * 4)
            }
        }

        #expect(errState.error != nil)
    }
}
