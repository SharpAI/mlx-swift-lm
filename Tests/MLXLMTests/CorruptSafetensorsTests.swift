import Foundation
import MLX
import MLXLMCommon
import Testing

@Suite
struct CorruptSafetensorsTests {
    @Test
    func testDeadlock() throws {
        let tempDir = FileManager.default.temporaryDirectory
        let pathUrl = tempDir.appendingPathComponent("dummy_corrupt_\(UUID().uuidString).safetensors")
        let path = pathUrl.path
        
        let dict: [String: MLXArray] = ["weight": MLXArray.zeros([256, 1024])]
        try MLX.save(arrays: dict, url: pathUrl)
        
        defer {
            try? FileManager.default.removeItem(at: pathUrl)
        }

        let fd = open(path, O_RDWR)
        #expect(fd != -1)
        guard fd != -1 else { return }
        
        defer {
            let closeResult = close(fd)
            #expect(closeResult == 0)
        }

        let truncateResult = ftruncate(fd, 100)
        #expect(truncateResult == 0)
        guard truncateResult == 0 else { return }

        let dst = MLXArray.zeros([256, 1024])
        dst.eval()

        let errState = ThreadSafeError()

        DispatchQueue.concurrentPerform(iterations: 16) { i in
            errState.catchError {
                MLXFast.preadIntoOffset(dst, safetensorsPath: path, tensorName: "weight", expertIndex: UInt32(i), dstOffset: i * 1024 * 4)
            }
        }

        #expect(errState.error != nil)
    }
}
