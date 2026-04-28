import Foundation
import MLX
import MLXLMCommon
import Testing

@Suite
struct CorruptSafetensorsTests {
    @Test
    func testThreadSafeErrorCheckPublishesToActiveLatch() throws {
        let latch = SSDStreamingErrorLatch()

        SSDStreamingErrorLatch.withActive(latch) {
            let errState = ThreadSafeError()
            errState.catchError {
                throw NSError(domain: "CorruptSafetensorsTests", code: 13, userInfo: [
                    NSLocalizedDescriptionKey: "truncated shard"
                ])
            }

            let latched = errState.check()
            #expect(latched != nil)
        }

        do {
            try latch.throwIfSet()
            Issue.record("Expected latch.throwIfSet() to surface an SSDStreamingError")
        } catch let error as SSDStreamingError {
            #expect(error.localizedDescription.contains("truncated shard"))
        } catch {
            Issue.record("Unexpected error type: \(error)")
        }
    }

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
        
        struct SendableArray: @unchecked Sendable {
            let array: MLXArray
        }
        let sendableDst = SendableArray(array: dst)

        let errState = ThreadSafeError()

        DispatchQueue.concurrentPerform(iterations: 16) { i in
            errState.catchError {
                MLXFast.preadIntoOffset(sendableDst.array, safetensorsPath: path, tensorName: "weight", expertIndex: UInt32(i), dstOffset: i * 1024 * 4)
            }
        }

        #expect(errState.error != nil)
    }
}
