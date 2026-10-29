// SL-28 A1: offline switch. Uses a stub Downloader (no real network) and a TaskLocal environment override,
// so no process env is mutated.

import Foundation
import MLXLMCommon
import Testing

private final class CallCounter: @unchecked Sendable {
    private let lock = NSLock()
    private var n = 0
    var count: Int { lock.lock(); defer { lock.unlock() }; return n }
    func bump() { lock.lock(); n += 1; lock.unlock() }
}

private struct StubDownloader: Downloader {
    let counter: CallCounter
    func download(
        id: String, revision: String?, matching patterns: [String], useLatest: Bool,
        progressHandler: @Sendable @escaping (Progress) -> Void
    ) async throws -> URL {
        counter.bump()
        return URL(fileURLWithPath: "/tmp/stub-\(id.replacingOccurrences(of: "/", with: "_"))")
    }
}

@Suite("ConformanceOfflineGuard")
struct ConformanceOfflineGuardTests {
    static let on = ["SWIFTLM_OFFLINE": "1"]

    @Test func envParsing() {
        #expect(!SharpAIOverlay.OfflineGuard.isOffline(environment: [:]))
        #expect(!SharpAIOverlay.OfflineGuard.isOffline(environment: ["SWIFTLM_OFFLINE": "0"]))
        #expect(!SharpAIOverlay.OfflineGuard.isOffline(environment: ["SWIFTLM_OFFLINE": ""]))
        #expect(SharpAIOverlay.OfflineGuard.isOffline(environment: ["SWIFTLM_OFFLINE": "1"]))
        #expect(SharpAIOverlay.OfflineGuard.isOffline(environment: ["HF_HUB_OFFLINE": "1"]))
        #expect(SharpAIOverlay.OfflineGuard.isOffline(environment: ["HF_HUB_OFFLINE": "TRUE"]))
    }

    @Test func resolveThrowsTypedErrorWhenOffline() async throws {
        let counter = CallCounter()
        let config = ModelConfiguration(id: "org/model")
        await SharpAIOverlay.OfflineGuard.$environmentOverride.withValue(Self.on) {
            do {
                _ = try await resolve(
                    configuration: config, from: StubDownloader(counter: counter),
                    useLatest: false, progressHandler: { _ in })
                Issue.record("expected networkBlocked, resolve succeeded")
            } catch let e as SharpAIOverlay.OfflineError {
                #expect(e == .networkBlocked(URL(string: "https://huggingface.co/org/model")!))
            } catch {
                Issue.record("wrong error type: \(error)")
            }
        }
        #expect(counter.count == 0)
    }

    @Test func resolveForModelConversionThrowsWhenOffline() async throws {
        let counter = CallCounter()
        let config = ModelConfiguration(id: "org/model")
        await SharpAIOverlay.OfflineGuard.$environmentOverride.withValue(Self.on) {
            do {
                _ = try await resolveForModelConversion(
                    configuration: config, from: StubDownloader(counter: counter),
                    useLatest: false, progressHandler: { _ in })
                Issue.record("expected networkBlocked")
            } catch is SharpAIOverlay.OfflineError {
            } catch {
                Issue.record("wrong error type: \(error)")
            }
        }
        #expect(counter.count == 0)
    }

    @Test func directorySourceStillWorksOffline() async throws {
        let dir = URL(fileURLWithPath: "/tmp/some-local-dir")
        let counter = CallCounter()
        let resolved = try await SharpAIOverlay.OfflineGuard.$environmentOverride.withValue(Self.on) {
            try await resolve(
                configuration: ModelConfiguration(directory: dir),
                from: StubDownloader(counter: counter), useLatest: false, progressHandler: { _ in })
        }
        #expect(resolved.modelDirectory == dir)
        #expect(counter.count == 0)
    }

    @Test func noOpWhenUnset() async throws {
        let counter = CallCounter()
        let stub = StubDownloader(counter: counter)
        // identity: the guard hands back the very same downloader type when online
        let g = SharpAIOverlay.OfflineGuard.$environmentOverride.withValue([:]) {
            SharpAIOverlay.guarded(stub)
        }
        #expect(g is StubDownloader)
        let resolved = try await SharpAIOverlay.OfflineGuard.$environmentOverride.withValue([:]) {
            try await resolve(
                configuration: ModelConfiguration(id: "org/model"), from: stub,
                useLatest: false, progressHandler: { _ in })
        }
        #expect(resolved.modelDirectory.path == "/tmp/stub-org_model")
        #expect(counter.count == 1)  // one download: tokenizer shares the model directory
    }
}
