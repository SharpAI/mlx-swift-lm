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

private final class FlagBox: @unchecked Sendable {
    private let lock = NSLock()
    private var v: Bool?
    var value: Bool? { lock.lock(); defer { lock.unlock() }; return v }
    func set(_ x: Bool) { lock.lock(); v = x; lock.unlock() }
}

/// Records the `useLatest` it was called with.
private struct RecordingDownloader: Downloader {
    let seen: FlagBox
    func download(
        id: String, revision: String?, matching patterns: [String], useLatest: Bool,
        progressHandler: @Sendable @escaping (Progress) -> Void
    ) async throws -> URL {
        seen.set(useLatest)
        return URL(fileURLWithPath: "/tmp/stub-recorded")
    }
}

/// A cache miss: the provider starts a transfer (progress strictly between 0 and total) and then returns.
private struct TransferringDownloader: Downloader {
    let started: FlagBox
    func download(
        id: String, revision: String?, matching patterns: [String], useLatest: Bool,
        progressHandler: @Sendable @escaping (Progress) -> Void
    ) async throws -> URL {
        let p = Progress(totalUnitCount: 100)
        p.completedUnitCount = 10
        progressHandler(p)
        started.set(true)
        return URL(fileURLWithPath: "/tmp/stub-downloaded")
    }
}

/// A cache miss whose provider ignores cancellation: reports partial progress at once, then keeps "transferring"
/// for `seconds` without ever looking at `Task.isCancelled`.
private struct StubbornDownloader: Downloader {
    let finished: FlagBox
    let seconds: Double
    func download(
        id: String, revision: String?, matching patterns: [String], useLatest: Bool,
        progressHandler: @Sendable @escaping (Progress) -> Void
    ) async throws -> URL {
        let p = Progress(totalUnitCount: 100)
        p.completedUnitCount = 10
        progressHandler(p)
        let end = Date().addingTimeInterval(seconds)
        while Date() < end { usleep(50_000) }
        finished.set(true)
        return URL(fileURLWithPath: "/tmp/stub-stubborn")
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
        // HF_HUB_OFFLINE is the standard Hugging Face "serve from the cache" switch, not the hard block.
        #expect(!SharpAIOverlay.OfflineGuard.isOffline(environment: ["HF_HUB_OFFLINE": "1"]))
        #expect(!SharpAIOverlay.OfflineGuard.isOffline(environment: ["HF_HUB_OFFLINE": "TRUE"]))
        #expect(SharpAIOverlay.OfflineGuard.isOffline(environment: ["SWIFTLM_OFFLINE": "1", "HF_HUB_OFFLINE": "1"]))
    }

    @Test func hubOfflineEnvParsing() {
        #expect(!SharpAIOverlay.OfflineGuard.isHubOffline(environment: [:]))
        #expect(!SharpAIOverlay.OfflineGuard.isHubOffline(environment: ["HF_HUB_OFFLINE": "0"]))
        #expect(SharpAIOverlay.OfflineGuard.isHubOffline(environment: ["HF_HUB_OFFLINE": "1"]))
        #expect(SharpAIOverlay.OfflineGuard.isHubOffline(environment: ["HF_HUB_OFFLINE": "TRUE"]))
        #expect(!SharpAIOverlay.OfflineGuard.isHubOffline(environment: ["SWIFTLM_OFFLINE": "1"]))
    }

    /// A cache hit: returns at once, reports no partial transfer.
    @Test func hfHubOfflineServesCacheHit() async throws {
        let counter = CallCounter()
        let resolved = try await SharpAIOverlay.OfflineGuard.$environmentOverride.withValue(["HF_HUB_OFFLINE": "1"]) {
            try await resolve(
                configuration: ModelConfiguration(id: "org/model"), from: StubDownloader(counter: counter),
                useLatest: true, progressHandler: { _ in })
        }
        #expect(resolved.modelDirectory.path == "/tmp/stub-org_model")
        #expect(counter.count == 1)
    }

    /// HF_HUB_OFFLINE never revalidates against the provider, even if the caller asked for the latest.
    @Test func hfHubOfflineForcesUseLatestOff() async throws {
        let seen = FlagBox()
        let stub = RecordingDownloader(seen: seen)
        _ = try await SharpAIOverlay.OfflineGuard.$environmentOverride.withValue(["HF_HUB_OFFLINE": "1"]) {
            try await resolve(
                configuration: ModelConfiguration(id: "org/model"), from: stub,
                useLatest: true, progressHandler: { _ in })
        }
        #expect(seen.value == false)
    }

    /// A cache miss starts a real transfer (partial progress): blocked with the typed error.
    @Test func hfHubOfflineBlocksCacheMiss() async throws {
        let seen = FlagBox()
        let stub = TransferringDownloader(started: seen)
        await SharpAIOverlay.OfflineGuard.$environmentOverride.withValue(["HF_HUB_OFFLINE": "1"]) {
            do {
                _ = try await resolve(
                    configuration: ModelConfiguration(id: "org/model"), from: stub,
                    useLatest: false, progressHandler: { _ in })
                Issue.record("expected networkBlocked on a cache miss")
            } catch is SharpAIOverlay.OfflineError {
            } catch {
                Issue.record("wrong error type: \(error)")
            }
        }
    }

    /// A provider that ignores cancellation must not hold the caller: the miss is reported as soon as it is seen,
    /// while the provider is still "transferring". Progress is reported synchronously on the provider's first
    /// instruction, so this also covers a miss that is recorded before the wrapper has finished setting up.
    @Test func hfHubOfflineThrowsWithoutWaitingForStubbornProvider() async throws {
        let finished = FlagBox()
        let stub = StubbornDownloader(finished: finished, seconds: 3)
        let start = Date()
        var thrown: Error?
        await SharpAIOverlay.OfflineGuard.$environmentOverride.withValue(["HF_HUB_OFFLINE": "1"]) {
            do {
                _ = try await SharpAIOverlay.guarded(stub).download(
                    id: "org/model", revision: nil, matching: [], useLatest: false, progressHandler: { _ in })
            } catch {
                thrown = error
            }
        }
        let elapsed = Date().timeIntervalSince(start)
        #expect(thrown is SharpAIOverlay.OfflineError, "got \(String(describing: thrown))")
        #expect(elapsed < 1.5, "waited \(elapsed)s for the provider")
        #expect(finished.value == nil, "the provider must still be running when the error is thrown")
        // let the leaked provider drain so it cannot spill into other tests
        for _ in 0 ..< 100 where finished.value == nil { try await Task.sleep(nanoseconds: 50_000_000) }
    }

    /// SWIFTLM_OFFLINE stays the hard block even when HF_HUB_OFFLINE is also set.
    @Test func swiftlmOfflineWinsOverHubOffline() async throws {
        let counter = CallCounter()
        await SharpAIOverlay.OfflineGuard.$environmentOverride.withValue(["SWIFTLM_OFFLINE": "1", "HF_HUB_OFFLINE": "1"]) {
            do {
                _ = try await resolve(
                    configuration: ModelConfiguration(id: "org/model"), from: StubDownloader(counter: counter),
                    useLatest: false, progressHandler: { _ in })
                Issue.record("expected networkBlocked")
            } catch is SharpAIOverlay.OfflineError {
            } catch {
                Issue.record("wrong error type: \(error)")
            }
        }
        #expect(counter.count == 0)
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

    @Test func adapterFactoryLoadThrowsWhenOffline() async throws {
        let counter = CallCounter()
        let factory = ModelAdapterFactory(registry: ModelAdapterTypeRegistry())
        await SharpAIOverlay.OfflineGuard.$environmentOverride.withValue(Self.on) {
            do {
                _ = try await factory.load(
                    from: StubDownloader(counter: counter), configuration: ModelConfiguration(id: "org/adapter"))
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
