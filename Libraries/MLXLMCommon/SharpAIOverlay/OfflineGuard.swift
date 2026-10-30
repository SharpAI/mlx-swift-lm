// SharpAI overlay (SL-28 step A1): explicit offline switch.
//
// SWIFTLM_OFFLINE=1 makes every `Downloader.download` issued by mlx-swift-lm throw
// `SharpAIOverlay.OfflineError.networkBlocked` before any I/O (hard block, cache hits included).
//
// HF_HUB_OFFLINE=1 is the standard Hugging Face "serve from the local cache" switch and is NOT a hard block
// (it used to be, which broke users who export it with a cached model). It wraps the downloader instead:
// `useLatest` is forced to false (no revalidation against the provider) and the call is served when it
// completes without a transfer. A cache miss shows up as a transfer in progress (progress strictly between
// 0 and its total); the wrapper then throws `networkBlocked` at once, without waiting for the base download.
//
// After that throw: the base download task is cancelled. A provider that ignores cancellation may keep running
// and may keep writing into the cache, but no further callback reaches the caller: progress is no longer
// forwarded to the caller's `progressHandler` once the call has produced its outcome.
// The first outcome wins: if the caller is cancelled before the base download finishes, the error is
// `CancellationError`, even if the base would have thrown something else, and an outcome recorded first is
// never replaced by a later miss or cancellation.
//
// Limits, stated precisely: cache-hit detection is this progress heuristic, nothing more. (a) A provider that
// reports no progress for a miss is not detected. (b) The HubClient cached-snapshot fast path may not match
// (for example a pattern set or revision it does not recognise); the call then goes to the network (file
// listing) BEFORE any progress is reported, so such a miss can surface as a transport error (for example a
// URLError) instead of `networkBlocked`, and a connection may already have been opened. (c) A miss may open a
// connection before it is cut. Only SWIFTLM_OFFLINE is a guarantee; HF_HUB_OFFLINE is best effort.
//
// With neither variable set, `guarded(_:)` returns the original downloader unchanged (identity), so
// behaviour is byte-for-byte the same.
//
// upstream-watch: after a sync, grep the tree for `downloader.download(` / `.download(id:`. Every call site
// in Libraries/ that is not behind `SharpAIOverlay.guarded` is an unguarded egress. Known hook sites (five):
//   1. MLXLMCommon/ModelFactory.swift `resolve`
//   2. MLXLMCommon/ModelConversion.swift `resolveForModelConversion`
//   3. MLXLMCommon/Adapters/ModelAdapterFactory.swift `load(from:)`
//   4. BenchmarkHelpers/BenchmarkHelpers.swift `resolveTokenizerDirectory`
//   5. BenchmarkHelpers/BenchmarkHelpers.swift `benchmarkDownloadCacheHit`
// ConformanceOfflineGuardCoverageTests scans for these statically. Also re-check whether upstream added its own
// offline flag.
//
// Scope: guards the `Downloader` protocol only. Under SWIFTLM_OFFLINE a Downloader that would have served a
// purely local cache hit is blocked too; sealed offline runs must use `.directory` sources. Direct URLSession
// use in BenchmarkHelpers / IntegrationTestHelpers is not guarded.

import Foundation

extension SharpAIOverlay {
    public enum OfflineError: Error, Equatable, CustomStringConvertible {
        case networkBlocked(URL)

        public var description: String {
            switch self {
            case .networkBlocked(let url):
                return "network access blocked (SWIFTLM_OFFLINE / HF_HUB_OFFLINE cache miss): \(url.absoluteString)"
            }
        }
    }

    public enum OfflineGuard {
        /// Test seam: when set, used instead of the process environment.
        @TaskLocal public static var environmentOverride: [String: String]?

        static func truthy(_ value: String?) -> Bool {
            guard let v = value?.trimmingCharacters(in: .whitespaces).lowercased() else { return false }
            return ["1", "true", "yes", "on"].contains(v)
        }

        /// The hard block: only the explicit SWIFTLM_OFFLINE switch.
        public static func isOffline(environment: [String: String]? = nil) -> Bool {
            let env = environment ?? environmentOverride ?? ProcessInfo.processInfo.environment
            return truthy(env["SWIFTLM_OFFLINE"])
        }

        /// The Hugging Face cache-only switch (HF_HUB_OFFLINE): serve cache hits, block misses.
        public static func isHubOffline(environment: [String: String]? = nil) -> Bool {
            let env = environment ?? environmentOverride ?? ProcessInfo.processInfo.environment
            return truthy(env["HF_HUB_OFFLINE"])
        }
    }

    /// Identity when online; a hard-blocking wrapper under SWIFTLM_OFFLINE; a cache-only wrapper under
    /// HF_HUB_OFFLINE.
    public static func guarded(_ downloader: any Downloader) -> any Downloader {
        if OfflineGuard.isOffline() { return OfflineBlockingDownloader() }
        if OfflineGuard.isHubOffline() { return HubOfflineDownloader(base: downloader) }
        return downloader
    }
}

/// Never delegates: throws before any I/O.
struct OfflineBlockingDownloader: Downloader {
    func download(
        id: String, revision: String?, matching patterns: [String], useLatest: Bool,
        progressHandler: @Sendable @escaping (Progress) -> Void
    ) async throws -> URL {
        throw SharpAIOverlay.OfflineError.networkBlocked(
            URL(string: "https://huggingface.co/\(id)") ?? URL(fileURLWithPath: id))
    }
}

/// HF_HUB_OFFLINE: delegates with `useLatest: false`; a transfer in progress means a cache miss, which is
/// cancelled and reported as `networkBlocked`.
struct HubOfflineDownloader: Downloader {
    let base: any Downloader

    /// First-outcome-wins hand-off between the base download, the miss signal and the caller. A result recorded
    /// before the caller starts waiting is kept and delivered when it does.
    private final class Outcome: @unchecked Sendable {
        private let lock = NSLock()
        private var result: Result<URL, Error>?
        private var waiter: CheckedContinuation<Result<URL, Error>, Never>?

        func finish(_ value: Result<URL, Error>) {
            lock.lock()
            guard result == nil else { lock.unlock(); return }
            result = value
            let w = waiter
            waiter = nil
            lock.unlock()
            w?.resume(returning: value)
        }

        var isFinished: Bool {
            lock.lock()
            defer { lock.unlock() }
            return result != nil
        }

        func wait() async -> Result<URL, Error> {
            await withCheckedContinuation { (c: CheckedContinuation<Result<URL, Error>, Never>) in
                lock.lock()
                if let r = result {
                    lock.unlock()
                    c.resume(returning: r)
                } else {
                    waiter = c
                    lock.unlock()
                }
            }
        }
    }

    func download(
        id: String, revision: String?, matching patterns: [String], useLatest: Bool,
        progressHandler: @Sendable @escaping (Progress) -> Void
    ) async throws -> URL {
        let blocked = SharpAIOverlay.OfflineError.networkBlocked(
            URL(string: "https://huggingface.co/\(id)") ?? URL(fileURLWithPath: id))
        let outcome = Outcome()
        let base = self.base
        let task = Task<Void, Never> {
            do {
                let url = try await base.download(
                    id: id, revision: revision, matching: patterns, useLatest: false,
                    progressHandler: { progress in
                        if progress.totalUnitCount > 0, progress.completedUnitCount > 0,
                            progress.completedUnitCount < progress.totalUnitCount
                        {
                            outcome.finish(.failure(blocked))
                            return
                        }
                        if outcome.isFinished { return }
                        progressHandler(progress)
                    })
                outcome.finish(.success(url))
            } catch {
                outcome.finish(.failure(error))
            }
        }
        // The caller gets the first outcome. A provider that ignores cancellation keeps running in the
        // background, but it no longer holds the caller.
        let result: Result<URL, Error> = await withTaskCancellationHandler {
            let r = await outcome.wait()
            if case .failure = r { task.cancel() }
            return r
        } onCancel: {
            task.cancel()
            outcome.finish(.failure(CancellationError()))
        }
        return try result.get()
    }
}
