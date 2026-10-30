// SharpAI overlay (SL-28 step A1): explicit offline switch.
//
// SWIFTLM_OFFLINE=1 makes every `Downloader.download` issued by mlx-swift-lm throw
// `SharpAIOverlay.OfflineError.networkBlocked` before any I/O (hard block, cache hits included).
//
// HF_HUB_OFFLINE=1 is the standard Hugging Face "serve from the local cache" switch and is NOT a hard block
// (it used to be, which broke users who export it with a cached model). It wraps the downloader instead:
// `useLatest` is forced to false (no revalidation against the provider) and the call is served when it
// completes without a transfer. A cache miss shows up as a transfer in progress (progress strictly between
// 0 and its total); the wrapper then cancels the call and throws `networkBlocked`. This is a heuristic at the
// `Downloader` layer (a cache hit is not otherwise distinguishable from a download here): a provider that
// reports no progress for a miss is not detected, and a miss may open a connection before it is cut.
// Use SWIFTLM_OFFLINE for a guarantee.
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

    private final class MissFlag: @unchecked Sendable {
        private let lock = NSLock()
        private var hit = false
        private var cancel: (@Sendable () -> Void)?
        var missed: Bool { lock.lock(); defer { lock.unlock() }; return hit }
        func arm(_ cancel: @escaping @Sendable () -> Void) { lock.lock(); self.cancel = cancel; lock.unlock() }
        func mark() {
            lock.lock(); hit = true; let c = cancel; lock.unlock()
            c?()
        }
    }

    func download(
        id: String, revision: String?, matching patterns: [String], useLatest: Bool,
        progressHandler: @Sendable @escaping (Progress) -> Void
    ) async throws -> URL {
        let blocked = SharpAIOverlay.OfflineError.networkBlocked(
            URL(string: "https://huggingface.co/\(id)") ?? URL(fileURLWithPath: id))
        let flag = MissFlag()
        let base = self.base
        let task = Task<URL, Error> {
            try await base.download(
                id: id, revision: revision, matching: patterns, useLatest: false,
                progressHandler: { progress in
                    if progress.totalUnitCount > 0, progress.completedUnitCount > 0,
                        progress.completedUnitCount < progress.totalUnitCount
                    {
                        flag.mark()
                        return
                    }
                    progressHandler(progress)
                })
        }
        flag.arm { task.cancel() }
        let result: Result<URL, Error> = await withTaskCancellationHandler {
            await task.result
        } onCancel: {
            task.cancel()
        }
        if flag.missed { throw blocked }
        return try result.get()
    }
}
