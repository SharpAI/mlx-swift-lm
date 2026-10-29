// SharpAI overlay (SL-28 step A1): explicit offline switch.
//
// SWIFTLM_OFFLINE=1 (or HF_HUB_OFFLINE=1) makes every `Downloader.download` issued by mlx-swift-lm throw
// `SharpAIOverlay.OfflineError.networkBlocked` before any I/O. With neither variable set, `guarded(_:)`
// returns the original downloader unchanged (identity), so behaviour is byte-for-byte the same.
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
// Scope: guards the `Downloader` protocol only. A Downloader that would have served a purely local cache hit
// is blocked too (we cannot tell cache hits from network I/O at this layer); offline runs must use
// `.directory` sources. Direct URLSession use in BenchmarkHelpers / IntegrationTestHelpers is not guarded.

import Foundation

extension SharpAIOverlay {
    public enum OfflineError: Error, Equatable, CustomStringConvertible {
        case networkBlocked(URL)

        public var description: String {
            switch self {
            case .networkBlocked(let url):
                return "network access blocked (SWIFTLM_OFFLINE / HF_HUB_OFFLINE): \(url.absoluteString)"
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

        public static func isOffline(environment: [String: String]? = nil) -> Bool {
            let env = environment ?? environmentOverride ?? ProcessInfo.processInfo.environment
            return truthy(env["SWIFTLM_OFFLINE"]) || truthy(env["HF_HUB_OFFLINE"])
        }
    }

    /// Identity when online; a blocking wrapper when offline.
    public static func guarded(_ downloader: any Downloader) -> any Downloader {
        OfflineGuard.isOffline() ? OfflineBlockingDownloader() : downloader
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
