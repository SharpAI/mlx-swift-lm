// SL-28 A4 (r3-offline): static coverage of the offline guard.
//
// Every `<expr>.download(` call on a `Downloader` under Libraries/ must sit in a function that first wraps its
// downloader with `SharpAIOverlay.guarded(`. BenchmarkHelpers calls the downloader directly, so a benchmark that
// injects a real Downloader reached the network even with SWIFTLM_OFFLINE=1 (this test was red before the hook).
// Static (source scan via #filePath) because the MLXLMTests target does not depend on BenchmarkHelpers and
// Package.swift is upstream-owned.
//
// upstream-watch: a new unguarded `.download(` after a sync fails here with file:line.

import Foundation
import Testing

@Suite("ConformanceOfflineGuardCoverage")
struct ConformanceOfflineGuardCoverageTests {
    static let librariesRoot: URL = {
        var url = URL(fileURLWithPath: #filePath)
        for _ in 0..<4 { url.deleteLastPathComponent() }
        return url.appendingPathComponent("Libraries")
    }()

    /// Non-Downloader `.download(` uses that are known and documented as unguarded (URLSession, not the protocol).
    static let knownNonDownloader = ["URLSession.shared.download("]

    struct Site: CustomStringConvertible {
        let file: String
        let line: Int
        let guarded: Bool
        var description: String { "\(file):\(line)" }
    }

    static func scan() throws -> [Site] {
        var sites: [Site] = []
        let fm = FileManager.default
        guard let e = fm.enumerator(at: librariesRoot, includingPropertiesForKeys: nil) else { return [] }
        for case let url as URL in e where url.pathExtension == "swift" {
            let rel = url.path.replacingOccurrences(of: librariesRoot.path + "/", with: "")
            if rel.hasPrefix("MLXLMCommon/SharpAIOverlay/") { continue }
            let lines = try String(contentsOf: url, encoding: .utf8).components(separatedBy: "\n")
            var funcStart = 0
            for (i, raw) in lines.enumerated() {
                let l = raw.trimmingCharacters(in: .whitespaces)
                if l.hasPrefix("//") { continue }
                if l.contains("func ") { funcStart = i }
                guard l.contains(".download("), !knownNonDownloader.contains(where: { l.contains($0) }) else {
                    continue
                }
                let g = lines[funcStart...i].contains { $0.contains("SharpAIOverlay.guarded(") }
                sites.append(Site(file: rel, line: i + 1, guarded: g))
            }
        }
        return sites
    }

    @Test func everyDownloaderCallSiteIsGuarded() throws {
        let sites = try Self.scan()
        #expect(sites.count >= 8, "scan found only \(sites.count) call sites; the scanner is broken or the tree moved")
        let bad = sites.filter { !$0.guarded }
        #expect(bad.isEmpty, "unguarded Downloader.download call sites: \(bad.map(\.description).joined(separator: ", "))")
    }

    @Test func benchmarkHelpersIsCovered() throws {
        let sites = try Self.scan().filter { $0.file.hasPrefix("BenchmarkHelpers/") }
        #expect(sites.count == 4)
        let allGuarded = sites.allSatisfy { $0.guarded }
        #expect(allGuarded)
    }
}
