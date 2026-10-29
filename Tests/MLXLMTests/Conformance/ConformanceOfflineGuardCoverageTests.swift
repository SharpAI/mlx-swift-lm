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
            sites += scanSource(try String(contentsOf: url, encoding: .utf8), file: rel)
        }
        return sites
    }

    /// Removes `//` and (nested) `/* */` comments, keeping newlines so line numbers stay valid. String literals
    /// are honoured so a `//` inside "https://..." is not taken as a comment.
    static func stripComments(_ text: String) -> String {
        let c = Array(text.unicodeScalars)
        var out = String.UnicodeScalarView()
        var i = 0, depth = 0, inString = false
        while i < c.count {
            let ch = c[i], nx: Unicode.Scalar? = i + 1 < c.count ? c[i + 1] : nil
            if depth > 0 {
                if ch == "/", nx == "*" { depth += 1; i += 2 }
                else if ch == "*", nx == "/" { depth -= 1; i += 2 }
                else { if ch == "\n" { out.append(ch) }; i += 1 }
            } else if inString {
                out.append(ch)
                if ch == "\\", let n = nx { out.append(n); i += 2; continue }
                if ch == "\"" || ch == "\n" { inString = false }
                i += 1
            } else if ch == "/", nx == "/" {
                while i < c.count, c[i] != "\n" { i += 1 }
            } else if ch == "/", nx == "*" { depth = 1; i += 2 }
            else { if ch == "\"" { inString = true }; out.append(ch); i += 1 }
        }
        return String(out)
    }

    /// Pure scanner over one file's text (a seam so the guard logic itself can be tested on literals).
    static func scanSource(_ text: String, file: String) -> [Site] {
        var sites: [Site] = []
        // Comments are stripped first, so a guard mentioned in a comment never satisfies the check. The guard must
        // then sit between the nearest preceding `func` line and the call, i.e. in the same function body.
        let lines = stripComments(text).components(separatedBy: "\n")
        var funcStart = 0
        for (i, raw) in lines.enumerated() {
            let l = raw.trimmingCharacters(in: .whitespaces)
            if l.contains("func ") { funcStart = i }
            guard l.contains(".download("), !knownNonDownloader.contains(where: { l.contains($0) }) else {
                continue
            }
            let g = lines[funcStart...i].contains { $0.contains("SharpAIOverlay.guarded(") }
            sites.append(Site(file: file, line: i + 1, guarded: g))
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
        #expect(sites.count >= 1, "BenchmarkHelpers has no Downloader call sites; scanner broken or sites moved")
        let allGuarded = sites.allSatisfy { $0.guarded }
        #expect(allGuarded)
    }

    // Guard logic itself, on literals.
    @Test func guardInCommentDoesNotCount() {
        let src = """
        func a() {
            // SharpAIOverlay.guarded(downloader)
            let x = try await downloader.download(id: "m")
        }
        func b() {
            /* SharpAIOverlay.guarded( */
            _ = try await downloader.download(id: "m")
        }
        func c() {
            let d = SharpAIOverlay.guarded(downloader)  // real guard, trailing comment
            _ = try await d.download(id: "m")
        }
        func d() {
            _ = try await downloader.download(id: "m")  // SharpAIOverlay.guarded(
        }
        """
        let sites = Self.scanSource(src, file: "x.swift")
        #expect(sites.count == 4)
        #expect(sites.map(\.guarded) == [false, false, true, false])
    }

    @Test func guardInOtherFunctionDoesNotCount() {
        let src = """
        func a() {
            let d = SharpAIOverlay.guarded(downloader)
            _ = d
        }
        func b() {
            _ = try await downloader.download(id: "m")
        }
        """
        #expect(Self.scanSource(src, file: "x.swift").map(\.guarded) == [false])
    }
}
