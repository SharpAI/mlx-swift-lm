// Guards that every known-gap body asserts through `ConformanceGap.expect`. A bare `#expect` inside a
// `ConformanceGap.knownIssue { }` body cannot be observed when it passes, so a fixed gap would be
// reported UNKNOWN and CONFORMANCE_STRICT=1 would never fail on it.
// upstream-watch: none (SharpAI-only).

import Foundation
import Testing

struct ConformanceGapMigrationTests {
    static let files = [
        "ConformanceGapGdnTests.swift",
        // ConformanceGapSoftcapTests.swift left the list: the softcap gap closed and the test is a plain regression test.
        // ConformanceGapDeepseekV4Tests.swift left the list in SL-35 4.3: every gap it pinned has landed and
        // its asserts are plain, so it has no knownIssue body for the scanner to find.
        "ConformanceSynthV4Tests.swift",
        // SL-36 golden suites with knownIssue bodies. ConformanceKdaGoldenTests / ConformanceKdaDifferentialTests have
        // none (the scanner fails a listed file with no body), so they are not listed.
        "ConformanceSwigluGoldenTests.swift", "Golden/dequant/DequantGoldenTests.swift",
        // SL-36 V4 golden suites that carry knownIssue bodies (the compressor golden suite has none).
        "ConformanceV4IndexerGoldenTests.swift", "ConformanceV4RouteGoldenTests.swift",
    ]

    /// Every test source under Tests/MLXLMTests/Conformance (recursive) whose text contains a
    /// `ConformanceGap.knownIssue(` call, as paths relative to the Conformance directory. This file itself
    /// (it names the marker in the scanner) and the wrapper definition are excluded.
    static let discovered: [String] = {
        let dir = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
        guard let en = FileManager.default.enumerator(at: dir, includingPropertiesForKeys: nil) else { return [] }
        var out: [String] = []
        for case let url as URL in en where url.pathExtension == "swift" {
            let rel = String(url.path.dropFirst(dir.path.count + 1))
            if rel == "ConformanceGapMigrationTests.swift" || rel == "GapReference/ConformanceGapSupport.swift" { continue }
            guard let src = try? String(contentsOf: url, encoding: .utf8) else { continue }
            if !bodies(src).isEmpty { out.append(rel) }
        }
        return out.sorted()
    }()

    /// Bodies of every `ConformanceGap.knownIssue(...) { ... }` call (brace matched, strings skipped naively).
    static func bodies(_ src: String) -> [String] {
        var out: [String] = []
        let chars = Array(src)
        let marker = Array("ConformanceGap.knownIssue(")
        var i = 0
        while i + marker.count <= chars.count {
            if Array(chars[i..<i + marker.count]) == marker {
                var j = i + marker.count, paren = 1, inStr = false
                while j < chars.count, paren > 0 {
                    let c = chars[j]
                    if c == "\\" && inStr { j += 2; continue }
                    if c == "\"" { inStr.toggle() }
                    if !inStr { if c == "(" { paren += 1 } else if c == ")" { paren -= 1 } }
                    j += 1
                }
                while j < chars.count, chars[j] != "{" { j += 1 }
                var depth = 0, start = j
                while j < chars.count {
                    if chars[j] == "{" { depth += 1; if depth == 1 { start = j + 1 } }
                    if chars[j] == "}" { depth -= 1; if depth == 0 { break } }
                    j += 1
                }
                if j <= chars.count, start <= j { out.append(String(chars[start..<j])) }
                i = j
            } else { i += 1 }
        }
        return out
    }

    @Test("the body scanner finds bare and wrapped expectations")
    func scannerSanity() {
        let b = Self.bodies("ConformanceGap.knownIssue(\"a (b)\") {\n #expect(x)\n}\nConformanceGap.knownIssue(\"c\") { ConformanceGap.expect(y) }")
        #expect(b.count == 2 && b[0].contains("#expect(") && !b[1].contains("#expect("))
    }

    @Test("every conformance file with a knownIssue body is scanned (the hand-written list is only a minimum)")
    func everyDiscoveredFileIsScanned() {
        #expect(!Self.discovered.isEmpty, "discovery found no files: the directory walk is broken")
        // The list is a minimum: each entry must exist and be discovered (a stale entry fails noBareExpect too).
        for f in Self.files { #expect(Self.discovered.contains(f), "\(f) is listed but no knownIssue body was discovered in it") }
        // Anything discovered but not in the list is scanned by `noBareExpect` through `scanned`.
        for f in Self.discovered { #expect(Self.scanned.contains(f), "\(f) has knownIssue bodies but is not scanned") }
    }

    /// What `noBareExpect` runs over: the discovered files plus the minimum list.
    static var scanned: [String] { Array(Set(discovered).union(files)).sorted() }

    @Test("no bare #expect inside a knownIssue body", arguments: scanned)
    func noBareExpect(file: String) throws {
        let dir = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
        let src = try String(contentsOf: dir.appendingPathComponent(file), encoding: .utf8)
        let bodies = Self.bodies(src)
        #expect(!bodies.isEmpty, "\(file): scanner found no knownIssue bodies")
        for b in bodies {
            let bare = b.components(separatedBy: "#expect(").count - 1
            #expect(bare == 0, "\(file): \(bare) bare #expect in a gap body (use ConformanceGap.expect): \(b.prefix(80))")
        }
    }
}
