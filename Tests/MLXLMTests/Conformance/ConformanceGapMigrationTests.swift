// Guards that every known-gap body asserts through `ConformanceGap.expect`. A bare `#expect` inside a
// `ConformanceGap.knownIssue { }` body cannot be observed when it passes, so a fixed gap would be
// reported UNKNOWN and CONFORMANCE_STRICT=1 would never fail on it.
// upstream-watch: none (SharpAI-only).

import Foundation
import Testing

struct ConformanceGapMigrationTests {
    static let files = [
        "ConformanceGapGdnTests.swift", "ConformanceGapSoftcapTests.swift",
        // ConformanceGapDeepseekV4Tests.swift left the list in SL-35 4.3: every gap it pinned has landed and
        // its asserts are plain, so it has no knownIssue body for the scanner to find.
        "ConformanceSynthV4Tests.swift",
        // SL-36 golden suites with knownIssue bodies. ConformanceKdaGoldenTests / ConformanceKdaDifferentialTests have
        // none (the scanner fails a listed file with no body), so they are not listed.
        "ConformanceSwigluGoldenTests.swift", "Golden/dequant/DequantGoldenTests.swift",
    ]

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

    @Test("no bare #expect inside a knownIssue body", arguments: files)
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
