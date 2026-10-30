// KDA differential harness, Swift side. LOCAL ONLY: the test is SKIPPED unless the environment variable KDA_DIFF_DIR
// names a directory prepared by `tools/conformance/differential/kda_diff.py gen` (CI is air-gapped and never sets it,
// so the default test run is unaffected; see tools/conformance/differential/README.md).
//
// The suite does no judging. For every case in DIR/cases.json it loads DIR/<file> (the golden fixture format of
// ConformanceKdaGoldenTests, so it reuses KdaGolden.Fixture / prepare / execute: same caller mirroring, same
// l2norm/scale prep), runs the Metal-kernel request AND the op-composition fallback, and writes
// DIR/out/<name>.json = {name, kernel: {y, state}, fallback: {y, state}} (or {name, error}). `kda_diff.py compare`
// then checks those files against the float64 oracle with the golden manifest's tolerances. A case whose execution
// throws is recorded as an error (compare fails it); the test itself fails only if the harness cannot do its job
// (unreadable index, unwritable out directory, no cases).
//
// upstream-watch: Libraries/MLXLMCommon/SharpAIOverlay/GatedDeltaKDA.swift.

import Foundation
import MLX
import Testing

@testable import MLXLMCommon

enum KdaDiff {
    /// Non-nil only when the local harness is armed.
    static let directory: URL? = {
        guard let v = ProcessInfo.processInfo.environment["KDA_DIFF_DIR"], !v.isEmpty else { return nil }
        return URL(fileURLWithPath: v, isDirectory: true)
    }()

    struct Index: Decodable {
        struct Entry: Decodable {
            let name: String
            let file: String
        }
        let schema: Int
        let cases: [Entry]
    }

    struct PathOutput: Encodable {
        let y: [Float]
        let state: [Float]
    }

    struct CaseOutput: Encodable {
        let name: String
        var kernel: PathOutput?
        var fallback: PathOutput?
        var error: String?
    }

    static func run(_ entry: Index.Entry, in dir: URL) -> CaseOutput {
        var out = CaseOutput(name: entry.name)
        do {
            let data = try Data(contentsOf: dir.appendingPathComponent(entry.file))
            let fixture = try JSONDecoder().decode(KdaGolden.Fixture.self, from: data)
            let prepared = try KdaGolden.prepare(fixture)
            let k = try KdaGolden.execute(fixture, prepared, useKernel: true)
            out.kernel = PathOutput(y: k.y, state: k.state)
            let f = try KdaGolden.execute(fixture, prepared, useKernel: false)
            out.fallback = PathOutput(y: f.y, state: f.state)
        } catch {
            out.error = "\(error)"
        }
        return out
    }
}

@Suite("ConformanceKdaDifferentialTests")
struct ConformanceKdaDifferentialTests {
    @Test(
        "differential: run every case in KDA_DIFF_DIR through the kernel and the fallback and write DIR/out/<case>.json",
        .enabled(if: KdaDiff.directory != nil, "SKIPPED: KDA_DIFF_DIR is not set (local-only differential harness, see tools/conformance/differential/README.md)"))
    func runAllCases() throws {
        let dir = try #require(KdaDiff.directory)
        let index = try JSONDecoder().decode(KdaDiff.Index.self, from: Data(contentsOf: dir.appendingPathComponent("cases.json")))
        #expect(index.schema == 1)
        #expect(!index.cases.isEmpty, "cases.json lists no cases")
        let outDir = dir.appendingPathComponent("out", isDirectory: true)
        try FileManager.default.createDirectory(at: outDir, withIntermediateDirectories: true)
        let encoder = JSONEncoder()
        // A NaN or inf from the kernel must reach the comparison (which fails it), not crash the encoder.
        encoder.nonConformingFloatEncodingStrategy = .convertToString(positiveInfinity: "Infinity", negativeInfinity: "-Infinity", nan: "NaN")
        var errors = 0
        for entry in index.cases {
            let result = KdaDiff.run(entry, in: dir)
            if result.error != nil { errors += 1 }
            try encoder.encode(result).write(to: outDir.appendingPathComponent(entry.name + ".json"))
        }
        print("KDA-DIFF wrote \(index.cases.count) case outputs to \(outDir.path), \(errors) execution errors")
    }
}
