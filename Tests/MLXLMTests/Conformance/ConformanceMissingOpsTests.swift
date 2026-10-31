// Conformance harness, round 3 (SL-29 phase 2): the "missing-ops" report.
//
// scripts/conformance/missing-ops/gen_report.py statically lists, for every offline fixture whose model_type is
// registered, the top-level config keys the registered Configuration type silently ignores, and writes
// REPORT.md / REPORT.csv. THIS FILE DOES NOT WRITE THE REPORT. It asserts that
//   * the generator runs (exit 0) into a scratch directory, and its output is internally consistent
//     (row counts, per-row counts, totals in CSV == JSON summary == REPORT.md);
//   * the number of (fixture, registry) rows equals what the REAL registries say at runtime, so the
//     script's static registry parse cannot drift from the code;
//   * for several families the script's ignored-key list equals the answer of a genuine Codable round trip
//     of the real Configuration type (decode the fixture, re-encode, diff the key sets).
// The committed REPORT.md being stale is only a printed warning (an upstream sync changes the sources).
// upstream-watch: the three *TypeRegistry.swift files and every model Configuration (the script reads them).

import Foundation
import MLXEmbedders
import MLXLLM
import MLXLMCommon
import MLXVLM
import Testing

@testable import MLXLLM

enum ConformanceMissingOps {
    static let repoRoot: URL = {
        var url = URL(fileURLWithPath: #filePath)
        for _ in 0..<4 { url.deleteLastPathComponent() }
        return url
    }()
    static var scriptDir: URL { repoRoot.appendingPathComponent("scripts/conformance/missing-ops") }

    struct RunResult {
        var status: Int32
        var stdout: String
        var stderr: String
    }

    static func python(_ args: [String]) throws -> RunResult {
        let p = Process()
        p.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        p.arguments = ["python3", "-I"] + args
        let out = Pipe(), err = Pipe()
        p.standardOutput = out
        p.standardError = err
        try p.run()
        // Drain before waiting: a full pipe buffer would deadlock waitUntilExit.
        let o = out.fileHandleForReading.readDataToEndOfFile()
        let e = err.fileHandleForReading.readDataToEndOfFile()
        p.waitUntilExit()
        return RunResult(
            status: p.terminationStatus, stdout: String(decoding: o, as: UTF8.self),
            stderr: String(decoding: e, as: UTF8.self))
    }

    /// Minimal CSV parser for the report (RFC 4180 quoting, no embedded newlines are produced).
    static func parseCSV(_ text: String) -> [[String: String]] {
        func fields(_ line: Substring) -> [String] {
            var out: [String] = [], cur = "", quoted = false
            var it = line.makeIterator()
            while let c = it.next() {
                if quoted {
                    if c == "\"" {
                        // doubled quote inside a quoted field is a literal quote
                        if let n = it.next() {
                            if n == "\"" { cur.append("\"") } else { quoted = false; if n == "," { out.append(cur); cur = "" } else { cur.append(n) } }
                        } else { quoted = false }
                    } else { cur.append(c) }
                } else if c == "\"" { quoted = true } else if c == "," { out.append(cur); cur = "" } else { cur.append(c) }
            }
            out.append(cur)
            return out
        }
        let lines = text.split(separator: "\n", omittingEmptySubsequences: true)
        guard let head = lines.first else { return [] }
        let names = fields(head)
        return lines.dropFirst().map { l in
            let f = fields(l)
            return Dictionary(uniqueKeysWithValues: zip(names, f))
        }
    }

    struct Generated {
        var rows: [[String: String]]
        var summary: [String: Int]
        var markdown: String
    }

    static func generate() throws -> Generated {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("conformance-missing-ops-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: dir) }
        let r = try python([
            scriptDir.appendingPathComponent("gen_report.py").path, "--root", repoRoot.path, "--out-dir", dir.path,
        ])
        guard r.status == 0 else {
            throw GeneratorFailure(message: "gen_report.py exit \(r.status): \(r.stderr.prefix(400))")
        }
        let csv = try String(contentsOf: dir.appendingPathComponent("REPORT.csv"), encoding: .utf8)
        let md = try String(contentsOf: dir.appendingPathComponent("REPORT.md"), encoding: .utf8)
        let obj = try JSONSerialization.jsonObject(with: Data(r.stdout.utf8)) as? [String: Int] ?? [:]
        return Generated(rows: parseCSV(csv), summary: obj, markdown: md)
    }

    struct GeneratorFailure: Error, CustomStringConvertible { var message: String; var description: String { message } }

    static func list(_ s: String?) -> [String] { (s ?? "").split(separator: ";").map(String.init) }
}

struct ConformanceMissingOpsTests {

    @Test("generator runs and its totals are internally consistent")
    func generatorRunsAndTotalsAreConsistent() throws {
        let g = try ConformanceMissingOps.generate()
        let rows = g.rows
        #expect(!rows.isEmpty)
        #expect(g.summary["rows"] == rows.count)

        var total = 0, meta = 0, cand = 0, withCand = 0
        for r in rows {
            let t = Int(r["ignored_total"] ?? "") ?? -1
            let m = Int(r["ignored_metadata"] ?? "") ?? -1
            let c = Int(r["ignored_candidates"] ?? "") ?? -1
            #expect(t == m + c, "\(r["model_type"] ?? "?") \(r["repo"] ?? "?"): total \(t) != \(m)+\(c)")
            #expect(ConformanceMissingOps.list(r["candidates"]).count == c, "candidate list length: \(r["repo"] ?? "?")")
            #expect(ConformanceMissingOps.list(r["metadata"]).count == m, "metadata list length: \(r["repo"] ?? "?")")
            #expect(Set(ConformanceMissingOps.list(r["candidates"])).isDisjoint(with: ConformanceMissingOps.list(r["metadata"])))
            #expect(t <= (Int(r["fixture_keys"] ?? "") ?? 0), "ignored more keys than the fixture has")
            total += t; meta += m; cand += c; if c > 0 { withCand += 1 }
        }
        #expect(g.summary["ignored_total"] == total && g.summary["ignored_metadata"] == meta)
        #expect(g.summary["ignored_candidates"] == cand && g.summary["fixtures_with_candidates"] == withCand)
        #expect(total == meta + cand)
        #expect(g.markdown.contains("**\(total)** = \(meta) metadata + **\(cand)** candidate ops"), "REPORT.md totals line")
        #expect(g.markdown.contains("= **\(rows.count)** rows"))
        #expect(g.summary["unresolved"] == rows.filter { $0["keys_source"] == "unresolved" }.count)
        print("[missing-ops] rows=\(rows.count) ignored=\(total) (metadata \(meta), candidate ops \(cand)) unresolved=\(g.summary["unresolved"] ?? -1)")
    }

    @Test("script's (fixture, registry) rows equal what the real registries contain")
    func rowsMatchRuntimeRegistries() async throws {
        let g = try ConformanceMissingOps.generate()
        var expected = Set<String>()  // "type|repo|registry"
        for e in try ConformanceBundle.manifest().fixtures {
            guard let mt = e.model_type else { continue }
            if await LLMTypeRegistry.shared.contains(mt) { expected.insert("\(mt)|\(e.repo)|llm") }
            if await VLMTypeRegistry.shared.contains(mt) { expected.insert("\(mt)|\(e.repo)|vlm") }
            if await EmbedderTypeRegistry.shared.contains(mt) { expected.insert("\(mt)|\(e.repo)|embedder") }
        }
        let got = Set(g.rows.map { "\($0["model_type"] ?? "")|\($0["repo"] ?? "")|\($0["registry"] ?? "")" })
        #expect(got.count == g.rows.count, "duplicate rows")
        #expect(
            got == expected,
            "only in script: \(got.subtracting(expected).sorted().prefix(5)); only at runtime: \(expected.subtracting(got).sorted().prefix(5))")
    }

    /// Keys of `data` that the Configuration silently ignores, from a real decode -> encode round trip.
    /// Keys whose fixture value is JSON null are left out of both sides (an absent optional re-encodes as nothing).
    static func roundTripIgnored<C: Codable>(_ type: C.Type, fixture dir: String, file: String? = nil) throws -> Set<String> {
        let obj = try ConformanceSynthV4Fixture.object(dir, file: file)
        let data = try JSONSerialization.data(withJSONObject: obj)
        let cfg = try JSONDecoder().decode(C.self, from: data)
        let encoder = JSONEncoder()
        // NemotronH defaults a time-step limit to +infinity, which plain JSONEncoder refuses to encode.
        encoder.nonConformingFloatEncodingStrategy = .convertToString(
            positiveInfinity: "inf", negativeInfinity: "-inf", nan: "nan")
        let enc = try #require(try JSONSerialization.jsonObject(with: encoder.encode(cfg)) as? [String: Any])
        return Set(obj.filter { !($0.value is NSNull) }.keys).subtracting(enc.keys)
    }

    @Test("script's ignored-key lists agree with a real Codable round trip (static parse validated)",
        arguments: [
            ("deepseek_v4", "DeepSeek-V4-Flash"), ("nemotron_h", ""), ("glm4_moe", ""), ("lfm2", "LFM2-ColBERT"),
            ("gemma3_text", ""),
        ])
    func staticParseAgreesWithRoundTrip(_ type: String, _ hint: String) throws {
        let g = try ConformanceMissingOps.generate()
        let file = try ConformanceSynthV4Fixture.files(type).first { hint.isEmpty || $0.lastPathComponent.contains(hint) }
        let url = try #require(file)
        let repo = try #require(
            try ConformanceBundle.manifest().fixtures.first { $0.path.hasSuffix("fixtures/\(type)/\(url.lastPathComponent)") }
        ).repo
        let row = try #require(g.rows.first { $0["model_type"] == type && $0["repo"] == repo && $0["registry"] == "llm" })
        let scripted = Set(ConformanceMissingOps.list(row["candidates"]) + ConformanceMissingOps.list(row["metadata"]))

        let dynamic: Set<String>
        switch type {
        case "deepseek_v4": dynamic = try Self.roundTripIgnored(SharpAIDeepseekV4Configuration.self, fixture: type, file: url.lastPathComponent)
        case "nemotron_h": dynamic = try Self.roundTripIgnored(NemotronHConfiguration.self, fixture: type, file: url.lastPathComponent)
        case "glm4_moe": dynamic = try Self.roundTripIgnored(GLM4MoEConfiguration.self, fixture: type, file: url.lastPathComponent)
        case "lfm2": dynamic = try Self.roundTripIgnored(LFM2Configuration.self, fixture: type, file: url.lastPathComponent)
        default:
            dynamic = try Self.roundTripIgnored(MLXLLM.Gemma3TextConfiguration.self, fixture: type, file: url.lastPathComponent)
        }
        let nulls = Set(try ConformanceSynthV4Fixture.object(type, file: url.lastPathComponent).filter { $0.value is NSNull }.keys)
        let s = scripted.subtracting(nulls)
        #expect(s == dynamic, "\(type): script-only \(s.subtracting(dynamic).sorted()), round-trip-only \(dynamic.subtracting(s).sorted())")
    }

    @Test("a Configuration that decodes through several CodingKey enums is not reported as ignoring those keys")
    func multipleCodingKeyEnumsAreRead() throws {
        let g = try ConformanceMissingOps.generate()
        // BertConfiguration decodes hidden_size & co. through `BertCodingKeys`, not `CodingKeys`.
        let bert = try #require(g.rows.first { $0["model_type"] == "bert" && $0["registry"] == "embedder" })
        let ignored = Set(ConformanceMissingOps.list(bert["candidates"]) + ConformanceMissingOps.list(bert["metadata"]))
        let decoded: Set<String> = [
            "hidden_size", "num_attention_heads", "intermediate_size", "num_hidden_layers", "type_vocab_size",
        ]
        #expect(ignored.isDisjoint(with: decoded), "bert keys wrongly reported ignored: \(ignored.intersection(decoded).sorted())")
        // The distilbert keys are decoded as well (DistilBertCodingKeys), so a bert-family row never lists them either.
        #expect(!ignored.contains("dim") && !ignored.contains("n_layers"))
    }

    /// Verdict on the committed report: `nil` when it is current, otherwise the message. The strictness is an
    /// injected flag so the rule is testable without mutating the process environment.
    static func staleReportProblem(committed: String?, fresh: String, strict: Bool) -> String? {
        guard let committed else { return "REPORT.md must be committed" }
        guard committed != fresh else { return nil }
        let msg = "scripts/conformance/missing-ops/REPORT.md differs from a fresh run; regenerate with gen_report.py"
        print("[missing-ops][STALE] \(msg)")
        return strict ? msg : nil
    }

    @Test("a stale report is a warning by default and a failure only under strict mode")
    func staleReportRule() {
        #expect(Self.staleReportProblem(committed: "a", fresh: "a", strict: true) == nil)
        #expect(Self.staleReportProblem(committed: "a", fresh: "b", strict: false) == nil, "default: warning only")
        #expect(Self.staleReportProblem(committed: "a", fresh: "b", strict: true) != nil, "strict: failure")
        #expect(Self.staleReportProblem(committed: nil, fresh: "b", strict: false) != nil, "missing file always fails")
    }

    @Test("committed REPORT.md is current (warning; failure under CONFORMANCE_STRICT=1)")
    func committedReportIsCurrent() throws {
        let g = try ConformanceMissingOps.generate()
        let committed = try? String(contentsOf: ConformanceMissingOps.scriptDir.appendingPathComponent("REPORT.md"), encoding: .utf8)
        let strict = ProcessInfo.processInfo.environment["CONFORMANCE_STRICT"] == "1"
        let problem = Self.staleReportProblem(committed: committed, fresh: g.markdown, strict: strict)
        #expect(problem == nil, "\(problem ?? "")")
    }
}
