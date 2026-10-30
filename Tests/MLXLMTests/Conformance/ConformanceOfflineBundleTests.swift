// Conformance offline bundle: proves the fixture loading mechanism.
//
// Fixtures live at <repo>/scripts/conformance/offline/fixtures/ (outside every SwiftPM target, so
// SwiftPM has no "unhandled files" to warn about for them; the Golden/kda and Golden/swiglu fixtures, which sit
// inside the test target dir, are listed as `exclude:` in Package.swift instead) and are located through
// `#filePath`. They are CONFIG-ONLY json files (no weights). Needs the source tree at test time, which an
// air-gapped `swift test` of a checkout always has. Generator: scripts/conformance/offline/make_bundle.py.

import CryptoKit
import Foundation
import MLXLLM
import MLXLMCommon
import Testing

@testable import MLXLLM

enum ConformanceBundle {
    /// <repo>/scripts/conformance/offline, derived from this file's path (Tests/MLXLMTests/Conformance/<file>).
    static let root: URL = {
        var url = URL(fileURLWithPath: #filePath)
        for _ in 0..<4 { url.deleteLastPathComponent() }
        return url.appendingPathComponent("scripts/conformance/offline")
    }()

    struct Entry: Decodable {
        let path: String
        let model_type: String?
        let repo: String
        let sha256: String
        let size: Int
    }
    struct Manifest: Decodable {
        let schema: Int
        let count: Int
        let fixtures: [Entry]
    }

    static func manifest() throws -> Manifest {
        try JSONDecoder().decode(
            Manifest.self, from: Data(contentsOf: root.appendingPathComponent("MANIFEST.json")))
    }

    static func data(_ entry: Entry) throws -> Data {
        try Data(contentsOf: root.appendingPathComponent(entry.path))
    }

    static func sha256(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }
}

struct ConformanceOfflineBundleTests {

    @Test("manifest is consistent with the fixture files (count, size, sha256)")
    func manifestMatchesFiles() throws {
        let m = try ConformanceBundle.manifest()
        #expect(m.schema == 1)
        #expect(m.count == m.fixtures.count)
        #expect(!m.fixtures.isEmpty)
        for e in m.fixtures {
            let d = try ConformanceBundle.data(e)
            #expect(d.count == e.size, "\(e.path): size")
            #expect(ConformanceBundle.sha256(d) == e.sha256, "\(e.path): sha256")
        }
    }

    /// A real, config-only fixture goes through the public registry path as far as config allows:
    /// registry lookup by `model_type`, then decode into the model's own Configuration type.
    /// (Building the model would allocate full-size random weights, so it stops at the config.)
    @Test("gemma3_text fixture decodes through the registry's configuration type")
    func gemma3TextFixtureDecodes() async throws {
        let e = try #require(
            try ConformanceBundle.manifest().fixtures.first { $0.repo == "EssentialAI/rnj-1-instruct" })
        #expect(e.model_type == "gemma3_text")
        #expect(await LLMTypeRegistry.shared.contains(try #require(e.model_type)))
        let cfg = try JSONDecoder().decode(Gemma3TextConfiguration.self, from: ConformanceBundle.data(e))
        #expect(cfg.modelType == "gemma3_text")
        #expect(cfg.intermediateSize == 16384)
    }
}
