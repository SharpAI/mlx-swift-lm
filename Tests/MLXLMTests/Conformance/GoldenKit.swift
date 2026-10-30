// Shared plumbing for the committed-golden conformance suites (ConformanceKdaGoldenTests, ConformanceSwigluGoldenTests,
// DequantGoldenTests). Replaces the three lane-private loaders.
//
// What every golden suite needs, and gets here once:
//  * Location through `#filePath` (no Package.swift resource, no bundle), so fixtures are read from the source tree.
//    `directory(_:overrideEnv:)` additionally honours an environment variable that points at another directory; that is
//    how a suite is shown RED against a perturbed copy (unset in normal runs).
//  * sha256 verification of a file against its MANIFEST entry (`verify`).
//  * An EXACT JSON parse. Foundation's JSONSerialization reads some decimals one ulp off (-0.024411658756434917, an
//    exact float32 rounding tie, came back as the other neighbour, so Float(parsed) flipped and an exact float32
//    comparison failed although MLX was right). `parseJSON` uses Swift's correctly rounded Double(String);
//    `decode` uses JSONDecoder, which is exact too (GoldenKitTests pins both against Double(String)).
//  * Integrity problems (missing file, unparsable JSON, hash mismatch) are REPORTED through `Issue.record` or thrown
//    as `LoadError`, never trapped: a trap would kill the whole test process and every unrelated suite after it.
//
// Nothing here runs Python or touches the network.

import CryptoKit  // macOS-only, like ConformanceOfflineBundleTests; CI for this repo is macos-15
import Foundation
import Testing

enum GoldenKit {

    struct LoadError: Error, CustomStringConvertible {
        let description: String
    }

    // MARK: locations

    /// Tests/MLXLMTests/Conformance, derived from this file's path.
    static let conformanceDir: URL = URL(fileURLWithPath: #filePath).deletingLastPathComponent()

    /// Repository root: Tests/MLXLMTests/Conformance/<this file> -> four components up.
    static let repoRoot: URL = {
        var url = URL(fileURLWithPath: #filePath)
        for _ in 0 ..< 4 { url.deleteLastPathComponent() }
        return url
    }()

    /// `relative` (below the repo root when it starts with "tools/", otherwise below Tests/MLXLMTests/Conformance), or
    /// the directory named by the environment variable `overrideEnv` when that is set and non-empty.
    static func directory(_ relative: String, overrideEnv: String? = nil) -> URL {
        if let name = overrideEnv, let value = ProcessInfo.processInfo.environment[name], !value.isEmpty {
            return URL(fileURLWithPath: value)
        }
        let base = relative.hasPrefix("tools/") ? repoRoot : conformanceDir
        return base.appendingPathComponent(relative)
    }

    // MARK: hashing and manifests

    static func sha256(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    /// nil when `data` matches what the manifest records for `name`; otherwise the problem. `bytes` is optional.
    static func verify(_ name: String, data: Data, sha256 expected: String?, bytes: Int? = nil) -> String? {
        guard let expected else { return "\(name) is not in MANIFEST.json" }
        let actual = sha256(data)
        if actual != expected { return "\(name): sha256 \(actual) != manifest \(expected)" }
        if let bytes, bytes != data.count { return "\(name): \(data.count) bytes, MANIFEST.json records \(bytes) bytes" }
        return nil
    }

    /// Records `problem` (if any) as an issue of the calling test.
    static func record(_ problem: String?, sourceLocation: SourceLocation = #_sourceLocation) {
        if let problem { Issue.record(Comment(rawValue: problem), sourceLocation: sourceLocation) }
    }

    // MARK: reading

    static func read(_ url: URL) throws -> Data {
        do { return try Data(contentsOf: url) } catch { throw LoadError(description: "cannot read \(url.path): \(error)") }
    }

    /// Typed decode (JSONDecoder reads Double exactly).
    static func decode<T: Decodable>(_ type: T.Type, from data: Data, name: String = "") throws -> T {
        do { return try JSONDecoder().decode(type, from: data) } catch {
            throw LoadError(description: "cannot decode \(name.isEmpty ? String(describing: type) : name): \(error)")
        }
    }

    /// Reads and parses a JSON object. On any failure it records an issue and returns nil.
    static func loadObject(_ url: URL, sourceLocation: SourceLocation = #_sourceLocation) -> [String: Any]? {
        do {
            guard let object = try parseJSON(try read(url)) as? [String: Any] else {
                throw LoadError(description: "\(url.lastPathComponent) is not a JSON object")
            }
            return object
        } catch {
            Issue.record(Comment(rawValue: "\(error)"), sourceLocation: sourceLocation)
            return nil
        }
    }

    /// Thread-safe memo (suites run their tests in parallel in debug runs).
    final class Memo<Value>: @unchecked Sendable {
        private let lock = NSLock()
        private var entries: [String: Value] = [:]
        init() {}
        func get(_ key: String, make: () -> Value) -> Value {
            lock.lock()
            defer { lock.unlock() }
            if let e = entries[key] { return e }
            let e = make()
            entries[key] = e
            return e
        }
    }

    // MARK: exact JSON

    /// Exact JSON parser (objects -> [String: Any], arrays -> [Any], integers -> NSNumber(Int), other numbers ->
    /// NSNumber(Double) via Swift's correctly rounded Double(String), strings, Bool, null -> NSNull). Malformed input
    /// throws instead of trapping.
    static func parseJSON(_ data: Data) throws -> Any {
        var p = JSONParser(b: [UInt8](data))
        let v = try p.value()
        p.skipWhitespace()
        guard p.i == p.b.count else { throw LoadError(description: "trailing bytes after JSON value at offset \(p.i)") }
        return v
    }

    private struct JSONParser {
        let b: [UInt8]
        var i = 0
        init(b: [UInt8]) { self.b = b }

        func fail(_ what: String) -> LoadError { LoadError(description: "JSON: \(what) at offset \(i)") }

        mutating func skipWhitespace() {
            while i < b.count, b[i] == 0x20 || b[i] == 0x0A || b[i] == 0x0D || b[i] == 0x09 { i += 1 }
        }

        func peek() throws -> UInt8 {
            guard i < b.count else { throw fail("unexpected end of input") }
            return b[i]
        }

        mutating func string() throws -> String {
            i += 1  // opening quote
            var bytes = [UInt8]()
            while try peek() != 0x22 {
                if b[i] == 0x5C {
                    i += 1
                    switch try peek() {
                    case 0x6E: bytes.append(0x0A)
                    case 0x74: bytes.append(0x09)
                    case 0x72: bytes.append(0x0D)
                    case 0x62: bytes.append(0x08)
                    case 0x66: bytes.append(0x0C)
                    case 0x75:
                        guard i + 4 < b.count,
                            let code = UInt32(String(decoding: b[(i + 1) ... (i + 4)], as: UTF8.self), radix: 16),
                            let scalar = UnicodeScalar(code)
                        else { throw fail("bad \\u escape") }
                        bytes.append(contentsOf: Array(String(scalar).utf8))
                        i += 4
                    default: bytes.append(b[i])  // \" \\ \/
                    }
                } else {
                    bytes.append(b[i])
                }
                i += 1
            }
            i += 1
            return String(decoding: bytes, as: UTF8.self)
        }

        mutating func value() throws -> Any {
            skipWhitespace()
            switch try peek() {
            case 0x7B:
                i += 1
                var o = [String: Any]()
                skipWhitespace()
                if try peek() == 0x7D { i += 1; return o }
                while true {
                    skipWhitespace()
                    guard try peek() == 0x22 else { throw fail("object key must be a string") }
                    let k = try string()
                    skipWhitespace()
                    guard try peek() == 0x3A else { throw fail("expected ':'") }
                    i += 1
                    o[k] = try value()
                    skipWhitespace()
                    let c = try peek()
                    i += 1
                    if c == 0x2C { continue }
                    guard c == 0x7D else { throw fail("expected ',' or '}'") }
                    return o
                }
            case 0x5B:
                i += 1
                var a = [Any]()
                skipWhitespace()
                if try peek() == 0x5D { i += 1; return a }
                while true {
                    a.append(try value())
                    skipWhitespace()
                    let c = try peek()
                    i += 1
                    if c == 0x2C { continue }
                    guard c == 0x5D else { throw fail("expected ',' or ']'") }
                    return a
                }
            case 0x22: return try string()
            case 0x74: i += 4; return NSNumber(value: true)
            case 0x66: i += 5; return NSNumber(value: false)
            case 0x6E: i += 4; return NSNull()
            default:
                let start = i
                while i < b.count, "+-0123456789.eE".utf8.contains(b[i]) { i += 1 }
                let tok = String(decoding: b[start ..< i], as: UTF8.self)
                if !tok.contains(where: { ".eE".contains($0) }), let n = Int(tok) { return NSNumber(value: n) }
                guard let d = Double(tok) else { throw fail("bad number '\(tok)'") }
                return NSNumber(value: d)
            }
        }
    }
}

struct ConformanceGoldenKitTests {
    @Test("GoldenKit: exact parsers agree with Double(String) on a float32 rounding tie and a subnormal")
    func exactNumbers() throws {
        let text = "[-0.024411658756434917, 0.1, 1e-39, 3.4028234663852886e38, 7, -3]"
        let want: [Double] = [-0.024411658756434917, 0.1, 1e-39, 3.4028234663852886e38, 7, -3]
        let viaParse = try #require(try GoldenKit.parseJSON(Data(text.utf8)) as? [Any]).map { ($0 as! NSNumber).doubleValue }
        let viaDecode = try GoldenKit.decode([Double].self, from: Data(text.utf8))
        #expect(viaParse.map(\.bitPattern) == want.map(\.bitPattern))
        #expect(viaDecode.map(\.bitPattern) == want.map(\.bitPattern))
    }

    @Test("GoldenKit: malformed JSON throws instead of trapping")
    func malformedThrows() {
        for bad in ["", "{", "[1,", "{\"a\" 1}", "[1 2]", "{\"a\":\"\\u12\"}", "1 2"] {
            #expect(throws: GoldenKit.LoadError.self, "\(bad)") { _ = try GoldenKit.parseJSON(Data(bad.utf8)) }
        }
    }

    @Test("GoldenKit: verify reports a flipped bit, a wrong size and an unknown file")
    func verifyReports() {
        let data = Data("fixture".utf8)
        let good = GoldenKit.sha256(data)
        #expect(GoldenKit.verify("a.json", data: data, sha256: good, bytes: data.count) == nil)
        #expect(GoldenKit.verify("a.json", data: Data("fixturf".utf8), sha256: good) != nil)
        #expect(GoldenKit.verify("a.json", data: data, sha256: good, bytes: 1)?.contains("bytes") == true)
        #expect(GoldenKit.verify("b.json", data: data, sha256: nil) != nil)
    }

    @Test("GoldenKit: env override redirects the fixture directory")
    func overrideEnvUnsetUsesDefault() {
        // The override variable name is arbitrary and unset here, so the default location is used.
        let url = GoldenKit.directory("Golden/kda", overrideEnv: "GOLDENKIT_TEST_UNSET_VARIABLE")
        #expect(url.path.hasSuffix("Tests/MLXLMTests/Conformance/Golden/kda"))
        #expect(GoldenKit.directory("tools/conformance/golden").path.hasSuffix("/tools/conformance/golden"))
    }
}
