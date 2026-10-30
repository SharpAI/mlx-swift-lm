// Dequantisation golden conformance (lane SL-36 r8-dequant).
//
// Oracle: float64 torch reference transcribed from vLLM 739e5945d (tools/conformance/golden/dequant/dequant_ref.py).
// CROSS-CHECKED against vLLM's own code on >= 200 random cases (MANIFEST.json "crosscheck"): nvfp4
// (nvfp4_emulation_utils.dequantize_to_dtype), mxfp8 e8m0 decode, fp8 scaled_dequantize (channel + block), int4 / AWQ
// pack-unpack, and the fp8 e4m3fn / e5m2 / e8m0 value tables against the torch dtypes.
// NOT cross-checked against any vLLM code: mxfp4 (vLLM delegates to amd-quark, absent from the snapshot; the golden is
// nvfp4's nibble order (I1) glued to the mxfp8 e8m0 decode (I2)) and the ue8m0 FP8 weight-scale composition (I4); both
// are the author's reading. Green mxfp4 / ue8m0 rows therefore show "MLX == this reading", not "MLX == a real
// checkpoint": check them against a real MXFP4 sample or the OCP MX spec vectors before relying on them.
// Fixtures: tools/conformance/golden/dequant/fixtures/*.json, loaded via #filePath (no Package.swift resource);
// the manifest sha256 is verified before any fixture is used. Tests never run Python.
//
// REAL code under test (nothing reimplemented here):
//   mxfp4  : SharpAIOverlay.mapMXFP4Experts (bit-preserving view + rename) -> MLX dequantized(mode: .mxfp4)
//   nvfp4  : MLX dequantized(mode: .nvfp4, globalScale:)
//   fp8    : MLXFast.fromFp8, SharpAIOverlay.dequantFP8Blocks (block + ue8m0), FP8Linear prefill path
//   awq    : convertAutoAWQ (ParoQuantLoader) -> MLX affine dequantized
//   int4 compressed-tensors pack-quantized, NVFP4 checkpoint loader, fp8 e5m2: no Swift path -> ConformanceGap (OPEN)
// Inferred points of the oracle (nibble order of mxfp4, ue8m0 composition, NVFP4 multiplier convention, ...) are
// listed at the top of dequant_ref.py.
// upstream-watch: MLX dequantize (mlx-swift), MXFP4Checkpoint.swift, FP8Linear.swift, ParoQuantLoader.swift.

import Foundation
import MLX
import Testing

@testable import MLXLMCommon

enum DequantGolden {
    /// Repo root (shared with the other golden suites).
    static let repoRoot: URL = GoldenKit.repoRoot
    /// DEQUANT_GOLDEN_DIR points the suite at another fixture directory (to show it red against a perturbed copy).
    static let dir = GoldenKit.directory("tools/conformance/golden/dequant/fixtures", overrideEnv: "DEQUANT_GOLDEN_DIR")

    static func manifest() -> [String: Any] { json("MANIFEST.json") }

    /// A fixture without its integrity check. An unreadable or unparsable file is recorded as an issue (never trapped)
    /// and reads as an empty object.
    static func json(_ name: String) -> [String: Any] {
        GoldenKit.loadObject(dir.appendingPathComponent(name)) ?? [:]
    }

    static func parseJSON(_ data: Data) -> Any { (try? GoldenKit.parseJSON(data)) ?? NSNull() }

    static func sha256(_ url: URL) -> String { GoldenKit.sha256((try? Data(contentsOf: url)) ?? Data()) }

    static func sha256(_ data: Data) -> String { GoldenKit.sha256(data) }

    /// nil when `data` is what `files` (the manifest's "files") records for `name`; otherwise the problem.
    static func verify(_ name: String, data: Data, files: [String: [String: Any]]) -> String? {
        let info = files[name]
        guard let problem = GoldenKit.verify(name, data: data, sha256: info?["sha256"] as? String, bytes: info?["bytes"] as? Int)
        else { return nil }
        return info == nil ? problem : problem + " (regenerate with gen_dequant_golden.py)"
    }

    /// Parsed fixtures and their integrity problem, read once per process (tests run in parallel).
    private static let cache = GoldenKit.Memo<(json: [String: Any], problem: String?)>()

    private static func read(_ name: String) -> (json: [String: Any], problem: String?) {
        do {
            let data = try GoldenKit.read(dir.appendingPathComponent(name))
            guard let object = try GoldenKit.parseJSON(data) as? [String: Any] else { return ([:], "\(name) is not a JSON object") }
            return (object, nil)
        } catch {
            return ([:], "\(error)")
        }
    }

    /// Loads a fixture after checking it against the manifest. A mismatch is RECORDED as an issue of the calling test
    /// (not trapped: a trap would kill the whole test process and every unrelated test after it).
    static func load(_ name: String) -> [String: Any] {
        let e = cache.get(name) {
            let r = read(name)
            if r.problem != nil { return r }
            let files = manifest()["files"] as? [String: [String: Any]] ?? [:]
            let data = (try? Data(contentsOf: dir.appendingPathComponent(name))) ?? Data()
            return (r.json, verify(name, data: data, files: files))
        }
        GoldenKit.record(e.problem)
        return e.json
    }

    /// Fixture content without the integrity check; for test ARGUMENT lists, which are evaluated outside any test.
    static func unverified(_ name: String) -> [String: Any] {
        cache.get("raw/" + name) { (read(name).json, nil) }.json
    }

    /// The `key` array of a fixture. A missing array is recorded as an issue and reads as empty.
    static func cases(_ file: String, _ key: String = "cases") -> [[String: Any]] {
        guard let c = load(file)[key] as? [[String: Any]] else {
            GoldenKit.record("\(file) has no '\(key)' array")
            return []
        }
        return c
    }

    /// Case ids for a parameterised test. When the fixture cannot be read the list holds one sentinel id, so the test
    /// runs once and goes red instead of silently running zero cases.
    static func ids(_ file: String, _ key: String = "cases") -> [String] {
        guard let c = unverified(file)[key] as? [[String: Any]], !c.isEmpty else { return ["LOAD-FAILED"] }
        return c.map { $0["id"] as? String ?? "MALFORMED-ID" }
    }

    /// The `byte` of every e8m0.json entry, for a parameterised test (evaluated outside any test, so unverified).
    /// An unreadable or malformed fixture yields the sentinel -1, so the test runs once and goes red (the entry
    /// lookup finds no byte -1) instead of the argument list trapping or silently running zero cases.
    static var e8m0ByteArguments: [Int] {
        guard let rows = unverified("e8m0.json")["bytes"] as? [[String: Any]], !rows.isEmpty else { return [-1] }
        return rows.map { ($0["byte"] as? NSNumber)?.intValue ?? -1 }
    }

    /// MLXArray from fixture data. A count that does not match the shape is RECORDED (not a precondition trap in MLX);
    /// the result is then zeros of the requested shape (or [0] for an invalid shape) so the caller's comparison fails.
    static func shaped<T: HasDType>(_ data: [T], _ shape: [Int], sourceLocation: SourceLocation = #_sourceLocation) -> MLXArray {
        let valid = shape.allSatisfy { $0 >= 0 }
        if valid, shape.reduce(1, *) == data.count { return MLXArray(data, shape) }
        GoldenKit.record("fixture array has \(data.count) elements but shape \(shape) needs \(valid ? shape.reduce(1, *) : -1)", sourceLocation: sourceLocation)
        return MLXArray.zeros(valid ? shape : [0], dtype: T.dtype)
    }

    static func entry(_ file: String, _ id: String, key: String = "cases") -> [String: Any] {
        guard let e = cases(file, key).first(where: { $0["id"] as? String == id }) else {
            GoldenKit.record("\(file): no case '\(id)' under '\(key)'")
            return [:]
        }
        return e
    }

    /// Typed field readers. A malformed or missing fixture value is RECORDED as an issue of the calling test and reads
    /// as a neutral value (0 / empty / NaN); it is never force-cast, because a trap would kill the whole test process.
    static func d(_ v: Any?, sourceLocation: SourceLocation = #_sourceLocation) -> Double {
        if let s = v as? String {
            switch s {
            case "nan": return .nan
            case "inf": return .infinity
            case "-inf": return -.infinity
            default:
                GoldenKit.record("bad number token \(s)", sourceLocation: sourceLocation)
                return .nan
            }
        }
        guard let n = v as? NSNumber else {
            GoldenKit.record("fixture value is not a number: \(String(describing: v))", sourceLocation: sourceLocation)
            return .nan
        }
        return n.doubleValue
    }

    static func doubles(_ v: Any?, sourceLocation: SourceLocation = #_sourceLocation) -> [Double] {
        guard let a = v as? [Any] else {
            GoldenKit.record("fixture value is not an array: \(String(describing: v))", sourceLocation: sourceLocation)
            return []
        }
        return a.map { d($0, sourceLocation: sourceLocation) }
    }

    static func ints(_ v: Any?, sourceLocation: SourceLocation = #_sourceLocation) -> [Int] {
        guard let a = v as? [Any] else {
            GoldenKit.record("fixture value is not an array: \(String(describing: v))", sourceLocation: sourceLocation)
            return []
        }
        return a.map { int($0, sourceLocation: sourceLocation) }
    }

    static func int(_ v: Any?, sourceLocation: SourceLocation = #_sourceLocation) -> Int {
        guard let n = v as? NSNumber else {
            GoldenKit.record("fixture value is not an integer: \(String(describing: v))", sourceLocation: sourceLocation)
            return 0
        }
        return n.intValue
    }

    static func bool(_ v: Any?, sourceLocation: SourceLocation = #_sourceLocation) -> Bool {
        guard let b = v as? Bool else {
            GoldenKit.record("fixture value is not a boolean: \(String(describing: v))", sourceLocation: sourceLocation)
            return false
        }
        return b
    }

    static func dict(_ v: Any?, sourceLocation: SourceLocation = #_sourceLocation) -> [String: Any] {
        guard let m = v as? [String: Any] else {
            GoldenKit.record("fixture value is not an object: \(String(describing: v))", sourceLocation: sourceLocation)
            return [:]
        }
        return m
    }

    /// float32_affine_cancellation. `scales[i]` is the scale of the group element i belongs to: the absolute
    /// tolerance of element i is atol_per_scale * scales[i] (its OWN group's scale, see MANIFEST "why").
    static func affineMismatch(_ got: [Float], _ want: [Double], scales: [Double]) -> String? {
        guard scales.count == got.count else { return "scales count \(scales.count) != \(got.count)" }
        guard let t = (manifest()["tolerances"] as? [String: [String: Any]])?["float32_affine_cancellation"],
            let atolPerScale = t["atol_per_scale"], let rtol = t["rtol"]
        else { return "MANIFEST.json has no float32_affine_cancellation tolerance" }
        let perScale = d(atolPerScale)
        return mismatch(got, want, rtol: d(rtol), atolAt: { perScale * scales[$0] })
    }

    static func tolerance(_ name: String) -> (rtol: Double, atol: Double) {
        guard let t = (manifest()["tolerances"] as? [String: [String: Any]])?[name], let r = t["rtol"], let a = t["atol"] else {
            GoldenKit.record("MANIFEST.json has no tolerance '\(name)'")
            return (0, 0)  // strictest: a missing tolerance can only make comparisons fail, never pass
        }
        return (d(r), d(a))
    }

    /// nil when `got` equals the float32 rounding of `want` within rtol/atol (NaN matches NaN, inf matches inf).
    static func mismatch(_ got: [Float], _ want: [Double], rtol: Double, atol: Double) -> String? {
        mismatch(got, want, rtol: rtol, atolAt: { _ in atol })
    }

    /// Same, with an absolute tolerance per element index.
    static func mismatch(_ got: [Float], _ want: [Double], rtol: Double, atolAt: (Int) -> Double) -> String? {
        guard got.count == want.count else { return "count \(got.count) != \(want.count)" }
        var bad = [String]()
        for i in got.indices {
            let w = Float(want[i]), g = got[i]
            if w.isNaN {
                if !g.isNaN { bad.append("[\(i)] got \(g) want NaN") }
            } else if g.isNaN {
                bad.append("[\(i)] got NaN want \(w)")
            } else if w.isInfinite || g.isInfinite {
                if g != w { bad.append("[\(i)] got \(g) want \(w)") }
            } else if Double(abs(g - w)) > atolAt(i) + rtol * Double(abs(w)) {
                bad.append("[\(i)] got \(g) want \(w)")
            }
        }
        return bad.isEmpty ? nil : "\(bad.count) of \(got.count) differ; first: " + bad.prefix(3).joined(separator: "; ")
    }

    static func floats(_ a: MLXArray) -> [Float] {
        let f = a.asType(.float32)
        eval(f)
        return f.asArray(Float.self)
    }

    /// Known (oracle != shipping) cases: id -> reason. EMPTY means shipping code agrees with the oracle everywhere.
    static let knownGaps: [String: String] = [
        "e8m0/0": subnormalFlush, "mxfp4/scale_byte_0": subnormalFlush, "mxfp4/scale_byte_1": subnormalFlush,
        "fp8_ue8m0_edge/0": ue8m0ByteZero,
    ]
    static let ue8m0ByteZero =
        "dequantFP8Blocks(ue8m0: true) decodes byte 0 as 0.0 (bits << 23) where the oracle gives 2^-127 (a float32 subnormal); real scales avoid byte 0 (OPEN)"
    static let subnormalFlush =
        "MLX mxfp4 dequantize flushes float32 SUBNORMAL results to zero (vLLM float32 keeps them); absolute error <= 2^-126 (1.2e-38), only for scale bytes 0/1 (OPEN)"

    /// Either a plain #expect, or (for a recorded disagreement) a tracked ConformanceGap.
    static func check(_ key: String, _ problem: @autoclosure () -> String?, sourceLocation: SourceLocation = #_sourceLocation) {
        let p = problem()
        if let why = knownGaps[key] {
            ConformanceGap.knownIssue("dequant[\(key)]: \(why)", sourceLocation: sourceLocation) {
                ConformanceGap.expect(p == nil, Comment(rawValue: p ?? "agrees"), sourceLocation: sourceLocation)
            }
        } else {
            #expect(p == nil, Comment(rawValue: "\(key): \(p ?? "")"), sourceLocation: sourceLocation)
        }
    }

    // MARK: real-path adapters

    /// mxfp4 through the shipping checkpoint mapping and MLX dequantize.
    static func mxfp4Dequant(rows: Int, cols: Int, packed: [Int], scales: [Int]) -> [Float] {
        let w = shaped(packed.map { UInt8($0) }, [rows, cols / 2])
        let s = shaped(scales.map { UInt8($0) }, [rows, cols / 32])
        let mapped = SharpAIOverlay.mapMXFP4Experts(
            ["m.layers.0.ffn.experts.0.w.weight": w, "m.layers.0.ffn.experts.0.w.scale": s], expertDtype: "fp4")
        let mw = mapped["m.layers.0.ffn.experts.0.w.weight"]!
        let ms = mapped["m.layers.0.ffn.experts.0.w.scales"]!
        guard mw.dtype == .uint32, mw.shape == [rows, cols / 8] else {
            Issue.record("mapMXFP4Experts returned dtype \(mw.dtype) shape \(mw.shape), expected uint32 [\(rows), \(cols / 8)]")
            return []
        }
        return floats(dequantized(mw, scales: ms, biases: nil, groupSize: 32, bits: 4, mode: .mxfp4, dtype: .float32))
    }

    /// MLX's nvfp4 `globalScale` is the ENCODER-side per-tensor value: the kernel decodes with
    /// `globalScale / (F8E4M3_MAX * F4E2M1_MAX)` (mlx backend/metal/kernels/fp_quantized.h:216), i.e. a multiplier m
    /// (the one vLLM's emulation applies, nvfp4_emulation_utils.py:63) is passed as m * 448 * 6. Found by the first
    /// run of this suite (values off by exactly 1/2688); not a defect, but a loader must convert.
    static let mlxNvfp4GlobalScaleUnit: Double = 448 * 6

    static func nvfp4Dequant(rows: Int, cols: Int, packed: [Int], scales: [Int], multiplier: Double) -> [Float] {
        let w = MLXArray(packed.map { UInt8($0) }, [rows, cols / 2]).view(dtype: .uint32)
        let s = shaped(scales.map { UInt8($0) }, [rows, cols / 16])
        return floats(
            dequantized(
                w, scales: s, biases: nil, groupSize: 16, bits: 4, mode: .nvfp4,
                globalScale: MLXArray(Float(multiplier * mlxNvfp4GlobalScaleUnit)), dtype: .float32))
    }

    static func fp8Weight(_ bytes: [Int], _ rows: Int, _ cols: Int) -> MLXArray {
        shaped(bytes.map { UInt8($0) }, [rows, cols])
    }
}

struct ConformanceDequantGoldenTests {
    typealias G = DequantGolden

    // MARK: manifest

    @Test("manifest: every fixture's sha256 matches, no stray files, generator inputs recorded")
    func manifestIntegrity() {
        let m = G.manifest()
        let files = m["files"] as? [String: [String: Any]] ?? [:]
        #expect(m["vllm_commit"] as? String == "739e5945d")
        #expect((m["torch_version"] as? String)?.isEmpty == false)
        #expect(files.count == 8)
        for name in files.keys.sorted() {
            let data = (try? Data(contentsOf: G.dir.appendingPathComponent(name))) ?? Data()
            #expect(G.verify(name, data: data, files: files) == nil, "\(G.verify(name, data: data, files: files) ?? "")")
        }
        let onDisk = Set(
            ((try? FileManager.default.contentsOfDirectory(atPath: G.dir.path)) ?? []).filter { $0.hasSuffix(".json") })
        #expect(onDisk == Set(files.keys).union(["MANIFEST.json"]), "fixtures on disk differ from the manifest")
        let tol = m["tolerances"] as? [String: [String: Any]] ?? [:]
        for k in ["float32_exact", "float32_product", "float32_affine_cancellation", "bfloat16_values"] {
            #expect((tol[k]?["why"] as? String)?.isEmpty == false, "tolerance \(k) needs a reason")
        }
        guard let cc = m["crosscheck"] as? [String: Any] else { Issue.record("MANIFEST.json has no crosscheck section"); return }
        #expect(Set(cc.keys).isSuperset(of: ["nvfp4", "mxfp8_e8m0_decode", "fp8_scaled_dequantize", "int4"]))
        // the crosscheck numbers are self-reported by the generator; at least pin that they claim real coverage
        for (k, field) in [("nvfp4", "cases"), ("mxfp8_e8m0_decode", "cases"), ("fp8_scaled_dequantize", "cases"), ("int4", "cases_each")] {
            let n = (cc[k] as? [String: Any])?[field] as? Int
            #expect((n ?? 0) >= 200, "crosscheck.\(k).\(field) is \(n.map(String.init) ?? "missing"), expected >= 200")
        }
    }

    // MARK: fp4 e2m1 / e8m0

    @Test("all 16 fp4 codes, in the low and the high nibble, through the real mxfp4 path")
    func fp4Codes() {
        let table = G.cases("fp4_e2m1_codes.json", "codes")
        #expect(table.count == 16)
        for e in table {
            let c = G.int(e["code"])
            let want = G.d(e["value"])
            // low nibble = c (element 0), high nibble = 2 (1.0, element 1); scale byte 127 = 1
            let lo = G.mxfp4Dequant(rows: 1, cols: 32, packed: [Int](repeating: c | (2 << 4), count: 16), scales: [127])
            let hi = G.mxfp4Dequant(rows: 1, cols: 32, packed: [Int](repeating: 2 | (c << 4), count: 16), scales: [127])
            #expect(lo[0] == Float(want) && lo[1] == 1, "code \(c) low nibble: got \(lo[0]) want \(want)")
            #expect(hi[1] == Float(want) && hi[0] == 1, "code \(c) high nibble: got \(hi[1]) want \(want)")
        }
    }

    @Test("e8m0 scale bytes 0/1/2/126/127/128/253/254 decode to 2^(byte-127)", arguments: G.e8m0ByteArguments)
    func e8m0Bytes(byte: Int) {
        guard let entry = G.cases("e8m0.json", "bytes").first(where: { $0["byte"] as? Int == byte }) else {
            Issue.record("e8m0.json has no entry for byte \(byte)")
            return
        }
        let want = G.d(entry["value"])
        // every element is code 2 (= 1.0) so the output equals the decoded scale
        let got = G.mxfp4Dequant(rows: 1, cols: 32, packed: [Int](repeating: 0x22, count: 16), scales: [byte])
        G.check("e8m0/\(byte)", G.mismatch(got, [Double](repeating: want, count: 32), rtol: 0, atol: 0))
    }

    @Test("e8m0 byte 255: vLLM's formula overflows float32 to inf (torch float8_e8m0fnu says NaN)")
    func e8m0Byte255() {
        let sp = G.dict(G.load("e8m0.json")["special"])
        #expect(sp["byte"] as? Int == 255)
        #expect(sp["vllm_formula_float32"] as? String == "inf")
        let got = G.mxfp4Dequant(rows: 1, cols: 32, packed: [Int](repeating: 0x22, count: 16), scales: [255])
        G.check("e8m0/255", G.mismatch(got, [Double](repeating: .infinity, count: 32), rtol: 0, atol: 0))
    }

    // MARK: mxfp4

    @Test("mxfp4 golden cases (nibble order, group boundaries, scale edges, random)", arguments: G.ids("mxfp4.json"))
    func mxfp4(id: String) {
        let c = G.entry("mxfp4.json", id)
        guard !c.isEmpty else { return }  // entry() recorded the problem
        let rows = G.int(c["rows"]), cols = G.int(c["cols"])
        let got = G.mxfp4Dequant(rows: rows, cols: cols, packed: G.ints(c["packed"]), scales: G.ints(c["scales"]))
        let t = G.tolerance("float32_exact")
        G.check("mxfp4/\(id)", G.mismatch(got, G.doubles(c["expected"]), rtol: t.rtol, atol: t.atol))
    }

    // MARK: nvfp4

    @Test("nvfp4 golden cases (e4m3 block scales, global multiplier, nibble order)", arguments: G.ids("nvfp4.json"))
    func nvfp4(id: String) {
        let c = G.entry("nvfp4.json", id)
        guard !c.isEmpty else { return }  // entry() recorded the problem
        let got = G.nvfp4Dequant(
            rows: G.int(c["rows"]), cols: G.int(c["cols"]), packed: G.ints(c["packed"]), scales: G.ints(c["scales"]),
            multiplier: G.d(c["global_scale"]))
        let t = G.tolerance("float32_exact")
        G.check("nvfp4/\(id)", G.mismatch(got, G.doubles(c["expected"]), rtol: t.rtol, atol: t.atol))
    }

    // MARK: fp8

    @Test("fp8 e4m3fn: all 256 codes (subnormals, max 448, NaN encodings) via MLXFast.fromFp8, float32 and bfloat16")
    func fp8E4m3Codes() {
        let want = G.doubles(G.load("fp8_codes.json")["e4m3fn"])
        #expect(want.count == 256 && want[0x7E] == 448 && want[0x7F].isNaN && want[0xFF].isNaN)
        #expect(want[0x01] == pow(2.0, -9) && want[0x07] == 7 * pow(2.0, -9) && want[0x08] == pow(2.0, -6))
        let all = MLXArray((0 ..< 256).map { UInt8($0) })
        let t = G.tolerance("bfloat16_values")
        for dt in [DType.float32, .bfloat16] {
            let got = G.floats(MLXFast.fromFp8(all, dtype: dt))
            G.check("fp8_codes/e4m3fn/\(dt)", G.mismatch(got, want, rtol: t.rtol, atol: t.atol))
        }
    }

    @Test("fp8 e5m2 codes: no Swift decoder (fromFp8 is e4m3 only)")
    func fp8E5m2Codes() {
        let want = G.doubles(G.load("fp8_codes.json")["e5m2"])
        #expect(want.count == 256 && want[0x7B] == 57344 && want[0x7C] == .infinity && want[0x7D].isNaN)
        let got = G.floats(MLXFast.fromFp8(MLXArray((0 ..< 256).map { UInt8($0) }), dtype: .float32))
        ConformanceGap.knownIssue("fp8 e5m2 weights/activations have no Swift decode path: MLXFast.fromFp8 reads every byte as e4m3 (OPEN)") {
            ConformanceGap.expect(G.mismatch(got, want, rtol: 0, atol: 0) == nil)
        }
    }

    @Test("fp8 per-channel scale: fromFp8 * scale[N,1], and dequantFP8Blocks(blockSize: 1) broadcast",
        arguments: G.ids("fp8_scaled.json", "channel"))
    func fp8Channel(id: String) {
        let c = G.entry("fp8_scaled.json", id, key: "channel")
        guard !c.isEmpty else { return }  // entry() recorded the problem
        let (n, k) = (G.int(c["rows"]), G.int(c["cols"]))
        let w = G.fp8Weight(G.ints(c["weight"]), n, k)
        let scale = G.shaped(G.doubles(c["scale"]).map { Float($0) }, [n, 1])
        let want = G.doubles(c["expected"])
        let t = G.tolerance("float32_product")
        G.check("fp8_channel/\(id)/fromFp8*scale", G.mismatch(G.floats(MLXFast.fromFp8(w, dtype: .float32) * scale), want, rtol: t.rtol, atol: t.atol))
        let blk = SharpAIOverlay.dequantFP8Blocks(
            weight: MLXFast.fromFp8(w, dtype: .float32), scaleInv: scale, blockSize: 1, ue8m0: false)
        G.check("fp8_channel/\(id)/dequantFP8Blocks1", G.mismatch(G.floats(blk), want, rtol: t.rtol, atol: t.atol))
    }

    @Test("fp8 block scale (weight_scale_inv): dequantFP8Blocks and the FP8Linear prefill path", arguments: G.ids("fp8_scaled.json", "block"))
    func fp8Block(id: String) {
        let c = G.entry("fp8_scaled.json", id, key: "block")
        guard !c.isEmpty else { return }  // entry() recorded the problem
        let (n, k, bs) = (G.int(c["rows"]), G.int(c["cols"]), G.int(c["block"]))
        let shape = G.ints(c["scale_shape"])
        let w = G.fp8Weight(G.ints(c["weight"]), n, k)
        let scale = G.shaped(G.doubles(c["scale"]).map { Float($0) }, shape)
        let want = G.doubles(c["expected"])
        let t = G.tolerance("float32_product")
        let deq = SharpAIOverlay.dequantFP8Blocks(
            weight: MLXFast.fromFp8(w, dtype: .float32), scaleInv: scale, blockSize: bs, ue8m0: false)
        G.check("fp8_block/\(id)/dequantFP8Blocks", G.mismatch(G.floats(deq), want, rtol: t.rtol, atol: t.atol))
        // FP8Linear prefill: out = x @ dequantised^T. x = identity [k, k] (k * k elements != inputDims = k, so the
        // multi-token branch runs) gives out = dequantised^T.
        let layer = FP8Linear(weight: w, weightScaleInv: scale, bias: nil, blockSize: bs)
        let out = layer(MLXArray.identity(k, dtype: .float32))
        #expect(out.shape == [k, n])
        G.check("fp8_block/\(id)/FP8Linear", G.mismatch(G.floats(out.transposed()), want, rtol: t.rtol, atol: t.atol))
    }

    /// One fp8 block case, optionally cropped to a NON-divisible shape (rows - (bs-1), cols - 1: the scale grid
    /// is unchanged because ceil(n'/bs) == n/bs). Elementwise dequantisation depends only on the element's own block
    /// scale, so the cropped golden is the corner of the full golden (no new maths; I6: vLLM asserts divisibility).
    struct BlockCase {
        let n: Int, k: Int, bs: Int
        let weight: [Int], scale: [Double], shape: [Int], want: [Double]
        init(_ id: String, cropped: Bool) {
            let c = G.entry("fp8_scaled.json", id, key: "block")
            let (fn, fk, bs) = (G.int(c["rows"]), G.int(c["cols"]), G.int(c["block"]))
            let (n, k) = cropped ? (fn - (bs - 1), fk - 1) : (fn, fk)
            let w = G.ints(c["weight"]), e = G.doubles(c["expected"])
            let keep = (0 ..< n).flatMap { r in (0 ..< k).map { r * fk + $0 } }
            self.n = n
            self.k = k
            self.bs = bs
            weight = keep.map { w[$0] }
            want = keep.map { e[$0] }
            scale = G.doubles(c["scale"])
            shape = G.ints(c["scale_shape"])
        }
        var weightArray: MLXArray { G.fp8Weight(weight, n, k) }
        func scaleArray(_ dt: DType = .float32) -> MLXArray { MLXArray(scale.map { Float($0) }, shape).asType(dt) }
    }

    /// FP8Linear's single-token branch (x.size == inputDims: the custom Metal GEMV with its own e4m3 decoder, bf16
    /// scale reads and block indexing). x = e_j picks column j of the dequantised weight.
    /// Tolerance: the kernel reads the scale as bfloat16 and writes a bfloat16 result: two roundings of <= 2^-9 each,
    /// so <= 2^-8 + 2^-18 relative (0.00392); 0.004 is that bound, a wrong exponent bias (factor 2) is far outside it.
    ///
    /// GATED: FP8Linear's kernel does not compile today. Its complete kernel text (includes, an `inline` helper, a `kernel void`
    /// with [[buffer]] parameters) is passed to MLXFast.metalKernel as the BODY, so the first single-token call dies in
    /// mlx::metal::Device ("redundant #include ... function definition is not allowed here", fatal, not catchable). Until
    /// that is fixed the decode comparisons only run with DEQUANT_GOLDEN_FP8_DECODE=1 and are otherwise recorded as a gap.
    /// The comparison harness itself is therefore UNVERIFIED against a compiling kernel.
    static let fp8DecodeEnabled = ProcessInfo.processInfo.environment["DEQUANT_GOLDEN_FP8_DECODE"] == "1"

    static func checkDecode(_ key: String, _ b: BlockCase) {
        guard fp8DecodeEnabled else {
            ConformanceGap.knownIssue(
                "FP8Linear single-token decode cannot run: its Metal kernel source is a complete kernel passed as a metalKernel BODY and fails to compile (fatal); set DEQUANT_GOLDEN_FP8_DECODE=1 to execute (OPEN)"
            ) {
                ConformanceGap.expect(false, "\(key): not executed")
            }
            return
        }
        G.check(key, G.mismatch(decodeColumns(b), b.want, rtol: 0.004, atol: 0))
    }

    static func decodeColumns(_ b: BlockCase) -> [Float] {
        let layer = FP8Linear(weight: b.weightArray, weightScaleInv: b.scaleArray(.bfloat16), bias: nil, blockSize: b.bs)
        var out = [Float](repeating: 0, count: b.n * b.k)
        for j in 0 ..< b.k {
            let x = MLXArray((0 ..< b.k).map { $0 == j ? Float(1) : Float(0) }, [1, b.k]).asType(.bfloat16)
            let col = G.floats(layer(x))
            precondition(col.count == b.n)
            for r in 0 ..< b.n { out[r * b.k + j] = col[r] }
        }
        return out
    }

    @Test("FP8Linear single-token decode (Metal GEMV) equals the golden within bfloat16 rounding", arguments: G.ids("fp8_scaled.json", "block"))
    func fp8Decode(id: String) {
        let b = BlockCase(id, cropped: false)
        Self.checkDecode("fp8_decode/\(id)", b)
    }

    @Test("non-divisible fp8 shapes (zero-pad and crop): dequantFP8Blocks, FP8Linear prefill and decode equal the cropped golden",
        arguments: G.ids("fp8_scaled.json", "block").filter { (G.entry("fp8_scaled.json", $0, key: "block")["block"] as? Int ?? 2) >= 2 })
    func fp8NonDivisible(id: String) {
        let b = BlockCase(id, cropped: true)
        #expect(b.n % b.bs != 0 && b.k % b.bs != 0 && b.n > 0 && b.k > 1, "case must be non-divisible")
        let exact = G.tolerance("float32_product")
        let deq = SharpAIOverlay.dequantFP8Blocks(
            weight: MLXFast.fromFp8(b.weightArray, dtype: .float32), scaleInv: b.scaleArray(), blockSize: b.bs, ue8m0: false)
        #expect(deq.shape == [b.n, b.k])
        G.check("fp8_nondiv/\(id)/dequantFP8Blocks", G.mismatch(G.floats(deq), b.want, rtol: exact.rtol, atol: exact.atol))
        let layer = FP8Linear(weight: b.weightArray, weightScaleInv: b.scaleArray(), bias: nil, blockSize: b.bs)
        let out = layer(MLXArray.identity(b.k, dtype: .float32))  // b.k * b.k != inputDims: prefill branch
        #expect(out.shape == [b.k, b.n])
        G.check("fp8_nondiv/\(id)/FP8Linear", G.mismatch(G.floats(out.transposed()), b.want, rtol: exact.rtol, atol: exact.atol))
        Self.checkDecode("fp8_nondiv/\(id)/decode", b)
    }

    @Test("fp8 ue8m0 block scales (byte -> 2^(byte-127)) through dequantFP8Blocks(ue8m0: true)",
        arguments: G.ids("fp8_scaled.json", "ue8m0_block"))
    func fp8Ue8m0(id: String) {
        let c = G.entry("fp8_scaled.json", id, key: "ue8m0_block")
        guard !c.isEmpty else { return }  // entry() recorded the problem
        let (n, k, bs) = (G.int(c["rows"]), G.int(c["cols"]), G.int(c["block"]))
        let w = G.fp8Weight(G.ints(c["weight"]), n, k)
        let sb = G.shaped(G.ints(c["scale_bytes"]).map { UInt8($0) }, G.ints(c["scale_shape"]))
        let got = SharpAIOverlay.dequantFP8Blocks(
            weight: MLXFast.fromFp8(w, dtype: .float32), scaleInv: sb, blockSize: bs, ue8m0: true)
        let t = G.tolerance("float32_exact")
        G.check("fp8_ue8m0/\(id)", G.mismatch(G.floats(got), G.doubles(c["expected"]), rtol: t.rtol, atol: t.atol))
    }

    @Test("ue8m0 edge scale bytes 0/1/254/255 through dequantFP8Blocks (oracle: 2^(byte-127); 255 = vLLM formula, inf)",
        arguments: [0, 1, 254, 255])
    func fp8Ue8m0EdgeBytes(byte: Int) {
        let e8 = G.load("e8m0.json")
        let want: Double
        if byte == 255 {
            want = G.d(G.dict(e8["special"])["vllm_formula_float32"])
        } else {
            guard let entry = (e8["bytes"] as? [[String: Any]])?.first(where: { $0["byte"] as? Int == byte }) else {
                Issue.record("e8m0.json has no entry for byte \(byte)")
                return
            }
            want = G.d(entry["value"])
        }
        let w = G.fp8Weight([0x38], 1, 1)  // e4m3 0x38 = 1.0, so the result is the decoded scale
        let sb = MLXArray([UInt8(byte)], [1, 1])
        let got = SharpAIOverlay.dequantFP8Blocks(
            weight: MLXFast.fromFp8(w, dtype: .float32), scaleInv: sb, blockSize: 1, ue8m0: true)
        G.check("fp8_ue8m0_edge/\(byte)", G.mismatch(G.floats(got), [want], rtol: 0, atol: 0))
    }

    // MARK: int4 (compressed-tensors pack-quantized) and AWQ

    @Test("int4 pack-quantized, group 32/64: the packing is MLX affine, w = q*s - zp*s (mapping feasibility, not a loader)",
        arguments: G.ids("int4_pack_quantized.json").filter { id in
            let g = G.entry("int4_pack_quantized.json", id)["group"] as? Int ?? 32
            return g == 32 || g == 64
        })
    func int4ViaAffine(id: String) {
        let c = G.entry("int4_pack_quantized.json", id)
        guard !c.isEmpty else { return }  // entry() recorded the problem
        let (got, scales) = Self.int4Dequant(c)
        G.check("int4/\(id)", G.affineMismatch(got, G.doubles(c["expected"]), scales: scales))
    }

    /// Row-major [n, k] result of the int4 case through MLX affine, and the scale of each element's group.
    static func int4Dequant(_ c: [String: Any]) -> (got: [Float], scales: [Double]) {
        let (n, k, g) = (G.int(c["rows"]), G.int(c["cols"]), G.int(c["group"]))
        // Malformed dimensions are an issue, not a divide-by-zero / shape trap; the caller's comparison then fails on the empty result.
        guard n > 0, k > 0, g > 0, k % 8 == 0, k % g == 0 else {
            Issue.record("int4 case has unusable dimensions rows=\(n) cols=\(k) group=\(g)")
            return ([], [])
        }
        let words = G.ints(c["weight_packed"]).map { UInt32(truncatingIfNeeded: $0) }
        let w = G.shaped(words, [n, k / 8])
        let s = G.shaped(G.doubles(c["scale"]).map { Float($0) }, [n, k / g])
        var zp = MLXArray.ones([n, k / g], dtype: .float32) * MLXArray(Float(8))  // uint4b8: symmetric zero point is 8
        if !(G.bool(c["symmetric"])) {
            // zero point nibbles are packed along dim 0 (dim-0 packing is stated in compressed_tensors_wNa16.py:158-183, see dequant_ref.py I7); unpack in test code
            let zw = G.ints(c["zero_packed"]).map { UInt32(truncatingIfNeeded: $0) }
            let cols = k / g
            var z = [Float](repeating: 0, count: n * cols)
            for r in 0 ..< n { for col in 0 ..< cols { z[r * cols + col] = Float((zw[(r / 8) * cols + col] >> UInt32(4 * (r % 8))) & 0xF) } }
            zp = G.shaped(z, [n, cols])
        }
        let got = dequantized(w, scales: s, biases: -(zp * s), groupSize: g, bits: 4, mode: .affine, dtype: .float32)
        let sc = G.doubles(c["scale"])
        return (G.floats(got), (0 ..< n * k).map { sc[($0 / k) * (k / g) + ($0 % k) / g] })
    }

    @Test("AWQ int4 (vLLM awq_pack layout) -> convertAutoAWQ -> MLX affine dequantize equals the golden [K,N]^T",
        arguments: G.ids("awq_int4.json"))
    func awq(id: String) {
        let c = G.entry("awq_int4.json", id)
        guard !c.isEmpty else { return }  // entry() recorded the problem
        let (got, scales) = Self.awqDequant(c)
        G.check("awq/\(id)", G.affineMismatch(got, G.doubles(c["expected_KxN"]), scales: scales))
    }

    /// Row-major [K, N] result of the AWQ case through convertAutoAWQ + MLX affine, and the scale of each element's group.
    static func awqDequant(_ c: [String: Any]) -> (got: [Float], scales: [Double]) {
        let (k, n, g) = (G.int(c["in_features"]), G.int(c["out_features"]), G.int(c["group"]))
        guard n > 0, k > 0, g > 0, k % 8 == 0, n % 8 == 0 else {
            Issue.record("AWQ case has unusable dimensions in=\(k) out=\(n) group=\(g)")
            return ([], [])
        }
        var weights: [String: MLXArray] = [
            "layer.qweight": G.shaped(G.ints(c["qweight"]).map { Int32(truncatingIfNeeded: $0) }, G.ints(c["qweight_shape"])),
            "layer.qzeros": G.shaped(G.ints(c["qzeros"]).map { Int32(truncatingIfNeeded: $0) }, G.ints(c["qzeros_shape"])),
            "layer.scales": G.shaped(G.doubles(c["scales"]).map { Float($0) }, G.ints(c["scales_shape"])),
            "other.theta": MLXArray.zeros([1], dtype: .float32),  // fixes the checkpoint float dtype (float32)
        ]
        convertAutoAWQ(&weights, groupSize: g)
        let w = weights["layer.weight"]!, s = weights["layer.scales"]!, b = weights["layer.biases"]!
        #expect(w.dtype == .uint32 && w.shape == [n, k / 8])
        let got = dequantized(w, scales: s, biases: b, groupSize: g, bits: 4, mode: .affine, dtype: .float32)  // [N, K]
        let sc = G.doubles(c["scales"])
        return (G.floats(got.transposed()), (0 ..< k * n).map { sc[(($0 / n) / g) * n + $0 % n] })
    }

    // MARK: ConformanceGap (no Swift path)

    /// `source` without `//` and `/* */` comments (block comments nest, as in Swift); string literals, including
    /// `"""` multi-line ones, are kept verbatim, so a `//` inside a literal is not taken for a comment.
    static func codeOnly(_ source: String) -> String {
        let c = Array(source.unicodeScalars)
        var out = String.UnicodeScalarView()
        var i = 0
        func at(_ j: Int) -> Unicode.Scalar? { j < c.count ? c[j] : nil }
        while i < c.count {
            if c[i] == "/", at(i + 1) == "/" {
                while i < c.count, c[i] != "\n" { i += 1 }
            } else if c[i] == "/", at(i + 1) == "*" {
                var depth = 1
                i += 2
                while i < c.count, depth > 0 {
                    if c[i] == "/", at(i + 1) == "*" { depth += 1; i += 2 }
                    else if c[i] == "*", at(i + 1) == "/" { depth -= 1; i += 2 }
                    else { i += 1 }
                }
                out.append(" ")
            } else if c[i] == "\"" {
                let multiline = at(i + 1) == "\"" && at(i + 2) == "\""
                let quote = multiline ? 3 : 1
                for _ in 0 ..< quote { out.append(c[i]); i += 1 }
                while i < c.count {
                    if c[i] == "\\", i + 1 < c.count { out.append(c[i]); out.append(c[i + 1]); i += 2; continue }
                    if c[i] == "\"" && (!multiline || (at(i + 1) == "\"" && at(i + 2) == "\"")) {
                        for _ in 0 ..< quote { out.append(c[i]); i += 1 }
                        break
                    }
                    if c[i] == "\n", !multiline { break }  // unterminated: resync at the line end
                    out.append(c[i]); i += 1
                }
            } else {
                out.append(c[i]); i += 1
            }
        }
        return String(out)
    }

    /// All Swift sources under Libraries/ (text, comments removed), for the "is there a loader for X" probes.
    static let librarySources: String = {
        let root = DequantGolden.repoRoot.appendingPathComponent("Libraries")
        var text = ""
        if let en = FileManager.default.enumerator(at: root, includingPropertiesForKeys: nil) {
            for case let u as URL in en where u.pathExtension == "swift" {
                text += codeOnly((try? String(contentsOf: u, encoding: .utf8)) ?? "")
            }
        }
        return text
    }()

    @Test("gap: compressed-tensors pack-quantized int4 checkpoint loader (weight_packed / weight_scale / weight_zero_point)")
    func gapCompressedTensorsInt4Loader() {
        let src = Self.librarySources
        #expect(src.contains("mxfp4"), "probe sanity: the library sources were read")
        ConformanceGap.knownIssue(
            "no loader maps compressed-tensors pack-quantized int4 (weight_packed, group 8/16 not supported by MLX affine either) onto MLX quantized layers (OPEN)"
        ) {
            ConformanceGap.expect(src.contains("weight_packed"))
        }
    }

    @Test("gap: NVFP4 checkpoint loader (weight_scale e4m3, weight_global_scale inverted at load)")
    func gapNvfp4Loader() {
        let src = Self.librarySources
        #expect(src.contains("mxfp4"), "probe sanity: the library sources were read")
        ConformanceGap.knownIssue(
            "no loader maps NVFP4 checkpoints (weight_global_scale -> 1/x, per-16 e4m3 weight_scale) onto MLX nvfp4 layers; only the dequantize op exists (OPEN)"
        ) {
            ConformanceGap.expect(src.contains("weight_global_scale"))
        }
    }

    // MARK: non-vacuity and coverage

    @Test("loader-gap probes read code, not comments (a comment mentioning weight_packed must not close the gap)")
    func probesIgnoreComments() {
        let src = "// weight_packed in a line comment\n/* weight_global_scale\n in a block */\nlet a = \"http://x\" // trailing note\nlet k = \"weight_packed\"\n"
        let code = Self.codeOnly(src)
        #expect(code.contains("let a") && code.contains("\"http://x\""), "string literals (even with //) stay")
        #expect(!code.contains("in a line comment") && !code.contains("in a block") && !code.contains("trailing note"))
        #expect(code.contains("\"weight_packed\""))
        #expect(!code.contains("weight_global_scale"))
    }

    @Test("the oracle and generator sources are pinned: editing dequant_ref / dequant_crosscheck / gen without regenerating fails")
    func toolsArePinned() {
        let tools = G.manifest()["tools"] as? [String: [String: Any]]
        #expect(tools != nil, "MANIFEST.json has no \"tools\" section (regenerate with gen_dequant_golden.py)")
        #expect(Set(tools?.keys ?? [:].keys) == ["dequant_ref.py", "dequant_crosscheck.py", "gen_dequant_golden.py"])
        for (name, info) in tools ?? [:] {
            let url = G.dir.deletingLastPathComponent().appendingPathComponent(name)
            guard let data = try? Data(contentsOf: url) else { Issue.record("\(name) is not readable"); continue }
            #expect(G.verify(name, data: data, files: [name: info]) == nil, "\(G.verify(name, data: data, files: [name: info]) ?? "")")
        }
    }

    @Test("fixture JSON numbers are parsed exactly (Foundation's parser is one ulp off on a float32 rounding tie)")
    func jsonNumbersAreExact() {
        guard let v = G.parseJSON(Data(#"{"a":[-0.024411658756434917, 0.1, 1e-39, 3, -7, true, "x\"y"], "b":{}}"#.utf8)) as? [String: Any],
              let a = v["a"] as? [Any], a.count == 7,
              let tie = (a[0] as? NSNumber)?.doubleValue,
              let a1 = (a[1] as? NSNumber)?.doubleValue, let a2 = (a[2] as? NSNumber)?.doubleValue
        else { Issue.record("parseJSON did not return the expected object"); return }
        #expect(tie == -0.024411658756434917 && a1 == 0.1 && a2 == 1e-39)
        // the tie is -0.01171875 * 2.0831282138824463 exactly; IEEE round-to-nearest-even of it is the CPU float32 product
        #expect(Float(tie) == Float(-0.01171875) * Float(2.0831282138824463))
        #expect(a[3] as? Int == 3 && a[4] as? Int == -7 && a[5] as? Bool == true && a[6] as? String == "x\"y")
        #expect((v["b"] as? [String: Any])?.isEmpty == true)
    }

    @Test("fixture integrity problems are reported as a problem string (no trap): tampered content, wrong size, unknown file")
    func integrityIsReported() {
        let files = G.manifest()["files"] as? [String: [String: Any]] ?? [:]
        let data = (try? Data(contentsOf: G.dir.appendingPathComponent("e8m0.json"))) ?? Data()
        #expect(G.verify("e8m0.json", data: data, files: files) == nil)
        var tampered = data
        tampered[tampered.startIndex + 10] ^= 1
        #expect(G.verify("e8m0.json", data: tampered, files: files) != nil, "one flipped bit must be reported")
        var wrongSize = files
        wrongSize["e8m0.json"]!["bytes"] = data.count + 1
        #expect(G.verify("e8m0.json", data: data, files: wrongSize)?.contains("bytes") == true)
        #expect(G.verify("nope.json", data: data, files: files) != nil)
    }

    @Test("tolerances are tight: a 1-ulp float32 product error, and an error just above one small-scale group's own bound, are rejected")
    func tolerancesAreTight() {
        // fp8 * float32 scale is one correctly rounded float32 product, so equality is required (no slack for 1 ulp)
        let ch = G.entry("fp8_scaled.json", "channel_4x8", key: "channel")
        guard !ch.isEmpty else { return }  // entry() recorded the problem
        let want = G.doubles(ch["expected"])
        let exact = want.map { Float($0) }
        let p = G.tolerance("float32_product")
        #expect(G.mismatch(exact, want, rtol: p.rtol, atol: p.atol) == nil)
        let ulp = exact.map { $0 != 0 && $0.isFinite ? $0.nextUp : $0 }
        #expect(G.mismatch(ulp, want, rtol: p.rtol, atol: p.atol) != nil, "a 1-ulp product error must be rejected")

        // affine: the bound is per group (15 * 2^-23 * that group's scale), not per case
        var qualified = 0
        let ratio = 4.0
        for (what, got, scales, want) in affineCases() {
            guard let lo = scales.indices.filter({ want[$0] != 0 }).min(by: { scales[$0] < scales[$1] }),
                scales.max()! > ratio * scales[lo]
            else { continue }
            qualified += 1
            #expect(G.affineMismatch(got, want, scales: scales) == nil, "\(what): baseline")
            var bumped = want
            bumped[lo] += 3 * 15 * pow(2.0, -23) * scales[lo]  // 3x the bound of its own group
            #expect(G.affineMismatch(got, bumped, scales: scales) != nil, "\(what): small-scale group error accepted")
        }
        #expect(qualified >= 3, "probe is vacuous: too few cases mix a small and a large scale")
    }

    func affineCases() -> [(String, [Float], [Double], [Double])] {
        var out = [(String, [Float], [Double], [Double])]()
        for id in G.ids("int4_pack_quantized.json") {
            let c = G.entry("int4_pack_quantized.json", id)
            guard !c.isEmpty else { continue }  // entry() recorded the problem
            guard [32, 64].contains(G.int(c["group"])) else { continue }
            let (got, scales) = Self.int4Dequant(c)
            out.append(("int4/\(id)", got, scales, G.doubles(c["expected"])))
        }
        for id in G.ids("awq_int4.json") {
            let c = G.entry("awq_int4.json", id)
            guard !c.isEmpty else { continue }  // entry() recorded the problem
            let (got, scales) = Self.awqDequant(c)
            out.append(("awq/\(id)", got, scales, G.doubles(c["expected_KxN"])))
        }
        return out
    }

    @Test("mutation: perturbed goldens and swapped nibbles must FAIL the comparison")
    func mutationsFail() {
        let c = G.entry("mxfp4.json", "all_bytes_unit_scale")
        guard !c.isEmpty else { return }  // entry() recorded the problem
        let rows = G.int(c["rows"]), cols = G.int(c["cols"])
        let packed = G.ints(c["packed"]), scales = G.ints(c["scales"])
        let want = G.doubles(c["expected"])
        let got = G.mxfp4Dequant(rows: rows, cols: cols, packed: packed, scales: scales)
        let t = G.tolerance("float32_exact")
        #expect(G.mismatch(got, want, rtol: t.rtol, atol: t.atol) == nil, "baseline must pass")
        // 1. one golden value nudged by 1e-6 relative (exact tolerance: must fail)
        var bumped = want
        guard let i = bumped.firstIndex(where: { $0 != 0 }) else {
            Issue.record("all_bytes_unit_scale has no non-zero golden value to perturb")
            return
        }
        bumped[i] *= 1 + 1e-6
        #expect(G.mismatch(got, bumped, rtol: t.rtol, atol: t.atol) != nil)
        // 2. a zero golden value made non-zero
        var z = want
        guard let zi = z.firstIndex(where: { $0 == 0 }) else {
            Issue.record("all_bytes_unit_scale has no zero golden value to perturb")
            return
        }
        z[zi] = 0.25
        #expect(G.mismatch(got, z, rtol: t.rtol, atol: t.atol) != nil)
        // 3. nibble order swapped in the INPUT: the real path then disagrees with the golden
        let swapped = packed.map { (($0 & 0xF) << 4) | ($0 >> 4) }
        let gotSwapped = G.mxfp4Dequant(rows: rows, cols: cols, packed: swapped, scales: scales)
        #expect(G.mismatch(gotSwapped, want, rtol: t.rtol, atol: t.atol) != nil, "nibble order must be observable")
        // 4. a group scale shifted by one exponent
        var sc = scales
        sc[0] += 1
        #expect(G.mismatch(G.mxfp4Dequant(rows: rows, cols: cols, packed: packed, scales: sc), want, rtol: t.rtol, atol: t.atol) != nil)
        // 5. tolerance-based comparison also rejects a 1e-3 perturbation
        let p = G.tolerance("float32_product")
        let ch = G.entry("fp8_scaled.json", "channel_4x8", key: "channel")
        guard !ch.isEmpty else { return }  // entry() recorded the problem
        var w2 = G.doubles(ch["expected"])
        guard let wi = w2.firstIndex(where: { $0 != 0 }) else {
            Issue.record("fp8 channel golden has no non-zero value to perturb")
            return
        }
        w2[wi] *= 1.001
        let g2 = G.floats(
            MLXFast.fromFp8(G.fp8Weight(G.ints(ch["weight"]), 4, 8), dtype: .float32)
                * MLXArray(G.doubles(ch["scale"]).map { Float($0) }, [4, 1]))
        #expect(G.mismatch(g2, G.doubles(ch["expected"]), rtol: p.rtol, atol: p.atol) == nil)
        #expect(G.mismatch(g2, w2, rtol: p.rtol, atol: p.atol) != nil)
        // 6. NaN golden against a finite result fails, and vice versa
        #expect(G.mismatch([1], [.nan], rtol: 0, atol: 0) != nil)
        #expect(G.mismatch([.nan], [1], rtol: 0, atol: 0) != nil)
    }

    @Test("every fixture in the manifest is loaded by this suite, and every case of a fixture is a test argument")
    func everyFixtureUsed() throws {
        let source = try String(contentsOf: URL(fileURLWithPath: #filePath), encoding: .utf8)
        let files = Array((G.manifest()["files"] as? [String: Any] ?? [:]).keys)
        for name in files {
            #expect(source.contains("\"\(name)\""), "\(name) is in the manifest but no test loads it")
        }
        // parametrised fixtures: the argument lists are derived from the fixture, so a case cannot be skipped;
        // pin the counts so shrinking a fixture is noticed
        #expect(G.ids("mxfp4.json").count >= 13)
        #expect(G.ids("nvfp4.json").count >= 12)
        #expect(G.ids("fp8_scaled.json", "channel").count >= 4)
        #expect(G.ids("fp8_scaled.json", "block").count >= 6)
        #expect(G.ids("fp8_scaled.json", "ue8m0_block").count >= 2)
        #expect(G.ids("int4_pack_quantized.json").count >= 7)
        #expect(G.ids("awq_int4.json").count >= 3)
        #expect(G.cases("fp4_e2m1_codes.json", "codes").count == 16)
        #expect(G.cases("e8m0.json", "bytes").count == 8)
        // every int4 case is either run through the affine mapping or documented as unsupported by MLX group sizes
        for id in G.ids("int4_pack_quantized.json") {
            let g = G.entry("int4_pack_quantized.json", id)["group"] as? Int ?? 0
            #expect([8, 16, 32, 64].contains(g))
        }
    }
}
