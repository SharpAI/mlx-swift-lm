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

import CryptoKit
import Foundation
import MLX
import Testing

@testable import MLXLMCommon

// TODO consolidate GoldenKit
enum DequantGolden {
    /// Repo root: Tests/MLXLMTests/Conformance/Golden/dequant/<this file> -> six components up.
    static let repoRoot: URL = {
        var url = URL(fileURLWithPath: #filePath)
        for _ in 0 ..< 6 { url.deleteLastPathComponent() }
        return url
    }()
    static let dir = repoRoot.appendingPathComponent("tools/conformance/golden/dequant/fixtures")

    static func manifest() -> [String: Any] { json("MANIFEST.json") }

    static func json(_ name: String) -> [String: Any] {
        let data = try! Data(contentsOf: dir.appendingPathComponent(name))
        return try! JSONSerialization.jsonObject(with: data) as! [String: Any]
    }

    static func sha256(_ url: URL) -> String {
        let data = try! Data(contentsOf: url)
        return SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    /// Loads a fixture after checking its manifest sha256.
    static func load(_ name: String) -> [String: Any] {
        let files = manifest()["files"] as! [String: [String: Any]]
        precondition(files[name] != nil, "\(name) is not in MANIFEST.json")
        precondition(
            sha256(dir.appendingPathComponent(name)) == files[name]!["sha256"] as! String,
            "\(name): sha256 differs from MANIFEST.json (regenerate with gen_dequant_golden.py)")
        return json(name)
    }

    static func cases(_ file: String, _ key: String = "cases") -> [[String: Any]] {
        load(file)[key] as! [[String: Any]]
    }

    static func ids(_ file: String, _ key: String = "cases") -> [String] {
        cases(file, key).map { $0["id"] as! String }
    }

    static func entry(_ file: String, _ id: String, key: String = "cases") -> [String: Any] {
        cases(file, key).first { $0["id"] as? String == id }!
    }

    static func d(_ v: Any) -> Double {
        if let s = v as? String {
            switch s {
            case "nan": return .nan
            case "inf": return .infinity
            case "-inf": return -.infinity
            default: preconditionFailure("bad number token \(s)")
            }
        }
        return (v as! NSNumber).doubleValue
    }

    static func doubles(_ v: Any?) -> [Double] { (v as! [Any]).map(d) }
    static func ints(_ v: Any?) -> [Int] { (v as! [Any]).map { ($0 as! NSNumber).intValue } }

    /// float32_affine_cancellation. `scales[i]` is the scale of the group element i belongs to.
    /// STUB (red): still uses the case-wide largest scale for every element.
    static func affineMismatch(_ got: [Float], _ want: [Double], scales: [Double]) -> String? {
        let t = (manifest()["tolerances"] as! [String: [String: Any]])["float32_affine_cancellation"]!
        return mismatch(got, want, rtol: d(t["rtol"]!), atol: d(t["atol_per_max_scale"]!) * scales.max()!)
    }

    static func tolerance(_ name: String) -> (rtol: Double, atol: Double) {
        let t = (manifest()["tolerances"] as! [String: [String: Any]])[name]!
        return (d(t["rtol"]!), d(t["atol"]!))
    }

    /// nil when `got` equals the float32 rounding of `want` within rtol/atol (NaN matches NaN, inf matches inf).
    static func mismatch(_ got: [Float], _ want: [Double], rtol: Double, atol: Double) -> String? {
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
            } else if Double(abs(g - w)) > atol + rtol * Double(abs(w)) {
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
    ]
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
        let w = MLXArray(packed.map { UInt8($0) }, [rows, cols / 2])
        let s = MLXArray(scales.map { UInt8($0) }, [rows, cols / 32])
        let mapped = SharpAIOverlay.mapMXFP4Experts(
            ["m.layers.0.ffn.experts.0.w.weight": w, "m.layers.0.ffn.experts.0.w.scale": s], expertDtype: "fp4")
        let mw = mapped["m.layers.0.ffn.experts.0.w.weight"]!
        let ms = mapped["m.layers.0.ffn.experts.0.w.scales"]!
        precondition(mw.dtype == .uint32 && mw.shape == [rows, cols / 8])
        return floats(dequantized(mw, scales: ms, biases: nil, groupSize: 32, bits: 4, mode: .mxfp4, dtype: .float32))
    }

    /// MLX's nvfp4 `globalScale` is the ENCODER-side per-tensor value: the kernel decodes with
    /// `globalScale / (F8E4M3_MAX * F4E2M1_MAX)` (mlx backend/metal/kernels/fp_quantized.h:216), i.e. a multiplier m
    /// (the one vLLM's emulation applies, nvfp4_emulation_utils.py:63) is passed as m * 448 * 6. Found by the first
    /// run of this suite (values off by exactly 1/2688); not a defect, but a loader must convert.
    static let mlxNvfp4GlobalScaleUnit: Double = 448 * 6

    static func nvfp4Dequant(rows: Int, cols: Int, packed: [Int], scales: [Int], multiplier: Double) -> [Float] {
        let w = MLXArray(packed.map { UInt8($0) }, [rows, cols / 2]).view(dtype: .uint32)
        let s = MLXArray(scales.map { UInt8($0) }, [rows, cols / 16])
        return floats(
            dequantized(
                w, scales: s, biases: nil, groupSize: 16, bits: 4, mode: .nvfp4,
                globalScale: MLXArray(Float(multiplier * mlxNvfp4GlobalScaleUnit)), dtype: .float32))
    }

    static func fp8Weight(_ bytes: [Int], _ rows: Int, _ cols: Int) -> MLXArray {
        MLXArray(bytes.map { UInt8($0) }, [rows, cols])
    }
}

struct DequantGoldenTests {
    typealias G = DequantGolden

    // MARK: manifest

    @Test("manifest: every fixture's sha256 matches, no stray files, generator inputs recorded")
    func manifestIntegrity() {
        let m = G.manifest()
        let files = m["files"] as! [String: [String: Any]]
        #expect(m["vllm_commit"] as? String == "739e5945d")
        #expect((m["torch_version"] as? String)?.isEmpty == false)
        #expect(files.count == 8)
        for (name, info) in files {
            #expect(G.sha256(G.dir.appendingPathComponent(name)) == info["sha256"] as? String, "\(name)")
        }
        let onDisk = Set(
            ((try? FileManager.default.contentsOfDirectory(atPath: G.dir.path)) ?? []).filter { $0.hasSuffix(".json") })
        #expect(onDisk == Set(files.keys).union(["MANIFEST.json"]), "fixtures on disk differ from the manifest")
        let tol = m["tolerances"] as! [String: [String: Any]]
        for k in ["float32_exact", "float32_product", "float32_affine_cancellation", "bfloat16_values"] {
            #expect((tol[k]?["why"] as? String)?.isEmpty == false, "tolerance \(k) needs a reason")
        }
        let cc = m["crosscheck"] as! [String: Any]
        #expect(Set(cc.keys).isSuperset(of: ["nvfp4", "mxfp8_e8m0_decode", "fp8_scaled_dequantize", "int4"]))
    }

    // MARK: fp4 e2m1 / e8m0

    @Test("all 16 fp4 codes, in the low and the high nibble, through the real mxfp4 path")
    func fp4Codes() {
        let table = G.cases("fp4_e2m1_codes.json", "codes")
        #expect(table.count == 16)
        for e in table {
            let c = e["code"] as! Int
            let want = G.d(e["value"]!)
            // low nibble = c (element 0), high nibble = 2 (1.0, element 1); scale byte 127 = 1
            let lo = G.mxfp4Dequant(rows: 1, cols: 32, packed: [Int](repeating: c | (2 << 4), count: 16), scales: [127])
            let hi = G.mxfp4Dequant(rows: 1, cols: 32, packed: [Int](repeating: 2 | (c << 4), count: 16), scales: [127])
            #expect(lo[0] == Float(want) && lo[1] == 1, "code \(c) low nibble: got \(lo[0]) want \(want)")
            #expect(hi[1] == Float(want) && hi[0] == 1, "code \(c) high nibble: got \(hi[1]) want \(want)")
        }
    }

    @Test("e8m0 scale bytes 0/1/2/126/127/128/253/254 decode to 2^(byte-127)", arguments: G.cases("e8m0.json", "bytes").map { $0["byte"] as! Int })
    func e8m0Bytes(byte: Int) {
        let want = G.d(G.cases("e8m0.json", "bytes").first { $0["byte"] as? Int == byte }!["value"]!)
        // every element is code 2 (= 1.0) so the output equals the decoded scale
        let got = G.mxfp4Dequant(rows: 1, cols: 32, packed: [Int](repeating: 0x22, count: 16), scales: [byte])
        G.check("e8m0/\(byte)", G.mismatch(got, [Double](repeating: want, count: 32), rtol: 0, atol: 0))
    }

    @Test("e8m0 byte 255: vLLM's formula overflows float32 to inf (torch float8_e8m0fnu says NaN)")
    func e8m0Byte255() {
        let sp = G.load("e8m0.json")["special"] as! [String: Any]
        #expect(sp["byte"] as? Int == 255)
        #expect(sp["vllm_formula_float32"] as? String == "inf")
        let got = G.mxfp4Dequant(rows: 1, cols: 32, packed: [Int](repeating: 0x22, count: 16), scales: [255])
        G.check("e8m0/255", G.mismatch(got, [Double](repeating: .infinity, count: 32), rtol: 0, atol: 0))
    }

    // MARK: mxfp4

    @Test("mxfp4 golden cases (nibble order, group boundaries, scale edges, random)", arguments: G.ids("mxfp4.json"))
    func mxfp4(id: String) {
        let c = G.entry("mxfp4.json", id)
        let rows = c["rows"] as! Int, cols = c["cols"] as! Int
        let got = G.mxfp4Dequant(rows: rows, cols: cols, packed: G.ints(c["packed"]), scales: G.ints(c["scales"]))
        let t = G.tolerance("float32_exact")
        G.check("mxfp4/\(id)", G.mismatch(got, G.doubles(c["expected"]), rtol: t.rtol, atol: t.atol))
    }

    // MARK: nvfp4

    @Test("nvfp4 golden cases (e4m3 block scales, global multiplier, nibble order)", arguments: G.ids("nvfp4.json"))
    func nvfp4(id: String) {
        let c = G.entry("nvfp4.json", id)
        let got = G.nvfp4Dequant(
            rows: c["rows"] as! Int, cols: c["cols"] as! Int, packed: G.ints(c["packed"]), scales: G.ints(c["scales"]),
            multiplier: G.d(c["global_scale"]!))
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
        let (n, k) = (c["rows"] as! Int, c["cols"] as! Int)
        let w = G.fp8Weight(G.ints(c["weight"]), n, k)
        let scale = MLXArray(G.doubles(c["scale"]).map { Float($0) }, [n, 1])
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
        let (n, k, bs) = (c["rows"] as! Int, c["cols"] as! Int, c["block"] as! Int)
        let shape = G.ints(c["scale_shape"])
        let w = G.fp8Weight(G.ints(c["weight"]), n, k)
        let scale = MLXArray(G.doubles(c["scale"]).map { Float($0) }, shape)
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

    @Test("fp8 ue8m0 block scales (byte -> 2^(byte-127)) through dequantFP8Blocks(ue8m0: true)",
        arguments: G.ids("fp8_scaled.json", "ue8m0_block"))
    func fp8Ue8m0(id: String) {
        let c = G.entry("fp8_scaled.json", id, key: "ue8m0_block")
        let (n, k, bs) = (c["rows"] as! Int, c["cols"] as! Int, c["block"] as! Int)
        let w = G.fp8Weight(G.ints(c["weight"]), n, k)
        let sb = MLXArray(G.ints(c["scale_bytes"]).map { UInt8($0) }, G.ints(c["scale_shape"]))
        let got = SharpAIOverlay.dequantFP8Blocks(
            weight: MLXFast.fromFp8(w, dtype: .float32), scaleInv: sb, blockSize: bs, ue8m0: true)
        let t = G.tolerance("float32_exact")
        G.check("fp8_ue8m0/\(id)", G.mismatch(G.floats(got), G.doubles(c["expected"]), rtol: t.rtol, atol: t.atol))
    }

    // MARK: int4 (compressed-tensors pack-quantized) and AWQ

    @Test("int4 pack-quantized, group 32/64: the packing is MLX affine, w = q*s - zp*s (mapping feasibility, not a loader)",
        arguments: G.ids("int4_pack_quantized.json").filter { id in
            let g = G.entry("int4_pack_quantized.json", id)["group"] as! Int
            return g == 32 || g == 64
        })
    func int4ViaAffine(id: String) {
        let c = G.entry("int4_pack_quantized.json", id)
        let (got, scales) = Self.int4Dequant(c)
        G.check("int4/\(id)", G.affineMismatch(got, G.doubles(c["expected"]), scales: scales))
    }

    /// Row-major [n, k] result of the int4 case through MLX affine, and the scale of each element's group.
    static func int4Dequant(_ c: [String: Any]) -> (got: [Float], scales: [Double]) {
        let (n, k, g) = (c["rows"] as! Int, c["cols"] as! Int, c["group"] as! Int)
        let words = G.ints(c["weight_packed"]).map { UInt32(truncatingIfNeeded: $0) }
        let w = MLXArray(words, [n, k / 8])
        let s = MLXArray(G.doubles(c["scale"]).map { Float($0) }, [n, k / g])
        var zp = MLXArray.ones([n, k / g], dtype: .float32) * MLXArray(Float(8))  // uint4b8: symmetric zero point is 8
        if !(c["symmetric"] as! Bool) {
            // zero point nibbles are packed along dim 0 (dim-0 packing is stated in compressed_tensors_wNa16.py:158-183, see dequant_ref.py I7); unpack in test code
            let zw = G.ints(c["zero_packed"]).map { UInt32(truncatingIfNeeded: $0) }
            let cols = k / g
            var z = [Float](repeating: 0, count: n * cols)
            for r in 0 ..< n { for col in 0 ..< cols { z[r * cols + col] = Float((zw[(r / 8) * cols + col] >> UInt32(4 * (r % 8))) & 0xF) } }
            zp = MLXArray(z, [n, cols])
        }
        let got = dequantized(w, scales: s, biases: -(zp * s), groupSize: g, bits: 4, mode: .affine, dtype: .float32)
        let sc = G.doubles(c["scale"])
        return (G.floats(got), (0 ..< n * k).map { sc[($0 / k) * (k / g) + ($0 % k) / g] })
    }

    @Test("AWQ int4 (vLLM awq_pack layout) -> convertAutoAWQ -> MLX affine dequantize equals the golden [K,N]^T",
        arguments: G.ids("awq_int4.json"))
    func awq(id: String) {
        let c = G.entry("awq_int4.json", id)
        let (got, scales) = Self.awqDequant(c)
        G.check("awq/\(id)", G.affineMismatch(got, G.doubles(c["expected_KxN"]), scales: scales))
    }

    /// Row-major [K, N] result of the AWQ case through convertAutoAWQ + MLX affine, and the scale of each element's group.
    static func awqDequant(_ c: [String: Any]) -> (got: [Float], scales: [Double]) {
        let (k, n, g) = (c["in_features"] as! Int, c["out_features"] as! Int, c["group"] as! Int)
        var weights: [String: MLXArray] = [
            "layer.qweight": MLXArray(G.ints(c["qweight"]).map { Int32(truncatingIfNeeded: $0) }, G.ints(c["qweight_shape"])),
            "layer.qzeros": MLXArray(G.ints(c["qzeros"]).map { Int32(truncatingIfNeeded: $0) }, G.ints(c["qzeros_shape"])),
            "layer.scales": MLXArray(G.doubles(c["scales"]).map { Float($0) }, G.ints(c["scales_shape"])),
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

    /// All Swift sources under Libraries/ (text), for the "is there a loader for X" probes.
    static let librarySources: String = {
        let root = DequantGolden.repoRoot.appendingPathComponent("Libraries")
        var text = ""
        if let en = FileManager.default.enumerator(at: root, includingPropertiesForKeys: nil) {
            for case let u as URL in en where u.pathExtension == "swift" {
                text += (try? String(contentsOf: u, encoding: .utf8)) ?? ""
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

    @Test("tolerances are tight: a 1-ulp float32 product error, and an error just above one small-scale group's own bound, are rejected")
    func tolerancesAreTight() {
        // fp8 * float32 scale is one correctly rounded float32 product, so equality is required (no slack for 1 ulp)
        let ch = G.entry("fp8_scaled.json", "channel_4x8", key: "channel")
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
            guard [32, 64].contains(c["group"] as! Int) else { continue }
            let (got, scales) = Self.int4Dequant(c)
            out.append(("int4/\(id)", got, scales, G.doubles(c["expected"])))
        }
        for id in G.ids("awq_int4.json") {
            let c = G.entry("awq_int4.json", id)
            let (got, scales) = Self.awqDequant(c)
            out.append(("awq/\(id)", got, scales, G.doubles(c["expected_KxN"])))
        }
        return out
    }

    @Test("mutation: perturbed goldens and swapped nibbles must FAIL the comparison")
    func mutationsFail() {
        let c = G.entry("mxfp4.json", "all_bytes_unit_scale")
        let rows = c["rows"] as! Int, cols = c["cols"] as! Int
        let packed = G.ints(c["packed"]), scales = G.ints(c["scales"])
        let want = G.doubles(c["expected"])
        let got = G.mxfp4Dequant(rows: rows, cols: cols, packed: packed, scales: scales)
        let t = G.tolerance("float32_exact")
        #expect(G.mismatch(got, want, rtol: t.rtol, atol: t.atol) == nil, "baseline must pass")
        // 1. one golden value nudged by 1e-6 relative (exact tolerance: must fail)
        var bumped = want
        let i = bumped.firstIndex { $0 != 0 }!
        bumped[i] *= 1 + 1e-6
        #expect(G.mismatch(got, bumped, rtol: t.rtol, atol: t.atol) != nil)
        // 2. a zero golden value made non-zero
        var z = want
        z[z.firstIndex { $0 == 0 }!] = 0.25
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
        var w2 = G.doubles(ch["expected"])
        w2[w2.firstIndex { $0 != 0 }!] *= 1.001
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
        let files = Array((G.manifest()["files"] as! [String: Any]).keys)
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
            let g = G.entry("int4_pack_quantized.json", id)["group"] as! Int
            #expect([8, 16, 32, 64].contains(g))
        }
    }
}
