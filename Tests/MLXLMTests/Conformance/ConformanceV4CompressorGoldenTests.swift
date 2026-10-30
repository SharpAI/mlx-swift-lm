// Golden conformance for the DeepSeek-V4 KV compressor (lane V4COMP, sl36/v4-compressor-golden).
//
// Fixtures: Tests/MLXLMTests/Conformance/Golden/v4comp/*.json, generated ONCE by
// tools/conformance/golden/v4comp/generate.py from a float64 torch oracle (v4comp_ref.py) transcribed from vLLM
// 2eae56c (compressor.cpp, cpu_compressor.py, compressor.py) and cross-checked against vLLM's own torch
// reference code on 240 random cases (MANIFEST.json "crosscheck"). Tests never run Python.
//
// Each case is run through the REAL shipping Swift code (nothing reimplemented here):
//   full sequence   SharpAIOverlay.V4Compressor.callAsFunction + v4RopeCompressed   (DeepseekV4.swift:489-492 path)
//   cached          SharpAIOverlay.V4Compressor.append + V4CompressorState          (DeepseekV4.swift:499-525 path),
//                   one shot, token by token, and several chunk plans that cross block boundaries
//   visibility      SharpAIOverlay.v4Visibility
// Weights, inputs and rope base are the fixture's; RoPE is `initializeRope(traditional: true)` like the layer's.

import CryptoKit
import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXLLM

// TODO consolidate GoldenKit
enum V4CompGolden {
    struct Case: Decodable {
        let name: String
        let dtype: String
        let ratio: Int
        let headDim: Int
        let ropeDim: Int
        let hidden: Int
        let tokens: Int
        let eps: Double
        let ropeTheta: Double
        let blocks: Int
        let pending: Int
        let x: [Double]
        let wkv: [Double]
        let wgate: [Double]
        let ape: [Double]
        let norm: [Double]
        let pooledNormed: [Double]
        let entries: [Double]
        let visible: [Int]
    }
    struct File: Decodable {
        let op: String
        let cases: [Case]
    }
    struct Manifest: Decodable {
        struct Entry: Decodable { let sha256: String; let cases: Int }
        struct Tol: Decodable { let rtol: Double; let atol: Double; let atolPerPosition: Double }
        struct Tols: Decodable { let float32: Tol; let bfloat16: Tol }
        let files: [String: Entry]
        let tolerances: Tols
    }

    static let decoder: JSONDecoder = {
        let d = JSONDecoder()
        d.keyDecodingStrategy = .convertFromSnakeCase
        return d
    }()
    static let dir = URL(fileURLWithPath: #filePath).deletingLastPathComponent().appendingPathComponent("Golden/v4comp")
    static let files = ["ratio4.json", "ratio128.json", "bf16.json"]

    static func manifest() throws -> Manifest {
        try decoder.decode(Manifest.self, from: Data(contentsOf: dir.appendingPathComponent("MANIFEST.json")))
    }

    /// Loads a fixture after verifying its sha256 and case count against MANIFEST.json.
    static func load(_ name: String) throws -> File {
        let data = try Data(contentsOf: dir.appendingPathComponent(name))
        let want = try #require(try manifest().files[name], "\(name) is not in MANIFEST.json")
        let got = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
        #expect(got == want.sha256, "\(name): sha256 \(got) != manifest \(want.sha256)")
        let f = try decoder.decode(File.self, from: data)
        #expect(f.cases.count == want.cases, "\(name): \(f.cases.count) cases != manifest \(want.cases)")
        return f
    }

    static func allCases() throws -> [Case] { try files.flatMap { try load($0).cases } }

    static func dtype(_ c: Case) -> DType { c.dtype == "bfloat16" ? .bfloat16 : .float32 }

    // MARK: building the shipping objects

    static func arr(_ v: [Double], _ shape: [Int]) -> MLXArray { MLXArray(v.map { Float($0) }, shape) }

    /// `as`: build the module (weights) in another dtype than the fixture's; bf16-exact values convert to float32 exactly.
    static func compressor(_ c: Case, as dt: DType? = nil) -> SharpAIOverlay.V4Compressor {
        let dt = dt ?? dtype(c)
        let comp = SharpAIOverlay.V4Compressor(hiddenSize: c.hidden, headDim: c.headDim, ratio: c.ratio, eps: Float(c.eps))
        let coff = c.ratio == 4 ? 2 : 1
        comp.update(parameters: ModuleParameters.unflattened([
            ("ape", arr(c.ape, [c.ratio, coff * c.headDim])),
            ("wkv.weight", arr(c.wkv, [coff * c.headDim, c.hidden]).asType(dt)),
            ("wgate.weight", arr(c.wgate, [coff * c.headDim, c.hidden]).asType(dt)),
            ("norm.weight", arr(c.norm, [c.headDim]).asType(dt)),
        ]))
        eval(comp)
        return comp
    }

    static func rope(_ c: Case) -> RoPELayer {
        initializeRope(dims: c.ropeDim, base: Float(c.ropeTheta), traditional: true, scalingConfig: nil, maxPositionEmbeddings: 4096)
    }

    static func input(_ c: Case, as dt: DType? = nil) -> MLXArray {
        arr(c.x, [1, c.tokens, c.hidden]).asType(dt ?? dtype(c))
    }

    static func floats(_ a: MLXArray) -> [Float] {
        let f = a.asType(.float32)
        eval(f)
        return f.reshaped(-1).asArray(Float.self)
    }

    // MARK: comparison

    struct Report {
        var compared = 0
        var worstExcess = 0.0   // max over elements of |err| / (atol + rtol |g|); <= 1 means within tolerance
        var maxAbs = 0.0
        var firstBad: String?
        var ok: Bool { worstExcess <= 1 && firstBad == nil }
    }

    /// Compares out [1, nb, headDim] with the first nb*headDim golden values. `positional`: the RoPE'd entries, whose
    /// tolerance grows with the entry's position b * ratio (angle error of float32 position * freq).
    static func compare(_ out: MLXArray?, golden: [Double], nb: Int, c: Case, positional: Bool, _ m: Manifest) -> Report {
        var r = Report()
        guard let out else {
            if nb != 0 { r.firstBad = "no output, expected \(nb) blocks" }
            return r
        }
        let o = floats(out)
        guard o.count == nb * c.headDim, out.dim(1) == nb else {
            r.firstBad = "shape \(out.shape), expected [1, \(nb), \(c.headDim)]"
            return r
        }
        let t = c.dtype == "bfloat16" ? m.tolerances.bfloat16 : m.tolerances.float32
        for i in 0 ..< o.count {
            let b = i / c.headDim
            let atol = t.atol + (positional ? t.atolPerPosition * Double(b * c.ratio) : 0)
            let g = golden[i], v = Double(o[i])
            let err = abs(v - g)
            r.compared += 1
            if !v.isFinite { r.worstExcess = .infinity; r.firstBad = r.firstBad ?? "non-finite at \(i)"; continue }
            r.maxAbs = max(r.maxAbs, err)
            let ex = err / (atol + t.rtol * abs(g))
            if ex > r.worstExcess { r.worstExcess = ex }
            if ex > 1, r.firstBad == nil { r.firstBad = "#\(i) (block \(b), ch \(i % c.headDim)) swift=\(v) golden=\(g)" }
        }
        return r
    }

    // MARK: shipping paths

    /// Uncached full-sequence forward as the layer hook runs it (the hook only runs it when L >= ratio).
    static func fullPath(_ c: Case) -> (pooled: MLXArray, entries: MLXArray?) {
        let comp = compressor(c)
        let pooled = comp(input(c))
        let entries = c.blocks > 0
            ? SharpAIOverlay.v4RopeCompressed(pooled, rope: rope(c), ropeDim: c.ropeDim, ratio: c.ratio) : nil
        return (pooled, entries)
    }

    /// Cached forward: feed `chunks` (the last size repeats) and check the state after every chunk.
    /// Returns the failures and the compared element count.
    static func cachedPath(_ c: Case, chunks: [Int], golden: [Double]? = nil, _ m: Manifest) -> (fails: [String], compared: Int) {
        let comp = compressor(c)
        let r = rope(c)
        let x = input(c)
        let g = golden ?? c.entries
        var state = SharpAIOverlay.V4CompressorState(ratio: c.ratio)
        var fails: [String] = []
        var compared = 0
        var pos = 0, i = 0
        while pos < c.tokens {
            let n = min(chunks[min(i, chunks.count - 1)], c.tokens - pos)
            let out = comp.append(x[0..., pos ..< (pos + n)], state: &state, rope: r, ropeDim: c.ropeDim)
            pos += n; i += 1
            let nb = pos / c.ratio   // vLLM emits an entry exactly when (position + 1) % ratio == 0 (compressor.cpp:289)
            if state.count != pos { fails.append("\(c.name) chunks \(chunks): count \(state.count) != \(pos)") }
            if state.blocks != nb { fails.append("\(c.name) chunks \(chunks): blocks \(state.blocks) != \(nb) after \(pos) tokens") }
            let rep = compare(out, golden: g, nb: nb, c: c, positional: true, m)
            compared += rep.compared
            if !rep.ok { fails.append("\(c.name) chunks \(chunks) after \(pos) tokens: worst excess \(rep.worstExcess), \(rep.firstBad ?? "")") }
        }
        return (fails, compared)
    }

    static func plans(_ c: Case) -> [[Int]] {
        [[1], [c.tokens], [c.ratio - 1, 1], [c.ratio + 1, c.ratio - 1, 3], [7, 1, 2, 5]]
    }
}

struct ConformanceV4CompressorGoldenTests {
    typealias G = V4CompGolden
    static let allCases: [G.Case] = (try? G.allCases()) ?? []
    static let cases: [G.Case] = allCases.filter { $0.dtype == "float32" }
    static let bf16Cases: [G.Case] = allCases.filter { $0.dtype == "bfloat16" }

    @Test("manifest: every fixture is listed, hashed (sha256) and consumed; vLLM commit and tolerances recorded")
    func manifestComplete() throws {
        let m = try G.manifest()
        #expect(Set(m.files.keys) == Set(G.files), "manifest files \(m.files.keys.sorted()) vs consumers \(G.files)")
        let onDisk = try FileManager.default.contentsOfDirectory(atPath: G.dir.path).filter { $0.hasSuffix(".json") && $0 != "MANIFEST.json" }
        #expect(Set(onDisk) == Set(m.files.keys), "fixture files on disk \(onDisk.sorted()) vs manifest")
        for f in G.files { _ = try G.load(f) }
        let raw = try JSONSerialization.jsonObject(with: Data(contentsOf: G.dir.appendingPathComponent("MANIFEST.json"))) as! [String: Any]
        #expect(raw["vllm_commit"] as? String == "2eae56c")
        let xc = raw["crosscheck"] as? [String: Any]
        #expect((xc?["cases"] as? Int ?? 0) >= 200, "crosscheck must cover >= 200 cases")
        #expect(Self.cases.count == 11 && Self.bf16Cases.count == 2)
        #expect(Set(Self.cases.map(\.ratio)) == [4, 128])
    }

    @Test("fixtures cover block boundaries, the overlap, pending tokens and the real head widths")
    func coverage() {
        let cs = Self.cases
        #expect(cs.contains { $0.ratio == 4 && $0.blocks >= 9 && $0.pending > 0 })           // several overlap windows + partial carry
        #expect(cs.contains { $0.ratio == 4 && $0.blocks == 1 && $0.pending == 0 })          // exactly one block, no previous block
        #expect(cs.contains { $0.ratio == 4 && $0.blocks == 0 && $0.pending == 3 })          // pending only
        #expect(cs.contains { $0.ratio == 128 && $0.blocks >= 2 && $0.pending > 0 })         // crosses 127 and 255
        #expect(cs.contains { $0.ratio == 128 && $0.tokens == 127 })                          // one short of the boundary
        #expect(cs.contains { $0.ratio == 128 && $0.tokens == 128 })                          // exactly on the boundary
        #expect(cs.contains { $0.headDim == 128 && $0.ropeDim == 64 && $0.ratio == 4 })
        #expect(cs.contains { $0.headDim == 128 && $0.ropeDim == 64 && $0.ratio == 128 })
    }

    @Test("full-sequence forward: pooled+normed (before RoPE) and RoPE'd entries equal the vLLM golden", arguments: cases)
    func fullSequence(_ c: G.Case) throws {
        let m = try G.manifest()
        let (pooled, entries) = G.fullPath(c)
        #expect(pooled.shape == [1, c.blocks, c.headDim])
        let p = G.compare(pooled, golden: c.pooledNormed, nb: c.blocks, c: c, positional: false, m)
        #expect(p.ok, "\(c.name) pooled_normed: worst excess \(p.worstExcess), max abs \(p.maxAbs), \(p.firstBad ?? "")")
        let e = G.compare(entries, golden: c.entries, nb: c.blocks, c: c, positional: true, m)
        #expect(e.ok, "\(c.name) entries: worst excess \(e.worstExcess), max abs \(e.maxAbs), \(e.firstBad ?? "")")
        print("[v4comp][stats] full \(c.name) pooled maxAbs=\(p.maxAbs) entries maxAbs=\(e.maxAbs) excess=\(e.worstExcess)")
    }

    @Test("cached append (one shot, token by token, chunk plans crossing block boundaries) equals the golden prefix after every chunk", arguments: cases)
    func cached(_ c: G.Case) throws {
        let m = try G.manifest()
        for plan in G.plans(c) {
            let (fails, _) = G.cachedPath(c, chunks: plan, m)
            #expect(fails.isEmpty, "\(fails.prefix(3))")
        }
    }

    @Test("visibility: entry b is readable at position t iff b < (t + 1) / ratio (sparse_mla.py:405,417; cpu_sparse.py:252-255)", arguments: cases)
    func visibility(_ c: G.Case) {
        guard c.blocks > 0 else { return }
        let mask = SharpAIOverlay.v4Visibility(length: c.tokens, blocks: c.blocks, ratio: c.ratio)
        eval(mask)
        let flat = mask.reshaped(-1).asArray(Bool.self)
        for t in 0 ..< c.tokens {
            let row = (0 ..< c.blocks).filter { flat[t * c.blocks + $0] }
            #expect(row == Array(0 ..< c.visible[t]), "\(c.name) t=\(t): readable \(row), golden count \(c.visible[t])")
        }
        // decode form: one query at `start`
        for t in stride(from: 0, to: c.tokens, by: max(1, c.ratio / 4)) {
            let one = SharpAIOverlay.v4Visibility(length: 1, blocks: c.blocks, ratio: c.ratio, start: t)
            eval(one)
            #expect(one.reshaped(-1).asArray(Bool.self).filter { $0 }.count == c.visible[t], "\(c.name) decode t=\(t)")
        }
    }

    @Test("non-vacuity: a perturbed golden is rejected on the full and the cached path, and wrong readings exceed the tolerance")
    func nonVacuity() throws {
        let m = try G.manifest()
        var rejected = 0
        for c in Self.cases where c.blocks > 0 {
            var bad = c.entries
            bad[bad.count - 1] += 1e-3
            let (_, entries) = G.fullPath(c)
            #expect(!G.compare(entries, golden: bad, nb: c.blocks, c: c, positional: true, m).ok, "\(c.name) full path accepted a perturbed golden")
            let (fails, _) = G.cachedPath(c, chunks: [c.tokens], golden: bad, m)
            #expect(!fails.isEmpty, "\(c.name) cached path accepted a perturbed golden")
            var badP = c.pooledNormed
            badP[0] += 1e-3
            let (pooled, _) = G.fullPath(c)
            #expect(!G.compare(pooled, golden: badP, nb: c.blocks, c: c, positional: false, m).ok, "\(c.name) pooled accepted a perturbed golden")
            rejected += 1
        }
        #expect(rejected == 9)
        // the three recorded wrong readings differ from the golden by >> tolerance
        let raw = try JSONSerialization.jsonObject(with: Data(contentsOf: G.dir.appendingPathComponent("MANIFEST.json"))) as! [String: Any]
        let disc = raw["discriminators_max_abs_diff_of_wrong_readings_vs_golden"] as! [String: [String: Double]]
        var n = 0
        for (_, d) in disc { for (_, v) in d { #expect(v > 1e-3); n += 1 } }
        #expect(n == 23)
    }

    @Test("bfloat16 (production dtype): fp32 projection and pooling like vLLM (full-sequence and cached)", arguments: bf16Cases)
    func bf16(_ c: G.Case) throws {
        let m = try G.manifest()
        let (pooled, entries) = G.fullPath(c)
        let p = G.compare(pooled, golden: c.pooledNormed, nb: c.blocks, c: c, positional: false, m)
        let e = G.compare(entries, golden: c.entries, nb: c.blocks, c: c, positional: true, m)
        print("[v4comp][stats] bf16 \(c.name) full pooled maxAbs=\(p.maxAbs) excess=\(p.worstExcess) | entries maxAbs=\(e.maxAbs) excess=\(e.worstExcess) | out dtype \(pooled.dtype)")
        // the output keeps the model dtype (the nb == 0 early return already does); float32 here would leak into
        // the attention keys and every later activation
        #expect(pooled.dtype == .bfloat16, "\(c.name) pooled dtype \(pooled.dtype)")
        #expect(entries?.dtype == .bfloat16, "\(c.name) entries dtype \(String(describing: entries?.dtype))")
        #expect(p.ok, "\(c.name) pooled_normed: worst excess \(p.worstExcess), max abs \(p.maxAbs), \(p.firstBad ?? "")")
        #expect(e.ok, "\(c.name) entries: worst excess \(e.worstExcess), max abs \(e.maxAbs), \(e.firstBad ?? "")")
        for plan in G.plans(c) {
            let (fails, _) = G.cachedPath(c, chunks: plan, m)
            #expect(fails.isEmpty, "\(fails.prefix(2))")
        }
    }

    @Test("bfloat16: internal precision is float32 and the output is that result rounded ONCE to bf16 (full and cached)", arguments: bf16Cases)
    func bf16RoundedOnce(_ c: G.Case) throws {
        // bf16-exact inputs and weights convert to float32 exactly, so the float32 module computes the same
        // float32 pooling; the bf16 module must return exactly that, rounded once. Rounding kv / score / the pooled
        // sum earlier would flip many bf16 values.
        let want = G.compressor(c, as: .float32)(G.input(c, as: .float32)).asType(.bfloat16)
        let got = G.compressor(c)(G.input(c))
        #expect(got.dtype == .bfloat16)
        #expect(want.shape == got.shape)
        let d = (want.asType(.float32) - got.asType(.float32)).abs().max()
        eval(d)
        #expect(d.item(Float.self) == 0, "\(c.name) bf16 output differs from the once-rounded float32 result by \(d.item(Float.self))")
        // cached path: entries keep the model dtype too
        var state = SharpAIOverlay.V4CompressorState(ratio: c.ratio)
        let out = G.compressor(c).append(G.input(c), state: &state, rope: G.rope(c), ropeDim: c.ropeDim)
        #expect(out?.dtype == .bfloat16, "\(c.name) cached entries dtype \(String(describing: out?.dtype))")
    }

    @Test("bfloat16 compressed entries do not promote the attention keys or output to float32")
    func bf16AttentionDtype() throws {
        let c = try #require(Self.bf16Cases.first { $0.ratio == 4 })
        let (_, entries) = G.fullPath(c)
        let compressed = try #require(entries)
        let l = c.tokens
        let q = MLXArray(Array(repeating: Float(0.25), count: 2 * l * c.headDim), [1, 2, l, c.headDim]).asType(.bfloat16)
        let k = MLXArray(Array(repeating: Float(0.5), count: l * c.headDim), [1, 1, l, c.headDim]).asType(.bfloat16)
        let vis = SharpAIOverlay.v4Visibility(length: l, blocks: c.blocks, ratio: c.ratio).expandedDimensions(axis: 0)
        let out = try #require(
            SharpAIOverlay.v4CompressedAttention(
                queries: q, localKeys: k, compressed: compressed, compressedMask: vis, localMask: .causal,
                scale: Float(0.25), sinks: nil))
        #expect(out.dtype == .bfloat16, "attention output dtype \(out.dtype)")
    }

    @Test("every case of every fixture is compared (no silent skipping)")
    func everyCaseCompared() throws {
        let m = try G.manifest()
        var pooledElems = 0, cachedElems = 0
        for c in Self.allCases {
            pooledElems += c.blocks * c.headDim
            #expect(c.pooledNormed.count == c.blocks * c.headDim && c.entries.count == c.blocks * c.headDim, "\(c.name)")
            let (_, e) = G.fullPath(c)
            let rep = G.compare(e, golden: c.entries, nb: c.blocks, c: c, positional: true, m)
            #expect(rep.compared == c.blocks * c.headDim, "\(c.name)")
            let (_, n) = G.cachedPath(c, chunks: [c.tokens], m)
            cachedElems += n
        }
        #expect(pooledElems > 0)
        #expect(cachedElems == pooledElems, "one-shot cached run compares exactly the golden elements: \(cachedElems) vs \(pooledElems)")
    }
}
