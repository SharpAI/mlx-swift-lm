// V4 sparse-indexer / sparse-attention golden fixtures: the SHIPPING Swift code (SharpAIOverlay.V4Indexer,
// v4RopeCompressed, v4Visibility, v4CompressedAttention, DeepseekV4Attention, DeepseekV4Model) against a float64 oracle
// transcribed from vLLM (commit 2eae56c) and committed as JSON. The tests never run python and never touch the network;
// fixtures are located through `#filePath` (no Package.swift resource change) and verified against MANIFEST.json (sha256
// per file, plus the generator and the oracle), so a changed fixture fails loudly instead of silently moving the target.
//
// Generator: tools/conformance/golden/v4idx/gen_v4idx_golden.py. Oracle: tools/conformance/golden/v4idx/v4idx_ref.py (every line
// cites vLLM file:line; what is inferred or not modelled is I1..I6 in its header). Cross-check against vLLM's own torch
// reference code: tools/conformance/golden/v4idx/v4idx_crosscheck.py.
//
// What this settles (see also the gap tests at the bottom; each gap is wrapped by ConformanceGap.knownIssue, run with
// CONFORMANCE_GAP_RAW=1 to see them red):
//   CONFIRMED  score = sum_h w * relu(q_h . k), w = weights_proj(x) * nHeads^-1/2 * headDim^-1/2     (vLLM attention.py:1004,1100)
//   CONFIRMED  RoPE on the LAST ropeDim channels of q (position t) and of k (position b * ratio)       (compressor.py:453)
//   CONFIRMED  complete blocks only: block b readable iff b < (t + 1) / ratio                          (indexer.py:549-551)
//   CONFIRMED  k = min(index_topk, #candidates), nothing selected when no block is complete            (cpu_sparse.py:246-261)
//   CONFIRMED  window branch and compressed branch share ONE softmax with the sink                     (cpu_sparse.py:630-641)
//   CONFIRMED  indexer only on ratio-4 layers, compressor on ratio > 1                                  (attention.py:302,369)
//   CONFIRMED  ratio-0 layers use rope_theta and plain RoPE, ratio>1 layers compress_rope_theta + YaRN  (common/rope.py:30-52; fixed in e56673c)
//   CONFIRMED  YaRN mscale is forced to 0 (cos/sin scale 1)                                             (deepseek_scaling_rope.py:52-55; fixed in e56673c)
//   CONFIRMED  YaRN inverse frequencies (find_correction_range, linear ramp, beta_fast/slow) of the compressed layers,
//              three `yarn` fixtures against a float64 transcription of vLLM's generic YaRN code (common.py:32-68,
//              deepseek_scaling_rope.py:73-101, vLLM 739e594). NOT covered: the rope_type selection of the current V4 tree
//              (deepseek_llama_scaling when attention_factor == 1.0, nested per-layer-type rope dicts), see v4idx_ref.py I4.
//   WRONG      the sliding window is never applied (sliding_window parsed, unused)                      (cpu_sparse.py:587-590)
//   WRONG      sinks are dropped when attn_sink.sum() == 0                                              (DeepseekV4.swift:484)
// COVERAGE, honestly: the sliding-window fixtures (swa_w1/w3/w4/w8, window < length) are NOT compared to Swift output;
// they are knownIssue gaps, and the passing swa tests are weak controls (no future leaks; in-window dependence is not
// silent). Only the window >= length case (swa_w8_l6) is compared to the oracle. With the 4 SWA gaps and the sink gap the
// suite reports 5 known issues: green does not mean the sliding window is right.
// The manifest sha256 checks detect an accidental or tampering edit of a fixture, the generator or the oracle. They are
// not an independent anchor: MANIFEST.json itself is not pinned, and the generator
// and oracle are hashed but never re-run in CI.
// upstream-watch: Libraries/MLXLLM/Models/DeepseekV4Indexer.swift, DeepseekV4.swift.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

// Loading, sha256, location and exact IO come from GoldenKit; the manifest schema, pins and comparators stay here.
enum V4IdxGolden {

    // MARK: file format

    struct Tol: Decodable, Sendable {
        let rtol: Double
        let atol: Double
    }

    struct ToleranceClass: Decodable, Sendable {
        let y: Tol
        /// "normwise": max|d| <= atol + rtol * max|want| over the tensor; "exact": equality.
        let mode: String
        let why: String
    }

    struct FileEntry: Decodable, Sendable {
        let sha256: String
        let kind: String
        let group: String
        let dtype: String
        let tolerance: String
    }

    struct Manifest: Decodable, Sendable {
        let schema: Int
        let vllmCommit: String
        let torch: String
        let generator: String
        let generatorSha256: String
        let oracle: String
        let oracleSha256: String
        let count: Int
        let files: [String: FileEntry]
        let tolerances: [String: ToleranceClass]
    }

    struct Selection: Decodable, Sendable, Equatable {
        let strict: [Int]
        let tie: [Int]
        let need: Int
    }

    struct LayerLayout: Decodable, Sendable {
        let layer: Int
        let ratio: Int
        let compressor: Bool
        let indexer: Bool
    }

    struct Inputs: Decodable, Sendable {
        let x: [Double]?
        let qr: [Double]?
        let wqB: [Double]?
        let wProj: [Double]?
        let keysRaw: [Double]?
        let keys: [Double]?
        let scores: [Double?]?
        let q: [Double]?
        let kv: [Double]?
        let comp: [Double]?
        let compVis: [[Bool]]?
        let winVis: [[Bool]]?
        let sink: [Double]?
    }

    struct Yarn: Decodable, Sendable {
        let factor: Double
        let originalMax: Int
        let betaFast: Double
        let betaSlow: Double
        let low: Int
        let high: Int
    }

    struct Outputs: Decodable, Sendable {
        let invFreq: [Double]?
        let keysRoped: [Double]?
        let scores: [Double?]?
        let select: [Selection]?
        let visible: [[Bool]]?
        let out: [Double]?
        let layers: [LayerLayout]?
    }

    struct Fixture: Decodable, Sendable {
        let name: String
        let kind: String
        let group: String
        let dtype: String
        let tolerance: String
        // score / topk
        let nHead: Int?
        let headDim: Int?
        let ropeDim: Int?
        let hidden: Int?
        let qLora: Int?
        let ratio: Int?
        let theta: Double?
        let start: Int?
        let topk: Int?
        let L: Int?
        let nb: Int?
        let R: Int?
        let checkSelect: Bool?
        let ropeCheck: Bool?
        // visibility / swa
        let length: Int?
        let blocks: Int?
        let window: Int?
        // merge
        let H: Int?
        let S: Int?
        let D: Int?
        let scale: Double?
        let localMask: String?
        // rope
        let offset: Int?
        // yarn
        let yarn: Yarn?
        // layout
        let ratios: [Int]?
        let inputs: Inputs?
        let outputs: Outputs
    }

    struct Loaded: Sendable {
        let manifest: Manifest
        let fixtures: [Fixture]
    }

    typealias LoadError = GoldenKit.LoadError

    // MARK: loader

    /// <repo>/Tests/MLXLMTests/Conformance/Golden/v4idx, derived from this file's path.
    static let defaultDirectory: URL = GoldenKit.directory("Golden/v4idx")

    /// <repo>/tools/conformance/golden/v4idx (generator and oracle), derived from this file's path.
    static let defaultToolsDirectory: URL = GoldenKit.directory("tools/conformance/golden/v4idx")

    /// sha256 of MANIFEST.json. Regenerating the fixtures (or changing the oracle and rerunning the generator) changes the
    /// manifest and so fails the default load until this line is edited: the change shows up in a second file.
    static let pinnedManifestSha256 = "fc7810addc142a72e760f6b2591d41036fb9ca098024c1ed2af62531428b62ef"

    /// Loads and VERIFIES a fixture directory. Throws (never skips) when anything is off.
    static func load(from dir: URL, toolsDir: URL = defaultToolsDirectory, pinnedManifestSha256: String? = nil) throws -> Loaded {
        let manifestURL = dir.appendingPathComponent("MANIFEST.json")
        let manifest: Manifest
        do {
            let manifestData = try GoldenKit.read(manifestURL)
            if let pinnedManifestSha256 {
                let actualManifest = GoldenKit.sha256(manifestData)
                guard actualManifest == pinnedManifestSha256 else {
                    throw LoadError(
                        description: "MANIFEST CHANGED: MANIFEST.json sha256 \(actualManifest) != the pin in this test file. "
                            + "Regenerated fixtures need a reviewed update of `pinnedManifestSha256`.")
                }
            }
            manifest = try GoldenKit.decode(Manifest.self, from: manifestData, name: "MANIFEST.json")
        } catch let error as LoadError {
            throw error
        } catch {
            throw LoadError(description: "cannot read \(manifestURL.path): \(error)")
        }
        guard manifest.schema == 1 else { throw LoadError(description: "unknown manifest schema \(manifest.schema)") }
        guard manifest.count == manifest.files.count, manifest.count > 0 else {
            throw LoadError(description: "manifest count \(manifest.count) != files \(manifest.files.count) (or empty)")
        }
        let onDisk = try FileManager.default.contentsOfDirectory(atPath: dir.path)
            .filter { $0.hasSuffix(".json") && $0 != "MANIFEST.json" }
        guard Set(onDisk) == Set(manifest.files.keys) else {
            let extra = Set(onDisk).subtracting(manifest.files.keys).sorted()
            let missing = Set(manifest.files.keys).subtracting(onDisk).sorted()
            throw LoadError(description: "fixture set differs from manifest; extra=\(extra) missing=\(missing)")
        }
        for (path, pinned) in [(manifest.generator, manifest.generatorSha256), (manifest.oracle, manifest.oracleSha256)] {
            let file = toolsDir.appendingPathComponent(URL(fileURLWithPath: path).lastPathComponent)
            let actual: String
            do {
                actual = GoldenKit.sha256(try GoldenKit.read(file))
            } catch {
                throw LoadError(description: "cannot read \(file.path) pinned by the manifest: \(error)")
            }
            guard actual == pinned else {
                throw LoadError(
                    description: "ORACLE/GENERATOR CHANGED: \(file.lastPathComponent) sha256 \(actual) != manifest \(pinned). "
                        + "Rerun tools/conformance/golden/v4idx/gen_v4idx_golden.py and commit the regenerated fixtures.")
            }
        }
        var fixtures = [Fixture]()
        for name in manifest.files.keys.sorted() {
            guard let entry = manifest.files[name] else { continue }
            let data = try GoldenKit.read(dir.appendingPathComponent(name))
            let actual = GoldenKit.sha256(data)
            guard actual == entry.sha256 else {
                throw LoadError(
                    description: "FIXTURE CHANGED: \(name) sha256 \(actual) != manifest \(entry.sha256). "
                        + "Regenerate with tools/conformance/golden/v4idx/gen_v4idx_golden.py and review the diff.")
            }
            let fixture = try GoldenKit.decode(Fixture.self, from: data, name: name)
            guard fixture.name + ".json" == name, fixture.kind == entry.kind, fixture.group == entry.group,
                fixture.dtype == entry.dtype, fixture.tolerance == entry.tolerance
            else {
                throw LoadError(description: "\(name): header disagrees with its manifest entry")
            }
            guard manifest.tolerances[fixture.tolerance] != nil else {
                throw LoadError(description: "\(name): unknown tolerance class \(fixture.tolerance)")
            }
            fixtures.append(fixture)
        }
        return Loaded(manifest: manifest, fixtures: fixtures)
    }

    /// Loaded once. `V4IDX_GOLDEN_DIR` points the loader at another directory (used to show the suite red against a
    /// perturbed copy; unset in normal runs).
    static let shared: Result<Loaded, any Error> = loadShared()

    private static func loadShared() -> Result<Loaded, any Error> {
        do {
            if let override = ProcessInfo.processInfo.environment["V4IDX_GOLDEN_DIR"], !override.isEmpty {
                return .success(try load(from: URL(fileURLWithPath: override)))
            }
            return .success(try load(from: defaultDirectory, pinnedManifestSha256: pinnedManifestSha256))
        } catch {
            return .failure(error)
        }
    }

    // MARK: test arguments

    struct CaseID: Sendable, CustomTestStringConvertible {
        let name: String
        let group: String
        var testDescription: String { "\(group)/\(name)" }
    }

    /// If loading failed the argument list holds one sentinel case whose test rethrows the load error: the suite goes
    /// red instead of running zero cases.
    static func ids(kind: String) -> [CaseID] {
        switch shared {
        case .success(let loaded):
            return loaded.fixtures.filter { $0.kind == kind }.map { CaseID(name: $0.name, group: $0.group) }
        case .failure:
            return [CaseID(name: "LOAD-FAILED", group: "loader")]
        }
    }

    static let kinds = ["score", "topk", "visibility", "swa", "merge", "layout", "rope", "yarn"]
    static let scoreIDs = ids(kind: "score")
    static let topkIDs = ids(kind: "topk")
    static let visibilityIDs = ids(kind: "visibility")
    static let swaIDs = ids(kind: "swa")
    static let mergeIDs = ids(kind: "merge")
    static let layoutIDs = ids(kind: "layout")
    static let ropeIDs = ids(kind: "rope")
    static let yarnIDs = ids(kind: "yarn")

    static func fixture(_ id: CaseID) throws -> (Fixture, ToleranceClass) {
        let loaded = try shared.get()
        guard let f = loaded.fixtures.first(where: { $0.name == id.name }) else {
            throw LoadError(description: "no fixture named \(id.name)")
        }
        guard let tol = loaded.manifest.tolerances[f.tolerance] else {
            throw LoadError(description: "no tolerance class \(f.tolerance)")
        }
        return (f, tol)
    }

    static func dtype(_ name: String) throws -> DType {
        switch name {
        case "float32": return .float32
        case "float16": return .float16
        case "bfloat16": return .bfloat16
        default: throw LoadError(description: "unknown dtype \(name)")
        }
    }

    static func floats(_ a: [Double]?, _ what: String) throws -> [Float] {
        guard let a else { throw LoadError(description: "missing \(what)") }
        return a.map { Float($0) }
    }

    static func req<T>(_ v: T?, _ what: String) throws -> T {
        guard let v else { throw LoadError(description: "missing \(what)") }
        return v
    }

    // MARK: comparison

    struct Check {
        /// worst |got - want| / allowed ; <= 1 means within tolerance. NaN, a size mismatch or an empty comparison never passes.
        let ratio: Double
        let maxAbsError: Double
        let count: Int
        var ok: Bool { count > 0 && ratio <= 1 }
    }

    /// `want` entries that are nil are -inf (masked): the got value must be -inf there and finite everywhere else.
    static func checkScores(_ got: [Float], _ want: [Double?], tol: Tol) -> Check {
        guard got.count == want.count, !want.isEmpty else {
            return Check(ratio: Double.infinity, maxAbsError: Double.infinity, count: 0)
        }
        var maxWant = Double(0)
        for w in want { if let w { maxWant = max(maxWant, abs(w)) } }
        var maxErr = Double(0)
        var worst = Double(0)
        let allowed = tol.atol + tol.rtol * maxWant
        for i in 0 ..< want.count {
            if let w = want[i] {
                let g = Double(got[i])
                guard g.isFinite else { return Check(ratio: Double.infinity, maxAbsError: Double.infinity, count: want.count) }
                let d = abs(g - w)
                maxErr = max(maxErr, d)
                worst = max(worst, d == 0 ? 0 : d / allowed)
            } else if got[i] != -Float.infinity {
                return Check(ratio: Double.infinity, maxAbsError: Double.infinity, count: want.count)
            }
        }
        return Check(ratio: worst, maxAbsError: maxErr, count: want.count)
    }

    static func check(_ got: [Float], _ want: [Double], tol: Tol) -> Check {
        checkScores(got, want.map { Optional($0) }, tol: tol)
    }

    /// The selection contract: every `strict` index, exactly `need` of the `tie` indices, nothing else.
    static func selectionOK(_ chosen: [Bool], _ s: Selection) -> Bool {
        let picked = Set(chosen.enumerated().filter { $0.element }.map { $0.offset })
        let strict = Set(s.strict), tie = Set(s.tie)
        guard strict.isSubset(of: picked) else { return false }
        let rest = picked.subtracting(strict)
        guard rest.isSubset(of: tie), rest.count == s.need else { return false }
        return true
    }

    static func rope(theta: Double, ropeDim: Int) -> RoPELayer {
        initializeRope(
            dims: ropeDim, base: Float(theta), traditional: true, scalingConfig: nil, maxPositionEmbeddings: nil)
    }

    // MARK: running cases

    /// Builds the shipping V4Indexer with the fixture's weights.
    static func indexer(_ f: Fixture) throws -> SharpAIOverlay.V4Indexer {
        let nh = try req(f.nHead, "nHead"), hd = try req(f.headDim, "headDim"), rd = try req(f.ropeDim, "ropeDim")
        let hid = try req(f.hidden, "hidden"), ql = try req(f.qLora, "qLora"), k = try req(f.topk, "topk")
        let ix = SharpAIOverlay.V4Indexer(
            hiddenSize: hid, qLoraRank: ql, nHeads: nh, headDim: hd, topk: k, ropeDim: rd, eps: Float(1e-6))
        let dt = try dtype(f.dtype)
        let inputs = try req(f.inputs, "inputs")
        let wqB = MLXArray(try floats(inputs.wqB, "wqB"), [nh * hd, ql]).asType(dt)
        let wProj = MLXArray(try floats(inputs.wProj, "wProj"), [nh, hid]).asType(dt)
        try ix.update(
            parameters: ModuleParameters.unflattened([("wq_b.weight", wqB), ("weights_proj.weight", wProj)]),
            verify: [.noUnusedKeys])
        eval(ix)
        return ix
    }

    struct ScoreRun {
        var scores: [Float]
        var selected: [Bool]
        var roped: [Float]
    }

    static func runScore(_ f: Fixture, mutate: (inout [Float]) -> Void = { _ in }) throws -> ScoreRun {
        let hd = try req(f.headDim, "headDim"), rd = try req(f.ropeDim, "ropeDim"), hid = try req(f.hidden, "hidden")
        let ql = try req(f.qLora, "qLora"), L = try req(f.L, "L"), nb = try req(f.nb, "nb"), start = try req(f.start, "start")
        let theta = try req(f.theta, "theta"), ratio = try req(f.ratio, "ratio")
        let dt = try dtype(f.dtype)
        let inputs = try req(f.inputs, "inputs")
        let ix = try indexer(f)
        var xs = try floats(inputs.x, "x")
        mutate(&xs)
        let x = MLXArray(xs, [1, L, hid]).asType(dt)
        let qr = MLXArray(try floats(inputs.qr, "qr"), [1, L, ql]).asType(dt)
        let keys = MLXArray(try floats(inputs.keys, "keys"), [1, nb, hd]).asType(dt)
        let rope = rope(theta: theta, ropeDim: rd)
        let s = ix.scores(x, qr: qr, rope: rope, keys: keys, start: start)
        let sel = ix.select(x, qr: qr, rope: rope, keys: keys, start: start)
        let raw = MLXArray(try floats(inputs.keysRaw, "keysRaw"), [1, nb, hd]).asType(dt)
        let roped = SharpAIOverlay.v4RopeCompressed(raw, rope: rope, ropeDim: rd, ratio: ratio)
        eval(s, sel, roped)
        guard s.shape == [1, L, nb], sel.shape == [1, L, nb] else {
            throw LoadError(description: "\(f.name): unexpected shapes \(s.shape) \(sel.shape)")
        }
        return ScoreRun(
            scores: s.asType(.float32).asArray(Float.self), selected: sel.asArray(Bool.self),
            roped: roped.asType(.float32).asArray(Float.self))
    }

    static func runMerge(_ f: Fixture, mutate: (inout [Float]) -> Void = { _ in }) throws -> [Float] {
        let H = try req(f.H, "H"), L = try req(f.L, "L"), S = try req(f.S, "S"), nb = try req(f.nb, "nb")
        let D = try req(f.D, "D"), scale = try req(f.scale, "scale")
        let dt = try dtype(f.dtype)
        let inputs = try req(f.inputs, "inputs")
        var qs = try floats(inputs.q, "q")
        mutate(&qs)
        let q = MLXArray(qs, [1, H, L, D]).asType(dt)
        let kv = MLXArray(try floats(inputs.kv, "kv"), [1, 1, S, D]).asType(dt)
        let comp = MLXArray(try floats(inputs.comp, "comp"), [1, nb, D]).asType(dt)
        let compVis = try req(inputs.compVis, "compVis").flatMap { $0 }
        let cmask = MLXArray(compVis, [1, L, nb])
        let local: MLXFast.ScaledDotProductAttentionMaskMode
        if f.localMask == "causal" {
            local = .causal
        } else {
            local = .array(MLXArray(try req(inputs.winVis, "winVis").flatMap { $0 }, [L, S]))
        }
        let sinks = inputs.sink.map { MLXArray($0.map { Float($0) }).asType(dt) }
        guard
            let out = SharpAIOverlay.v4CompressedAttention(
                queries: q, localKeys: kv, compressed: comp, compressedMask: cmask, localMask: local,
                scale: Float(scale), sinks: sinks)
        else { throw LoadError(description: "\(f.name): v4CompressedAttention returned nil") }
        eval(out)
        guard out.shape == [1, H, L, D] else { throw LoadError(description: "\(f.name): unexpected shape \(out.shape)") }
        return out.asType(.float32).asArray(Float.self)
    }

    static func tinyConfigJSON(_ edit: (inout [String: Any]) -> Void) throws -> Data {
        guard var o = try JSONSerialization.jsonObject(with: ConformanceDeepseekV4.tinyJSON()) as? [String: Any] else {
            throw LoadError(description: "tinyJSON is not an object")
        }
        edit(&o)
        return try JSONSerialization.data(withJSONObject: o)
    }

    static func tinyConfig(_ edit: (inout [String: Any]) -> Void) throws -> DeepseekV4Configuration {
        try JSONDecoder().decode(DeepseekV4Configuration.self, from: tinyConfigJSON(edit))
    }

    /// A one-layer tiny model whose only layer has the given compress ratio and sliding window.
    static func oneLayerModel(window: Int, ratio: Int = 0) async throws -> DeepseekV4Model {
        let data = try tinyConfigJSON { o in
            o["num_hidden_layers"] = 1
            o["num_nextn_predict_layers"] = 1
            o["num_hash_layers"] = 1
            o["sliding_window"] = window
            o["compress_ratios"] = [ratio, 0]
        }
        let m = try await LLMTypeRegistry.shared.createModel(configuration: data, modelType: "deepseek_v4")
        let model = try #require(m as? DeepseekV4Model)
        eval(model)
        return model
    }

    /// dep[t][j]: does changing token j change the logits at position t (|d| > 1e-6)? One forward per token.
    static func dependence(_ m: DeepseekV4Model, length T: Int) -> [[Bool]] {
        let tokens = (0 ..< T).map { Int32(($0 * 7 + 3) % 64) }
        func logits(_ tk: [Int32]) -> [Float] {
            let out = m(MLXArray(tk).reshaped(1, T), cache: nil)
            eval(out)
            return out.asArray(Float.self)
        }
        let base = logits(tokens)
        let vocab = base.count / T
        var dep = [[Bool]](repeating: [Bool](repeating: false, count: T), count: T)
        for j in 0 ..< T {
            var tk = tokens
            tk[j] = (tk[j] + 17) % 64
            let other = logits(tk)
            for t in 0 ..< T {
                var worst = Float(0)
                for v in 0 ..< vocab { worst = max(worst, abs(other[t * vocab + v] - base[t * vocab + v])) }
                dep[t][j] = worst > Float(1e-6)
            }
        }
        return dep
    }
}

@Suite("ConformanceV4IndexerGoldenTests")
struct ConformanceV4IndexerGoldenTests {
    typealias G = V4IdxGolden

    // MARK: indexer scoring, RoPE of the compressed keys, selection

    @Test("index scores: sum_h w * relu(q_h . k), w = weights_proj(x) / sqrt(nHeads * headDim), -inf for incomplete blocks", arguments: G.scoreIDs)
    func scores(_ id: G.CaseID) throws {
        let (f, tol) = try G.fixture(id)
        let run = try G.runScore(f)
        let want = try G.req(f.outputs.scores, "scores")
        let c = G.checkScores(run.scores, want, tol: tol.y)
        print("V4IDX-GOLDEN \(f.name) [scores] err=\(c.maxAbsError) ratio=\(c.ratio)")
        #expect(c.ok, "\(f.name): max|d|=\(c.maxAbsError) = \(c.ratio)x the allowed error (n=\(c.count))")
    }

    @Test("compressed-key RoPE: entry b is rotated at position b * ratio on the last ropeDim channels", arguments: G.scoreIDs)
    func ropedKeys(_ id: G.CaseID) throws {
        let (f, tol) = try G.fixture(id)
        guard f.ropeCheck == true else { return }  // fixtures that overwrite keys (ties) have no raw/roped relation
        let run = try G.runScore(f)
        let c = G.check(run.roped, try G.req(f.outputs.keysRoped, "keysRoped"), tol: tol.y)
        print("V4IDX-GOLDEN \(f.name) [keys rope] err=\(c.maxAbsError) ratio=\(c.ratio)")
        #expect(c.ok, "\(f.name): keys rope max|d|=\(c.maxAbsError) = \(c.ratio)x the allowed error")
    }

    @Test("top-k selection through the whole indexer (scores -> select): strict / tie / need contract", arguments: G.scoreIDs)
    func selectEndToEnd(_ id: G.CaseID) throws {
        let (f, _) = try G.fixture(id)
        guard f.checkSelect == true else { return }
        let run = try G.runScore(f)
        let L = try G.req(f.L, "L"), nb = try G.req(f.nb, "nb")
        let want = try G.req(f.outputs.select, "select")
        #expect(run.selected.count == L * nb)
        var bad = [Int]()
        for t in 0 ..< L where !G.selectionOK(Array(run.selected[(t * nb) ..< ((t + 1) * nb)]), want[t]) { bad.append(t) }
        #expect(bad.isEmpty, "\(f.name): rows whose selection violates the contract: \(bad)")
    }

    @Test("selection step on a score matrix: ties, k > available, empty rows, negative scores", arguments: G.topkIDs)
    func topk(_ id: G.CaseID) throws {
        let (f, _) = try G.fixture(id)
        let R = try G.req(f.R, "R"), nb = try G.req(f.nb, "nb"), k = try G.req(f.topk, "topk")
        let raw = try G.req(f.inputs?.scores, "scores")
        let ix = SharpAIOverlay.V4Indexer(
            hiddenSize: 4, qLoraRank: 4, nHeads: 1, headDim: 4, topk: k, ropeDim: 2, eps: Float(1e-6))
        let s = MLXArray(raw.map { $0.map { Float($0) } ?? -Float.infinity }, [1, R, nb])
        let sel = ix.select(s)
        eval(sel)
        #expect(sel.shape == [1, R, nb])
        let got = sel.asArray(Bool.self)
        let want = try G.req(f.outputs.select, "select")
        var bad = [Int]()
        for r in 0 ..< R where !G.selectionOK(Array(got[(r * nb) ..< ((r + 1) * nb)]), want[r]) { bad.append(r) }
        #expect(bad.isEmpty, "\(f.name): rows whose selection violates the contract: \(bad)")
    }

    // MARK: visibility

    @Test("compressed-block visibility: block b readable iff b < (position + 1) / ratio", arguments: G.visibilityIDs)
    func visibility(_ id: G.CaseID) throws {
        let (f, _) = try G.fixture(id)
        let length = try G.req(f.length, "length"), blocks = try G.req(f.blocks, "blocks")
        let ratio = try G.req(f.ratio, "ratio"), start = try G.req(f.start, "start")
        let v = SharpAIOverlay.v4Visibility(length: length, blocks: blocks, ratio: ratio, start: start)
        eval(v)
        #expect(v.shape == [length, blocks])
        let want = try G.req(f.outputs.visible, "visible").flatMap { $0 }
        #expect(v.asArray(Bool.self) == want, "\(f.name)")
    }

    // MARK: window + compressed branch merge

    @Test("merge: window keys, selected compressed entries and the sink share one softmax", arguments: G.mergeIDs)
    func merge(_ id: G.CaseID) throws {
        let (f, tol) = try G.fixture(id)
        let got = try G.runMerge(f)
        let c = G.check(got, try G.req(f.outputs.out, "out"), tol: tol.y)
        print("V4IDX-GOLDEN \(f.name) [merge] err=\(c.maxAbsError) ratio=\(c.ratio)")
        #expect(c.ok, "\(f.name): max|d|=\(c.maxAbsError) = \(c.ratio)x the allowed error (n=\(c.count))")
    }

    // MARK: which layers have what

    @Test("compressor on ratio > 1, indexer on ratio == 4 only", arguments: G.layoutIDs)
    func layout(_ id: G.CaseID) throws {
        let (f, _) = try G.fixture(id)
        let ratios = try G.req(f.ratios, "ratios")
        let cfg = try G.tinyConfig { $0["compress_ratios"] = ratios }
        for want in try G.req(f.outputs.layers, "layers") {
            let attn = DeepseekV4Attention(config: cfg, layerIndex: want.layer)
            #expect((attn.compressor != nil) == want.compressor, "\(f.name) layer \(want.layer) ratio \(want.ratio): compressor")
            #expect((attn.indexer != nil) == want.indexer, "\(f.name) layer \(want.layer) ratio \(want.ratio): indexer")
        }
    }

    // MARK: RoPE of the layers

    @Test("ratio > 1 layers rotate with compress_rope_theta (plain RoPE where no YaRN is configured)", arguments: G.ropeIDs)
    func ropeCompressedLayer(_ id: G.CaseID) throws {
        let (f, tol) = try G.fixture(id)
        let theta = try G.req(f.theta, "theta")
        let cfg = try G.tinyConfig { o in o["compress_rope_theta"] = theta; o["rope_theta"] = 12345.0 }
        let attn = DeepseekV4Attention(config: cfg, layerIndex: 1)  // ratio 4
        try ropeCheck(f, tol, attn)
    }

    func ropeCheck(_ f: G.Fixture, _ tol: G.ToleranceClass, _ attn: DeepseekV4Attention) throws {
        let rd = try G.req(f.ropeDim, "ropeDim"), L = try G.req(f.L, "L"), off = try G.req(f.offset, "offset")
        let x = MLXArray(try G.floats(f.inputs?.x, "x"), [1, 1, L, rd])
        let y = attn.rope(x, offset: off)
        eval(y)
        let c = G.check(y.asArray(Float.self), try G.req(f.outputs.out, "out"), tol: tol.y)
        #expect(c.ok, "\(f.name): layer rope max|d|=\(c.maxAbsError) = \(c.ratio)x the allowed error")
    }

    // MARK: fixture hygiene

    @Test("every fixture in the manifest is exercised, and the cases the task is about exist")
    func everyFixtureIsUsed() throws {
        let loaded = try G.shared.get()
        #expect(loaded.manifest.count > 0)
        #expect(loaded.fixtures.count == loaded.manifest.count)
        let all = G.scoreIDs + G.topkIDs + G.visibilityIDs + G.swaIDs + G.mergeIDs + G.layoutIDs + G.ropeIDs + G.yarnIDs
        let used = Set(all.map(\.name))
        #expect(used.count == all.count, "duplicate case names")
        #expect(used == Set(loaded.fixtures.map(\.name)), "a fixture has no test")
        #expect(used.count == loaded.manifest.count, "tests cover \(used.count) of \(loaded.manifest.count) manifest entries")
        for f in loaded.fixtures { #expect(G.kinds.contains(f.kind), "\(f.name): unknown kind \(f.kind)") }
        let names = Set(loaded.fixtures.map(\.name))
        for required in [
            "score_tie_dup_keys", "score_tie_all_equal", "score_k_gt_blocks", "score_rows_without_blocks", "score_decode_start37",
            "topk_ties_at_threshold", "topk_k_gt_visible", "topk_none_visible", "topk_k_gt_columns", "vis_r128_prefill",
            "swa_w4_l12", "merge_decode_cache_window8", "merge_window_only", "layout_flash_like",
        ] { #expect(names.contains(required), "required case \(required) missing") }
        // The suite must cover the shapes it claims: real ties that straddle the cut, k larger than what is available.
        let ties = loaded.fixtures.filter { $0.kind == "topk" }.flatMap { $0.outputs.select ?? [] }
        #expect(ties.contains { $0.need > 0 && $0.tie.count > $0.need }, "no topk row has a tie straddling the cut")
        #expect(ties.contains { $0.strict.isEmpty && $0.tie.isEmpty }, "no topk row with nothing visible")
    }

    @Test("the manifest checksum is real: a changed byte in a copy of the fixtures fails the loader")
    func loaderRejectsAChangedFixture() throws {
        let loaded = try G.shared.get()
        let fm = FileManager.default
        let tmp = fm.temporaryDirectory.appendingPathComponent("v4idx-golden-\(UUID().uuidString)")
        try fm.copyItem(at: G.defaultDirectory, to: tmp)
        defer { try? fm.removeItem(at: tmp) }
        #expect(try G.load(from: tmp).fixtures.count == loaded.manifest.count)
        let victim = tmp.appendingPathComponent("score_basic.json")
        var text = try String(contentsOf: victim, encoding: .utf8)
        let idx = try #require(text.range(of: "\"scores\":["))
        let digitRange = idx.upperBound ..< text.index(idx.upperBound, offsetBy: 1)
        text.replaceSubrange(digitRange, with: text[digitRange] == "9" ? "8" : "9")
        try text.write(to: victim, atomically: true, encoding: .utf8)
        #expect(throws: G.LoadError.self) { try G.load(from: tmp) }
        try fm.removeItem(at: victim)
        try fm.copyItem(at: G.defaultDirectory.appendingPathComponent("score_basic.json"), to: victim)
        try Data("{}".utf8).write(to: tmp.appendingPathComponent("stray.json"))
        #expect(throws: G.LoadError.self) { try G.load(from: tmp) }
    }

    @Test("the manifest itself is pinned in this file: a regenerated (self-consistent) manifest fails the default load")
    func loaderRejectsARegeneratedManifest() throws {
        let fm = FileManager.default
        let tmp = fm.temporaryDirectory.appendingPathComponent("v4idx-manifest-\(UUID().uuidString)")
        defer { try? fm.removeItem(at: tmp) }
        try fm.copyItem(at: G.defaultDirectory, to: tmp)
        #expect(try G.load(from: tmp, pinnedManifestSha256: G.pinnedManifestSha256).fixtures.count > 0, "the pin matches the committed manifest")
        let url = tmp.appendingPathComponent("MANIFEST.json")
        let original = try Data(contentsOf: url)
        try (original + Data("\n".utf8)).write(to: url)
        #expect(throws: G.LoadError.self, "a changed manifest must fail against the pin") {
            try G.load(from: tmp, pinnedManifestSha256: G.pinnedManifestSha256)
        }
    }

    @Test("the manifest pins the generator and the oracle: a changed byte in either fails the loader")
    func loaderRejectsADriftedOracle() throws {
        let loaded = try G.shared.get()
        let fm = FileManager.default
        let tmp = fm.temporaryDirectory.appendingPathComponent("v4idx-tools-\(UUID().uuidString)")
        try fm.createDirectory(at: tmp, withIntermediateDirectories: true)
        defer { try? fm.removeItem(at: tmp) }
        let names = [loaded.manifest.generator, loaded.manifest.oracle].map { URL(fileURLWithPath: $0).lastPathComponent }
        for n in names { try fm.copyItem(at: G.defaultToolsDirectory.appendingPathComponent(n), to: tmp.appendingPathComponent(n)) }
        #expect(try G.load(from: G.defaultDirectory, toolsDir: tmp).fixtures.count == loaded.manifest.count)
        for n in names {
            let url = tmp.appendingPathComponent(n)
            let original = try Data(contentsOf: url)
            try (original + Data("\n# drift\n".utf8)).write(to: url)
            #expect(throws: G.LoadError.self, "a changed \(n) must fail the loader") {
                try G.load(from: G.defaultDirectory, toolsDir: tmp)
            }
            try original.write(to: url)
        }
    }

    // MARK: non-vacuity (a perturbed golden / input must FAIL)

    @Test("mutation self-test: perturbing the golden or an input makes every comparison FAIL")
    func comparisonIsNotVacuous() throws {
        // scores
        let (sf, stol) = try G.fixture(G.CaseID(name: "score_basic", group: "score"))
        let run = try G.runScore(sf)
        let want = try G.req(sf.outputs.scores, "scores")
        #expect(G.checkScores(run.scores, want, tol: stol.y).ok, "unperturbed baseline must pass")
        let i0 = try #require(want.firstIndex { $0 != nil })
        var moved = want
        let absWant: [Double] = want.compactMap { $0 }.map { abs($0) }
        let maxWant: Double = absWant.max() ?? Double(1)
        let allowed: Double = stol.y.atol + stol.y.rtol * maxWant
        let base: Double = want[i0] ?? Double(0)
        moved[i0] = base + Double(3) * allowed
        #expect(!G.checkScores(run.scores, moved, tol: stol.y).ok, "a score moved by 3x the allowed error must fail")
        var unmasked = want
        let m0 = try #require(want.firstIndex { $0 == nil })
        unmasked[m0] = 0
        #expect(!G.checkScores(run.scores, unmasked, tol: stol.y).ok, "an unmasked entry must fail")
        let rows: Int = try G.req(sf.L, "L")
        let hidden: Int = try G.req(sf.hidden, "hidden")
        let lastRow: Int = (rows - 1) * hidden  // row 0 sees no block: perturb the last row
        let inputPerturbed = try G.runScore(sf) { $0[lastRow] += Float(0.05) }
        #expect(!G.checkScores(inputPerturbed.scores, want, tol: stol.y).ok, "a perturbed x must change the scores")
        #expect(!G.checkScores([], [], tol: stol.y).ok, "an empty comparison must not pass")
        #expect(!G.checkScores([Float.nan], [Double(1)], tol: stol.y).ok, "NaN must not pass")
        // selection contract
        let sel = try G.req(sf.outputs.select, "select")
        let nb = try G.req(sf.nb, "nb")
        let row = try #require(sel.firstIndex { !$0.strict.isEmpty })
        var chosen = Array(run.selected[(row * nb) ..< ((row + 1) * nb)])
        #expect(G.selectionOK(chosen, sel[row]))
        chosen[sel[row].strict[0]] = false
        #expect(!G.selectionOK(chosen, sel[row]), "dropping a strict winner must fail")
        // a tie contract: choosing a non-tied index in place of a tied one must fail
        let (tf, _) = try G.fixture(G.CaseID(name: "topk_ties_at_threshold", group: "ties"))
        let tsel = try G.req(tf.outputs.select, "select")[0]
        var picked = [Bool](repeating: false, count: 5)
        for i in tsel.strict { picked[i] = true }
        for i in tsel.tie.prefix(tsel.need) { picked[i] = true }
        #expect(G.selectionOK(picked, tsel))
        picked[4] = true
        #expect(!G.selectionOK(picked, tsel), "an extra index must fail")
        // merge
        let (mf, mtol) = try G.fixture(G.CaseID(name: "merge_window4_sink", group: "merge"))
        let mout = try G.runMerge(mf)
        let mwant = try G.req(mf.outputs.out, "out")
        #expect(G.check(mout, mwant, tol: mtol.y).ok)
        #expect(!G.check(try G.runMerge(mf) { $0[0] += Float(0.05) }, mwant, tol: mtol.y).ok, "a perturbed q must fail")
        var mmoved = mwant
        let mMax: Double = mwant.map { abs($0) }.max() ?? Double(1)
        let mAllowed: Double = mtol.y.atol + mtol.y.rtol * mMax
        mmoved[0] += Double(4) * mAllowed
        #expect(!G.check(mout, mmoved, tol: mtol.y).ok)
        // visibility
        let (vf, _) = try G.fixture(G.CaseID(name: "vis_r4_prefill", group: "ratio4"))
        var cells = try G.req(vf.outputs.visible, "visible").flatMap { $0 }
        let v = SharpAIOverlay.v4Visibility(length: 24, blocks: 6, ratio: 4, start: 0)
        eval(v)
        #expect(v.asArray(Bool.self) == cells)
        cells[7].toggle()
        #expect(v.asArray(Bool.self) != cells, "a flipped visibility cell must be detected")
        // the visibility golden is not trivially all-true / all-false
        #expect(cells.contains(true) && cells.contains(false))
    }

    // MARK: sliding-window visibility (golden) -- the shipping model, measured by perturbation

    @Test("SWA control: no position ever reads the future, and positions inside the window always matter", arguments: G.swaIDs)
    func swaControl(_ id: G.CaseID) async throws {
        let (f, _) = try G.fixture(id)
        let T = try G.req(f.length, "length"), w = try G.req(f.window, "window")
        let want = try G.req(f.outputs.visible, "visible")
        let model = try await G.oneLayerModel(window: w)
        let dep = G.dependence(model, length: T)
        var leaks = 0, silentInWindow = 0
        for t in 0 ..< T {
            for j in 0 ..< T where j != t {
                if j > t && dep[t][j] { leaks += 1 }
                if j < t && want[t][j] && !dep[t][j] { silentInWindow += 1 }
            }
        }
        print("V4IDX-GOLDEN \(f.name) [swa control] future leaks=\(leaks) in-window silent=\(silentInWindow)")
        #expect(leaks == 0, "\(f.name): \(leaks) (t, j > t) pairs depend on the future")
        #expect(silentInWindow == 0, "\(f.name): \(silentInWindow) in-window (t, j < t) pairs show no dependence (probe too weak)")
    }

    @Test("gap: keys outside the sliding window must not influence the output (sliding_window is parsed but never applied)", arguments: G.swaIDs)
    func swaWindowNotApplied(_ id: G.CaseID) async throws {
        let (f, _) = try G.fixture(id)
        let T = try G.req(f.length, "length"), w = try G.req(f.window, "window")
        let want = try G.req(f.outputs.visible, "visible")
        let model = try await G.oneLayerModel(window: w)
        let dep = G.dependence(model, length: T)
        var outside = 0, wrong = 0
        for t in 0 ..< T {
            for j in 0 ..< t where !want[t][j] {
                outside += 1
                if dep[t][j] { wrong += 1 }
            }
        }
        print("V4IDX-GOLDEN \(f.name) [swa window] outside-window pairs=\(outside), still read by the model=\(wrong)")
        if outside == 0 {
            // window >= length: the golden is plain causal, which the model implements -- a green control.
            #expect(wrong == 0)
            return
        }
        ConformanceGap.knownIssue("DeepseekV4 never applies sliding_window: \(wrong) of \(outside) out-of-window (t, j) pairs are still attended (vLLM cpu_sparse.py:587-590)") {
            ConformanceGap.expect(wrong == 0, "\(f.name): \(wrong) of \(outside) pairs outside the window still change the output")
        }
    }

    // MARK: RoPE of the sliding-window layers and YaRN magnitude (gaps)

    @Test("ratio-0 (sliding-window) layers rotate with rope_theta and plain RoPE, not compress_rope_theta", arguments: G.ropeIDs)
    func ropeSlidingLayer(_ id: G.CaseID) throws {
        let (f, tol) = try G.fixture(id)
        let theta = try G.req(f.theta, "theta")
        // rope_theta = the fixture's theta, compress_rope_theta = something else: only a layer that picks rope_theta matches.
        let cfg = try G.tinyConfig { o in o["rope_theta"] = theta; o["compress_rope_theta"] = 777777.0 }
        let attn = DeepseekV4Attention(config: cfg, layerIndex: 0)  // ratio 0
        let rd = try G.req(f.ropeDim, "ropeDim"), L = try G.req(f.L, "L"), off = try G.req(f.offset, "offset")
        let x = MLXArray(try G.floats(f.inputs?.x, "x"), [1, 1, L, rd])
        let want = try G.req(f.outputs.out, "out")
        let y = attn.rope(x, offset: off)
        eval(y)
        let c = G.check(y.asArray(Float.self), want, tol: tol.y)
        #expect(c.ok, "\(f.name): ratio-0 layer rope max|d|=\(c.maxAbsError) = \(c.ratio)x the allowed error (vLLM common/rope.py:42-44: rope_theta, plain RoPE)")
    }

    @Test("compressed layers rotate with vLLM's YaRN-blended inverse frequencies", arguments: G.yarnIDs)
    func yarnFrequencies(_ id: G.CaseID) throws {
        let (f, tol) = try G.fixture(id)
        let y = try G.req(f.yarn, "yarn")
        let theta = try G.req(f.theta, "theta"), rd = try G.req(f.ropeDim, "ropeDim")
        let L = try G.req(f.L, "L"), off = try G.req(f.offset, "offset")
        let cfg = try G.tinyConfig { o in
            o["qk_rope_head_dim"] = rd
            o["head_dim"] = 2 * rd
            o["index_head_dim"] = rd
            o["compress_rope_theta"] = theta
            // the production shape of Flash's rope_scaling (type yarn); no mscale keys: the layer forces them to 0.
            o["rope_scaling"] = [
                "type": "yarn", "factor": y.factor, "original_max_position_embeddings": y.originalMax,
                "beta_fast": y.betaFast, "beta_slow": y.betaSlow,
            ] as [String: Any]
        }
        let attn = DeepseekV4Attention(config: cfg, layerIndex: 1)  // ratio 4: a compressed layer
        let x = MLXArray(try G.floats(f.inputs?.x, "x"), [1, 1, L, rd])
        let want = try G.req(f.outputs.out, "out")
        let out = attn.rope(x, offset: off)
        eval(out)
        let c = G.check(out.asArray(Float.self), want, tol: tol.y)
        print("V4IDX-GOLDEN \(f.name): yarn rope max|d|=\(c.maxAbsError) ratio=\(c.ratio)")
        #expect(c.ok, "\(f.name): YaRN rope max|d|=\(c.maxAbsError) = \(c.ratio)x the allowed error (vLLM deepseek_scaling_rope.py:73-101, ramp pairs \(y.low)..\(y.high))")
    }

    @Test("YaRN must not rescale the rotated channels (vLLM forces mscale 0, so position 0 is the identity)")
    func yarnMscale() throws {
        let flash = try ConformanceDeepseekV4.flashObject()
        let cfg = try G.tinyConfig { o in
            // keep the production rope_scaling of Flash (yarn, factor 16); tinyJSON removes it.
            o["rope_scaling"] = flash["rope_scaling"]
            o["compress_rope_theta"] = flash["compress_rope_theta"]
        }
        #expect(cfg.ropeScaling != nil, "the probe needs a rope_scaling config")
        let attn = DeepseekV4Attention(config: cfg, layerIndex: 1)  // ratio 4: the layer type that uses YaRN in vLLM
        let x = MLXArray.ones([1, 1, 3, cfg.qkRopeHeadDim])
        let y = attn.rope(x, offset: 0)
        eval(y)
        let first = y[0, 0, 0, 0...].asArray(Float.self)  // position 0: cos = 1, sin = 0
        let err = first.map { abs($0 - Float(1)) }.max() ?? 0
        print("V4IDX-GOLDEN yarn position-0 probe: |rope(1) - 1| = \(err)")
        #expect(err < Float(1e-6), "position 0 is not the identity: |rope(1) - 1| = \(err) (vLLM common/rope.py:55-56 forces mscale 0)")
    }

    // MARK: attention sink

    @Test("gap: a sink is applied even when attn_sink sums to exactly zero")
    func sinkWithZeroSum() throws {
        let cfg = try G.tinyConfig { $0["compress_ratios"] = [0, 0, 0, 0] }
        let attn = DeepseekV4Attention(config: cfg, layerIndex: 0)
        ConformanceV4CompressorTests.randomize(attn, seed: 5, scale: 0.4, skip: ["attn_sink"])
        let x = MLXRandom.normal([1, 6, cfg.hiddenSize], key: MLXRandom.key(9))
        func run(_ sink: [Float]) -> MLXArray {
            attn.attn_sink = MLXArray(sink)
            let y = attn(x, mask: .causal, cache: nil)
            eval(y)
            return y
        }
        // head 0: a huge sink (all softmax mass goes to the sink -> that head outputs ~0); head 1: effectively none.
        let withSink = run([40, -40])  // sums to exactly 0
        let noSink = run([-40, -40])  // sums to -80, applied; -40 is no sink at all
        let diff = abs(withSink - noSink).max().item(Float.self)
        print("V4IDX-GOLDEN sink [40,-40] vs [-40,-40]: max|d| = \(diff)")
        // Precondition: the probe can see a sink at all.
        let alsoSink = run([40, 40])
        #expect(abs(alsoSink - noSink).max().item(Float.self) > Float(1e-3), "probe cannot see a sink")
        ConformanceGap.knownIssue("DeepseekV4Attention drops attn_sink when attn_sink.sum() == 0 (DeepseekV4.swift:484); vLLM adds the sink logit unconditionally (-inf == none)") {
            ConformanceGap.expect(diff > Float(1e-3), "sink [40, -40] changes nothing: it was dropped (max|d| = \(diff))")
        }
    }
}
