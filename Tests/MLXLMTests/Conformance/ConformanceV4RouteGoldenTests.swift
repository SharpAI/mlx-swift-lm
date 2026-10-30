// V4 routing / clamped SwiGLU / mHC golden fixtures: the REAL shipping Swift code (DeepseekV4Gate, DeepseekV4Expert,
// DeepseekV4MoE, hcPre / hcPost / hcHead in DeepseekV4.swift) against a float64 oracle transcribed from vLLM
// (commit 2eae56c) and committed as JSON. The tests never run python and never touch the network; the fixtures are
// located through `#filePath` (no Package.swift resource change) and verified against MANIFEST.json (sha256 per
// file, plus the sha256 of the generator and the oracle) at load, so a changed fixture fails loudly.
//
// Generator: tools/conformance/golden/gen_v4route_golden.py (run once, commit outputs). Oracle:
// tools/conformance/golden/v4route_ref.py (every line cites vLLM file:line; the INFERRED points are I1..I5 in its
// header). Tolerances, with their reasoning, are in MANIFEST.json.
//
// What is compared: per token the chosen expert ids (sorted ascending, exact) and the routing weights; the shared
// expert and the whole MoE output; the hyper-connection pre / post / head / pre->post chain. Where the shipping code
// disagrees with the oracle the case is listed in `V4RouteGolden.gaps` with the reason and runs through
// `ConformanceGap.knownIssue`; CONFORMANCE_GAP_RAW=1 shows the raw failure, CONFORMANCE_STRICT=1 fails once a gap is
// fixed. upstream-watch: Libraries/MLXLLM/Models/DeepseekV4.swift.

import CryptoKit
import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXLLM

// TODO consolidate GoldenKit (the loader, manifest verification and comparators below are lane-private copies)
enum V4RouteGolden {

    // MARK: file format

    struct ToleranceClass: Decodable, Sendable {
        let mode: String  // "normwise" | "elementwise"
        let rtol: Double
        let atol: Double
        let why: String
    }

    struct FileEntry: Decodable, Sendable {
        let sha256: String
        let kind: String
        let group: String
        let dtype: String
        let tolerance: [String: String]
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

    struct Tensor: Decodable, Sendable {
        let shape: [Int]
        let data: [Double]
    }

    struct Params: Decodable, Sendable {
        let B: Int?
        let S: Int?
        let H: Int?
        let E: Int?
        let I: Int?
        let topk: Int?
        let vocab: Int?
        let normTopkProb: Bool?
        let routedScalingFactor: Double?
        let swigluLimit: Double?
        let hash: Bool?
        let hc: Int?
        let D: Int?
        let sinkhornIters: Int?
        let rmsEps: Double?
        let hcEps: Double?
        let sublayerScale: Double?
        let sublayerShift: Double?
    }

    struct Fixture: Decodable, Sendable {
        let name: String
        let kind: String
        let group: String
        let dtype: String
        let tolerance: [String: String]
        let params: Params
        let inputs: [String: Tensor]
        var outputs: [String: Tensor]
    }

    struct Loaded: Sendable {
        let manifest: Manifest
        let fixtures: [Fixture]
    }

    struct LoadError: Error, CustomStringConvertible {
        let description: String
    }

    // MARK: loader

    static let defaultDirectory: URL = {
        var url = URL(fileURLWithPath: #filePath)
        url.deleteLastPathComponent()
        return url.appendingPathComponent("Golden/v4route")
    }()

    static let defaultToolsDirectory: URL = {
        var url = URL(fileURLWithPath: #filePath)
        for _ in 0 ..< 4 { url.deleteLastPathComponent() }
        return url.appendingPathComponent("tools/conformance/golden")
    }()

    static func sha256(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    /// Loads and VERIFIES a fixture directory. Throws (never skips) on a missing or extra file, a sha256 that
    /// differs from the manifest, a count that does not add up, an unknown tolerance class.
    static func load(from dir: URL, toolsDir: URL = defaultToolsDirectory) throws -> Loaded {
        let manifestURL = dir.appendingPathComponent("MANIFEST.json")
        let manifest: Manifest
        do {
            manifest = try JSONDecoder().decode(Manifest.self, from: Data(contentsOf: manifestURL))
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
                actual = sha256(try Data(contentsOf: file))
            } catch {
                throw LoadError(description: "cannot read \(file.path) pinned by the manifest: \(error)")
            }
            guard actual == pinned else {
                throw LoadError(
                    description: "ORACLE/GENERATOR CHANGED: \(file.lastPathComponent) sha256 \(actual) != manifest \(pinned). "
                        + "Rerun tools/conformance/golden/gen_v4route_golden.py and commit the regenerated fixtures.")
            }
        }
        var fixtures = [Fixture]()
        for name in manifest.files.keys.sorted() {
            guard let entry = manifest.files[name] else { continue }
            let data = try Data(contentsOf: dir.appendingPathComponent(name))
            let actual = sha256(data)
            guard actual == entry.sha256 else {
                throw LoadError(
                    description: "FIXTURE CHANGED: \(name) sha256 \(actual) != manifest \(entry.sha256). "
                        + "Regenerate with tools/conformance/golden/gen_v4route_golden.py and review the diff.")
            }
            let fixture: Fixture
            do {
                fixture = try JSONDecoder().decode(Fixture.self, from: data)
            } catch {
                throw LoadError(description: "cannot decode \(name): \(error)")
            }
            guard fixture.name + ".json" == name, fixture.kind == entry.kind, fixture.group == entry.group,
                fixture.dtype == entry.dtype, fixture.tolerance == entry.tolerance
            else {
                throw LoadError(description: "\(name): header disagrees with its manifest entry")
            }
            for cls in fixture.tolerance.values where manifest.tolerances[cls] == nil {
                throw LoadError(description: "\(name): unknown tolerance class \(cls)")
            }
            fixtures.append(fixture)
        }
        return Loaded(manifest: manifest, fixtures: fixtures)
    }

    /// Loaded once. `V4ROUTE_GOLDEN_DIR` points the loader at another directory (used to show the suite red
    /// against a perturbed copy; unset in normal runs).
    static let shared: Result<Loaded, any Error> = Result {
        if let override = ProcessInfo.processInfo.environment["V4ROUTE_GOLDEN_DIR"], !override.isEmpty {
            return try load(from: URL(fileURLWithPath: override))
        }
        return try load(from: defaultDirectory)
    }

    // MARK: test arguments

    struct CaseID: Sendable, CustomTestStringConvertible {
        let name: String
        let group: String
        var testDescription: String { "\(group)/\(name)" }
    }

    /// If loading failed the list holds one sentinel whose test rethrows the load error: red, never zero cases.
    static func ids(kind: String) -> [CaseID] {
        switch shared {
        case .success(let loaded):
            return loaded.fixtures.filter { $0.kind == kind }.map { CaseID(name: $0.name, group: $0.group) }
        case .failure:
            return [CaseID(name: "LOAD-FAILED", group: "loader")]
        }
    }

    static let handledKinds = ["gate", "hash", "expert", "moe", "hcpre", "hcpost", "hchead", "hcchain"]

    static func fixture(_ id: CaseID) throws -> Fixture {
        let loaded = try shared.get()
        guard let f = loaded.fixtures.first(where: { $0.name == id.name }) else {
            throw LoadError(description: "no fixture named \(id.name)")
        }
        return f
    }

    // MARK: known gaps (shipping code vs oracle). Remove an entry in the commit that fixes it.

    static let k1Gap =
        "top_k == 1: vLLM still renormalises (kernels divide by the sum, CU:191/469; reference TK:77-78), DeepseekV4Gate "
        + "skips normalisation when topK <= 1. Unreachable for shipped V4 (num_experts_per_tok = 6)."
    static let gaps: [String: String] = {
        var g: [String: String] = [
            "gate_fp32_k1_renorm": k1Gap, "hash_fp32_k1_renorm": k1Gap,
        ]
        return g
    }()

    // MARK: tensors

    static func array(_ t: Tensor, _ dtype: DType) -> MLXArray {
        MLXArray(t.data.map { Float($0) }, t.shape).asType(dtype)
    }

    static func ints(_ t: Tensor) -> MLXArray {
        MLXArray(t.data.map { Int32($0) }, t.shape)
    }

    static func dtype(_ f: Fixture) -> DType { f.dtype == "bfloat16" ? .bfloat16 : .float32 }

    // MARK: comparison (pure: returns violation strings, so the non-vacuity test can call it too)

    final class Worst: @unchecked Sendable {
        private let lock = NSLock()
        private var table: [String: Double] = [:]
        func note(_ key: String, _ usage: Double) {
            if ProcessInfo.processInfo.environment["V4ROUTE_USAGE"] == "1" { print("USAGE \(key) \(usage)") }
            lock.lock(); defer { lock.unlock() }
            table[key] = max(table[key] ?? 0, usage)
        }
        var snapshot: [String: Double] { lock.lock(); defer { lock.unlock() }; return table }
    }
    static let worst = Worst()

    static func compare(
        _ got: MLXArray, _ want: Tensor, tol: ToleranceClass, label: String, tag: String
    ) -> [String] {
        let g = got.asType(.float32).asArray(Float.self).map { Double($0) }
        guard got.shape == want.shape, g.count == want.data.count else {
            return ["\(label): shape \(got.shape) != golden \(want.shape)"]
        }
        if g.contains(where: { !$0.isFinite }) { return ["\(label): non-finite output"] }
        let maxWant = want.data.map { abs($0) }.max() ?? 0
        var out: [String] = []
        switch tol.mode {
        case "normwise":
            let err = zip(g, want.data).map { abs($0 - $1) }.max() ?? 0
            let bound = tol.atol + tol.rtol * maxWant
            worst.note("\(tag)/\(label)", bound > 0 ? err / bound : 0)
            if err > bound {
                out.append("\(label): normwise max|d| \(err) > \(bound) (rtol \(tol.rtol) x max|want| \(maxWant))")
            }
        default:
            var usage = 0.0, bad = 0, first = ""
            for (i, (a, b)) in zip(g, want.data).enumerated() {
                let bound = tol.atol + tol.rtol * abs(b)
                let err = abs(a - b)
                usage = max(usage, bound > 0 ? err / bound : (err == 0 ? 0 : .infinity))
                if err > bound {
                    bad += 1
                    if first.isEmpty { first = "element \(i): got \(a) want \(b) (|d| \(err) > \(bound))" }
                }
            }
            worst.note("\(tag)/\(label)", usage)
            if bad > 0 { out.append("\(label): \(bad) of \(g.count) elements out of tolerance; first: \(first)") }
        }
        return out
    }

    static func tol(_ m: Manifest, _ f: Fixture, _ output: String) throws -> ToleranceClass {
        guard let cls = f.tolerance[output], let t = m.tolerances[cls] else {
            throw LoadError(description: "\(f.name): no tolerance class for output \(output)")
        }
        return t
    }

    static func input(_ f: Fixture, _ k: String) throws -> Tensor {
        guard let t = f.inputs[k] else { throw LoadError(description: "\(f.name): missing input \(k)") }
        return t
    }

    static func output(_ f: Fixture, _ k: String) throws -> Tensor {
        guard let t = f.outputs[k] else { throw LoadError(description: "\(f.name): missing output \(k)") }
        return t
    }

    // MARK: running the real code

    /// The shipped Flash config with dimensions overridden (the same trick the other V4 suites use).
    static func config(_ overrides: [String: Any]) throws -> DeepseekV4Configuration {
        var o = try #require(try JSONSerialization.jsonObject(with: ConformanceDeepseekV4.tinyJSON()) as? [String: Any])
        for (k, v) in overrides { o[k] = v }
        return try JSONDecoder().decode(
            DeepseekV4Configuration.self, from: JSONSerialization.data(withJSONObject: o))
    }

    static func violations(_ f: Fixture) throws -> [String] {
        let loaded = try shared.get()
        switch f.kind {
        case "gate", "hash": return try runGate(f, loaded.manifest)
        case "expert": return try runExpert(f, loaded.manifest)
        case "moe": return try runMoE(f, loaded.manifest)
        case "hcpre": return try runHcPre(f, loaded.manifest)
        case "hcpost": return try runHcPost(f, loaded.manifest)
        case "hchead": return try runHcHead(f, loaded.manifest)
        case "hcchain": return try runHcChain(f, loaded.manifest)
        default: throw LoadError(description: "\(f.name): kind \(f.kind) has no runner")
        }
    }

    static func runGate(_ f: Fixture, _ m: Manifest) throws -> [String] {
        let p = f.params
        let isHash = f.inputs["table"] != nil
        let cfg = try config([
            "hidden_size": try #require(p.H), "n_routed_experts": try #require(p.E),
            "num_experts_per_tok": try #require(p.topk), "vocab_size": try #require(p.vocab),
            "norm_topk_prob": try #require(p.normTopkProb),
            "routed_scaling_factor": try #require(p.routedScalingFactor),
        ])
        let gate = DeepseekV4Gate(config: cfg, layerIndex: isHash ? 0 : 1)  // num_hash_layers = 1
        let dt = dtype(f)
        var params: [(String, MLXArray)] = [("weight", array(try input(f, "weight"), dt))]
        if isHash {
            params.append(("tid2eid", ints(try input(f, "table"))))
        } else {
            params.append(("e_score_correction_bias", array(try input(f, "bias"), .float32)))
        }
        gate.update(parameters: ModuleParameters.unflattened(params))
        let x = array(try input(f, "x"), dt)
        let (inds, w) = gate(x, tokenIds: isHash ? ints(try input(f, "tokenIds")) : nil)
        eval(inds, w)

        let k = try #require(p.topk)
        let wantIds = try output(f, "ids"), wantW = try output(f, "weights")
        let gi = inds.asType(.int32).asArray(Int32.self)
        let gw = w.asType(.float32).asArray(Float.self)
        guard gi.count == wantIds.data.count, gw.count == wantW.data.count else {
            return ["shape: ids \(inds.shape) weights \(w.shape) vs golden \(wantIds.shape)"]
        }
        // sort each token's (id, weight) pairs by id: the order inside the top-k is not part of the contract.
        var sortedIds: [Int32] = [], sortedW: [Float] = []
        for row in 0 ..< gi.count / k {
            let pairs = (0 ..< k).map { (gi[row * k + $0], gw[row * k + $0]) }.sorted { $0.0 < $1.0 }
            sortedIds += pairs.map { $0.0 }
            sortedW += pairs.map { $0.1 }
        }
        var out: [String] = []
        let wantI = wantIds.data.map { Int32($0) }
        if sortedIds != wantI {
            let bad = zip(sortedIds, wantI).enumerated().filter { $0.element.0 != $0.element.1 }
            out.append("ids: \(bad.count) of \(wantI.count) differ; first at \(bad[0].offset): got \(bad[0].element.0) want \(bad[0].element.1)")
        }
        out += compare(
            MLXArray(sortedW, wantW.shape), wantW, tol: try tol(m, f, "weights"), label: "weights", tag: f.name)
        return out
    }

    static func runExpert(_ f: Fixture, _ m: Manifest) throws -> [String] {
        let p = f.params
        let expert = DeepseekV4Expert(
            hiddenSize: try #require(p.H), intermediateSize: try #require(p.I),
            swiguLimit: Float(try #require(p.swigluLimit)))
        expert.update(parameters: ModuleParameters.unflattened([
            ("gate_proj.weight", array(try input(f, "wg"), .float32)),
            ("up_proj.weight", array(try input(f, "wu"), .float32)),
            ("down_proj.weight", array(try input(f, "wd"), .float32)),
        ]))
        let y = expert(array(try input(f, "x"), .float32))
        eval(y)
        return compare(y, try output(f, "y"), tol: try tol(m, f, "y"), label: "y", tag: f.name)
    }

    static func runMoE(_ f: Fixture, _ m: Manifest) throws -> [String] {
        let p = f.params
        let isHash = try #require(p.hash)
        let cfg = try config([
            "hidden_size": try #require(p.H), "moe_intermediate_size": try #require(p.I),
            "n_routed_experts": try #require(p.E), "num_experts_per_tok": try #require(p.topk),
            "vocab_size": try #require(p.vocab), "swiglu_limit": try #require(p.swigluLimit),
            "norm_topk_prob": try #require(p.normTopkProb),
            "routed_scaling_factor": try #require(p.routedScalingFactor), "n_shared_experts": 1,
        ])
        let moe = DeepseekV4MoE(config: cfg, layerIndex: isHash ? 0 : 1)
        var params: [(String, MLXArray)] = [
            ("gate.weight", array(try input(f, "gateWeight"), .float32)),
            ("switch_mlp.gate_proj.weight", array(try input(f, "wg"), .float32)),
            ("switch_mlp.up_proj.weight", array(try input(f, "wu"), .float32)),
            ("switch_mlp.down_proj.weight", array(try input(f, "wd"), .float32)),
            ("shared_experts.gate_proj.weight", array(try input(f, "sg"), .float32)),
            ("shared_experts.up_proj.weight", array(try input(f, "su"), .float32)),
            ("shared_experts.down_proj.weight", array(try input(f, "sd"), .float32)),
        ]
        if isHash {
            params.append(("gate.tid2eid", ints(try input(f, "table"))))
        } else {
            params.append(("gate.e_score_correction_bias", array(try input(f, "bias"), .float32)))
        }
        moe.update(parameters: ModuleParameters.unflattened(params))
        let y = moe(
            array(try input(f, "x"), .float32), tokenIds: isHash ? ints(try input(f, "tokenIds")) : nil)
        eval(y)
        return compare(y, try output(f, "y"), tol: try tol(m, f, "y"), label: "y", tag: f.name)
    }

    static func runHcPre(_ f: Fixture, _ m: Manifest) throws -> [String] {
        let p = f.params
        let (y, post, comb) = hcPre(
            x: array(try input(f, "x"), .float32), hcFn: array(try input(f, "fn"), .float32),
            hcScale: array(try input(f, "scale"), .float32), hcBase: array(try input(f, "base"), .float32),
            hcMult: try #require(p.hc), sinkhornIters: try #require(p.sinkhornIters),
            rmsEps: Float(try #require(p.rmsEps)), eps: Float(try #require(p.hcEps)))
        eval(y, post, comb)
        return compare(y, try output(f, "y"), tol: try tol(m, f, "y"), label: "y", tag: f.name)
            + compare(post, try output(f, "post"), tol: try tol(m, f, "post"), label: "post", tag: f.name)
            + compare(comb, try output(f, "comb"), tol: try tol(m, f, "comb"), label: "comb", tag: f.name)
    }

    static func runHcPost(_ f: Fixture, _ m: Manifest) throws -> [String] {
        let dt = dtype(f)
        let y = hcPost(
            x: array(try input(f, "x"), dt), residual: array(try input(f, "residual"), dt),
            post: array(try input(f, "post"), .float32), comb: array(try input(f, "comb"), .float32))
        eval(y)
        return compare(y, try output(f, "y"), tol: try tol(m, f, "y"), label: "y", tag: f.name)
    }

    static func runHcHead(_ f: Fixture, _ m: Manifest) throws -> [String] {
        let p = f.params
        let y = hcHead(
            x: array(try input(f, "x"), .float32), hcFn: array(try input(f, "fn"), .float32),
            hcScale: array(try input(f, "scale"), .float32), hcBase: array(try input(f, "base"), .float32),
            rmsEps: Float(try #require(p.rmsEps)), eps: Float(try #require(p.hcEps)))
        eval(y)
        return compare(y, try output(f, "y"), tol: try tol(m, f, "y"), label: "y", tag: f.name)
    }

    static func runHcChain(_ f: Fixture, _ m: Manifest) throws -> [String] {
        let p = f.params
        let x = array(try input(f, "x"), .float32)
        let (y, post, comb) = hcPre(
            x: x, hcFn: array(try input(f, "fn"), .float32),
            hcScale: array(try input(f, "scale"), .float32), hcBase: array(try input(f, "base"), .float32),
            hcMult: try #require(p.hc), sinkhornIters: try #require(p.sinkhornIters),
            rmsEps: Float(try #require(p.rmsEps)), eps: Float(try #require(p.hcEps)))
        let sub = y * Float(try #require(p.sublayerScale)) + Float(try #require(p.sublayerShift))
        let out = hcPost(x: sub, residual: x, post: post, comb: comb)
        eval(out)
        return compare(out, try output(f, "y"), tol: try tol(m, f, "y"), label: "y", tag: f.name)
    }

    // MARK: the per-case check

    static func check(_ id: CaseID, sourceLocation: SourceLocation = #_sourceLocation) throws {
        let f = try fixture(id)
        let v = try violations(f)
        if let why = gaps[f.name] {
            ConformanceGap.knownIssue(Comment(rawValue: "\(f.name): \(why)"), sourceLocation: sourceLocation) {
                ConformanceGap.expect(v.isEmpty, "\(f.name): \(v.joined(separator: "; "))")
            }
        } else {
            #expect(v.isEmpty, "\(f.name): \(v.joined(separator: "; "))", sourceLocation: sourceLocation)
        }
    }

    /// The golden with every output pushed away from the truth: the check must then fail.
    static func perturbed(_ f: Fixture) -> Fixture {
        var g = f
        for (k, t) in f.outputs {
            let isIds = k == "ids"
            let maxWant = t.data.map { abs($0) }.max() ?? 0
            var data = t.data
            if isIds {
                data[0] = data[0] + 1  // a different expert
            } else {
                data[0] += 0.2 * maxWant + 0.01
            }
            g.outputs[k] = Tensor(shape: t.shape, data: data)
        }
        return g
    }
}

struct ConformanceV4RouteGoldenTests {

    @Test("learned gate (bias-shifted selection, unbiased weights) vs the vLLM oracle",
          arguments: V4RouteGolden.ids(kind: "gate"))
    func gate(_ id: V4RouteGolden.CaseID) throws { try V4RouteGolden.check(id) }

    @Test("hash gate (tid2eid lookup, learned weights) vs the vLLM oracle",
          arguments: V4RouteGolden.ids(kind: "hash"))
    func hash(_ id: V4RouteGolden.CaseID) throws { try V4RouteGolden.check(id) }

    @Test("shared expert (clamped SwiGLU) vs the vLLM oracle", arguments: V4RouteGolden.ids(kind: "expert"))
    func expert(_ id: V4RouteGolden.CaseID) throws { try V4RouteGolden.check(id) }

    @Test("MoE layer (gate + routed experts + shared expert) vs the vLLM oracle",
          arguments: V4RouteGolden.ids(kind: "moe"))
    func moe(_ id: V4RouteGolden.CaseID) throws { try V4RouteGolden.check(id) }

    @Test("mHC pre (Sinkhorn) vs the vLLM oracle", arguments: V4RouteGolden.ids(kind: "hcpre"))
    func hcpre(_ id: V4RouteGolden.CaseID) throws { try V4RouteGolden.check(id) }

    @Test("mHC post vs the vLLM oracle", arguments: V4RouteGolden.ids(kind: "hcpost"))
    func hcpost(_ id: V4RouteGolden.CaseID) throws { try V4RouteGolden.check(id) }

    @Test("mHC head vs the vLLM oracle", arguments: V4RouteGolden.ids(kind: "hchead"))
    func hchead(_ id: V4RouteGolden.CaseID) throws { try V4RouteGolden.check(id) }

    @Test("mHC pre -> sublayer -> post chain vs the vLLM oracle", arguments: V4RouteGolden.ids(kind: "hcchain"))
    func hcchain(_ id: V4RouteGolden.CaseID) throws { try V4RouteGolden.check(id) }

    // MARK: the harness itself

    @Test("every fixture in the manifest is covered by a runner, and the argument lists add up")
    func everyFixtureIsUsed() throws {
        let loaded = try V4RouteGolden.shared.get()
        let kinds = Set(loaded.fixtures.map { $0.kind })
        #expect(kinds.isSubset(of: Set(V4RouteGolden.handledKinds)), "kinds without a runner: \(kinds.subtracting(V4RouteGolden.handledKinds))")
        #expect(Set(V4RouteGolden.handledKinds).isSubset(of: kinds), "a runner without fixtures")
        let used = V4RouteGolden.handledKinds.flatMap { V4RouteGolden.ids(kind: $0) }.map { $0.name }
        #expect(used.count == loaded.fixtures.count && Set(used) == Set(loaded.fixtures.map { $0.name }))
        #expect(loaded.manifest.count == loaded.fixtures.count)
        // every gap entry names a real fixture
        #expect(Set(V4RouteGolden.gaps.keys).isSubset(of: Set(loaded.fixtures.map { $0.name })))
    }

    @Test("non-vacuity: a perturbed golden fails the same check", arguments: V4RouteGolden.handledKinds)
    func perturbedGoldenFails(kind: String) throws {
        let loaded = try V4RouteGolden.shared.get()
        let cases = loaded.fixtures.filter { $0.kind == kind && V4RouteGolden.gaps[$0.name] == nil }
        for f in cases {
            let bad = try V4RouteGolden.violations(V4RouteGolden.perturbed(f))
            #expect(!bad.isEmpty, "\(f.name): a golden moved by 20% of its range still passed")
        }
    }

    @Test("the fixtures exercise what they claim (selection flips, clamp active, tie, repeated tokens)")
    func fixturesAreNotVacuous() throws {
        let loaded = try V4RouteGolden.shared.get()
        func fx(_ n: String) throws -> V4RouteGolden.Fixture {
            try #require(loaded.fixtures.first { $0.name == n })
        }
        // clamp-active expert cases really have pre-activation values outside the limit; the inactive one has none.
        let data = { (n: String) throws -> [String: Int] in
            let url = V4RouteGolden.defaultDirectory.appendingPathComponent(n + ".json")
            let o = try #require(
                try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any])
            return try #require(o["stats"] as? [String: Int])
        }
        let inactive = try data("expert_limit_inactive")
        #expect(inactive.values.allSatisfy { $0 == 0 })
        let below = try data("expert_gate_below")
        #expect((below["gateBelowNegLimit"] ?? 0) > 0)
        let above = try data("expert_gate_above")
        #expect((above["gateAboveLimit"] ?? 0) > 0)
        let up = try data("expert_up_outside")
        #expect((up["upOutsideLimit"] ?? 0) > 0)
        // repeated token ids give identical id rows
        let rep = try fx("hash_fp32_repeated_token")
        let ids = try V4RouteGolden.output(rep, "ids").data
        let k = try #require(rep.params.topk)
        #expect(Array(ids[0 ..< k]) == Array(ids[2 * k ..< 3 * k]) && Array(ids[k ..< 2 * k]) == Array(ids[3 * k ..< 4 * k]))
    }

    @Test("a fixture that was edited after generation is rejected (sha256)")
    func tamperedFixtureRejected() throws {
        let fm = FileManager.default
        let tmp = fm.temporaryDirectory.appendingPathComponent("v4route-tamper-\(UUID().uuidString)")
        try fm.copyItem(at: V4RouteGolden.defaultDirectory, to: tmp)
        defer { try? fm.removeItem(at: tmp) }
        let victim = tmp.appendingPathComponent("gate_fp32_renorm_scale.json")
        var d = try Data(contentsOf: victim)
        d.append(contentsOf: [0x20])
        try d.write(to: victim)
        #expect(throws: V4RouteGolden.LoadError.self) { try V4RouteGolden.load(from: tmp) }
    }
}
