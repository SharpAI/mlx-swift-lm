// KDA golden fixtures: the SharpAI KDA Metal kernel (and its op-composition fallback) against a float64 oracle
// transcribed from vLLM (commit 739e5945d) and committed as JSON. The tests never run python and never touch the
// network; the fixtures are located through `#filePath` (no Package.swift resource changes) and verified against
// MANIFEST.json (sha256 per file) at load, so a changed fixture fails loudly instead of silently moving the target.
//
// Generator: tools/conformance/golden/gen_kda_golden.py (run once, commit outputs). Oracle:
// tools/conformance/golden/kda_ref.py (every line cites vLLM file:line; items NOT in vLLM are listed there as I1..I6:
// the per-token mask semantics and the lower_bound "safe gate" are INFERRED).
//
// What the test does for a case mirrors the CALLER of the kernel: l2-normalise q and k (if `l2norm`), multiply q by
// `scale`, then call SharpAIOverlayGatedDeltaKDA.update with g = exp(logDecay). q/k/v are already rounded to the
// case dtype in the fixture, so a tolerance only has to cover compute error. The tolerances (with their reasoning)
// live in the manifest. upstream-watch: Libraries/MLXLMCommon/SharpAIOverlay/GatedDeltaKDA.swift.

import CryptoKit
import Foundation
import MLX
import Testing

@testable import MLXLMCommon

enum KdaGolden {

    // MARK: file format

    struct Tol: Decodable, Sendable {
        let rtol: Double
        let atol: Double
    }

    struct ToleranceClass: Decodable, Sendable {
        let y: Tol
        let state: Tol?
        /// "normwise": max|d| <= atol + rtol * max|want| ; "elementwise": |d_i| <= atol + rtol * |want_i| for every i.
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

    struct Inputs: Decodable, Sendable {
        let q: [Double]?
        let k: [Double]?
        let v: [Double]?
        let logDecay: [Double]?
        let logDecayConst: Double?
        let beta: [Double]?
        let betaConst: Double?
        let state0: [Double]?
        let mask: [[Bool]]?
        let g: [Double]?
        let aLog: [Double]?
        let dtBias: [Double]?
    }

    struct Outputs: Decodable, Sendable {
        let y: [Double]?
        let state1: [Double]?
        let logDecay: [Double]?
    }

    struct Fixture: Decodable, Sendable {
        let name: String
        let kind: String
        let group: String
        let dtype: String
        let tolerance: String
        let expectedPath: String?
        let B: Int?
        let T: Int?
        let Hk: Int?
        let Hv: Int?
        let Dk: Int?
        let Dv: Int?
        let H: Int?
        let D: Int?
        let scale: Double?
        let l2norm: Bool?
        let softplusBeta: Double?
        let threshold: Double?
        let lowerBound: Double?
        let inputs: Inputs
        let outputs: Outputs
    }

    struct Loaded: Sendable {
        let manifest: Manifest
        let fixtures: [Fixture]
    }

    struct LoadError: Error, CustomStringConvertible {
        let description: String
    }

    // MARK: loader

    /// <repo>/Tests/MLXLMTests/Conformance/Golden/kda, derived from this file's path.
    static let defaultDirectory: URL = {
        var url = URL(fileURLWithPath: #filePath)
        url.deleteLastPathComponent()
        return url.appendingPathComponent("Golden/kda")
    }()

    /// <repo>/tools/conformance/golden (generator and oracle), derived from this file's path.
    static let defaultToolsDirectory: URL = {
        var url = URL(fileURLWithPath: #filePath)
        for _ in 0 ..< 4 { url.deleteLastPathComponent() }
        return url.appendingPathComponent("tools/conformance/golden")
    }()

    static func sha256(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    /// Loads and VERIFIES a fixture directory. Throws (never skips) when anything is off: a missing or extra
    /// file, a sha256 that differs from the manifest, a count that does not add up, an unknown tolerance class.
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
        // The manifest pins the generator and the oracle it was produced with: editing either without regenerating
        // the fixtures is drift, and fails here. (Regenerating fixtures AND manifest together is self-consistent by
        // design; review the diff of kda_ref.py for that.)
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
                        + "Rerun tools/conformance/golden/gen_kda_golden.py and commit the regenerated fixtures.")
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
                        + "Regenerate with tools/conformance/golden/gen_kda_golden.py and review the diff.")
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
            guard manifest.tolerances[fixture.tolerance] != nil else {
                throw LoadError(description: "\(name): unknown tolerance class \(fixture.tolerance)")
            }
            fixtures.append(fixture)
        }
        return Loaded(manifest: manifest, fixtures: fixtures)
    }

    /// Loaded once. `KDA_GOLDEN_DIR` points the loader at another directory (used to show the suite red against a
    /// perturbed copy; unset in normal runs).
    static let shared: Result<Loaded, any Error> = Result {
        if let override = ProcessInfo.processInfo.environment["KDA_GOLDEN_DIR"], !override.isEmpty {
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

    /// If loading failed the argument list holds one sentinel case whose test rethrows the load error: the suite
    /// goes red instead of running zero cases.
    static func ids(kind: String) -> [CaseID] {
        switch shared {
        case .success(let loaded):
            return loaded.fixtures.filter { $0.kind == kind }.map { CaseID(name: $0.name, group: $0.group) }
        case .failure:
            return [CaseID(name: "LOAD-FAILED", group: "loader")]
        }
    }

    static let recurrentIDs: [CaseID] = ids(kind: "recurrent")
    static let gateIDs: [CaseID] = ids(kind: "gate")

    static func fixture(_ id: CaseID) throws -> (Fixture, ToleranceClass, Manifest) {
        let loaded = try shared.get()
        guard let f = loaded.fixtures.first(where: { $0.name == id.name }) else {
            throw LoadError(description: "no fixture named \(id.name)")
        }
        guard let tol = loaded.manifest.tolerances[f.tolerance] else {
            throw LoadError(description: "no tolerance class \(f.tolerance)")
        }
        return (f, tol, loaded.manifest)
    }

    // MARK: comparison

    struct Check {
        /// worst |got - want| / allowed ; <= 1 means within tolerance. NaN or a size mismatch never passes.
        let ratio: Double
        let maxAbsError: Double
        let count: Int
        var ok: Bool { count > 0 && ratio <= 1 }
    }

    static func check(_ got: [Float], _ want: [Double], tol: Tol, mode: String) -> Check {
        guard got.count == want.count, !want.isEmpty else {
            return Check(ratio: Double.infinity, maxAbsError: Double.infinity, count: 0)
        }
        var maxErr = Double(0)
        var worst = Double(0)
        var maxWant = Double(0)
        for w in want { maxWant = max(maxWant, abs(w)) }
        for i in 0 ..< want.count {
            let d = abs(Double(got[i]) - want[i])
            if d.isNaN { return Check(ratio: Double.nan, maxAbsError: Double.nan, count: want.count) }
            maxErr = max(maxErr, d)
            let allowed = mode == "elementwise" ? tol.atol + tol.rtol * abs(want[i]) : tol.atol + tol.rtol * maxWant
            let r: Double = d == 0 ? 0 : d / allowed
            worst = max(worst, r)
        }
        return Check(ratio: worst, maxAbsError: maxErr, count: want.count)
    }

    // MARK: running a recurrent case

    /// The fixture's inputs as plain arrays, so a test can perturb them in memory before running.
    struct Prepared {
        var q: [Float]
        var k: [Float]
        var v: [Float]
        var logDecay: [Float]
        var beta: [Float]
        var state0: [Float]?
        var mask: [Bool]?
    }

    static func prepare(_ f: Fixture) throws -> Prepared {
        func floats(_ a: [Double]?, _ what: String) throws -> [Float] {
            guard let a else { throw LoadError(description: "\(f.name): missing input \(what)") }
            return a.map { Float($0) }
        }
        guard let B = f.B, let T = f.T, let Hv = f.Hv, let Dk = f.Dk else {
            throw LoadError(description: "\(f.name): missing shape")
        }
        let logDecay: [Float]
        if let c = f.inputs.logDecayConst {
            logDecay = [Float](repeating: Float(c), count: B * T * Hv * Dk)
        } else {
            logDecay = try floats(f.inputs.logDecay, "logDecay")
        }
        let beta: [Float]
        if let c = f.inputs.betaConst {
            beta = [Float](repeating: Float(c), count: B * T * Hv)
        } else {
            beta = try floats(f.inputs.beta, "beta")
        }
        return Prepared(
            q: try floats(f.inputs.q, "q"), k: try floats(f.inputs.k, "k"), v: try floats(f.inputs.v, "v"),
            logDecay: logDecay, beta: beta,
            state0: f.inputs.state0.map { $0.map { Float($0) } },
            mask: f.inputs.mask.map { $0.flatMap { $0 } })
    }

    static func dtype(_ name: String) throws -> DType {
        switch name {
        case "float32": return .float32
        case "float16": return .float16
        case "bfloat16": return .bfloat16
        default: throw LoadError(description: "unknown dtype \(name)")
        }
    }

    /// Mirrors the caller: l2norm (vLLM FR:128-129: x / sqrt(sum(x^2) + 1e-6)), then q * scale, then the kernel.
    static func execute(_ f: Fixture, _ p: Prepared, useKernel: Bool) throws -> (y: [Float], state: [Float]) {
        guard let B = f.B, let T = f.T, let Hk = f.Hk, let Hv = f.Hv, let Dk = f.Dk, let Dv = f.Dv else {
            throw LoadError(description: "\(f.name): missing shape")
        }
        let dt = try dtype(f.dtype)
        var q = MLXArray(p.q, [B, T, Hk, Dk])
        var k = MLXArray(p.k, [B, T, Hk, Dk])
        let v = MLXArray(p.v, [B, T, Hv, Dv])
        if f.l2norm == true {
            q = q / sqrt(sum(q * q, axis: -1, keepDims: true) + Float(1e-6))
            k = k / sqrt(sum(k * k, axis: -1, keepDims: true) + Float(1e-6))
        }
        q = q * Float(f.scale ?? Double(1))
        let g = exp(MLXArray(p.logDecay, [B, T, Hv, Dk]))
        let beta = MLXArray(p.beta, [B, T, Hv])
        let state: MLXArray? = p.state0.map { MLXArray($0, [B, Hv, Dv, Dk]) }
        let mask: MLXArray? = p.mask.map { MLXArray($0, [B, T]) }
        let r = SharpAIOverlayGatedDeltaKDA.update(
            q: q.asType(dt), k: k.asType(dt), v: v.asType(dt), g: g, beta: beta,
            state: state, mask: mask, useKernel: useKernel)
        eval(r.y, r.state)
        guard r.y.dtype == dt, r.state.dtype == .float32, r.y.shape == [B, T, Hv, Dv] else {
            throw LoadError(description: "\(f.name): unexpected output dtype/shape \(r.y.dtype) \(r.y.shape) \(r.state.dtype)")
        }
        return (r.y.asType(.float32).asArray(Float.self), r.state.asArray(Float.self))
    }

    /// Compares y and the final state to the golden. Returns both checks.
    static func compare(
        _ f: Fixture, _ tol: ToleranceClass, _ out: (y: [Float], state: [Float])
    ) throws -> (y: Check, state: Check) {
        guard let wantY = f.outputs.y, let wantS = f.outputs.state1, let stTol = tol.state else {
            throw LoadError(description: "\(f.name): missing golden outputs or state tolerance")
        }
        return (
            check(out.y, wantY, tol: tol.y, mode: tol.mode),
            check(out.state, wantS, tol: stTol, mode: tol.mode)
        )
    }
}

@Suite("ConformanceKdaGoldenTests")
struct ConformanceKdaGoldenTests {
    typealias G = KdaGolden

    @Test("recurrent golden: Metal kernel AND op-composition fallback match the vLLM-derived oracle", arguments: G.recurrentIDs)
    func recurrent(_ id: G.CaseID) throws {
        let (f, tol, _) = try G.fixture(id)
        let p = try G.prepare(f)
        // The fixture says which path the kernel dispatch takes for its shape: keep that in step with
        // SharpAIOverlayGatedDeltaKDA.update (Dk % 32 == 0 && Dv % 4 == 0 -> kernel, else fallback).
        let kernelShape = (f.Dk ?? 0) % 32 == 0 && (f.Dv ?? 0) % 4 == 0
        #expect((f.expectedPath == "kernel") == kernelShape, "\(f.name): expectedPath disagrees with the dispatch rule")
        for useKernel in [true, false] {
            let out = try G.execute(f, p, useKernel: useKernel)
            let c = try G.compare(f, tol, out)
            let path = useKernel ? (kernelShape ? "kernel" : "kernel-requested(fallback shape)") : "fallback"
            print(
                "KDA-GOLDEN \(f.name) [\(path)] y: err=\(c.y.maxAbsError) ratio=\(c.y.ratio) | state: err=\(c.state.maxAbsError) ratio=\(c.state.ratio)"
            )
            #expect(c.y.ok, "\(f.name) [\(path)] y: max|d|=\(c.y.maxAbsError) = \(c.y.ratio)x the allowed error (n=\(c.y.count))")
            #expect(
                c.state.ok,
                "\(f.name) [\(path)] state: max|d|=\(c.state.maxAbsError) = \(c.state.ratio)x the allowed error (n=\(c.state.count))")
        }
    }

    @Test("gate golden: the repo's KDA log-decay reference matches the vLLM-derived gate", arguments: G.gateIDs)
    func gate(_ id: G.CaseID) throws {
        let (f, tol, _) = try G.fixture(id)
        guard let T = f.T, let H = f.H, let D = f.D, let g = f.inputs.g, let aLog = f.inputs.aLog,
            let want = f.outputs.logDecay
        else { throw G.LoadError(description: "\(f.name): incomplete gate fixture") }
        // SharpAIOverlayGdnReference.kdaLogDecay implements softplus with beta 1 (logAddExp); the fixtures must not ask for more.
        #expect(f.softplusBeta == 1 && f.threshold == 20, "\(f.name): reference supports softplus beta 1 only")
        let bias = f.inputs.dtBias ?? [Double](repeating: 0, count: H * D)
        let got = SharpAIOverlayGdnReference.kdaLogDecay(
            aLog: MLXArray(aLog.map { Float($0) }, [H]),
            g: MLXArray(g.map { Float($0) }, [1, T, H, D]),
            bias: MLXArray(bias.map { Float($0) }, [H, D]),
            lowerBound: f.lowerBound.map { Float($0) })
        eval(got)
        let c = G.check(got.asArray(Float.self), want, tol: tol.y, mode: tol.mode)
        print("KDA-GOLDEN \(f.name) [gate\(f.lowerBound == nil ? "" : ", lower_bound INFERRED")] err=\(c.maxAbsError) ratio=\(c.ratio)")
        #expect(c.ok, "\(f.name): max|d|=\(c.maxAbsError) = \(c.ratio)x the allowed error (n=\(c.count))")
    }

    @Test("every fixture in the manifest is exercised by exactly one parametrised case")
    func everyFixtureIsUsed() throws {
        let loaded = try G.shared.get()
        #expect(loaded.manifest.count > 0)
        #expect(loaded.fixtures.count == loaded.manifest.count)
        let used = Set((G.recurrentIDs + G.gateIDs).map(\.name))
        #expect(used.count == G.recurrentIDs.count + G.gateIDs.count, "duplicate case names")
        #expect(used == Set(loaded.fixtures.map(\.name)), "a fixture has no test")
        #expect(used.count == loaded.manifest.count, "tests cover \(used.count) of \(loaded.manifest.count) manifest entries")
        for f in loaded.fixtures { #expect(f.kind == "recurrent" || f.kind == "gate", "\(f.name): unknown kind \(f.kind)") }
        // The suite must cover the shapes the task is about, not just any N.
        let names = Set(loaded.fixtures.map(\.name))
        for required in ["mask_all_false_state", "full_forget_neg1000", "long_t256_nodecay_beta_half", "gqa_hk1_hv4", "t3_state_dk128", "fb_dk16", "dv8_state_dk32", "dv16_state_dk64", "b3_hk2_hv4_mask_dv8", "b2_hk4_hv8_nomask", "fb_t40_b2_hk2_hv4_mask"] {
            #expect(names.contains(required), "required case \(required) missing")
        }
    }

    @Test("the manifest checksum is real: a changed byte in a copy of the fixtures fails the loader")
    func loaderRejectsAChangedFixture() throws {
        let loaded = try G.shared.get()
        let fm = FileManager.default
        let tmp = fm.temporaryDirectory.appendingPathComponent("kda-golden-\(UUID().uuidString)")
        try fm.copyItem(at: G.defaultDirectory, to: tmp)
        defer { try? fm.removeItem(at: tmp) }
        // untouched copy loads
        #expect(try G.load(from: tmp).fixtures.count == loaded.manifest.count)
        // flip one digit in one fixture
        let victim = tmp.appendingPathComponent("t1_state_dk32.json")
        var text = try String(contentsOf: victim, encoding: .utf8)
        let idx = try #require(text.range(of: "\"state1\":["))
        let digitRange = idx.upperBound ..< text.index(idx.upperBound, offsetBy: 1)
        text.replaceSubrange(digitRange, with: text[digitRange] == "9" ? "8" : "9")
        try text.write(to: victim, atomically: true, encoding: .utf8)
        #expect(throws: G.LoadError.self) { try G.load(from: tmp) }
        // a stray extra fixture also fails
        try fm.removeItem(at: victim)
        try fm.copyItem(at: G.defaultDirectory.appendingPathComponent("t1_state_dk32.json"), to: victim)
        try Data("{}".utf8).write(to: tmp.appendingPathComponent("stray.json"))
        #expect(throws: G.LoadError.self) { try G.load(from: tmp) }
    }

    @Test("the manifest pins the generator and the oracle: a changed byte in either fails the loader")
    func loaderRejectsADriftedOracle() throws {
        let loaded = try G.shared.get()
        let fm = FileManager.default
        let tmp = fm.temporaryDirectory.appendingPathComponent("kda-tools-\(UUID().uuidString)")
        try fm.createDirectory(at: tmp, withIntermediateDirectories: true)
        defer { try? fm.removeItem(at: tmp) }
        let names = [loaded.manifest.generator, loaded.manifest.oracle].map { URL(fileURLWithPath: $0).lastPathComponent }
        for n in names { try fm.copyItem(at: G.defaultToolsDirectory.appendingPathComponent(n), to: tmp.appendingPathComponent(n)) }
        // untouched copies load
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

    @Test("mutation self-test: perturbing any one golden input slightly makes the comparison FAIL (kernel and fallback)")
    func comparisonIsNotVacuous() throws {
        let id = G.CaseID(name: "t7_state_dk32", group: "multi_step")
        let (f, tol, _) = try G.fixture(id)
        let clean = try G.prepare(f)
        for useKernel in [true, false] {
            let base = try G.compare(f, tol, try G.execute(f, clean, useKernel: useKernel))
            #expect(base.y.ok && base.state.ok, "unperturbed baseline must pass (kernel=\(useKernel))")
            let mutations: [(String, (inout G.Prepared) -> Void)] = [
                ("q[0] += 0.01", { $0.q[0] += Float(0.01) }),
                ("k[0] += 0.01", { $0.k[0] += Float(0.01) }),
                ("v[0] += 0.01", { $0.v[0] += Float(0.01) }),
                ("logDecay[0] += 0.01", { $0.logDecay[0] += Float(0.01) }),
                ("beta[0] += 0.01", { $0.beta[0] += Float(0.01) }),
                ("state0[0] += 0.01", { $0.state0?[0] += Float(0.01) }),
            ]
            for (label, mutate) in mutations {
                var p = clean
                mutate(&p)
                let c = try G.compare(f, tol, try G.execute(f, p, useKernel: useKernel))
                let failed = !(c.y.ok && c.state.ok)
                print("KDA-GOLDEN mutation [\(useKernel ? "kernel" : "fallback")] \(label): y ratio=\(c.y.ratio) state ratio=\(c.state.ratio)")
                #expect(failed, "mutation \"\(label)\" went undetected (kernel=\(useKernel)): y ratio \(c.y.ratio), state ratio \(c.state.ratio)")
            }
        }
        // And the comparison itself: a golden output moved by 2x the allowed error must fail, one moved by 0.5x must pass.
        let want = try #require(f.outputs.y)
        let out = try G.execute(f, clean, useKernel: true)
        let allowed = tol.y.atol + tol.y.rtol * (want.map { abs($0) }.max() ?? 0)
        var off = want
        off[0] += 2 * allowed
        #expect(!G.check(out.y, off, tol: tol.y, mode: tol.mode).ok)
        var near = want
        near[0] += allowed / 2
        #expect(G.check(out.y, near, tol: tol.y, mode: tol.mode).ok)
        #expect(!G.check([Float](), [Double](), tol: tol.y, mode: tol.mode).ok, "an empty comparison must not pass")
        #expect(!G.check([Float.nan], [Double(1)], tol: tol.y, mode: tol.mode).ok, "NaN must not pass")
    }
}
