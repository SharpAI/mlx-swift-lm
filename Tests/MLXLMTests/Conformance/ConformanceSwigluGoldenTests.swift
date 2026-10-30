// Golden conformance for the SwiGLU family (lane SWIGLU, sl36/swiglu-golden).
//
// Fixtures: Tests/MLXLMTests/Conformance/Golden/swiglu/*.json, generated ONCE by
// tools/conformance/golden/swiglu/generate.py from a float64 torch oracle transcribed from vLLM
// 739e5945d activation.py. Tests never run Python. Each case is run through the REAL shipping Swift
// module that the model uses (no reimplementation of the activation here):
//   silu(gate)*up          -> compiledSiluProduct, SwitchGLU default init, DeepseekV4Expert (limit 0)
//   swigluoai (GPT-OSS)    -> GPTOSS.swift SwiGLUSwitchGLU (compiledSwiglu inside)
//   V4 dense expert        -> DeepseekV4Expert(swiguLimit:)
//   V4 routed experts      -> DeepseekV4MoE.switchMLP (SwitchGLU with the activation closure)
//   Step silu-then-clamp   -> no Swift path exists (pinned as a gap)
// The projections are 0/1 selection matrices, so the only arithmetic in the output is the activation
// (a single non-zero product per dot product is exact in every dtype).

import CryptoKit
import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

// TODO consolidate GoldenKit
enum SwigluGolden {
    struct Case: Decodable {
        let name: String
        let dtype: String
        let limit: Double
        let rows: Int
        let d: Int
        let x: [Double]
        let golden: [Double]
        let goldenGateMaxOnly: [Double]?
        enum CodingKeys: String, CodingKey {
            case name, dtype, limit, rows, d, x, golden
            case goldenGateMaxOnly = "golden_gate_max_only"
        }
    }
    struct File: Decodable {
        let op: String
        let layout: String
        let cases: [Case]
    }
    struct Manifest: Decodable {
        struct Entry: Decodable { let sha256: String; let cases: Int }
        struct Tol: Decodable { let rtol: Double; let atol: Double }
        struct Tols: Decodable { let float32: Tol; let bfloat16: Tol; let float16: Tol }
        let files: [String: Entry]
        let tolerances: Tols
    }

    static let dir = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
        .appendingPathComponent("Golden/swiglu")

    static func manifest() throws -> Manifest {
        try JSONDecoder().decode(Manifest.self, from: Data(contentsOf: dir.appendingPathComponent("MANIFEST.json")))
    }

    /// Loads a fixture after verifying its sha256 against MANIFEST.json.
    static func load(_ name: String) throws -> File {
        let data = try Data(contentsOf: dir.appendingPathComponent(name))
        let m = try manifest()
        let want = try #require(m.files[name], "\(name) is not in MANIFEST.json")
        let got = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
        #expect(got == want.sha256, "\(name): sha256 \(got) != manifest \(want.sha256)")
        let f = try JSONDecoder().decode(File.self, from: data)
        #expect(f.cases.count == want.cases, "\(name): \(f.cases.count) cases != manifest \(want.cases)")
        return f
    }

    static func dtype(_ s: String) -> DType {
        switch s {
        case "float32": return .float32
        case "bfloat16": return .bfloat16
        default: return .float16
        }
    }

    static func tol(_ s: String, _ m: Manifest) -> (rtol: Double, atol: Double) {
        let t = s == "float32" ? m.tolerances.float32 : (s == "bfloat16" ? m.tolerances.bfloat16 : m.tolerances.float16)
        return (t.rtol, t.atol)
    }

    /// Indices where |out - golden| > atol + rtol*|golden| (NaN / inf always count).
    static func mismatches(_ out: [Float], _ golden: [Double], dtype: String, _ m: Manifest) -> [Int] {
        let (rtol, atol) = tol(dtype, m)
        var bad: [Int] = []
        for (i, g) in golden.enumerated() {
            let o = Double(out[i])
            if !o.isFinite || abs(o - g) > atol + rtol * abs(g) { bad.append(i) }
        }
        return bad
    }

    /// 0/1 projections that route the fixture layout into gate / up and read the first d outputs back.
    /// layout "chunk": gate = x[:d], up = x[d:];  "interleave": gate = x[::2], up = x[1::2].
    static func selections(layout: String, d: Int, dt: DType) -> (gate: MLXArray, up: MLXArray, down: MLXArray) {
        var g = [Float](repeating: 0, count: d * 2 * d)
        var u = [Float](repeating: 0, count: d * 2 * d)
        var o = [Float](repeating: 0, count: 2 * d * d)
        for i in 0 ..< d {
            let gc = layout == "chunk" ? i : 2 * i
            let uc = layout == "chunk" ? d + i : 2 * i + 1
            g[i * 2 * d + gc] = 1
            u[i * 2 * d + uc] = 1
            o[i * d + i] = 1   // down: [2d, d]; first d rows are the identity
        }
        return (MLXArray(g, [d, 2 * d]).asType(dt), MLXArray(u, [d, 2 * d]).asType(dt), MLXArray(o, [2 * d, d]).asType(dt))
    }

    static func input(_ c: Case, dt: DType) -> MLXArray {
        MLXArray(c.x.map { Float($0) }, [c.rows, 2 * c.d]).asType(dt)
    }

    static func floats(_ a: MLXArray) -> [Float] {
        let f = a.asType(.float32)
        eval(f)
        return f.reshaped(-1).asArray(Float.self)
    }

    /// V4 config whose hidden size is 2d and expert width d, so a MoE block's SwitchGLU maps [.., 2d] -> [.., d] -> [.., 2d].
    static func v4Config(d: Int, limit: Double) throws -> DeepseekV4Configuration {
        let json = """
            {"vocab_size": 256, "hidden_size": \(2 * d), "moe_intermediate_size": \(d), "num_hidden_layers": 4,
             "num_attention_heads": 4, "head_dim": 16, "q_lora_rank": 32, "qk_rope_head_dim": 4,
             "rms_norm_eps": 1e-6, "o_groups": 2, "o_lora_rank": 16, "sliding_window": 128,
             "compress_ratios": [0, 0, 4, 128, 0, 0], "compress_rope_theta": 160000.0,
             "n_routed_experts": 1, "n_shared_experts": 1, "num_experts_per_tok": 1,
             "scoring_func": "sqrtsoftplus", "routed_scaling_factor": 1.5, "swiglu_limit": \(limit),
             "num_hash_layers": 1, "num_nextn_predict_layers": 1, "norm_topk_prob": true, "hc_mult": 2,
             "hc_sinkhorn_iters": 3, "hc_eps": 1e-6, "rope_theta": 10000.0, "max_position_embeddings": 4096,
             "index_head_dim": 16, "index_n_heads": 4, "index_topk": 4}
            """
        return try JSONDecoder().decode(DeepseekV4Configuration.self, from: Data(json.utf8))
    }

    // MARK: shipping paths (each returns the activation output [rows * d] as Float)

    static func viaCompiledSiluProduct(_ c: Case, layout: String) -> [Float] {
        let dt = dtype(c.dtype), x = input(c, dt: dt)
        return floats(compiledSiluProduct(x[0..., ..<c.d], x[0..., c.d...]))
    }

    /// SwitchGLU default init: the activation every silu-based MoE in the repo uses.
    static func viaSwitchGLUDefault(_ c: Case, layout: String) -> [Float] {
        let dt = dtype(c.dtype), s = selections(layout: layout, d: c.d, dt: dt)
        let m = SwitchGLU(inputDims: 2 * c.d, hiddenDims: c.d, numExperts: 1)
        m.update(parameters: ModuleParameters.unflattened([
            "gate_proj.weight": s.gate.expandedDimensions(axis: 0),
            "up_proj.weight": s.up.expandedDimensions(axis: 0),
            "down_proj.weight": s.down.expandedDimensions(axis: 0),
        ]))
        return floats(runSwitch(m, c, dt)[.ellipsis, ..<c.d])
    }

    static func runSwitch(_ m: SwitchGLU, _ c: Case, _ dt: DType) -> MLXArray {
        let x = input(c, dt: dt).expandedDimensions(axis: 0)           // [1, rows, 2d]
        let idx = MLXArray.zeros([1, c.rows, 1], dtype: .int32)
        return m(x, idx).squeezed(axis: -2)                           // [1, rows, 2d]
    }

    static func viaGPTOSS(_ c: Case, layout: String) -> [Float] {
        let dt = dtype(c.dtype), s = selections(layout: layout, d: c.d, dt: dt)
        let m = SwiGLUSwitchGLU(inputDims: 2 * c.d, hiddenDims: c.d, numExperts: 1)
        m.update(parameters: ModuleParameters.unflattened([
            "gate_proj.weight": s.gate.expandedDimensions(axis: 0),
            "up_proj.weight": s.up.expandedDimensions(axis: 0),
            "down_proj.weight": s.down.expandedDimensions(axis: 0),
        ]))
        let x = input(c, dt: dt).expandedDimensions(axis: 0)
        let idx = MLXArray.zeros([1, c.rows, 1], dtype: .int32)
        return floats(m(x, idx).squeezed(axis: -2)[.ellipsis, ..<c.d])
    }

    static func viaV4Dense(_ c: Case, layout: String, limit: Float) -> [Float] {
        let dt = dtype(c.dtype), s = selections(layout: layout, d: c.d, dt: dt)
        let m = DeepseekV4Expert(hiddenSize: 2 * c.d, intermediateSize: c.d, swiguLimit: limit)
        m.update(parameters: ModuleParameters.unflattened([
            "gate_proj.weight": s.gate, "up_proj.weight": s.up, "down_proj.weight": s.down,
        ]))
        return floats(m(input(c, dt: dt))[.ellipsis, ..<c.d])
    }

    static func viaV4Routed(_ c: Case, layout: String) throws -> [Float] {
        let dt = dtype(c.dtype), s = selections(layout: layout, d: c.d, dt: dt)
        let moe = DeepseekV4MoE(config: try v4Config(d: c.d, limit: c.limit), layerIndex: 1)
        let m = moe.switchMLP
        m.update(parameters: ModuleParameters.unflattened([
            "gate_proj.weight": s.gate.expandedDimensions(axis: 0),
            "up_proj.weight": s.up.expandedDimensions(axis: 0),
            "down_proj.weight": s.down.expandedDimensions(axis: 0),
        ]))
        return floats(runSwitch(m, c, dt)[.ellipsis, ..<c.d])
    }
}

struct ConformanceSwigluGoldenTests {
    typealias S = SwigluGolden

    /// Fixture file -> the shipping paths that must consume every one of its cases.
    static let consumers: [String: [String]] = [
        "silu_and_mul.json": ["compiledSiluProduct", "SwitchGLU(default)", "DeepseekV4Expert(limit 0)"],
        "swigluoai_and_mul.json": ["GPTOSS SwiGLUSwitchGLU"],
        "v4_clamped.json": ["DeepseekV4Expert(limit)", "DeepseekV4MoE.switchMLP"],
        "swiglustep_and_mul.json": ["gap: no Swift Step path"],
    ]

    /// Runs every case of `file` through `run`, returns the failures as readable strings.
    static func check(_ file: String, golden: KeyPath<S.Case, [Double]?>? = nil,
                      mutate: Bool = false, run: (S.Case, String) throws -> [Float]) throws -> (failures: [String], compared: Int, nonzero: Int) {
        let f = try S.load(file), m = try S.manifest()
        var fails: [String] = [], nonzero = 0
        for c in f.cases {
            let out = try run(c, f.layout)
            #expect(out.count == c.golden.count, "\(c.name): output size \(out.count) != \(c.golden.count)")
            var want = golden.flatMap { c[keyPath: $0] } ?? c.golden
            if mutate {   // non-vacuity: a golden perturbed far beyond tolerance must be rejected
                let (rtol, atol) = S.tol(c.dtype, m)
                want[0] = want[0] + 50 * (atol + rtol * abs(want[0])) + 1e-3
            }
            nonzero += out.filter { $0 != 0 }.count
            let bad = S.mismatches(out, want, dtype: c.dtype, m)
            if !bad.isEmpty {
                let i = bad[0]
                fails.append("\(c.name): \(bad.count)/\(want.count) out of tolerance, first #\(i) swift=\(out[i]) golden=\(want[i]) x=\(c.x.count > i ? c.x[i] : 0)")
            }
        }
        return (fails, f.cases.count, nonzero)
    }

    @Test("manifest: every fixture is listed, hashed, and has a consumer")
    func manifestComplete() throws {
        let m = try S.manifest()
        #expect(Set(m.files.keys) == Set(Self.consumers.keys), "manifest files \(m.files.keys.sorted()) vs consumers \(Self.consumers.keys.sorted())")
        let onDisk = try FileManager.default.contentsOfDirectory(atPath: S.dir.path).filter { $0.hasSuffix(".json") && $0 != "MANIFEST.json" }
        #expect(Set(onDisk) == Set(m.files.keys), "fixture files on disk \(onDisk.sorted()) vs manifest")
        for (file, paths) in Self.consumers { #expect(!paths.isEmpty, "\(file) has no consumer") }
        for file in m.files.keys { _ = try S.load(file) }   // verifies sha256 + case counts
    }

    @Test("fixtures cover dtypes, boundary values, negative gates, last-dim 1 and odd sizes")
    func coverage() throws {
        for file in Self.consumers.keys {
            let f = try S.load(file)
            #expect(Set(f.cases.map(\.dtype)) == ["float32", "bfloat16", "float16"], "\(file) dtypes")
            #expect(f.cases.contains { $0.d == 1 } && f.cases.contains { $0.d % 2 == 1 && $0.d > 1 }, "\(file): d == 1 and odd d")
            let sp = f.cases.filter { $0.name.hasSuffix("special") }
            #expect(!sp.isEmpty, "\(file) special cases")
            for c in sp {   // gate varies by row, so the special block must hold values at +-limit, beyond it, and a negative gate
                #expect(c.x.contains(c.limit) || file == "silu_and_mul.json" || c.x.contains(7.0))
                #expect(c.x.contains { $0 < -20 } && c.x.contains { $0 > 100 })
            }
        }
    }

    @Test("silu(gate)*up through compiledSiluProduct")
    func siluCompiledProduct() throws {
        let r = try Self.check("silu_and_mul.json") { c, l in S.viaCompiledSiluProduct(c, layout: l) }
        #expect(r.failures.isEmpty, "\(r.failures.joined(separator: "\n"))")
        #expect(r.nonzero > 0)
    }

    @Test("silu(gate)*up through SwitchGLU default init (all silu MoEs)")
    func siluSwitchGLU() throws {
        let r = try Self.check("silu_and_mul.json") { c, l in S.viaSwitchGLUDefault(c, layout: l) }
        #expect(r.failures.isEmpty, "\(r.failures.joined(separator: "\n"))")
    }

    @Test("silu(gate)*up through DeepseekV4Expert with limit 0 (clamp disabled)")
    func siluV4DenseUnclamped() throws {
        let r = try Self.check("silu_and_mul.json") { c, l in S.viaV4Dense(c, layout: l, limit: 0) }
        #expect(r.failures.isEmpty, "\(r.failures.joined(separator: "\n"))")
    }

    @Test("swigluoai (alpha 1.702, limit 7) through GPT-OSS SwiGLUSwitchGLU")
    func gptossSwigluOAI() throws {
        let r = try Self.check("swigluoai_and_mul.json") { c, l in S.viaGPTOSS(c, layout: l) }
        #expect(r.failures.isEmpty, "\(r.failures.joined(separator: "\n"))")
        #expect(r.nonzero > 0)
    }

    @Test("V4 dense (shared) expert: silu(clamp(gate)) * clamp(up), both two-sided")
    func v4Dense() throws {
        let r = try Self.check("v4_clamped.json") { c, l in S.viaV4Dense(c, layout: l, limit: Float(c.limit)) }
        #expect(r.failures.isEmpty, "\(r.failures.joined(separator: "\n"))")
        #expect(r.nonzero > 0)
    }

    @Test("V4 routed experts (DeepseekV4MoE.switchMLP): up must be clamped too")
    func v4Routed() throws {
        let r = try Self.check("v4_clamped.json") { c, l in try S.viaV4Routed(c, layout: l) }
        #expect(r.failures.isEmpty, "\(r.failures.joined(separator: "\n"))")
    }

    @Test("V4 fixture discriminates: the two gate-clamp readings differ, and the dense expert matches the symmetric one")
    func v4SidednessIsDiscriminating() throws {
        let f = try S.load("v4_clamped.json")
        var differing = 0
        for c in f.cases {
            let alt = try #require(c.goldenGateMaxOnly)
            differing += zip(c.golden, alt).filter { $0 != $1 }.count
        }
        #expect(differing > 0, "negative-gate cases must separate clamp(gate, -L, L) from clamp(gate, max: L)")
        let r = try Self.check("v4_clamped.json", golden: \.goldenGateMaxOnly) { c, l in S.viaV4Dense(c, layout: l, limit: Float(c.limit)) }
        #expect(!r.failures.isEmpty, "the dense expert is two-sided; the max-only golden must be rejected (else the fixture cannot tell them apart)")
    }

    @Test("non-vacuity: a perturbed golden is rejected on every path")
    func perturbedGoldenFails() throws {
        let runs: [(String, (S.Case, String) throws -> [Float])] = [
            ("silu_and_mul.json", { c, l in S.viaCompiledSiluProduct(c, layout: l) }),
            ("swigluoai_and_mul.json", { c, l in S.viaGPTOSS(c, layout: l) }),
            ("v4_clamped.json", { c, l in S.viaV4Dense(c, layout: l, limit: Float(c.limit)) }),
        ]
        for (file, run) in runs {
            let ok = try Self.check(file, run: run)
            let mutated = try Self.check(file, mutate: true, run: run)
            #expect(ok.failures.isEmpty, "\(file) baseline must pass: \(ok.failures.prefix(3))")
            #expect(mutated.failures.count == mutated.compared, "\(file): perturbed golden accepted in \(mutated.compared - mutated.failures.count)/\(mutated.compared) cases")
        }
    }

    @Test("every case of every consumed fixture is compared (no silent skipping)")
    func everyCaseCompared() throws {
        for (file, run) in [("silu_and_mul.json", { (c: S.Case, l: String) in S.viaCompiledSiluProduct(c, layout: l) }),
                            ("swigluoai_and_mul.json", { (c: S.Case, l: String) in S.viaGPTOSS(c, layout: l) })] {
            let r = try Self.check(file, run: run)
            #expect(r.compared == (try S.manifest().files[file]?.cases ?? -1), "\(file)")
        }
    }

    @Test("Step silu-then-clamp has no shipping Swift path (gap)")
    func stepStyleHasNoPath() async throws {
        let f = try S.load("swiglustep_and_mul.json")   // fixture is loaded and hash-checked; there is nothing to feed it to
        #expect(f.cases.count == 30)
        let registered = await LLMTypeRegistry.shared.contains("step3p5")
        ConformanceGap.knownIssue("no Swift implementation of the Step-3.5 clamped SwiGLU: clamp(silu(gate), max: limit) * clamp(up, -limit, limit); model_type step3p5 is not registered") {
            ConformanceGap.expect(registered, "step3p5 must be registered with a Step activation tested against swiglustep_and_mul.json")
        }
    }
}
