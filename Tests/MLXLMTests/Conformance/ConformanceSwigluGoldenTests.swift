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
//
// Limits of this suite (read before trusting a green run):
//  * The interleaved gate/up layout of GPT-OSS is injected by the selection matrices below. GPTOSS.swift
//    sanitize (stride(by: 2) split of gate_up_proj) is NOT exercised: a gate/up swap there would not be noticed.
//  * Tolerances are constants in this file (SwigluGolden.tolerances). MANIFEST.json repeats them and the test
//    manifestComplete fails if the two differ, so loosening one without the other is caught.
//  * The V4 oracle is INFERRED (vLLM 739e5945d has no DeepSeek-V4 model; no V4 reference code is in this repo
//    or in 3rd_party). v4Dense/v4Routed therefore characterize the shipping code against an inferred formula;
//    only the region where gate >= -limit and |up| <= limit is independent of the unknown clamp sidedness
//    (v4RoutedGateOnly).
//  * GPT-OSS swiglu runs in float32 for bfloat16/float16 input (see gptossDTypePromotion), so it is held to the
//    float32 tolerance in every dtype; the bfloat16/float16 rows of the tolerance table are not used for it.

import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

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
        struct Generator: Decodable { let scripts: [String: String] }
        let files: [String: Entry]
        let tolerances: Tols
        let generator: Generator
    }

    /// The tolerances the suite enforces. These are the authority; MANIFEST.json only documents them (manifestComplete
    /// fails if they differ). |swift - golden| <= atol + (rtol + extra) * |golden|.
    /// atol 1e-30 (float32, bfloat16) is far below any golden value that matters: the earlier 1e-20 made the check
    /// vacuous for the ~78 swigluoai golden values below 5e-15. float16 keeps 1e-7 (fp16 subnormal step 6e-8).
    static let tolerances: [String: (rtol: Double, atol: Double)] = [
        "float32": (2e-6, 1e-30),
        "bfloat16": (2e-2, 1e-30),
        "float16": (2e-3, 1e-7),
    ]

    /// Repository root, derived from this file: Tests/MLXLMTests/Conformance/<this file>.
    static let repoRoot = GoldenKit.repoRoot
    static let scriptsDir = repoRoot.appendingPathComponent("tools/conformance/golden/swiglu")

    static func sha256(_ data: Data) -> String { GoldenKit.sha256(data) }

    /// SWIGLU_GOLDEN_DIR points the suite at another directory (to show it red against a perturbed copy).
    static let dir = GoldenKit.directory("Golden/swiglu", overrideEnv: "SWIGLU_GOLDEN_DIR")

    static func manifest() throws -> Manifest {
        try GoldenKit.decode(Manifest.self, from: GoldenKit.read(dir.appendingPathComponent("MANIFEST.json")), name: "MANIFEST.json")
    }

    /// Loads a fixture after verifying its sha256 against MANIFEST.json.
    static func load(_ name: String) throws -> File {
        let data = try GoldenKit.read(dir.appendingPathComponent(name))
        let m = try manifest()
        let want = try #require(m.files[name], "\(name) is not in MANIFEST.json")
        let got = sha256(data)
        #expect(got == want.sha256, "\(name): sha256 \(got) != manifest \(want.sha256)")
        let f = try GoldenKit.decode(File.self, from: data, name: name)
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

    static func tol(_ s: String) -> (rtol: Double, atol: Double) { tolerances[s]! }

    /// gate / up of output element `i` (row-major [rows, d]) read out of the fixture input `x` ([rows, 2d]).
    static func gateUp(_ c: Case, layout: String, _ i: Int) -> (gate: Double, up: Double) {
        let row = i / c.d, col = i % c.d, base = row * 2 * c.d
        return layout == "chunk" ? (c.x[base + col], c.x[base + c.d + col]) : (c.x[base + 2 * col], c.x[base + 2 * col + 1])
    }

    /// Extra relative budget for the swigluoai sigmoid tail (INFERRED, standard conditioning argument, not measured):
    /// sigmoid(a) ~ exp(a) for very negative a = alpha*gate, and exp has relative condition number |a|. The argument is
    /// perturbed by Float(1.702) != 1.702 (<= 2^-24 relative) and by the rounding of the float32 product (<= 2^-24
    /// relative), so the result moves by about |a| * 2 * 2^-24 relative. Zero for |gate| small, so the base rtol rules there.
    static func oaiTailRtol(gate: Double) -> Double { abs(1.702 * gate) * Double(2) / Double(1 << 24) }

    struct Bad { let c: Case; let index: Int; let out: Float; let golden: Double; let gate: Double; let up: Double }

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
        floats(gptossRaw(c, layout: layout)[.ellipsis, ..<c.d])
    }

    /// GPT-OSS SwiGLUSwitchGLU output [1, rows, 2d] with its own dtype (not cast), so dtype promotion is observable.
    static func gptossRaw(_ c: Case, layout: String) -> MLXArray {
        let dt = dtype(c.dtype), s = selections(layout: layout, d: c.d, dt: dt)
        let m = SwiGLUSwitchGLU(inputDims: 2 * c.d, hiddenDims: c.d, numExperts: 1)
        m.update(parameters: ModuleParameters.unflattened([
            "gate_proj.weight": s.gate.expandedDimensions(axis: 0),
            "up_proj.weight": s.up.expandedDimensions(axis: 0),
            "down_proj.weight": s.down.expandedDimensions(axis: 0),
        ]))
        let x = input(c, dt: dt).expandedDimensions(axis: 0)
        let idx = MLXArray.zeros([1, c.rows, 1], dtype: .int32)
        return m(x, idx).squeezed(axis: -2)
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

    typealias Result = (failures: [String], compared: Int, nonzero: Int, bad: [S.Bad])

    /// Runs every case of `file` through `run`, returns the failures as readable strings plus every out-of-tolerance element.
    /// `tolDType` overrides which dtype's tolerance row applies; `include` restricts which elements are compared;
    /// `extraRtol` widens the relative tolerance per element (see `SwigluGolden.oaiTailRtol`).
    static func check(_ file: String, label: String = "", golden: KeyPath<S.Case, [Double]?>? = nil,
                      mutate: Bool = false, only: (S.Case) -> Bool = { _ in true },
                      tolDType: String? = nil,
                      include: (S.Case, String, Int) -> Bool = { _, _, _ in true },
                      extraRtol: (S.Case, String, Int) -> Double = { _, _, _ in 0 },
                      run: (S.Case, String) throws -> [Float]) throws -> Result {
        let f = try S.load(file)
        var fails: [String] = [], nonzero = 0, compared = 0
        var allBad: [S.Bad] = []
        var worst: [String: (abs: Double, rel: Double)] = [:]   // per dtype, for the log
        for c in f.cases where only(c) {
            compared += 1
            let out = try run(c, f.layout)
            #expect(out.count == c.golden.count, "\(c.name): output size \(out.count) != \(c.golden.count)")
            var want = golden.flatMap { c[keyPath: $0] } ?? c.golden
            let (rtol, atol) = S.tol(tolDType ?? c.dtype)
            if mutate {   // non-vacuity: a golden perturbed far beyond tolerance must be rejected
                want[0] = want[0] + 50 * (atol + rtol * abs(want[0])) + 1e-3
            }
            nonzero += out.filter { $0 != 0 }.count
            var bad: [Int] = []
            for (i, g) in want.enumerated() where include(c, f.layout, i) {
                let o = Double(out[i])
                if !mutate && o.isFinite {
                    var w = worst[c.dtype] ?? (0, 0)
                    let a = abs(o - g)
                    w.abs = max(w.abs, a)
                    if abs(g) > 1e-30 { w.rel = max(w.rel, a / abs(g)) }
                    worst[c.dtype] = w
                }
                if !o.isFinite || abs(o - g) > atol + (rtol + extraRtol(c, f.layout, i)) * abs(g) {
                    bad.append(i)
                    let gu = S.gateUp(c, layout: f.layout, i)
                    allBad.append(S.Bad(c: c, index: i, out: out[i], golden: g, gate: gu.gate, up: gu.up))
                }
            }
            if let i = bad.first {
                let gu = S.gateUp(c, layout: f.layout, i)
                fails.append("\(c.name): \(bad.count)/\(want.count) out of tolerance, first #\(i) swift=\(out[i]) golden=\(want[i]) gate=\(gu.gate) up=\(gu.up)")
            }
        }
        if !mutate {
            for (dt, w) in worst.sorted(by: { $0.key < $1.key }) {
                print("[swiglu][stats] \(file) \(label) \(dt) max|abs err|=\(w.abs) max rel err=\(w.rel)")
            }
        }
        return (fails, compared, nonzero, allBad)
    }

    /// Extra budget used by every GPT-OSS comparison.
    static let oaiExtra: @Sendable (S.Case, String, Int) -> Double = { c, l, i in S.oaiTailRtol(gate: S.gateUp(c, layout: l, i).gate) }

    @Test("manifest: every fixture is listed, hashed, has a consumer; tolerances and generator scripts match")
    func manifestComplete() throws {
        let m = try S.manifest()
        #expect(Set(m.files.keys) == Set(Self.consumers.keys), "manifest files \(m.files.keys.sorted()) vs consumers \(Self.consumers.keys.sorted())")
        let onDisk = try FileManager.default.contentsOfDirectory(atPath: S.dir.path).filter { $0.hasSuffix(".json") && $0 != "MANIFEST.json" }
        #expect(Set(onDisk) == Set(m.files.keys), "fixture files on disk \(onDisk.sorted()) vs manifest")
        for (file, paths) in Self.consumers { #expect(!paths.isEmpty, "\(file) has no consumer") }
        for file in m.files.keys { _ = try S.load(file) }   // verifies sha256 + case counts
        // The manifest only documents the tolerances; the constants in S.tolerances are what the tests enforce.
        for (name, t) in [("float32", m.tolerances.float32), ("bfloat16", m.tolerances.bfloat16), ("float16", m.tolerances.float16)] {
            #expect(t.rtol == S.tol(name).rtol && t.atol == S.tol(name).atol, "\(name): MANIFEST.json \(t) differs from the enforced \(S.tol(name))")
        }
        // The generator scripts that produced the fixtures are the ones in the repo.
        #expect(Set(m.generator.scripts.keys) == ["generate.py", "swiglu_ref.py"])
        for (script, want) in m.generator.scripts {
            let data = try Data(contentsOf: S.scriptsDir.appendingPathComponent(script))
            #expect(S.sha256(data) == want, "tools/conformance/golden/swiglu/\(script) changed since the fixtures were generated; regenerate")
        }
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
                #expect(c.x.contains(c.limit) && c.x.contains(-c.limit))
                #expect(c.x.contains { $0 < -20 } && c.x.contains { $0 > 100 })
            }
        }
    }

    /// Cases a path is held to without a gap wrapper. fp16 silu(gate)*up is a pinned gap (see `siluFloat16Gap`).
    static let notFloat16: @Sendable (S.Case) -> Bool = { $0.dtype != "float16" }

    @Test("silu(gate)*up through compiledSiluProduct (float32, bfloat16)")
    func siluCompiledProduct() throws {
        let r = try Self.check("silu_and_mul.json", label: "compiledSiluProduct", only: Self.notFloat16) { c, l in S.viaCompiledSiluProduct(c, layout: l) }
        #expect(r.failures.isEmpty, "\(r.failures.joined(separator: "\n"))")
        #expect(r.nonzero > 0)
    }

    @Test("silu(gate)*up through SwitchGLU default init (float32, bfloat16)")
    func siluSwitchGLU() throws {
        let r = try Self.check("silu_and_mul.json", label: "SwitchGLU", only: Self.notFloat16) { c, l in S.viaSwitchGLUDefault(c, layout: l) }
        #expect(r.failures.isEmpty, "\(r.failures.joined(separator: "\n"))")
    }

    @Test("silu(gate)*up through DeepseekV4Expert with limit 0 (float32, bfloat16)")
    func siluV4DenseUnclamped() throws {
        let r = try Self.check("silu_and_mul.json", label: "V4Expert", only: Self.notFloat16) { c, l in S.viaV4Dense(c, layout: l, limit: 0) }
        #expect(r.failures.isEmpty, "\(r.failures.joined(separator: "\n"))")
    }

    /// sigmoid(gate) ~ exp(gate) leaves float16's normal range (min normal 2^-14 = 6.1e-5) at gate = ln(2^-14) = -9.70, and
    /// rounds to zero once it is below half the smallest subnormal (2^-25 = 3e-8) at gate = ln(2^-25) = -17.33.
    static let fp16SigmoidSubnormalGate = -9.70
    static let fp16SigmoidZeroGate = -17.33

    @Test("gap: float16 silu(gate)*up is off for strongly negative gates (all three silu paths)")
    func siluFloat16Gap() throws {
        // Measured (run log, 32 bad elements of the float16 cases, identical in all three paths): every bad element has
        // gate <= -11.9; relative error grows with |gate| (0.4% at -11.9, 3.8% at -14.3, 27% at -16.3, 96% at -17.3) and the output
        // is exactly 0.0 for the 10 gates between -17.9 and -19.3. That fits float16 ROUNDING OF sigmoid(gate): below gate -9.7
        // sigmoid is a float16 subnormal (absolute step 2^-24 = 6e-8, so the relative step grows like e^|gate|) and below -17.33
        // it rounds to zero. It does NOT fit exp overflow (exp(-gate) > 65504 at gate < -11.09 would zero every output from
        // -11.09 on; the outputs at -11.9 .. -17.3 are non-zero). That sigmoid is rounded to float16 inside the fused silu*up is
        // INFERRED from this fit, not read from the MLX kernel. vLLM's CUDA silu computes in fp32, so the oracle's hardware does
        // not show it. The formula is right; the dtype of the intermediate is the issue, and an fp32 intermediate would be the
        // fix (not tried). Absolute errors are ~1e-6 .. 1e-4 (the 0.72 abs error in the stats line is a ~4e4 output, within rtol).
        let only: (S.Case) -> Bool = { $0.dtype == "float16" }
        let a = try Self.check("silu_and_mul.json", label: "compiledSiluProduct fp16", only: only) { c, l in S.viaCompiledSiluProduct(c, layout: l) }
        let b = try Self.check("silu_and_mul.json", label: "SwitchGLU fp16", only: only) { c, l in S.viaSwitchGLUDefault(c, layout: l) }
        let d = try Self.check("silu_and_mul.json", label: "V4Expert fp16", only: only) { c, l in S.viaV4Dense(c, layout: l, limit: 0) }
        // Always-on classification of the out-of-tolerance elements (the evidence behind the text above).
        let zeroed = a.bad.filter { $0.out == 0 }
        print("[swiglu][fp16-silu] bad=\(a.bad.count) zero-output=\(zeroed.count) max gate of bad=\(a.bad.map(\.gate).max() ?? 0) max gate of zero-output=\(zeroed.map(\.gate).max() ?? 0)")
        #expect(!a.bad.isEmpty, "the gap went away: remove the knownIssue")
        #expect(a.bad.allSatisfy { $0.gate < Self.fp16SigmoidSubnormalGate }, "bad element with sigmoid still in float16's normal range: \(a.bad.filter { $0.gate >= Self.fp16SigmoidSubnormalGate }.prefix(3))")
        #expect(zeroed.allSatisfy { $0.gate < Self.fp16SigmoidZeroGate }, "zero output where sigmoid(gate) is representable: \(zeroed.filter { $0.gate >= Self.fp16SigmoidZeroGate }.prefix(3))")
        #expect(a.bad.count == b.bad.count && a.bad.count == d.bad.count, "the three silu paths fail on different elements")
        ConformanceGap.knownIssue("float16 silu(gate)*up misses rtol 2e-3 / atol 1e-7 for strongly negative gates (compiledSiluProduct, SwitchGLU, DeepseekV4Expert limit 0)") {
            ConformanceGap.expect(a.failures.isEmpty, "compiledSiluProduct: \(a.failures.prefix(2))")
            ConformanceGap.expect(b.failures.isEmpty, "SwitchGLU: \(b.failures.prefix(2))")
            ConformanceGap.expect(d.failures.isEmpty, "DeepseekV4Expert: \(d.failures.prefix(2))")
        }
    }

    @Test("swigluoai (alpha 1.702, limit 7) through GPT-OSS SwiGLUSwitchGLU, float32 tolerance in every dtype")
    func gptossSwigluOAI() throws {
        // GPT-OSS computes in float32 whatever the input dtype (gptossDTypePromotion), so all three dtypes are held to the
        // float32 row; the exp tail gets the conditioning term S.oaiTailRtol (INFERRED, see there).
        let r = try Self.check("swigluoai_and_mul.json", label: "GPTOSS", tolDType: "float32", extraRtol: Self.oaiExtra) { c, l in S.viaGPTOSS(c, layout: l) }
        #expect(r.failures.isEmpty, "\(r.failures.joined(separator: "\n"))")
        #expect(r.nonzero > 0)
    }

    @Test("GPT-OSS swiglu keeps the activation dtype (float32 holds; bfloat16/float16 are promoted to float32: gap)")
    func gptossDTypePromotion() throws {
        let f = try S.load("swigluoai_and_mul.json")
        var promoted: [String] = []
        for name in ["float32", "bfloat16", "float16"] {
            let c = try #require(f.cases.first { $0.dtype == name })
            let out = S.gptossRaw(c, layout: f.layout)
            if name == "float32" { #expect(out.dtype == .float32) } else if out.dtype != S.dtype(name) { promoted.append("\(name) -> \(out.dtype)") }
        }
        // swiglu() in GPTOSS.swift clips with MLXArray(limit), a strongly typed float32 array, which promotes the activation.
        // Whether that is intended is not established; mlx-lm's Python port clips with a Python scalar (not verified here).
        ConformanceGap.knownIssue("GPT-OSS swiglu promotes bfloat16/float16 activations to float32 (clip with MLXArray(limit)); output dtype != input dtype") {
            ConformanceGap.expect(promoted.isEmpty, "\(promoted)")
        }
    }

    @Test("V4 dense (shared) expert: shipping silu(clamp(gate)) * clamp(up), both two-sided (oracle INFERRED)")
    func v4Dense() throws {
        // Characterization: the oracle's sidedness is inferred (no V4 reference available), so this pins the shipping
        // behaviour; it does not prove it right.
        let r = try Self.check("v4_clamped.json", label: "V4 dense") { c, l in S.viaV4Dense(c, layout: l, limit: Float(c.limit)) }
        #expect(r.failures.isEmpty, "\(r.failures.joined(separator: "\n"))")
        #expect(r.nonzero > 0)
    }

    /// Elements where the unknown V4 clamp sidedness cannot matter: gate >= -limit (both gate readings agree) and |up| <= limit.
    /// float16 only: the routed path does not clamp up, so a huge up in ANOTHER column of the row makes that hidden value overflow
    /// (|silu(clamp(gate))| <= limit, so limit * |up| > 65504 is a sufficient overflow test); the down projection then computes
    /// 0 * inf = NaN and poisons the whole row. Those rows are symptoms of the up gap (v4Routed), not of the gate path.
    static func rowOverflowsFloat16(_ c: S.Case, _ i: Int) -> Bool {
        guard c.dtype == "float16" else { return false }
        let base = (i / c.d) * 2 * c.d + c.d   // chunk layout: up of column k is x[base + k]
        return (0 ..< c.d).contains { c.limit * abs(c.x[base + $0]) > 65504 }
    }

    static let v4Independent: @Sendable (S.Case, String, Int) -> Bool = { c, l, i in
        let gu = S.gateUp(c, layout: l, i)
        return gu.gate >= -c.limit && abs(gu.up) <= c.limit && !rowOverflowsFloat16(c, i)
    }

    @Test("V4 routed experts: silu(clamp(gate)) * up is exact in the region independent of the inferred oracle")
    func v4RoutedGateOnly() throws {
        let f = try S.load("v4_clamped.json")
        var region = 0, negativeGates = 0, disagree = 0
        for c in f.cases {
            let alt = try #require(c.goldenGateMaxOnly)
            for i in 0 ..< c.golden.count where Self.v4Independent(c, f.layout, i) {
                region += 1
                if S.gateUp(c, layout: f.layout, i).gate < 0 { negativeGates += 1 }
                if c.golden[i] != alt[i] { disagree += 1 }
            }
        }
        #expect(disagree == 0, "the two gate-clamp readings differ inside the region meant to be independent of them")
        #expect(region > 0 && negativeGates > 0, "region \(region), negative gates \(negativeGates)")
        let r = try Self.check("v4_clamped.json", label: "V4 routed gate-only", include: Self.v4Independent) { c, l in try S.viaV4Routed(c, layout: l) }
        #expect(r.failures.isEmpty, "\(r.failures.joined(separator: "\n"))")
    }

    @Test("V4 routed experts (DeepseekV4MoE.switchMLP): up must be clamped too (oracle INFERRED)")
    func v4Routed() throws {
        let r = try Self.check("v4_clamped.json", label: "V4 routed") { c, l in try S.viaV4Routed(c, layout: l) }
        // Precondition (outside the gap): the path ran, produced output, and the failures are not all-zero garbage.
        #expect(r.compared == 90 && r.nonzero > 0)
        // Evidence is internal, not external: DeepseekV4Expert (same file, :566-568) clamps up two-sided and this closure does not.
        // No DeepSeek-V4 reference is available, so treat this gap as provisional.
        ConformanceGap.knownIssue("PROVISIONAL (inferred oracle): DeepseekV4MoE routed experts (SwitchGLU activation closure, DeepseekV4.swift:~676) clamp gate but NOT up, while DeepseekV4Expert clamps both: |up| > swiglu_limit is passed through") {
            ConformanceGap.expect(r.failures.isEmpty, "\(r.failures.prefix(3))")
        }
    }

    @Test("V4 fixture separates the two gate-clamp readings (which one is right is NOT asserted: oracle INFERRED)")
    func v4FixtureSeparatesReadings() throws {
        let f = try S.load("v4_clamped.json")
        var differing = 0
        for c in f.cases {
            let alt = try #require(c.goldenGateMaxOnly)
            differing += zip(c.golden, alt).filter { $0 != $1 }.count
        }
        #expect(differing > 0, "negative-gate cases must separate clamp(gate, -L, L) from clamp(gate, max: L)")
    }

    @Test("non-vacuity: a perturbed golden is rejected on every path")
    func perturbedGoldenFails() throws {
        let runs: [(String, (S.Case) -> Bool, String?, (S.Case, String) throws -> [Float])] = [
            ("silu_and_mul.json", Self.notFloat16, nil, { c, l in S.viaCompiledSiluProduct(c, layout: l) }),
            ("swigluoai_and_mul.json", { _ in true }, "float32", { c, l in S.viaGPTOSS(c, layout: l) }),
            ("v4_clamped.json", { _ in true }, nil, { c, l in S.viaV4Dense(c, layout: l, limit: Float(c.limit)) }),
        ]
        for (file, only, tolD, run) in runs {
            var extra: (S.Case, String, Int) -> Double = { _, _, _ in 0 }
            if file.hasPrefix("swigluoai") { extra = Self.oaiExtra }
            let ok = try Self.check(file, label: "nonvacuity", only: only, tolDType: tolD, extraRtol: extra, run: run)
            let mutated = try Self.check(file, mutate: true, only: only, tolDType: tolD, extraRtol: extra, run: run)
            #expect(ok.failures.isEmpty, "\(file) baseline must pass: \(ok.failures.prefix(3))")
            #expect(mutated.failures.count == mutated.compared, "\(file): perturbed golden accepted in \(mutated.compared - mutated.failures.count)/\(mutated.compared) cases")
        }
    }

    @Test("every case of every consumed fixture is compared (no silent skipping)")
    func everyCaseCompared() throws {
        for (file, run) in [("silu_and_mul.json", { (c: S.Case, l: String) in S.viaCompiledSiluProduct(c, layout: l) }),
                            ("swigluoai_and_mul.json", { (c: S.Case, l: String) in S.viaGPTOSS(c, layout: l) })] {
            let r = try Self.check(file, label: "coverage", run: run)   // failures ignored here: this only counts cases
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
