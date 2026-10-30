// Conformance gap pins, round 3: the GDN / KDA gaps in the CURRENT code (everything else is in
// ConformanceGapGdnReferenceTests.swift). Each gap is wrapped by `ConformanceGap.knownIssue`, so the suite
// is green now AND after the fix; `CONFORMANCE_GAP_RAW=1` runs the bare assertions to show the red state.
// Delete a wrapper in the PR that fixes its gap. upstream-watch: Libraries/MLXLMCommon/GatedDelta.swift
// (a `g` of rank 4 in the kernel dispatch), Libraries/MLXLLM/Models/Qwen35.swift (`output_gate_type`).

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

struct ConformanceGapGdnTests {
    typealias G = ConformanceGdn
    typealias Ref = SharpAIOverlayGdnReference

    @Test("the per-key-gate (KDA) kernel SharpAIOverlayGatedDeltaKDA equals the reference (was a known gap: the scalar kernel misread rank-4 g)")
    func kernelPerKeyGate() {
        let c = G.makeCase(T: 8, Hk: 2, Hv: 4, Dk: 32, Dv: 16, perKey: true, seed: 71)
        let want = Ref.recurrent(q: c.q, k: c.k, v: c.v, logDecay: c.logDecay, beta: c.beta)
        // Precondition (outside the known issue): the case is discriminating -- the slow ops path IS right,
        // and a per-key gate really differs from a scalar one on this data.
        let ops = gatedDeltaOps(q: c.q, k: c.k, v: c.v, g: exp(c.logDecay), beta: c.beta)
        eval(ops.0)
        #expect(G.maxDiff(ops.0, want.y) < 1e-4, "ops path must be the working reference for this case")
        let scalar = Ref.recurrent(
            q: c.q, k: c.k, v: c.v, logDecay: c.logDecay.mean(axis: -1, keepDims: true), beta: c.beta)
        #expect(G.maxDiff(scalar.y, want.y) > 1e-2, "per-key and mean-scalar gate must give different outputs")

        let zero = MLXArray.zeros([1, 4, 16, 32])
        let (y, state) = SharpAIOverlayGatedDeltaKDA.update(
            q: c.q, k: c.k, v: c.v, g: exp(c.logDecay), beta: c.beta, state: zero)
        eval(y, state)
        let dy = G.maxDiff(y, want.y), ds = G.maxDiff(state, want.state)
        #expect(dy < 1e-3 && ds < 1e-3, "kernel vs per-key reference: |dy|=\(dy) |dstate|=\(ds)")
    }

    @Test("gap: Qwen3.5 GDN config does not carry Qwen4Exp's output_gate_type (silu | sigmoid); the norm is hard-wired to silu")
    func outputGateTypeNotDecoded() throws {
        let json = Data(#"{"model_type":"qwen3_5_text","output_gate_type":"sigmoid"}"#.utf8)
        let cfg = try JSONDecoder().decode(Qwen35TextConfiguration.self, from: json)
        let enc = try #require(
            try JSONSerialization.jsonObject(with: JSONEncoder().encode(cfg)) as? [String: Any])
        // Precondition: the round trip is a valid probe -- a key the struct does carry survives it.
        #expect(enc["model_type"] as? String == "qwen3_5_text")
        ConformanceGap.knownIssue("Qwen35TextConfiguration ignores output_gate_type; Qwen3NextRMSNormGated always uses silu") {
            ConformanceGap.expect(enc["output_gate_type"] as? String == "sigmoid", "output_gate_type survives decode")
        }
    }
}
