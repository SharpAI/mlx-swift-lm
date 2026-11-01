// SharpAI fork overlay tests for the cosmos3_omni generation tower and joint attention. Not part of upstream.
//
// Air-gapped: tiny synthetic config, deterministic fills (no MLXRandom), no downloads.

import Foundation
import MLX
import MLXNN
import XCTest

@testable import MLXVLM

final class SharpAICosmos3OmniGenTowerTests: XCTestCase {

    private let hidden = 64
    private let heads = 4
    private let kvHeads = 2
    private let headDim = 16
    private let layersCount = 2
    private let batch = 2
    private let undLength = 5
    private let genLength = 3

    private var towerConfig: SharpAICosmos3OmniGenTowerConfiguration {
        SharpAICosmos3OmniGenTowerConfiguration(
            hiddenSize: hidden, intermediateSize: 96, numHiddenLayers: layersCount,
            numAttentionHeads: heads, numKeyValueHeads: kvHeads, headDim: headDim)
    }

    private func tinyQwen() throws -> Qwen3VL {
        let json = """
            {
                "model_type": "qwen3_vl",
                "text_config": {
                    "model_type": "qwen3", "hidden_size": \(hidden), "intermediate_size": 96,
                    "num_hidden_layers": \(layersCount), "num_attention_heads": \(heads),
                    "num_key_value_heads": \(kvHeads), "head_dim": \(headDim),
                    "rope_theta": 5000000.0, "max_position_embeddings": 512,
                    "rms_norm_eps": 1e-6, "tie_word_embeddings": true, "vocab_size": 128
                },
                "vision_config": {
                    "model_type": "qwen3_vl_vision", "depth": 1, "hidden_size": 32,
                    "intermediate_size": 64, "out_hidden_size": \(hidden), "num_heads": 4,
                    "patch_size": 16, "spatial_merge_size": 2, "temporal_patch_size": 1,
                    "num_position_embeddings": 16, "in_channels": 3
                }
            }
            """
        let config = try JSONDecoder().decode(Qwen3VLConfiguration.self, from: Data(json.utf8))
        return Qwen3VL(config)
    }

    /// Deterministic parameters: norms near 1, matrices small and varied per parameter.
    private func fill(_ module: Module, seed: Float) {
        var flat = [(String, MLXArray)]()
        for (index, entry) in module.parameters().flattened().enumerated() {
            let ramp = MLXArray(0 ..< entry.1.size).asType(.float32)
            let wave = sin(ramp * MLXArray(Float(0.37)) + MLXArray(seed + Float(index)))
                .reshaped(entry.1.shape)
            if entry.1.ndim == 1 {
                flat.append((entry.0, MLXArray(Float(1)) + wave * MLXArray(Float(0.1))))
            } else {
                flat.append((entry.0, wave * MLXArray(Float(0.15))))
            }
        }
        module.update(parameters: ModuleParameters.unflattened(flat))
    }

    private func sequence(_ length: Int, seed: Float) -> MLXArray {
        let n = batch * length * hidden
        return sin(MLXArray(0 ..< n).asType(.float32) * MLXArray(Float(0.11)) + MLXArray(seed))
            .reshaped(batch, length, hidden)
    }

    /// 3 x B x N ids: temporal starts at `offset`, height and width are distinct ramps.
    private func positions(_ length: Int, offset: Int32) -> MLXArray {
        var values = [Int32]()
        for axis in 0 ..< 3 {
            for _ in 0 ..< batch {
                for i in 0 ..< length {
                    values.append(offset + Int32(i * (axis + 1)))
                }
            }
        }
        return MLXArray(values).reshaped(3, batch, length)
    }

    private struct Fixture {
        let qwen: Qwen3VL
        let tower: SharpAICosmos3OmniGenTower
        let undLayers: [Qwen3VLLanguage.DecoderLayer]
    }

    private func makeFixture() throws -> Fixture {
        let qwen = try tinyQwen()
        let layers = SharpAICosmos3OmniJointAttention.understandingLayers(of: qwen)
        for (i, layer) in layers.enumerated() { fill(layer, seed: Float(i) * 3) }
        let tower = SharpAICosmos3OmniGenTower(towerConfig)
        fill(tower, seed: 50)
        return Fixture(qwen: qwen, tower: tower, undLayers: layers)
    }

    // MARK: structure

    func testParameterPathsMatchKeyContract() {
        let tower = SharpAICosmos3OmniGenTower(towerConfig)
        let keys = Set(tower.parameters().flattened().map { $0.0 })
        var expected: Set<String> = ["norm.weight"]
        for n in 0 ..< layersCount {
            for name in ["q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm"] {
                expected.insert("layers.\(n).self_attn.\(name).weight")
            }
            for name in ["gate_proj", "up_proj", "down_proj"] {
                expected.insert("layers.\(n).mlp.\(name).weight")
            }
            expected.insert("layers.\(n).input_layernorm.weight")
            expected.insert("layers.\(n).post_attention_layernorm.weight")
        }
        XCTAssertEqual(keys, expected)
    }

    func testFindsUnderstandingLayersInOrder() throws {
        let qwen = try tinyQwen()
        let layers = SharpAICosmos3OmniJointAttention.understandingLayers(of: qwen)
        XCTAssertEqual(layers.count, layersCount)
        let params = Dictionary(
            uniqueKeysWithValues: qwen.parameters().flattened().map { ($0.0, $0.1) })
        for (i, layer) in layers.enumerated() {
            let reference = params["language_model.model.layers.\(i).self_attn.q_proj.weight"]!
            XCTAssertTrue(allClose(layer.attention.wq.weight, reference).item(Bool.self))
        }
    }

    // MARK: attention against a naive reference

    /// Softmax attention with explicit matmul; `causal` masks j > i.
    private func naiveAttention(
        q: MLXArray, k: MLXArray, v: MLXArray, causal: Bool
    ) -> MLXArray {
        let groups = q.dim(1) / k.dim(1)
        let kFull = repeated(k, count: groups, axis: 1)
        let vFull = repeated(v, count: groups, axis: 1)
        var scores =
            matmul(q, kFull.transposed(0, 1, 3, 2)) * MLXArray(Float(1) / Float(headDim).squareRoot())
        if causal {
            var mask = [Float]()
            for i in 0 ..< q.dim(2) {
                for j in 0 ..< k.dim(2) { mask.append(j > i ? -1e9 : 0) }
            }
            scores = scores + MLXArray(mask).reshaped(1, 1, q.dim(2), k.dim(2))
        }
        return matmul(softmax(scores, axis: -1), vFull)
    }

    private func rotateHalf(_ x: MLXArray) -> MLXArray {
        let half = x.dim(-1) / 2
        return concatenated([-x[.ellipsis, half...], x[.ellipsis, 0 ..< half]], axis: -1)
    }

    func testAttentionMatchesNaiveReference() throws {
        let fx = try makeFixture()
        let undNormed = sequence(undLength, seed: 1)
        let genNormed = sequence(genLength, seed: 2)
        let undPos = positions(undLength, offset: 0)
        let genPos = positions(genLength, offset: 100)

        let layer = fx.undLayers[0]
        let genAttention = fx.tower.layers[0].attention
        let actual = SharpAICosmos3OmniJointAttention.attend(
            und: layer.attention, gen: genAttention, undNormed: undNormed, genNormed: genNormed,
            undPositions: undPos, genPositions: genPos)

        // Reference, built from the same weights.
        let a = layer.attention
        func splitHeads(_ x: MLXArray, _ count: Int) -> MLXArray {
            x.reshaped(batch, x.dim(1), count, headDim)
        }
        let (cosU, sinU) = a.rotaryEmbedding(positionIds: undPos, dtype: .float32)
        let (cosG, sinG) = a.rotaryEmbedding(positionIds: genPos, dtype: .float32)
        func rope(_ x: MLXArray, _ c: MLXArray, _ s: MLXArray) -> MLXArray {
            x * expandedDimensions(c, axis: 1) + rotateHalf(x) * expandedDimensions(s, axis: 1)
        }
        let qU = rope(
            a.qNorm(splitHeads(a.wq(undNormed), self.heads)).transposed(0, 2, 1, 3), cosU, sinU)
        let kU = rope(
            a.kNorm(splitHeads(a.wk(undNormed), kvHeads)).transposed(0, 2, 1, 3), cosU, sinU)
        let vU = splitHeads(a.wv(undNormed), kvHeads).transposed(0, 2, 1, 3)
        let g = genAttention
        let qG = rope(
            g.qNorm(splitHeads(g.wq(genNormed), self.heads)).transposed(0, 2, 1, 3), cosG, sinG)
        let kG = rope(
            g.kNorm(splitHeads(g.wk(genNormed), kvHeads)).transposed(0, 2, 1, 3), cosG, sinG)
        let vG = splitHeads(g.wv(genNormed), kvHeads).transposed(0, 2, 1, 3)

        let undRef = a.wo(
            naiveAttention(q: qU, k: kU, v: vU, causal: true)
                .transposed(0, 2, 1, 3).reshaped(batch, undLength, -1))
        let genRef = g.wo(
            naiveAttention(
                q: qG, k: concatenated([kU, kG], axis: 2), v: concatenated([vU, vG], axis: 2),
                causal: false
            ).transposed(0, 2, 1, 3).reshaped(batch, genLength, -1))

        XCTAssertEqual(actual.und.shape, [batch, undLength, hidden])
        XCTAssertEqual(actual.gen.shape, [batch, genLength, hidden])
        XCTAssertTrue(allClose(actual.und, undRef, rtol: 1e-3, atol: 1e-4).item(Bool.self))
        XCTAssertTrue(allClose(actual.gen, genRef, rtol: 1e-3, atol: 1e-4).item(Bool.self))
    }

    // MARK: whole stack

    func testJointForwardShapesAndFiniteness() throws {
        let fx = try makeFixture()
        let result = fx.tower.jointForward(
            understanding: fx.qwen, undSeq: sequence(undLength, seed: 1),
            genSeq: sequence(genLength, seed: 2),
            undPositions: positions(undLength, offset: 0),
            genPositions: positions(genLength, offset: 100))
        XCTAssertEqual(result.und.shape, [batch, undLength, hidden])
        XCTAssertEqual(result.gen.shape, [batch, genLength, hidden])
        for array in [result.und, result.gen] {
            // NaN and infinity both propagate into the sum.
            XCTAssertTrue(array.sum().item(Float.self).isFinite)
        }
    }

    func testUnderstandingIsIndependentOfGeneration() throws {
        let fx = try makeFixture()
        let undSeq = sequence(undLength, seed: 1)
        let undPos = positions(undLength, offset: 0)
        let genPos = positions(genLength, offset: 100)
        let a = fx.tower.jointForward(
            undLayers: fx.undLayers, undSeq: undSeq, genSeq: sequence(genLength, seed: 2),
            undPositions: undPos, genPositions: genPos)
        let b = fx.tower.jointForward(
            undLayers: fx.undLayers, undSeq: undSeq, genSeq: sequence(genLength, seed: 9) * MLXArray(Float(3)),
            undPositions: undPos, genPositions: genPos)
        XCTAssertTrue(allClose(a.und, b.und, rtol: 1e-5, atol: 1e-6).item(Bool.self))
        XCTAssertFalse(allClose(a.gen, b.gen, rtol: 1e-3, atol: 1e-4).item(Bool.self))
    }

    func testGenerationDependsOnUnderstanding() throws {
        let fx = try makeFixture()
        let genSeq = sequence(genLength, seed: 2)
        let undPos = positions(undLength, offset: 0)
        let genPos = positions(genLength, offset: 100)
        let a = fx.tower.jointForward(
            undLayers: fx.undLayers, undSeq: sequence(undLength, seed: 1), genSeq: genSeq,
            undPositions: undPos, genPositions: genPos)
        let b = fx.tower.jointForward(
            undLayers: fx.undLayers, undSeq: sequence(undLength, seed: 7), genSeq: genSeq,
            undPositions: undPos, genPositions: genPos)
        XCTAssertFalse(allClose(a.gen, b.gen, rtol: 1e-3, atol: 1e-4).item(Bool.self))
    }

    func testUnderstandingIsCausal() throws {
        // Changing the LAST understanding token must not change earlier understanding outputs.
        let fx = try makeFixture()
        let genSeq = sequence(genLength, seed: 2)
        let undPos = positions(undLength, offset: 0)
        let genPos = positions(genLength, offset: 100)
        let base = sequence(undLength, seed: 1)
        // Out-of-place perturbation (MLXArray subscript assignment can alias the source).
        var lastOnly = [Float](repeating: 0, count: undLength)
        lastOnly[undLength - 1] = 5
        let perturbed = base + MLXArray(lastOnly).reshaped(1, undLength, 1)
        let a = fx.tower.jointForward(
            undLayers: fx.undLayers, undSeq: base, genSeq: genSeq,
            undPositions: undPos, genPositions: genPos)
        let b = fx.tower.jointForward(
            undLayers: fx.undLayers, undSeq: perturbed, genSeq: genSeq,
            undPositions: undPos, genPositions: genPos)
        XCTAssertTrue(
            allClose(
                a.und[0..., 0 ..< (undLength - 1), 0...], b.und[0..., 0 ..< (undLength - 1), 0...],
                rtol: 1e-5, atol: 1e-6
            ).item(Bool.self))
    }
}
