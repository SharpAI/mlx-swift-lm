//
//  SharpAISarvamMoeMoE.swift
//  MLXLLM
//
//  Dense MLP, sigmoid router with expert bias, and sparse MoE block for sarvam_moe.
//  Selection uses the biased scores in grouped top-k. Routing weights use the
//  unbiased scores, renormalized when top_k > 1 (as in modeling_sarvam_moe.py),
//  then scaled by routed_scaling_factor.
//  Router keys `mlp.gate.weight` and `mlp.gate.expert_bias` and the MLP projections
//  (no bias) were checked against sarvamai/sarvam-30b's index and modeling source.
//

import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// Dense SwiGLU MLP. modeling_sarvam_moe.py builds every projection with `bias=False`,
/// regardless of `use_bias`.
class SharpAISarvamMoeMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "up_proj") var up: Linear
    @ModuleInfo(key: "down_proj") var down: Linear

    init(hiddenSize: Int, intermediateSize: Int) {
        _gate.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        _up.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        _down.wrappedValue = Linear(intermediateSize, hiddenSize, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        down(silu(gate(x)) * up(x))
    }
}

class SharpAISarvamMoeGate: Module {
    let topK: Int
    let nGroup: Int
    let topkGroup: Int
    let numExperts: Int
    let routedScalingFactor: Float
    let normTopkProb: Bool

    var weight: MLXArray
    var expert_bias: MLXArray

    init(_ config: SharpAISarvamMoeConfiguration) {
        self.topK = config.numExpertsPerTok
        self.nGroup = config.nGroup
        self.topkGroup = config.topkGroup
        self.numExperts = config.numExperts
        self.routedScalingFactor = config.routedScalingFactor
        self.normTopkProb = config.normTopkProb
        self.weight = zeros([config.numExperts, config.hiddenSize])
        self.expert_bias = zeros([config.numExperts])
    }

    /// Returns (expert indices [B, L, topK], routing weights [B, L, topK]).
    func callAsFunction(_ x: MLXArray) -> (MLXArray, MLXArray) {
        let (bsz, seqLen) = (x.dim(0), x.dim(1))
        let groupSize = numExperts / nGroup

        let scores = sigmoid(x.matmul(weight.T).asType(.float32))
        let biased = scores + expert_bias

        let groupScores = biased.reshaped(bsz, seqLen, nGroup, groupSize)
        let groupTop2 = top(groupScores, k: 2, axis: -1).sum(axis: -1)
        let selectedGroups = argPartition(-groupTop2, kth: topkGroup - 1, axis: -1)[
            .ellipsis, ..<topkGroup]
        let groupMask = putAlong(
            MLXArray.zeros(like: groupTop2), selectedGroups, values: MLXArray(1.0), axis: -1)
        let expertKeep = broadcast(
            groupMask.expandedDimensions(axis: -1),
            to: [bsz, seqLen, nGroup, groupSize]
        ).reshaped(bsz, seqLen, numExperts)
        let masked = MLX.where(
            expertKeep.asType(.bool), biased, MLXArray(-Float.infinity))

        let inds = argPartition(-masked, kth: topK - 1, axis: -1)[.ellipsis, ..<topK]
        var weights = takeAlong(scores, inds, axis: -1)
        // modeling_sarvam_moe.py renormalizes only when top_k > 1; a single expert keeps
        // its raw sigmoid score.
        if normTopkProb && topK > 1 {
            weights = weights / (weights.sum(axis: -1, keepDims: true) + 1e-20)
        }
        weights = weights * routedScalingFactor
        return (inds, weights.asType(x.dtype))
    }
}

class SharpAISarvamMoeSparseMoE: Module, UnaryLayer {
    @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU
    var gate: SharpAISarvamMoeGate
    @ModuleInfo(key: "shared_experts") var sharedExperts: SharpAISarvamMoeMLP?

    init(_ config: SharpAISarvamMoeConfiguration) {
        _switchMLP.wrappedValue = SwitchGLU(
            inputDims: config.hiddenSize,
            hiddenDims: config.moeIntermediateSize,
            numExperts: config.numExperts,
            bias: false
        )
        self.gate = SharpAISarvamMoeGate(config)

        if config.numSharedExperts > 0 {
            _sharedExperts.wrappedValue = SharpAISarvamMoeMLP(
                hiddenSize: config.hiddenSize,
                intermediateSize: config.moeSharedExpertIntermediateSize * config.numSharedExperts
            )
        } else {
            _sharedExperts.wrappedValue = nil
        }
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (indices, weights) = gate(x)
        var y = switchMLP(x, indices)
        y = weightedExpertSum(y, weights)
        if let sharedExperts {
            y = y + sharedExperts(x)
        }
        return y
    }
}
