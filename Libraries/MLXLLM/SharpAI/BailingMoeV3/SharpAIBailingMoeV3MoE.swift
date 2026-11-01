//
//  SharpAIBailingMoeV3MoE.swift
//  MLXLLM
//
//  Dense MLP, router and sparse MoE block for bailing_moe_v3.
//  Routing follows vLLM grouped_topk with sigmoid scores: top-`topkGroup` groups by
//  the sum of their top-2 biased scores, then top-k experts on the biased scores.
//  Routing weights use the unbiased scores, renormalized, then scaled.
//

import Foundation
import MLX
import MLXLMCommon
import MLXNN

class SharpAIBailingMoeV3MLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "up_proj") var up: Linear
    @ModuleInfo(key: "down_proj") var down: Linear

    init(hiddenSize: Int, intermediateSize: Int, bias: Bool) {
        _gate.wrappedValue = Linear(hiddenSize, intermediateSize, bias: bias)
        _up.wrappedValue = Linear(hiddenSize, intermediateSize, bias: bias)
        _down.wrappedValue = Linear(intermediateSize, hiddenSize, bias: bias)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        down(silu(gate(x)) * up(x))
    }
}

/// Router. Checkpoint keys: `mlp.gate.weight` and `mlp.gate.expert_bias`.
class SharpAIBailingMoeV3Gate: Module {
    let topK: Int
    let nGroup: Int
    let topkGroup: Int
    let numExperts: Int
    let routedScalingFactor: Float

    var weight: MLXArray
    var expert_bias: MLXArray

    init(_ config: SharpAIBailingMoeV3Configuration) {
        self.topK = config.numExpertsPerTok
        self.nGroup = config.nGroup
        self.topkGroup = config.topkGroup
        self.numExperts = config.numExperts
        self.routedScalingFactor = config.routedScalingFactor
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
        weights = weights / weights.sum(axis: -1, keepDims: true)
        weights = weights * routedScalingFactor
        return (inds, weights.asType(x.dtype))
    }
}

class SharpAIBailingMoeV3SparseMoE: Module, UnaryLayer {
    @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU
    var gate: SharpAIBailingMoeV3Gate
    @ModuleInfo(key: "shared_experts") var sharedExperts: SharpAIBailingMoeV3MLP?

    init(_ config: SharpAIBailingMoeV3Configuration) {
        _switchMLP.wrappedValue = SwitchGLU(
            inputDims: config.hiddenSize,
            hiddenDims: config.moeIntermediateSize,
            numExperts: config.numExperts,
            bias: config.useBias
        )
        self.gate = SharpAIBailingMoeV3Gate(config)

        if config.numSharedExperts > 0 {
            _sharedExperts.wrappedValue = SharpAIBailingMoeV3MLP(
                hiddenSize: config.hiddenSize,
                intermediateSize: config.moeSharedExpertIntermediateSize * config.numSharedExperts,
                bias: config.useBias
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
