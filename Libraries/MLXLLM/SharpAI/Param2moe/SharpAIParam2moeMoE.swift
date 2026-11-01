//
//  SharpAIParam2moeMoE.swift
//  MLXLLM
//
//  SwiGLU MLP, the sigmoid router with a per-expert score bias, and the sparse MoE block.
//  Selection uses the biased scores, with grouped top-k when n_group > 1. The routing weights use
//  the unbiased scores, renormalized when norm_topk_prob is set, then scaled by
//  routed_scaling_factor. This follows vLLM grouped_topk in param2moe.py.
//  Router keys: mlp.gate.weight and mlp.gate.expert_bias (HF names, no remap needed).
//

import Foundation
import MLX
import MLXLMCommon
import MLXNN

class SharpAIParam2moeMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(hiddenSize: Int, intermediateSize: Int, bias: Bool) {
        _gateProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: bias)
        _upProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: bias)
        _downProj.wrappedValue = Linear(intermediateSize, hiddenSize, bias: bias)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(silu(gateProj(x)) * upProj(x))
    }
}

class SharpAIParam2moeRouter: Module {
    let topK: Int
    let nGroup: Int
    let topkGroup: Int
    let normTopkProb: Bool
    let routedScalingFactor: Float
    let useSoftmax: Bool

    @ParameterInfo(key: "weight") var weight: MLXArray
    @ParameterInfo(key: "expert_bias") var expertBias: MLXArray

    init(_ config: SharpAIParam2moeConfiguration) {
        self.topK = config.numExpertsPerTok
        self.nGroup = config.nGroup
        self.topkGroup = config.topkGroup
        self.normTopkProb = config.normTopkProb
        self.routedScalingFactor = config.routedScalingFactor
        self.useSoftmax = config.scoreFunction == "softmax"

        _weight.wrappedValue = zeros([config.numExperts, config.hiddenSize])
        _expertBias.wrappedValue = zeros([config.numExperts])

        super.init()
    }

    /// Returns (expert indices [B, L, topK], routing weights [B, L, topK]).
    func callAsFunction(_ x: MLXArray) -> (MLXArray, MLXArray) {
        // Logits in float32, as vLLM's fp32 gate does.
        let logits = x.asType(.float32).matmul(weight.asType(.float32).T)
        let scores = useSoftmax ? softmax(logits, axis: -1) : sigmoid(logits)

        var selection = scores + expertBias
        if nGroup > 1 {
            let (B, L) = (x.dim(0), x.dim(1))
            let grouped = selection.reshaped(B, L, nGroup, -1)
            let groupScores = top(grouped, k: 2, axis: -1).sum(axis: -1)
            let keptGroups = argPartition(-groupScores, kth: topkGroup - 1, axis: -1)[
                .ellipsis, ..<topkGroup]
            let groupIds = MLXArray(Int32(0) ..< Int32(nGroup))
            let groupMask = (keptGroups.expandedDimensions(axis: -1) .== groupIds).any(axis: -2)
            selection = MLX.where(
                groupMask.expandedDimensions(axis: -1), grouped, MLXArray(-Float.infinity)
            )
            .reshaped(B, L, -1)
        }

        let inds = argPartition(-selection, kth: topK - 1, axis: -1)[.ellipsis, ..<topK]
        var weights = takeAlong(scores, inds, axis: -1)
        if normTopkProb {
            weights = weights / weights.sum(axis: -1, keepDims: true)
        }
        weights = weights * routedScalingFactor
        return (inds, weights.asType(x.dtype))
    }
}

class SharpAIParam2moeSparseMoE: Module, UnaryLayer {
    @ModuleInfo(key: "gate") var gate: SharpAIParam2moeRouter
    @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU
    @ModuleInfo(key: "shared_experts") var sharedExperts: SharpAIParam2moeMLP?

    init(_ config: SharpAIParam2moeConfiguration) {
        _gate.wrappedValue = SharpAIParam2moeRouter(config)

        _switchMLP.wrappedValue = SwitchGLU(
            inputDims: config.hiddenSize,
            hiddenDims: config.moeIntermediateSize,
            numExperts: config.numExperts
        )

        if config.numSharedExperts > 0 {
            _sharedExperts.wrappedValue = SharpAIParam2moeMLP(
                hiddenSize: config.hiddenSize,
                intermediateSize: config.moeSharedExpertIntermediateSize,
                bias: config.useBias)
        } else {
            _sharedExperts.wrappedValue = nil
        }
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (inds, scores) = gate(x)
        let expertOutputs = switchMLP(x, inds)
        var y = (expertOutputs * scores[.ellipsis, .newAxis]).sum(axis: -2)
        if let sharedExperts {
            y = y + sharedExperts(x)
        }
        return y
    }
}
