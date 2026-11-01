//
//  SharpAIMimoV2MoE.swift
//  MLXLLM
//
//  Dense SiLU MLP and the sparse MoE block for mimo_v2. Routing follows vLLM:
//  sigmoid scores in float32, a per-expert correction bias for selection only,
//  optional group-limited top-k, then renormalization and routed_scaling_factor.
//  UNVERIFIED: vLLM does not pass routed_scaling_factor to FusedMoEFactory, so
//  applying it here is a choice. It is 1.0 unless the checkpoint sets it.
//

import Foundation
import MLX
import MLXLMCommon
import MLXNN

class SharpAIMimoV2MLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(hiddenSize: Int, intermediateSize: Int) {
        _gateProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        _upProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        _downProj.wrappedValue = Linear(intermediateSize, hiddenSize, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(silu(gateProj(x)) * upProj(x))
    }
}

/// Group-limited top-k selection over sigmoid scores. Weights come from the unbiased scores.
private func sharpAIMimoV2SelectExperts(
    gates: MLXArray,
    eScoreCorrectionBias: MLXArray,
    topK: Int,
    nGroup: Int,
    topkGroup: Int,
    routedScalingFactor: Float,
    normTopkProb: Bool
) -> (MLXArray, MLXArray) {
    var scores = sigmoid(gates.asType(.float32))
    let originalScores = scores
    scores = scores + eScoreCorrectionBias

    if nGroup > 1 {
        scores = unflatten(scores, axis: -1, shape: [nGroup, -1])
        let groupScores = top(scores, k: 2, axis: -1).sum(axis: -1, keepDims: true)
        let k = nGroup - topkGroup
        let groupIdx = argPartition(groupScores, kth: k - 1, axis: -2)[.ellipsis, ..<k, 0...]
        scores = putAlong(
            scores,
            stopGradient(groupIdx),
            values: MLXArray(0.0),
            axis: -2
        )
        scores = flattened(scores, start: -2, end: -1)
    }

    let inds = argPartition(-scores, kth: topK - 1, axis: -1)[.ellipsis, ..<topK]
    var selected = takeAlong(originalScores, inds, axis: -1)
    if topK > 1, normTopkProb {
        let denominator = selected.sum(axis: -1, keepDims: true)
        selected = selected / (denominator + 1e-20)
    }
    selected = selected * routedScalingFactor

    return (inds, selected)
}

/// Router. Checkpoint keys: `mlp.gate.weight` and `mlp.gate.e_score_correction_bias`.
class SharpAIMimoV2Gate: Module {
    let topK: Int
    let normTopkProb: Bool
    let routedScalingFactor: Float
    let nGroup: Int
    let topkGroup: Int

    @ParameterInfo(key: "weight") var weight: MLXArray
    @ParameterInfo(key: "e_score_correction_bias") var eScoreCorrectionBias: MLXArray

    init(_ config: SharpAIMimoV2Configuration) {
        let experts = config.nRoutedExperts ?? 0
        self.topK = config.numExpertsPerTok ?? 0
        self.normTopkProb = config.normTopkProb
        self.routedScalingFactor = config.routedScalingFactor
        self.nGroup = config.nGroup
        self.topkGroup = config.topkGroup

        _weight.wrappedValue = zeros([experts, config.hiddenSize])
        _eScoreCorrectionBias.wrappedValue = zeros([experts])

        super.init()
    }

    /// Returns (expert indices [B, L, topK], routing weights [B, L, topK]).
    func callAsFunction(_ x: MLXArray) -> (MLXArray, MLXArray) {
        let gates = x.asType(.float32).matmul(weight.T.asType(.float32))
        return sharpAIMimoV2SelectExperts(
            gates: gates,
            eScoreCorrectionBias: eScoreCorrectionBias,
            topK: topK,
            nGroup: nGroup,
            topkGroup: topkGroup,
            routedScalingFactor: routedScalingFactor,
            normTopkProb: normTopkProb
        )
    }
}

class SharpAIMimoV2MoE: Module, UnaryLayer {
    let gate: SharpAIMimoV2Gate

    @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU

    init(_ config: SharpAIMimoV2Configuration) {
        guard let experts = config.nRoutedExperts, let moeSize = config.moeIntermediateSize else {
            fatalError("SharpAIMimoV2MoE requires n_routed_experts and moe_intermediate_size.")
        }

        self.gate = SharpAIMimoV2Gate(config)

        _switchMLP.wrappedValue = SwitchGLU(
            inputDims: config.hiddenSize,
            hiddenDims: moeSize,
            numExperts: experts
        )

        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (inds, scores) = gate(x)
        var y = switchMLP(x, inds)
        y = weightedExpertSum(y, scores).asType(y.dtype)
        return y
    }
}
