//
//  SharpAILagunaMoE.swift
//  MLXLLM
//
//  Dense MLP, sigmoid router with an aux-loss-free bias, and the sparse MoE block for laguna.
//  Top-k selects experts on (sigmoid score + bias). Weights use the unbiased sigmoid scores,
//  renormalized, then scaled by moe_routed_scaling_factor. The shared expert is added after.
//  UNVERIFIED: the bias key location is taken from a vLLM comment, not from a checkpoint.
//

import Foundation
import MLX
import MLXLMCommon
import MLXNN

class SharpAILagunaMLP: Module, UnaryLayer {
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

/// Router. Checkpoint keys: `mlp.gate.weight`. The bias is stored by the checkpoint at
/// `mlp.experts.e_score_correction_bias`, and sanitize moves it to `mlp.gate.e_score_correction_bias`.
class SharpAILagunaRouter: Module {
    let topK: Int
    let softcapping: Float
    let routedScalingFactor: Float

    @ParameterInfo(key: "weight") var weight: MLXArray
    @ParameterInfo(key: "e_score_correction_bias") var eScoreCorrectionBias: MLXArray

    init(_ config: SharpAILagunaConfiguration) {
        self.topK = config.numExpertsPerTok
        self.softcapping = config.routerLogitSoftcapping
        self.routedScalingFactor = config.routedScalingFactor

        _weight.wrappedValue = zeros([config.numExperts, config.hiddenSize])
        _eScoreCorrectionBias.wrappedValue = zeros([config.numExperts])

        super.init()
    }

    /// Returns (expert indices [B, L, topK], routing weights [B, L, topK]).
    func callAsFunction(_ x: MLXArray) -> (MLXArray, MLXArray) {
        var logits = x.matmul(weight.T).asType(.float32)
        if softcapping > 0 {
            logits = tanh(logits / softcapping) * softcapping
        }

        let scores = sigmoid(logits)
        let selection = scores + eScoreCorrectionBias

        let inds = argPartition(-selection, kth: topK - 1, axis: -1)[.ellipsis, ..<topK]
        var weights = takeAlong(scores, inds, axis: -1)
        weights = weights / weights.sum(axis: -1, keepDims: true)
        weights = weights * routedScalingFactor
        return (inds, weights.asType(x.dtype))
    }
}

class SharpAILagunaSparseMoE: Module, UnaryLayer {
    @ModuleInfo(key: "gate") var gate: SharpAILagunaRouter
    @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU
    @ModuleInfo(key: "shared_expert") var sharedExpert: SharpAILagunaMLP?

    init(_ config: SharpAILagunaConfiguration) {
        _gate.wrappedValue = SharpAILagunaRouter(config)

        _switchMLP.wrappedValue = SwitchGLU(
            inputDims: config.hiddenSize,
            hiddenDims: config.moeIntermediateSize,
            numExperts: config.numExperts
        )

        if config.sharedExpertIntermediateSize > 0 {
            _sharedExpert.wrappedValue = SharpAILagunaMLP(
                hiddenSize: config.hiddenSize,
                intermediateSize: config.sharedExpertIntermediateSize)
        }
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (inds, scores) = gate(x)
        let expertOutputs = switchMLP(x, inds)
        var y = (expertOutputs * scores[.ellipsis, .newAxis]).sum(axis: -2)
        if let sharedExpert {
            y = y + sharedExpert(x)
        }
        return y
    }
}
