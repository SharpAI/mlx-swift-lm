//
//  SharpAIHyV4MoE.swift
//  MLXLLM
//
//  Dense SwiGLU MLP, sigmoid router with expert bias, and the sparse MoE block for
//  hy_v4. Port of HYV4MLP, HYV4TopkRouter, Glm5NextTextExperts and HYV4MoE.
//
//  Routing: top-k on (sigmoid score + e_score_correction_bias). Weights use the
//  unbiased sigmoid scores, renormalized when norm_topk_prob is set, then scaled by
//  routed_scaling_factor. n_group and topk_group are 1 for hy_v4, so the group
//  selection is a no-op and routing is a global top-k.
//
//  Routed experts clamp the gate to at most `swiglu_limit` and the up projection to
//  [-limit, limit] before silu(gate) * up. Shared experts and the dense layer are
//  plain SwiGLU with no clamp.
//
//  UNVERIFIED: checkpoint keys (`mlp.gate.*`, `mlp.experts.*`, `mlp.shared_experts.*`)
//  follow the transformers module names. See SharpAIHyV4Model.sanitize.
//

import Foundation
import MLX
import MLXLMCommon
import MLXNN

class SharpAIHyV4MLP: Module, UnaryLayer {
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

/// Router. Checkpoint keys: `mlp.gate.weight` and `mlp.gate.e_score_correction_bias`.
class SharpAIHyV4Gate: Module {
    let topK: Int
    let routeNorm: Bool
    let routedScalingFactor: Float

    @ParameterInfo(key: "weight") var weight: MLXArray
    @ParameterInfo(key: "e_score_correction_bias") var eScoreCorrectionBias: MLXArray

    init(_ config: SharpAIHyV4Configuration) {
        self.topK = config.numExpertsPerTok
        self.routeNorm = config.resolvedNormTopKProb
        self.routedScalingFactor = config.resolvedRoutedScalingFactor

        _weight.wrappedValue = zeros([config.numRoutedExperts, config.hiddenSize])
        _eScoreCorrectionBias.wrappedValue = zeros([config.numRoutedExperts])

        super.init()
    }

    /// Returns (expert indices [B, L, topK], routing weights [B, L, topK]).
    func callAsFunction(_ x: MLXArray) -> (MLXArray, MLXArray) {
        let scores = sigmoid(matmul(x.asType(.float32), weight.asType(.float32).T))
        let choice = scores + eScoreCorrectionBias.asType(.float32)

        let inds = argPartition(-choice, kth: topK - 1, axis: -1)[.ellipsis, ..<topK]
        var weights = takeAlong(scores, inds, axis: -1)
        if routeNorm {
            weights = weights / (weights.sum(axis: -1, keepDims: true) + 1e-20)
        }
        weights = weights * routedScalingFactor
        return (inds, weights.asType(x.dtype))
    }
}

class SharpAIHyV4SparseMoE: Module, UnaryLayer {
    let gate: SharpAIHyV4Gate

    @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU
    @ModuleInfo(key: "shared_experts") var sharedExperts: SharpAIHyV4MLP?

    init(_ config: SharpAIHyV4Configuration) {
        self.gate = SharpAIHyV4Gate(config)

        let limit = config.resolvedSwigluLimit
        let product: @Sendable (MLXArray, MLXArray) -> MLXArray
        if limit > 0 {
            // Clamp gate from above and up to [-limit, limit], as Glm5NextTextExperts._apply_gate.
            product = { gateIn, upIn in
                silu(minimum(gateIn, limit)) * minimum(maximum(upIn, -limit), limit)
            }
        } else {
            product = { gateIn, upIn in silu(gateIn) * upIn }
        }

        _switchMLP.wrappedValue = SwitchGLU(
            inputDims: config.hiddenSize,
            hiddenDims: config.moeIntermediateSize,
            numExperts: config.numRoutedExperts,
            activation: { silu($0) },
            activationProduct: product
        )

        if config.resolvedSharedExperts > 0 {
            _sharedExperts.wrappedValue = SharpAIHyV4MLP(
                hiddenSize: config.hiddenSize,
                intermediateSize: config.moeIntermediateSize * config.resolvedSharedExperts)
        }

        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (inds, scores) = gate(x)
        var y = switchMLP(x, inds)
        y = weightedExpertSum(y, scores).asType(y.dtype)
        if let sharedExperts {
            y = y + sharedExperts(x)
        }
        return y
    }
}
