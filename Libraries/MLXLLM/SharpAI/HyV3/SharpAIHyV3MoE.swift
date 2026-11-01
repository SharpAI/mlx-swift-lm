//
//  SharpAIHyV3MoE.swift
//  MLXLLM
//
//  Dense MLP, sigmoid router with expert bias, and sparse MoE block for hy_v3.
//  Routing: top-k on (sigmoid score + expert bias). Weights use the unbiased
//  sigmoid scores, renormalized when route_norm is set, then scaled.
//  Verified against: https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/hy_v3/modeling_hy_v3.py
//  (HYV3TopKRouter.forward, HYV3MoE.__init__ and forward: no grouped routing,
//  e_score_correction_bias is a buffer on HYV3MoE, shared width is
//  moe_intermediate_size * num_shared_experts, expert activation is silu).
//  The checkpoint key names for the non-expert tensors are not confirmed against
//  the safetensors index (the index fetch was truncated before them).
//

import Foundation
import MLX
import MLXLMCommon
import MLXNN

class SharpAIHyV3MLP: Module, UnaryLayer {
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

/// Router weight only. The expert-selection bias is a buffer on the MoE block
/// (`mlp.e_score_correction_bias`), per HYV3MoE in the Transformers source.
class SharpAIHyV3Gate: Module {
    let topK: Int
    let routeNorm: Bool
    let routerScalingFactor: Float

    @ParameterInfo(key: "weight") var weight: MLXArray

    init(_ config: SharpAIHyV3Configuration) {
        self.topK = config.numExpertsPerTok
        self.routeNorm = config.routeNorm
        self.routerScalingFactor = config.routerScalingFactor

        _weight.wrappedValue = zeros([config.numExperts, config.hiddenSize])

        super.init()
    }

    /// Returns (expert indices [B, L, topK], routing weights [B, L, topK]).
    /// Bias is added for selection only; weights come from the unbiased scores.
    func callAsFunction(_ x: MLXArray, _ eScoreCorrectionBias: MLXArray) -> (MLXArray, MLXArray) {
        let scores = sigmoid(x.matmul(weight.T).asType(.float32))
        let selection = scores + eScoreCorrectionBias

        let inds = argPartition(-selection, kth: topK - 1, axis: -1)[.ellipsis, ..<topK]
        var weights = takeAlong(scores, inds, axis: -1)
        if routeNorm {
            // 1e-20 matches the Transformers HYV3TopKRouter normalization.
            weights = weights / (weights.sum(axis: -1, keepDims: true) + 1e-20)
        }
        weights = weights * routerScalingFactor
        return (inds, weights.asType(x.dtype))
    }
}

class SharpAIHyV3SparseMoE: Module, UnaryLayer {
    let gate: SharpAIHyV3Gate

    @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU
    @ModuleInfo(key: "shared_experts") var sharedExperts: SharpAIHyV3MLP?
    @ParameterInfo(key: "e_score_correction_bias") var eScoreCorrectionBias: MLXArray

    init(_ config: SharpAIHyV3Configuration) {
        self.gate = SharpAIHyV3Gate(config)
        _eScoreCorrectionBias.wrappedValue = zeros([config.numExperts])

        _switchMLP.wrappedValue = SwitchGLU(
            inputDims: config.hiddenSize,
            hiddenDims: config.moeIntermediateSize,
            numExperts: config.numExperts
        )

        if config.numSharedExperts > 0 {
            _sharedExperts.wrappedValue = SharpAIHyV3MLP(
                hiddenSize: config.hiddenSize,
                intermediateSize: config.moeIntermediateSize * config.numSharedExperts)
        }

        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (inds, scores) = gate(x, eScoreCorrectionBias)
        var y = switchMLP(x, inds)
        y = weightedExpertSum(y, scores).asType(y.dtype)
        if let sharedExperts {
            y = y + sharedExperts(x)
        }
        return y
    }
}
