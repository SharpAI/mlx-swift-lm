//
//  SharpAIStep3p5Layers.swift
//  MLXLLM
//
//  Norm, SwiGLU MLP, sigmoid router, and MoE block for step3p5.
//  All RMSNorm weights are zero-centered (scale = 1 + weight), as vLLM GemmaRMSNorm does.
//  Router: sigmoid scores, top-k on (score + router_bias), weights from the unbiased scores,
//  renormalized when norm_expert_weight is set, then scaled by moe_router_scaling_factor.
//  Clamped SwiGLU follows vLLM SwigluStepAndMul: silu(gate) capped at the limit, up clipped
//  to [-limit, limit].
//  UNVERIFIED: no checkpoint was run. The layout comes from vLLM step3p5.py.
//

import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// RMSNorm with zero-centered scale: `x * rsqrt(mean(x^2)) * (1 + weight)`.
class SharpAIStep3p5RMSNorm: Module {
    let eps: Float

    @ParameterInfo(key: "weight") var weight: MLXArray

    init(dimensions: Int, eps: Float) {
        _weight.wrappedValue = zeros([dimensions])
        self.eps = eps
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        MLXFast.rmsNorm(x, weight: 1.0 + weight, eps: eps)
    }
}

/// Dense SwiGLU MLP. Also used for the shared expert. A positive `limit` clamps the gate and up
/// halves.
class SharpAIStep3p5MLP: Module, UnaryLayer {
    let limit: Float

    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(hiddenSize: Int, intermediateSize: Int, limit: Float = 0) {
        self.limit = limit
        _gateProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        _upProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        _downProj.wrappedValue = Linear(intermediateSize, hiddenSize, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var gate = silu(gateProj(x))
        var up = upProj(x)
        if limit > 0 {
            gate = clip(gate, max: limit)
            up = clip(up, min: -limit, max: limit)
        }
        return downProj(gate * up)
    }
}

/// Sigmoid router. Kept in FP32 for numerical stability, as vLLM need_fp32_gate requires.
class SharpAIStep3p5Router: Module {
    @ParameterInfo(key: "weight") var weight: MLXArray

    init(hiddenSize: Int, numExperts: Int) {
        _weight.wrappedValue = zeros([numExperts, hiddenSize])
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        x.asType(.float32).matmul(weight.asType(.float32).T)
    }
}

/// Sparse MoE block with a shared expert. Checkpoint keys: `moe.gate.weight`,
/// `moe.router_bias`, `moe.switch_mlp.*` (stacked by sanitize), `moe.share_expert.*`.
class SharpAIStep3p5MoE: Module {
    let topK: Int
    let routedScalingFactor: Float
    let renormalize: Bool

    @ModuleInfo(key: "gate") var gate: SharpAIStep3p5Router
    @ParameterInfo(key: "router_bias") var routerBias: MLXArray
    @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU
    @ModuleInfo(key: "share_expert") var shareExpert: SharpAIStep3p5MLP

    init(_ config: SharpAIStep3p5Configuration, layer: Int) {
        topK = config.moeTopK
        routedScalingFactor = config.moeRouterScalingFactor
        renormalize = config.normExpertWeight

        _gate.wrappedValue = SharpAIStep3p5Router(
            hiddenSize: config.hiddenSize, numExperts: config.moeNumExperts)
        _routerBias.wrappedValue = zeros([config.moeNumExperts])

        let limit = config.routedSwigluLimit(layer: layer)
        if limit > 0 {
            _switchMLP.wrappedValue = SwitchGLU(
                inputDims: config.hiddenSize,
                hiddenDims: config.moeIntermediateSize,
                numExperts: config.moeNumExperts,
                activation: { x in clip(silu(x), max: limit) },
                activationProduct: { g, u in
                    clip(silu(g), max: limit) * clip(u, min: -limit, max: limit)
                }
            )
        } else {
            _switchMLP.wrappedValue = SwitchGLU(
                inputDims: config.hiddenSize,
                hiddenDims: config.moeIntermediateSize,
                numExperts: config.moeNumExperts
            )
        }

        _shareExpert.wrappedValue = SharpAIStep3p5MLP(
            hiddenSize: config.hiddenSize,
            intermediateSize: config.shareExpertDim,
            limit: config.sharedSwigluLimit(layer: layer)
        )

        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let scores = sigmoid(gate(x))
        let selection = scores + routerBias

        let inds = argPartition(-selection, kth: topK - 1, axis: -1)[.ellipsis, ..<topK]
        var weights = takeAlong(scores, inds, axis: -1)
        if renormalize {
            weights = weights / weights.sum(axis: -1, keepDims: true)
        }
        weights = (weights * routedScalingFactor).asType(x.dtype)

        let routed = (switchMLP(x, inds) * weights[.ellipsis, .newAxis]).sum(axis: -2)
        return routed + shareExpert(x)
    }
}
