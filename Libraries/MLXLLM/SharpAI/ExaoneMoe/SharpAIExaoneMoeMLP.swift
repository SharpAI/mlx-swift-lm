//
//  SharpAIExaoneMoeMLP.swift
//  MLXLLM
//
//  Dense MLP, clamped SwiGLU helper, routed experts and the sigmoid router for
//  exaone_moe. Routing follows vLLM grouped top-k: group score is the sum of each
//  group's top-2 biased scores. Weights come from the unbiased scores.
//

import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// SwiGLU with the vLLM `SiluAndMulWithClamp` clamp (alpha 1, beta 0):
/// `silu(min(gate, limit)) * clip(up, -limit, limit)`.
/// Source: vLLM `model_executor/layers/activation.py`. The MLX side is not verified
/// against a checkpoint.
func sharpAIExaoneMoeClampedSwiGLU(gate: MLXArray, up: MLXArray, limit: Float) -> MLXArray {
    let clampedGate = MLX.minimum(gate, MLXArray(limit).asType(gate.dtype))
    let clampedUp = MLX.minimum(
        MLX.maximum(up, MLXArray(-limit).asType(up.dtype)),
        MLXArray(limit).asType(up.dtype))
    return silu(clampedGate) * clampedUp
}

/// Dense SwiGLU MLP. Also used for shared experts and MTP layers.
class SharpAIExaoneMoeMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "up_proj") var up: Linear
    @ModuleInfo(key: "down_proj") var down: Linear

    let swigluLimit: Float?

    init(hiddenSize: Int, intermediateSize: Int, bias: Bool, swigluLimit: Float?) {
        _gate.wrappedValue = Linear(hiddenSize, intermediateSize, bias: bias)
        _up.wrappedValue = Linear(hiddenSize, intermediateSize, bias: bias)
        _down.wrappedValue = Linear(intermediateSize, hiddenSize, bias: bias)
        self.swigluLimit = swigluLimit
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let g = gate(x)
        let u = up(x)
        let activated: MLXArray
        if let swigluLimit {
            activated = sharpAIExaoneMoeClampedSwiGLU(gate: g, up: u, limit: swigluLimit)
        } else {
            activated = silu(g) * u
        }
        return down(activated)
    }
}

/// Routed experts with the SwiGLU clamp. Used only for layers whose clamp is > 0.
/// Layers without a clamp use `SwitchGLU`, which keeps the upstream fast paths.
/// Keys match `SwitchGLU` (`gate_proj`, `up_proj`, `down_proj`), so checkpoints load
/// the same way for both classes.
class SharpAIExaoneMoeSwitchFFN: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: SwitchLinear
    @ModuleInfo(key: "up_proj") var upProj: SwitchLinear
    @ModuleInfo(key: "down_proj") var downProj: SwitchLinear

    let swigluLimit: Float

    init(inputDims: Int, hiddenDims: Int, numExperts: Int, swigluLimit: Float) {
        _gateProj.wrappedValue = SwitchLinear(
            inputDims: inputDims, outputDims: hiddenDims, numExperts: numExperts, bias: false)
        _upProj.wrappedValue = SwitchLinear(
            inputDims: inputDims, outputDims: hiddenDims, numExperts: numExperts, bias: false)
        _downProj.wrappedValue = SwitchLinear(
            inputDims: hiddenDims, outputDims: inputDims, numExperts: numExperts, bias: false)
        self.swigluLimit = swigluLimit
        super.init()
    }

    /// Same routing and sort/unsort flow as `SwitchGLU.callAsFunction`.
    func callAsFunction(_ x: MLXArray, _ indices: MLXArray) -> MLXArray {
        var x = MLX.expandedDimensions(x, axes: [-2, -3])

        let doSort = indices.size >= 64

        var idx = indices
        var inverseOrder = MLXArray()

        if doSort {
            (x, idx, inverseOrder) = gatherSort(x: x, indices: indices)
        }

        let gate = gateProj(x, idx, sortedIndices: doSort)
        let up = upProj(x, idx, sortedIndices: doSort)
        let activated = sharpAIExaoneMoeClampedSwiGLU(gate: gate, up: up, limit: swigluLimit)
        x = downProj(activated, idx, sortedIndices: doSort)

        if doSort {
            x = scatterUnsort(x: x, invOrder: inverseOrder, shape: indices.shape)
        }

        return MLX.squeezed(x, axis: -2)
    }
}

/// Sparse MoE block. Checkpoint keys: `mlp.gate.weight`, `mlp.switch_mlp.*` (stacked by
/// sanitize), `mlp.shared_experts.*`. The router bias is read from `mlp.gate.e_score_correction_bias`
/// and moved here by sanitize (UNVERIFIED against a checkpoint).
class SharpAIExaoneMoeSparseMoE: Module, UnaryLayer {
    let topK: Int
    let nGroup: Int
    let topkGroup: Int
    let numExperts: Int
    let normTopkProb: Bool
    let routedScalingFactor: Float

    @ModuleInfo(key: "gate") var gate: Linear
    @ParameterInfo(key: "e_score_correction_bias") var eScoreCorrectionBias: MLXArray
    @ModuleInfo(key: "switch_mlp") var switchMLP: Module
    @ModuleInfo(key: "shared_experts") var sharedExperts: SharpAIExaoneMoeMLP?

    init(_ config: SharpAIExaoneMoeConfiguration, layer: Int) {
        self.topK = config.numExpertsPerTok
        self.nGroup = config.nGroup
        self.topkGroup = config.topkGroup
        self.numExperts = config.numExperts
        self.normTopkProb = config.normTopkProb
        self.routedScalingFactor = config.routedScalingFactor

        _gate.wrappedValue = Linear(config.hiddenSize, config.numExperts, bias: false)
        _eScoreCorrectionBias.wrappedValue = zeros([config.numExperts])

        if let limit = config.swigluLimit(layer: layer) {
            _switchMLP.wrappedValue = SharpAIExaoneMoeSwitchFFN(
                inputDims: config.hiddenSize,
                hiddenDims: config.moeIntermediateSize,
                numExperts: config.numExperts,
                swigluLimit: limit)
        } else {
            _switchMLP.wrappedValue = SwitchGLU(
                inputDims: config.hiddenSize,
                hiddenDims: config.moeIntermediateSize,
                numExperts: config.numExperts)
        }

        if config.numSharedExperts > 0 {
            _sharedExperts.wrappedValue = SharpAIExaoneMoeMLP(
                hiddenSize: config.hiddenSize,
                intermediateSize: config.moeIntermediateSize * config.numSharedExperts,
                bias: false,
                swigluLimit: config.swigluLimit(layer: layer))
        } else {
            _sharedExperts.wrappedValue = nil
        }

        super.init()
    }

    /// Returns (expert indices [B, L, topK], routing weights [B, L, topK]).
    func route(_ x: MLXArray) -> (MLXArray, MLXArray) {
        let (bsz, seqLen) = (x.dim(0), x.dim(1))

        let scores = sigmoid(gate(x).asType(.float32))
        let biased = scores + eScoreCorrectionBias

        var selection = biased
        if nGroup > 1 {
            let groupSize = numExperts / nGroup
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
            selection = MLX.where(
                expertKeep.asType(.bool), biased, MLXArray(-Float.infinity))
        }

        let inds = argPartition(-selection, kth: topK - 1, axis: -1)[.ellipsis, ..<topK]
        var weights = takeAlong(scores, inds, axis: -1)
        if normTopkProb {
            weights = weights / weights.sum(axis: -1, keepDims: true)
        }
        weights = weights * routedScalingFactor
        return (inds, weights.asType(x.dtype))
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (indices, weights) = route(x)

        let routed: MLXArray
        if let ffn = switchMLP as? SharpAIExaoneMoeSwitchFFN {
            routed = ffn(x, indices)
        } else if let glu = switchMLP as? SwitchGLU {
            routed = glu(x, indices)
        } else {
            fatalError("SharpAIExaoneMoe: unexpected routed expert module")
        }

        var y = weightedExpertSum(routed, weights).asType(x.dtype)
        if let sharedExperts {
            y = y + sharedExperts(x)
        }
        return y
    }
}
