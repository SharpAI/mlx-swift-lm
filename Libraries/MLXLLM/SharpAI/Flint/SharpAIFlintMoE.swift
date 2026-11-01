// SharpAI fork overlay: flint dense MLP and MoE block. Not part of upstream mlx-swift-lm.
//
// Router: sigmoid scores (softmax if router_score_function says so), top-k on those scores
// with no bias term, then renormalization over the selected experts when norm_topk_prob is set.
// The shared expert is added without a gate. Ref: vllm/models/flint/model.py.
//
// UNVERIFIED: the top-k selection on raw sigmoid scores (no correction bias) is taken from
// the vLLM docstring ("sigmoid top-k routing with renormalized weights"). No checkpoint was
// available to confirm it against logits.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

class SharpAIFlintMLP: Module, UnaryLayer {
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

class SharpAIFlintSparseMoeBlock: Module, UnaryLayer {
    let topK: Int
    let normTopkProb: Bool
    let usesSigmoid: Bool

    @ModuleInfo(key: "gate") var gate: Linear
    @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU
    @ModuleInfo(key: "shared_expert") var sharedExpert: SharpAIFlintMLP

    init(_ config: SharpAIFlintConfiguration) {
        self.topK = config.numExpertsPerTok
        self.normTopkProb = config.normTopkProb
        self.usesSigmoid = config.routerScoreFunction == "sigmoid"

        _gate.wrappedValue = Linear(config.hiddenSize, config.numExperts, bias: false)
        _switchMLP.wrappedValue = SwitchGLU(
            inputDims: config.hiddenSize,
            hiddenDims: config.moeIntermediateSize,
            numExperts: config.numExperts
        )
        _sharedExpert.wrappedValue = SharpAIFlintMLP(
            hiddenSize: config.hiddenSize,
            intermediateSize: config.sharedExpertIntermediateSize
        )
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let logits = gate(x).asType(.float32)
        let scores = usesSigmoid ? sigmoid(logits) : MLX.softmax(logits, axis: -1, precise: true)

        let inds = MLX.argPartition(-scores, kth: topK - 1, axis: -1)[.ellipsis, ..<topK]
        var weights = MLX.takeAlong(scores, inds, axis: -1)
        if normTopkProb {
            weights = weights / MLX.sum(weights, axis: -1, keepDims: true)
        }

        let y = switchMLP(x, inds)
        return weightedExpertSum(y, weights).asType(y.dtype) + sharedExpert(x)
    }
}
