import Foundation
import MLX
import MLXLMCommon
import MLXNN

// Ported from vLLM `vllm/models/beam/nvidia/moe.py` and the grouped sigmoid routing in
// `vllm/model_executor/layers/fused_moe/router/grouped_topk_router.py`.

/// SwiGLU MLP for `mlp_layer_types` entries set to `dense`.
final class SharpAIBeamMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(hiddenSize: Int, intermediateSize: Int) {
        self._gateProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        self._upProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        self._downProj.wrappedValue = Linear(intermediateSize, hiddenSize, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(silu(gateProj(x)) * upProj(x))
    }
}

/// Router for `mlp_layer_types` entries set to `sparse`.
///
/// Scores are `sigmoid(logits)`. Experts are chosen on `scores + e_score_correction_bias`
/// inside the best `topk_group` groups. Routing weights come from the unbiased scores.
final class SharpAIBeamRouter: Module {
    let numExperts: Int
    let topK: Int
    let nGroup: Int?
    let topkGroup: Int?
    let normTopkProb: Bool
    let routedScalingFactor: Float

    var weight: MLXArray
    var e_score_correction_bias: MLXArray

    init(_ config: SharpAIBeamConfiguration) {
        self.numExperts = config.numExperts
        self.topK = config.numExpertsPerTok
        self.nGroup = config.nGroup
        self.topkGroup = config.topkGroup
        self.normTopkProb = config.normTopkProb
        self.routedScalingFactor = config.routedScalingFactor
        self.weight = zeros([config.numExperts, config.hiddenSize])
        self.e_score_correction_bias = zeros([config.numExperts])
        super.init()
    }

    /// Returns expert indices `[B, L, K]` and routing weights `[B, L, K]`.
    func callAsFunction(_ x: MLXArray) -> (indices: MLXArray, weights: MLXArray) {
        let (B, L) = (x.dim(0), x.dim(1))
        let logits = x.asType(.float32).matmul(weight.asType(.float32).T)
        let scores = sigmoid(logits)
        var choice = scores + e_score_correction_bias.asType(.float32)

        if let nGroup, let topkGroup {
            let perGroup = numExperts / nGroup
            let groups = choice.reshaped(B, L, nGroup, perGroup)
            // Group score is the sum of the top two expert choices in that group.
            let groupScores = top(groups, k: 2, axis: -1).sum(axis: -1)
            let groupRank = argSort(argSort(-groupScores, axis: -1), axis: -1)
            let groupKeep = groupRank .< topkGroup
            let expertKeep = broadcast(groupKeep[.ellipsis, .newAxis], to: groups.shape)
                .reshaped(B, L, numExperts)
            choice = which(expertKeep, choice, MLXArray(-Float.infinity))
        }

        let indices = argPartition(-choice, kth: topK - 1, axis: -1)[.ellipsis, ..<topK]
        var weights = takeAlong(scores, indices, axis: -1)
        if normTopkProb {
            weights = weights / weights.sum(axis: -1, keepDims: true)
        }
        return (indices, weights * routedScalingFactor)
    }
}

/// Sparse MoE block. Expert weights are stacked into `switch_mlp` by `sanitize`.
final class SharpAIBeamSparseMoE: Module, UnaryLayer {
    var gate: SharpAIBeamRouter
    @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU

    init(_ config: SharpAIBeamConfiguration) {
        self.gate = SharpAIBeamRouter(config)
        self._switchMLP.wrappedValue = SwitchGLU(
            inputDims: config.hiddenSize,
            hiddenDims: config.moeIntermediateSize,
            numExperts: config.numExperts
        )
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (indices, weights) = gate(x)
        let expertOutput = switchMLP(x, indices)
        return weightedExpertSum(expertOutput, weights).asType(x.dtype)
    }
}
