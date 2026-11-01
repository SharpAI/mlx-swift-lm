// SharpAI fork overlay: longcat_flash router and MoE block with identity zero-experts.
// Mirrors LongcatFlashTopkRouter and LongcatFlashMoE in mlx-lm longcat_flash.py.
// Not part of upstream.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

class SharpAILongcatFlashRouter: Module {
    let topK: Int
    let totalExperts: Int
    let routedScalingFactor: Float
    let normTopkProb: Bool

    @ModuleInfo(key: "classifier") var classifier: Linear
    var e_score_correction_bias: MLXArray

    init(_ config: SharpAILongcatFlashConfiguration) {
        let total = config.nRoutedExperts + config.zeroExpertNum

        self.topK = config.moeTopk
        self.totalExperts = total
        self.routedScalingFactor = config.routedScalingFactor
        self.normTopkProb = config.normTopkProb

        self._classifier.wrappedValue = Linear(
            config.hiddenSize, total, bias: config.routerBias)
        self.e_score_correction_bias = zeros([total])
    }

    /// Returns `(indices, weights)`, each shaped `[batch, seq, topK]`.
    func callAsFunction(_ x: MLXArray) -> (MLXArray, MLXArray) {
        let scores = softmax(classifier(x), axis: -1)
        let corrected = scores + e_score_correction_bias

        let kth = totalExperts - topK
        let indices = argPartition(corrected, kth: kth, axis: -1)[
            .ellipsis, kth ..< totalExperts]
        var weights = takeAlong(scores, indices, axis: -1)

        if normTopkProb {
            weights = weights / (weights.sum(axis: -1, keepDims: true) + 1e-20)
        }
        weights = weights * routedScalingFactor

        return (indices, weights.asType(x.dtype))
    }
}

class SharpAILongcatFlashMoE: Module {
    let nRoutedExperts: Int

    @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU
    var router: SharpAILongcatFlashRouter

    init(_ config: SharpAILongcatFlashConfiguration) {
        self.nRoutedExperts = config.nRoutedExperts
        self._switchMLP.wrappedValue = SwitchGLU(
            inputDims: config.hiddenSize,
            hiddenDims: config.expertFfnHiddenSize,
            numExperts: config.nRoutedExperts)
        self.router = SharpAILongcatFlashRouter(config)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (indices, weights) = router(x)

        // Indices at or above nRoutedExperts select identity (zero) experts.
        let isZeroExpert = greaterEqual(indices, nRoutedExperts)
        let regularIndices = which(isZeroExpert, 0, indices)
        let regularWeights = which(isZeroExpert, Float(0), weights)

        let regularOutput = switchMLP(x, regularIndices)
        let routed = (regularOutput * regularWeights[.ellipsis, .newAxis]).sum(axis: -2)

        // Identity experts return the input scaled by their summed routing weight.
        let identityWeights = which(isZeroExpert, weights, Float(0))
            .sum(axis: -1, keepDims: true)
        return routed + x * identityWeights
    }
}
