// SharpAI fork overlay: audioflamingo3 audio encoder. Not part of upstream mlx-swift-lm.
// Follows vLLM's audioflamingo3.py: conv stem, pre-LN blocks, 2x average pool, final LayerNorm.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// Output tokens for `melFrames` input frames: conv (stride 2), then average pool (stride 2).
func sharpAIAudioflamingo3PooledLength(_ melFrames: Int) -> Int {
    guard melFrames > 0 else { return 0 }
    let convLength = (melFrames - 1) / 2 + 1
    guard convLength >= 2 else { return 0 }
    return (convLength - 2) / 2 + 1
}

final class SharpAIAudioflamingo3EncoderAttention: Module {
    let heads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear

    init(hiddenSize: Int, heads: Int) {
        let headDim = hiddenSize / heads
        self.heads = heads
        self.headDim = headDim
        self.scale = pow(Float(headDim), -0.5)
        _qProj.wrappedValue = Linear(hiddenSize, hiddenSize, bias: true)
        _kProj.wrappedValue = Linear(hiddenSize, hiddenSize, bias: false)
        _vProj.wrappedValue = Linear(hiddenSize, hiddenSize, bias: true)
        _outProj.wrappedValue = Linear(hiddenSize, hiddenSize, bias: true)
    }

    /// Bidirectional attention. `mask` is additive, broadcastable to `[batch, heads, len, len]`.
    func callAsFunction(_ x: MLXArray, mask: MLXArray?) -> MLXArray {
        let (batch, length) = (x.dim(0), x.dim(1))
        let queries = qProj(x).reshaped(batch, length, heads, headDim).transposed(0, 2, 1, 3)
        let keys = kProj(x).reshaped(batch, length, heads, headDim).transposed(0, 2, 1, 3)
        let values = vProj(x).reshaped(batch, length, heads, headDim).transposed(0, 2, 1, 3)

        let attentionMask: MLXFast.ScaledDotProductAttentionMaskMode
        if let mask {
            attentionMask = .array(mask)
        } else {
            attentionMask = .none
        }
        let output = attentionWithCacheUpdate(
            queries: queries, keys: keys, values: values, cache: nil, scale: scale,
            mask: attentionMask)
        return outProj(output.transposed(0, 2, 1, 3).reshaped(batch, length, -1))
    }
}

final class SharpAIAudioflamingo3EncoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttention: SharpAIAudioflamingo3EncoderAttention
    @ModuleInfo(key: "self_attn_layer_norm") var selfAttentionLayerNorm: LayerNorm
    @ModuleInfo(key: "fc1") var fc1: Linear
    @ModuleInfo(key: "fc2") var fc2: Linear
    @ModuleInfo(key: "final_layer_norm") var finalLayerNorm: LayerNorm

    init(_ config: SharpAIAudioflamingo3AudioConfiguration) {
        _selfAttention.wrappedValue = SharpAIAudioflamingo3EncoderAttention(
            hiddenSize: config.hiddenSize, heads: config.attentionHeads)
        _selfAttentionLayerNorm.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: 1e-5)
        _fc1.wrappedValue = Linear(config.hiddenSize, config.intermediateSize)
        _fc2.wrappedValue = Linear(config.intermediateSize, config.hiddenSize)
        _finalLayerNorm.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: 1e-5)
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray?) -> MLXArray {
        var h = x + selfAttention(selfAttentionLayerNorm(x), mask: mask)
        h = h + fc2(gelu(fc1(finalLayerNorm(h))))
        return h
    }
}

final class SharpAIAudioflamingo3AudioEncoder: Module {
    let numMelBins: Int

    @ModuleInfo(key: "conv1") var conv1: Conv1d
    @ModuleInfo(key: "conv2") var conv2: Conv1d
    @ModuleInfo(key: "embed_positions") var embedPositions: Embedding
    let layers: [SharpAIAudioflamingo3EncoderLayer]
    @ModuleInfo(key: "layer_norm") var layerNorm: LayerNorm

    init(_ config: SharpAIAudioflamingo3AudioConfiguration) {
        let hiddenSize = config.hiddenSize
        self.numMelBins = config.numMelBins
        _conv1.wrappedValue = Conv1d(
            inputChannels: config.numMelBins, outputChannels: hiddenSize, kernelSize: 3,
            padding: 1)
        _conv2.wrappedValue = Conv1d(
            inputChannels: hiddenSize, outputChannels: hiddenSize, kernelSize: 3, stride: 2,
            padding: 1)
        _embedPositions.wrappedValue = Embedding(
            embeddingCount: config.maxSourcePositions, dimensions: hiddenSize)
        self.layers = (0 ..< config.hiddenLayers).map { _ in
            SharpAIAudioflamingo3EncoderLayer(config)
        }
        _layerNorm.wrappedValue = LayerNorm(dimensions: hiddenSize, eps: 1e-5)
    }

    /// - Parameters:
    ///   - features: log-mel features `[chunks, numMelBins, frames]`.
    ///   - featureMask: `[chunks, frames]`, 1 for valid frames. `nil` means all frames are valid.
    /// - Returns: `[chunks, pooled, hiddenSize]`. `pooled` is `frames` after two stride-2 steps.
    func callAsFunction(_ features: MLXArray, featureMask: MLXArray?) -> MLXArray {
        let dtype = conv1.weight.dtype
        var x = features.transposed(0, 2, 1).asType(dtype)
        x = gelu(conv1(x))
        x = gelu(conv2(x))

        let positions = x.dim(1)
        x = x + embedPositions.weight[0 ..< positions]

        let mask = attentionMask(
            featureMask: featureMask, batch: features.dim(0), positions: positions)?
            .asType(dtype)
        for layer in layers {
            x = layer(x, mask: mask)
        }

        // Average pool over time (kernel 2, stride 2), then the final LayerNorm.
        let pooled = positions / 2
        x = x[0..., 0 ..< (pooled * 2), 0...]
        x = x.reshaped(x.dim(0), pooled, 2, x.dim(2)).mean(axis: 2)
        return layerNorm(x)
    }

    /// Additive mask `[batch, 1, 1, positions]`: `-inf` on positions past each row's valid length.
    private func attentionMask(featureMask: MLXArray?, batch: Int, positions: Int) -> MLXArray? {
        guard let featureMask else { return nil }
        let lengths = featureMask.asType(.int32).sum(axis: -1).asArray(Int.self)
        var values = [Float](repeating: 0, count: batch * positions)
        for (row, length) in lengths.enumerated() where row < batch {
            let convLength = length > 0 ? (length - 1) / 2 + 1 : 0
            for position in min(convLength, positions) ..< positions {
                values[row * positions + position] = -Float.infinity
            }
        }
        return MLXArray(values, [batch, 1, 1, positions])
    }
}

/// Maps the pooled audio hidden states into the text hidden size.
final class SharpAIAudioflamingo3Projector: Module {
    @ModuleInfo(key: "linear_1") var linear1: Linear
    @ModuleInfo(key: "linear_2") var linear2: Linear

    init(audioHidden: Int, textHidden: Int, bias: Bool) {
        _linear1.wrappedValue = Linear(audioHidden, textHidden, bias: bias)
        _linear2.wrappedValue = Linear(textHidden, textHidden, bias: bias)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        linear2(gelu(linear1(x)))
    }
}
