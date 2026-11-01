// SharpAI fork overlay: axk1 (A.X K1) model. Not part of upstream mlx-swift-lm.
//
// Reuses the DeepSeek-V3 attention, dense MLP and MoE blocks. Differences from V3, taken
// from the axk1 gap list:
// - post_mlp_layernorm exists only on MoE layers.
// - e_score_correction_bias may be absent from the checkpoint; sanitize fills zeros.
//
// UNVERIFIED: the placement of post_mlp_layernorm (applied to the MLP output before the
// residual add) is assumed. No reference implementation was available to check it.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// True when a layer uses the MoE block. Mirrors the rule in `DeepseekV3DecoderLayer`.
func sharpAIAxk1IsMoELayer(_ layerIdx: Int, config: DeepseekV3Configuration) -> Bool {
    config.nRoutedExperts != nil
        && layerIdx >= config.firstKDenseReplace
        && layerIdx % config.moeLayerFreq == 0
}

class SharpAIAxk1DecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: DeepseekV3Attention
    var mlp: UnaryLayer
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm
    @ModuleInfo(key: "post_mlp_layernorm") var postMlpLayerNorm: RMSNorm?

    init(config: DeepseekV3Configuration, layerIdx: Int) {
        self._selfAttn.wrappedValue = DeepseekV3Attention(config: config)

        if sharpAIAxk1IsMoELayer(layerIdx, config: config) {
            self.mlp = DeepseekV3MoE(config: config)
            self._postMlpLayerNorm.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize, eps: config.rmsNormEps)
        } else {
            self.mlp = DeepseekV3MLP(config: config)
            self._postMlpLayerNorm.wrappedValue = nil
        }

        self._inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let r = selfAttn(inputLayerNorm(x), mask: mask, cache: cache)
        let h = x + r
        var r2 = mlp(postAttentionLayerNorm(h))
        if let norm = postMlpLayerNorm {
            r2 = norm(r2)
        }
        return h + r2
    }
}

public class SharpAIAxk1ModelInner: Module, LayerPartitionable, StreamableMoE {
    var config: DeepseekV3Configuration
    var vocabSize: Int
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    var layers: [SharpAIAxk1DecoderLayer]
    var startIdx: Int
    var endIdx: Int
    var numLayers: Int
    @ModuleInfo(key: "norm") var norm: RMSNorm
    var pipelineRank: Int
    var pipelineSize: Int

    public var gpuLayerCount: Int? = nil
    public var streamExperts: Bool = false
    public var totalLayerCount: Int { layers.count }

    init(config: DeepseekV3Configuration) {
        self.config = config
        self.vocabSize = config.vocabSize
        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabSize, dimensions: config.hiddenSize)
        self.layers = (0 ..< config.numHiddenLayers).map {
            SharpAIAxk1DecoderLayer(config: config, layerIdx: $0)
        }
        self.startIdx = 0
        self.endIdx = layers.count
        self.numLayers = endIdx
        self._norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self.pipelineRank = 0
        self.pipelineSize = 1
    }

    func callAsFunction(_ x: MLXArray, cache: [KVCache]?) -> MLXArray {
        var h = embedTokens(x)

        let attentionMask = createAttentionMask(h: h, cache: cache?.first)

        for (i, layer) in layers.enumerated() {
            h = partitionedLayerCall(index: i, gpuLayerCount: gpuLayerCount, stream: streamExperts) {
                layer(h, mask: attentionMask, cache: cache?[i])
            }
        }

        return norm(h)
    }
}

public class SharpAIAxk1Model: Module, LLMModel, KVCacheDimensionProvider, LoRAModel {
    public var kvHeads: [Int]

    var args: DeepseekV3Configuration
    public var model: SharpAIAxk1ModelInner
    @ModuleInfo(key: "lm_head") var lmHead: Linear

    public init(_ configuration: SharpAIAxk1Configuration) {
        let args = configuration.base
        self.args = args
        self.kvHeads = Array(repeating: args.numKeyValueHeads, count: args.numHiddenLayers)
        self.model = SharpAIAxk1ModelInner(config: args)
        self._lmHead.wrappedValue = Linear(args.hiddenSize, args.vocabSize, bias: false)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        let out = model(inputs, cache: cache)
        return lmHead(out)
    }

    /// Maps HF checkpoint keys to module keys.
    ///
    /// - Dequantizes FP8 `weight_scale_inv` pairs (block size 128).
    /// - Stacks per-expert MoE weights into `switch_mlp` on MoE layers.
    /// - Fills zeros for a missing `e_score_correction_bias` on MoE layers.
    /// - Drops MTP layers past `numHiddenLayers` and `rotary_emb.inv_freq`.
    ///
    /// UNVERIFIED: the MTP drop and the expert key names assume the V3 layout.
    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var newWeights = weights

        func dequant(weight: MLXArray, scaleInv: MLXArray) -> MLXArray {
            let bs = 128
            let (m, n) = (weight.dim(0), weight.dim(1))
            let padBottom = (bs - m % bs) % bs
            let padSide = (bs - n % bs) % bs

            var paddedWeight = padded(weight, widths: [.init((0, padBottom)), .init((0, padSide))])
            paddedWeight = paddedWeight.reshaped([(m + padBottom) / bs, bs, (n + padSide) / bs, bs])
            let scaled = paddedWeight * scaleInv[0..., .newAxis, 0..., .newAxis]
            return scaled.reshaped([m + padBottom, n + padSide])[0 ..< m, 0 ..< n]
        }

        for (key, value) in weights where key.contains("weight_scale_inv") {
            let weightKey = key.replacingOccurrences(of: "_scale_inv", with: "")
            if let weight = weights[weightKey] {
                newWeights[weightKey] = dequant(weight: weight, scaleInv: value)
                newWeights[key] = nil
            }
        }

        for l in 0 ..< args.numHiddenLayers where sharpAIAxk1IsMoELayer(l, config: args) {
            let prefix = "model.layers.\(l)"
            for projName in ["gate_proj", "down_proj", "up_proj"] {
                for suffix in ["weight", "scales", "biases"] {
                    let firstKey = "\(prefix).mlp.experts.0.\(projName).\(suffix)"
                    guard newWeights[firstKey] != nil else { continue }
                    let expertCount = args.nRoutedExperts ?? 1
                    let expertKeys = (0 ..< expertCount).map {
                        "\(prefix).mlp.experts.\($0).\(projName).\(suffix)"
                    }
                    let joined = expertKeys.map { newWeights[$0]! }
                    newWeights["\(prefix).mlp.switch_mlp.\(projName).\(suffix)"] = stacked(joined)
                    for expertKey in expertKeys {
                        newWeights[expertKey] = nil
                    }
                }
            }

            let biasKey = "\(prefix).mlp.gate.e_score_correction_bias"
            if newWeights[biasKey] == nil {
                newWeights[biasKey] = MLXArray.zeros([args.nRoutedExperts ?? 1])
            }
        }

        return newWeights.filter { key, _ in
            !isMultiTokenPredictionKey(key) && !key.contains("rotary_emb.inv_freq")
        }
    }

    private func isMultiTokenPredictionKey(_ key: String) -> Bool {
        guard key.starts(with: "model.layers.") else { return false }
        let parts = key.split(separator: ".")
        guard parts.count >= 3, let layerIdx = Int(parts[2]) else { return false }
        return layerIdx >= args.numHiddenLayers
    }

    public var loraLayers: [Module] {
        model.layers
    }
}
