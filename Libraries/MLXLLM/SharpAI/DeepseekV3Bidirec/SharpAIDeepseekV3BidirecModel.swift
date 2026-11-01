// SharpAI fork overlay: deepseek_v3_bidirec model. Not part of upstream mlx-swift-lm.
//
// Same module layout as DeepseekV3Model, with two differences:
//   - q_lora_rank is optional, so the attention takes the direct q_proj path when it is nil.
//   - Attention is non-causal. Every layer runs with no mask, so each position sees the
//     whole sequence (and the whole cache).
//
// UNVERIFIED: the bidirectional behaviour is inferred from the model name and spec. Confirm
// against the checkpoint's modeling code before relying on generation. Chunked prefill and
// KV-cache decoding change what a position can see, because later tokens are not in the cache yet.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

private func sharpAIBidirecClippedSilu(_ x: MLXArray) -> MLXArray {
    clip(x * sigmoid(x), min: -100, max: 100)
}

final class SharpAIDeepseekV3BidirecAttention: Module {
    let numHeads: Int
    let qLoraRank: Int?
    let qkRopeHeadDim: Int
    let kvLoraRank: Int
    let vHeadDim: Int
    let qkNopeHeadDim: Int
    let qHeadDim: Int
    let scale: Float
    let rope: RoPELayer

    @ModuleInfo(key: "q_proj") var qProj: Linear?
    @ModuleInfo(key: "q_a_proj") var qAProj: Linear?
    @ModuleInfo(key: "q_a_layernorm") var qALayerNorm: RMSNorm?
    @ModuleInfo(key: "q_b_proj") var qBProj: Linear?
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "kv_a_proj_with_mqa") var kvAProjWithMqa: Linear
    @ModuleInfo(key: "kv_a_layernorm") var kvALayerNorm: RMSNorm
    @ModuleInfo(key: "kv_b_proj") var kvBProj: Linear

    init(_ config: SharpAIDeepseekV3BidirecConfiguration) {
        let qHeadDim = config.qkNopeHeadDim + config.qkRopeHeadDim
        let heads = config.numAttentionHeads

        var scale = pow(Float(qHeadDim), -0.5)
        if let ropeScaling = config.ropeScaling {
            let mScaleAllDim = ropeScaling["mscale_all_dim"]?.asFloat() ?? 0.0
            if mScaleAllDim != 0 {
                let scalingFactor = ropeScaling["factor"]?.asFloat() ?? 1.0
                if scalingFactor > 1 {
                    let s = 0.1 * mScaleAllDim * log(scalingFactor) + 1.0
                    scale = scale * s * s
                }
            }
        }

        self.numHeads = heads
        self.qLoraRank = config.qLoraRank
        self.qkRopeHeadDim = config.qkRopeHeadDim
        self.kvLoraRank = config.kvLoraRank
        self.vHeadDim = config.vHeadDim
        self.qkNopeHeadDim = config.qkNopeHeadDim
        self.qHeadDim = qHeadDim
        self.scale = scale

        if let qLoraRank = config.qLoraRank {
            self._qAProj.wrappedValue = Linear(
                config.hiddenSize, qLoraRank, bias: config.attentionBias)
            self._qALayerNorm.wrappedValue = RMSNorm(dimensions: qLoraRank)
            self._qBProj.wrappedValue = Linear(qLoraRank, heads * qHeadDim, bias: false)
        } else {
            self._qProj.wrappedValue = Linear(config.hiddenSize, heads * qHeadDim, bias: false)
        }

        self._kvAProjWithMqa.wrappedValue = Linear(
            config.hiddenSize,
            config.kvLoraRank + config.qkRopeHeadDim,
            bias: config.attentionBias
        )
        self._kvALayerNorm.wrappedValue = RMSNorm(dimensions: config.kvLoraRank)
        self._kvBProj.wrappedValue = Linear(
            config.kvLoraRank,
            heads * (config.qkNopeHeadDim + config.vHeadDim),
            bias: false
        )
        self._oProj.wrappedValue = Linear(
            heads * config.vHeadDim, config.hiddenSize, bias: config.attentionBias)

        self.rope = initializeRope(
            dims: config.qkRopeHeadDim, base: config.ropeTheta, traditional: true,
            scalingConfig: config.ropeScaling, maxPositionEmbeddings: config.maxPositionEmbeddings)
    }

    /// Full (non-causal) attention. The mask is always `.none`.
    func callAsFunction(_ x: MLXArray, cache: KVCache?) -> MLXArray {
        let B = x.dim(0)
        let L = x.dim(1)

        let q: MLXArray
        if qLoraRank == nil {
            q = qProj!(x)
        } else {
            q = qBProj!(qALayerNorm!(qAProj!(x)))
        }

        let qHeads = q.reshaped(B, L, numHeads, qHeadDim).transposed(0, 2, 1, 3)
        let splitQ = split(qHeads, indices: [qkNopeHeadDim], axis: -1)
        let qNope = splitQ[0]
        var qPe = splitQ[1]

        let compressedKv = kvAProjWithMqa(x)
        let splitCompressedKv = split(compressedKv, indices: [kvLoraRank], axis: -1)
        let latentKv = splitCompressedKv[0]
        var kPe = splitCompressedKv[1]
        kPe = kPe.reshaped(B, L, 1, qkRopeHeadDim).transposed(0, 2, 1, 3)

        let kv = kvBProj(kvALayerNorm(latentKv))
            .reshaped(B, L, numHeads, -1)
            .transposed(0, 2, 1, 3)
        let splitKv = split(kv, indices: [qkNopeHeadDim], axis: -1)
        let kNope = splitKv[0]
        let values = splitKv[1]

        let offset = cache?.ropeOffset
        qPe = applyRotaryPosition(rope, to: qPe, offset: offset)
        kPe = applyRotaryPosition(rope, to: kPe, offset: offset)
        kPe = repeated(kPe, count: numHeads, axis: 1)

        let keys = concatenated([kNope, kPe], axis: -1)
        let queries = concatenated([qNope, qPe], axis: -1)

        let output = attentionWithCacheUpdate(
            queries: queries,
            keys: keys,
            values: values,
            cache: cache,
            scale: scale,
            mask: MLXFast.ScaledDotProductAttentionMaskMode.none
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)

        return oProj(output)
    }
}

final class SharpAIDeepseekV3BidirecMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(hiddenSize: Int, intermediateSize: Int) {
        self._gateProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        self._upProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        self._downProj.wrappedValue = Linear(intermediateSize, hiddenSize, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(silu(gateProj(x)) * upProj(x))
    }
}

/// Group-limited top-k router. Mirrors the sigmoid gate of DeepseekV3's MoEGate.
final class SharpAIDeepseekV3BidirecGate: Module {
    let topK: Int
    let normTopkProb: Bool
    let nRoutedExperts: Int
    let routedScalingFactor: Float
    let nGroup: Int
    let topkGroup: Int

    var weight: MLXArray
    var e_score_correction_bias: MLXArray

    init(_ config: SharpAIDeepseekV3BidirecConfiguration) {
        let experts = config.nRoutedExperts ?? 1
        self.topK = config.numExpertsPerTok ?? 1
        self.normTopkProb = config.normTopkProb
        self.routedScalingFactor = config.routedScalingFactor
        self.nGroup = config.nGroup ?? 1
        self.topkGroup = config.topkGroup ?? 1
        self.nRoutedExperts = experts
        self.weight = zeros([experts, config.hiddenSize])
        self.e_score_correction_bias = zeros([experts])
    }

    func callAsFunction(_ x: MLXArray) -> (MLXArray, MLXArray) {
        let bsz = x.dim(0)
        let seqLen = x.dim(1)

        let scores = sigmoid(x.matmul(weight.T))
        var choice = scores + e_score_correction_bias

        let dropGroups = nGroup - topkGroup
        if dropGroups > 0 {
            var groupScores = choice.reshaped(bsz, seqLen, nGroup, -1)
            let groupTop = top(groupScores, k: 2, axis: -1).sum(axis: -1, keepDims: true)
            let dropIdx = argPartition(groupTop, kth: dropGroups - 1, axis: -2)[
                .ellipsis, ..<dropGroups, 0...]
            let dropBroadcast = broadcast(
                dropIdx, to: [bsz, seqLen, dropGroups, nRoutedExperts / nGroup])
            groupScores = putAlong(
                groupScores, stopGradient(dropBroadcast), values: MLXArray(0.0), axis: -2)
            choice = flattened(groupScores, start: -2, end: -1)
        }

        let inds = argPartition(-choice, kth: topK - 1, axis: -1)[.ellipsis, ..<topK]
        var weights = takeAlong(scores, inds, axis: -1)
        if topK > 1, normTopkProb {
            let denominator = weights.sum(axis: -1, keepDims: true) + 1e-20
            weights = weights / denominator
        }
        weights = weights * MLXArray(routedScalingFactor)

        return (inds, weights)
    }
}

final class SharpAIDeepseekV3BidirecMoE: Module, UnaryLayer {
    let numExpertsPerTok: Int
    @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU
    var gate: SharpAIDeepseekV3BidirecGate
    @ModuleInfo(key: "shared_experts") var sharedExperts: SharpAIDeepseekV3BidirecMLP?

    init(_ config: SharpAIDeepseekV3BidirecConfiguration) {
        self.numExpertsPerTok = config.numExpertsPerTok ?? 1
        self._switchMLP.wrappedValue = SwitchGLU(
            inputDims: config.hiddenSize,
            hiddenDims: config.moeIntermediateSize,
            numExperts: config.nRoutedExperts ?? 1,
            activation: sharpAIBidirecClippedSilu
        )
        self.gate = SharpAIDeepseekV3BidirecGate(config)
        if let sharedCount = config.nSharedExperts {
            self._sharedExperts.wrappedValue = SharpAIDeepseekV3BidirecMLP(
                hiddenSize: config.hiddenSize,
                intermediateSize: config.moeIntermediateSize * sharedCount)
        }
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (indices, scores) = gate(x)
        var y = switchMLP(x, indices)
        y = weightedExpertSum(y, scores)
        if let shared = sharedExperts {
            y = y + shared(x)
        }
        return y
    }
}

final class SharpAIDeepseekV3BidirecDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: SharpAIDeepseekV3BidirecAttention
    var mlp: UnaryLayer
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    init(_ config: SharpAIDeepseekV3BidirecConfiguration, layerIdx: Int) {
        self._selfAttn.wrappedValue = SharpAIDeepseekV3BidirecAttention(config)

        if config.nRoutedExperts != nil,
            layerIdx >= config.firstKDenseReplace,
            layerIdx % config.moeLayerFreq == 0
        {
            self.mlp = SharpAIDeepseekV3BidirecMoE(config)
        } else {
            self.mlp = SharpAIDeepseekV3BidirecMLP(
                hiddenSize: config.hiddenSize, intermediateSize: config.intermediateSize)
        }

        self._inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    func callAsFunction(_ x: MLXArray, cache: KVCache?) -> MLXArray {
        let r = selfAttn(inputLayerNorm(x), cache: cache)
        let h = x + r
        let r2 = mlp(postAttentionLayerNorm(h))
        return h + r2
    }
}

final class SharpAIDeepseekV3BidirecModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    var layers: [SharpAIDeepseekV3BidirecDecoderLayer]
    @ModuleInfo(key: "norm") var norm: RMSNorm

    init(_ config: SharpAIDeepseekV3BidirecConfiguration) {
        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabSize, dimensions: config.hiddenSize)
        self.layers = (0 ..< config.numHiddenLayers).map {
            SharpAIDeepseekV3BidirecDecoderLayer(config, layerIdx: $0)
        }
        self._norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    func callAsFunction(_ x: MLXArray, cache: [KVCache]?) -> MLXArray {
        var h = embedTokens(x)
        for (i, layer) in layers.enumerated() {
            h = layer(h, cache: cache?[i])
        }
        return norm(h)
    }
}

final class SharpAIDeepseekV3BidirecModel: Module, LLMModel, KVCacheDimensionProvider, LoRAModel {
    var kvHeads: [Int]

    let args: SharpAIDeepseekV3BidirecConfiguration
    var model: SharpAIDeepseekV3BidirecModelInner
    @ModuleInfo(key: "lm_head") var lmHead: Linear

    init(_ args: SharpAIDeepseekV3BidirecConfiguration) {
        self.args = args
        self.kvHeads = Array(repeating: args.numKeyValueHeads, count: args.numHiddenLayers)
        self.model = SharpAIDeepseekV3BidirecModelInner(args)
        self._lmHead.wrappedValue = Linear(args.hiddenSize, args.vocabSize, bias: false)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        lmHead(model(inputs, cache: cache))
    }

    func sanitize(weights: [String: MLXArray]) throws -> [String: MLXArray] {
        try SharpAIDeepseekV3BidirecSanitizer.sanitize(weights, configuration: args)
    }

    var loraLayers: [Module] {
        model.layers
    }
}
