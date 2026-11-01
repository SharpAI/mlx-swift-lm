//
//  SharpAIVaultgemmaModel.swift
//  MLXLLM
//
//  Text model for `vaultgemma` (VaultGemmaForCausalLM), a Gemma 2 layout with two changes.
//  Each block has no post-attention or post-feedforward norm. Even layers use a sliding
//  window and odd layers use full attention. Attention and final logits are soft-capped.
//  Weight keys follow the HF checkpoint names.
//  UNVERIFIED: no real checkpoint was run. Softcap and mask behaviour follow the HF eager path.
//

import Foundation
import MLX
import MLXLMCommon
import MLXNN

class SharpAIVaultgemmaAttention: Module {
    let nHeads: Int
    let nKVHeads: Int
    let headDim: Int
    let scale: Float
    let logitSoftcapping: Float?

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear

    let rope: RoPE

    init(_ config: SharpAIVaultgemmaConfiguration) {
        self.nHeads = config.attentionHeads
        self.nKVHeads = config.kvHeads
        self.headDim = config.headDim
        self.scale = pow(config.queryPreAttnScalar, -0.5)
        self.logitSoftcapping = config.attnLogitSoftcapping

        let dim = config.hiddenSize
        _qProj.wrappedValue = Linear(dim, nHeads * headDim, bias: config.attentionBias)
        _kProj.wrappedValue = Linear(dim, nKVHeads * headDim, bias: config.attentionBias)
        _vProj.wrappedValue = Linear(dim, nKVHeads * headDim, bias: config.attentionBias)
        _oProj.wrappedValue = Linear(nHeads * headDim, dim, bias: config.attentionBias)

        self.rope = RoPE(dimensions: headDim, traditional: false, base: config.ropeTheta)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))

        var queries = qProj(x).reshaped(B, L, nHeads, -1).transposed(0, 2, 1, 3)
        var keys = kProj(x).reshaped(B, L, nKVHeads, -1).transposed(0, 2, 1, 3)
        var values = vProj(x).reshaped(B, L, nKVHeads, -1).transposed(0, 2, 1, 3)

        let offset = cache?.ropeOffset
        queries = applyRotaryPosition(rope, to: queries, offset: offset)
        keys = applyRotaryPosition(rope, to: keys, offset: offset)

        if let quantizedCache = cache as? QuantizedKVCacheProtocol {
            // Softcapped scores need the full matrices, so dequantize the cached keys and values.
            let (qKeys, qValues) = quantizedCache.updateQuantized(keys: keys, values: values)
            keys = dequantized(
                qKeys.0, scales: qKeys.1, biases: qKeys.2,
                groupSize: quantizedCache.groupSize, bits: quantizedCache.bits,
                mode: quantizedCache.mode, dtype: queries.dtype)
            values = dequantized(
                qValues.0, scales: qValues.1, biases: qValues.2,
                groupSize: quantizedCache.groupSize, bits: quantizedCache.bits,
                mode: quantizedCache.mode, dtype: queries.dtype)
        } else if let cache {
            (keys, values) = cache.update(keys: keys, values: values)
        }

        let repeats = nHeads / nKVHeads
        if repeats > 1 {
            keys = MLX.repeated(keys, count: repeats, axis: 1)
            values = MLX.repeated(values, count: repeats, axis: 1)
        }

        var scores = matmul(queries * scale, keys.swappedAxes(-1, -2))
        if let logitSoftcapping {
            scores = tanh(scores / logitSoftcapping) * logitSoftcapping
        }
        scores = applyAttentionMask(scores: scores, mask: mask)
        let weights = softmax(scores, axis: -1, precise: true)

        let output = matmul(weights, values)
        return oProj(output.transposed(0, 2, 1, 3).reshaped(B, L, -1))
    }
}

class SharpAIVaultgemmaMLP: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    let approximateGelu: Bool

    init(_ config: SharpAIVaultgemmaConfiguration) {
        _gateProj.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        _upProj.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        _downProj.wrappedValue = Linear(config.intermediateSize, config.hiddenSize, bias: false)
        self.approximateGelu = config.hiddenAct == "gelu_pytorch_tanh"
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let gate = gateProj(x)
        let activated = approximateGelu ? geluApproximate(gate) : gelu(gate)
        return downProj(activated * upProj(x))
    }
}

/// Pre-norm block with no post-attention or post-feedforward norm.
class SharpAIVaultgemmaDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var attention: SharpAIVaultgemmaAttention
    @ModuleInfo(key: "mlp") var mlp: SharpAIVaultgemmaMLP
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: Gemma.RMSNorm
    @ModuleInfo(key: "pre_feedforward_layernorm") var preFeedforwardLayerNorm: Gemma.RMSNorm

    /// Nil for full-attention layers, and for sliding layers when the config has no window.
    let slidingWindow: Int?

    init(_ config: SharpAIVaultgemmaConfiguration, layerIndex: Int) {
        _attention.wrappedValue = SharpAIVaultgemmaAttention(config)
        _mlp.wrappedValue = SharpAIVaultgemmaMLP(config)
        _inputLayerNorm.wrappedValue = Gemma.RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _preFeedforwardLayerNorm.wrappedValue = Gemma.RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self.slidingWindow =
            config.isSlidingLayer(layerIndex) ? config.slidingWindow : nil
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let h = x + attention(inputLayerNorm(x), mask: mask, cache: cache)
        return h + mlp(preFeedforwardLayerNorm(h))
    }
}

class SharpAIVaultgemmaModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    let layers: [SharpAIVaultgemmaDecoderLayer]
    let norm: Gemma.RMSNorm
    let hiddenScale: Float

    init(_ config: SharpAIVaultgemmaConfiguration) {
        precondition(config.vocabularySize > 0)

        _embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize, dimensions: config.hiddenSize)
        self.layers = (0 ..< config.hiddenLayers).map {
            SharpAIVaultgemmaDecoderLayer(config, layerIndex: $0)
        }
        self.norm = Gemma.RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self.hiddenScale = pow(Float(config.hiddenSize), 0.5)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var h = embedTokens(inputs)
        h = h * MLXArray(hiddenScale, dtype: h.dtype)

        for (i, layer) in layers.enumerated() {
            // Each layer builds its own mask. Sliding layers use their window.
            let layerCache = cache?[i]
            let mask = createAttentionMask(h: h, cache: layerCache, windowSize: layer.slidingWindow)
            h = layer(h, mask: mask, cache: layerCache)
        }

        return norm(h)
    }
}

class SharpAIVaultgemmaModel: Module, LLMModel, KVCacheDimensionProvider {
    let vocabularySize: Int
    let kvHeads: [Int]
    let model: SharpAIVaultgemmaModelInner
    let configuration: SharpAIVaultgemmaConfiguration

    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    init(_ config: SharpAIVaultgemmaConfiguration) {
        self.configuration = config
        self.vocabularySize = config.vocabularySize
        self.kvHeads = (0 ..< config.hiddenLayers).map { _ in config.kvHeads }
        self.model = SharpAIVaultgemmaModelInner(config)

        if !config.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabularySize, bias: false)
        }
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let out = model(inputs, cache: cache)

        var logits: MLXArray
        if let lmHead {
            logits = lmHead(out)
        } else {
            logits = model.embedTokens.asLinear(out)
        }

        if let cap = configuration.finalLogitSoftcapping {
            logits = tanh(logits / cap) * cap
        }
        return logits
    }

    func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        SharpAIVaultgemmaSanitizer.sanitize(weights, configuration: configuration)
    }

    /// Sliding layers get a rotating cache of the window size. Full layers get a normal cache.
    func newCache(parameters: GenerateParameters?) throws -> [KVCache] {
        try model.layers.map { (layer) throws -> KVCache in
            if let window = layer.slidingWindow {
                return try makeSlidingWindowKVCache(parameters: parameters, window: window)
            }
            return try makeAttentionKVCache(parameters: parameters)
        }
    }

    func messageGenerator(tokenizer: any Tokenizer) -> any MessageGenerator {
        NoSystemMessageGenerator()
    }
}

extension SharpAIVaultgemmaModel: LoRAModel {
    var loraLayers: [Module] {
        model.layers
    }
}
