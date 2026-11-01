//
//  SharpAIIquestloopcoderModel.swift
//  MLXLLM
//
//  SharpAI fork overlay for `iquestloopcoder` (IQuestLoopCoderForCausalLM). Not upstream.
//
//  Math follows the IQuestLab reference (modeling_iquestloopcoder.py) and the vLLM port
//  (iquest_loopcoder.py), both Apache/Modified-MIT, re-implemented here in MLX Swift:
//
//  - The decoder stack runs `loop_num` times over the same layers.
//  - Pass 0 is ordinary causal attention. Its keys and values go to a shared, full-history
//    cache per layer ("global" context).
//  - Pass 1+ computes two attentions for each layer and mixes them with a per-head gate:
//      local  = causal sliding-window attention over this pass's own keys/values
//               (window = loop_window_size tokens, including the current token)
//      global = causal attention over the shared pass-0 keys/values
//      gate   = sigmoid(einsum("bhsd,hd->bhs", rope(q), W) + b), W: [heads, head_dim]
//      out    = local * (1 - gate) + global * gate
//  - Final RMSNorm, then the LM head.
//
//  Cache layout (count = layers * loop_num):
//    index i                                  shared (pass 0) cache of layer i, KVCacheSimple
//    index layers + (loop - 1) * layers + i   local window cache of pass `loop` (1...), RotatingKVCache
//

import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// Per-head sigmoid gate computed from the rotated query (IQuestLoopCoder `LoopGateProjection`).
final class SharpAIIquestloopcoderLoopGate: Module {
    /// [numHeads, headDim]
    @ParameterInfo(key: "weight") var weight: MLXArray
    /// [numHeads]
    @ParameterInfo(key: "bias") var bias: MLXArray

    /// Reference initialization is zeros, which gives gate = 0.5 until a checkpoint is loaded.
    init(numHeads: Int, headDim: Int) {
        self._weight.wrappedValue = MLXArray.zeros([numHeads, headDim])
        self._bias.wrappedValue = MLXArray.zeros([numHeads])
    }

    init(weight: MLXArray, bias: MLXArray) {
        self._weight.wrappedValue = weight
        self._bias.wrappedValue = bias
    }

    /// - Parameter queries: `[B, H, L, D]`, after RoPE.
    /// - Returns: gate `[B, H, L, 1]` with values in (0, 1).
    func callAsFunction(_ queries: MLXArray) -> MLXArray {
        let heads = weight.dim(0)
        let headDim = weight.dim(1)
        let w = weight.reshaped(1, heads, 1, headDim)
        let logits = (queries * w).sum(axis: -1) + bias.reshaped(1, heads, 1)
        return sigmoid(logits).expandedDimensions(axis: -1)
    }

    /// `local * (1 - gate) + global * gate`. Shapes broadcast as `[B, H, L, D]` against `[B, H, L, 1]`.
    static func mix(local: MLXArray, global: MLXArray, gate: MLXArray) -> MLXArray {
        local * (1 - gate) + global * gate
    }
}

final class SharpAIIquestloopcoderAttention: Module {
    let numHeads: Int
    let numKVHeads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var wq: Linear
    @ModuleInfo(key: "k_proj") var wk: Linear
    @ModuleInfo(key: "v_proj") var wv: Linear
    @ModuleInfo(key: "o_proj") var wo: Linear

    let rope: RoPELayer

    init(_ c: SharpAIIquestloopcoderConfiguration) {
        numHeads = c.attentionHeads
        numKVHeads = c.kvHeads
        headDim = c.headDim
        scale = pow(Float(headDim), -0.5)

        self._wq.wrappedValue = Linear(c.hiddenSize, numHeads * headDim, bias: c.attentionBias)
        self._wk.wrappedValue = Linear(c.hiddenSize, numKVHeads * headDim, bias: c.attentionBias)
        self._wv.wrappedValue = Linear(c.hiddenSize, numKVHeads * headDim, bias: c.attentionBias)
        self._wo.wrappedValue = Linear(numHeads * headDim, c.hiddenSize, bias: c.attentionBias)

        self.rope = initializeRope(
            dims: headDim, base: c.ropeTheta, traditional: false,
            scalingConfig: nil, maxPositionEmbeddings: c.maxPositionEmbeddings)
    }

    /// - Parameters:
    ///   - ropeOffset: absolute position of the first new token. Captured once per forward,
    ///     before any pass updates the caches.
    ///   - shared: this layer's pass-0 cache.
    ///   - sharedMask: mask for pass 0, built before the shared cache is updated.
    ///   - local: this layer's window cache for `loop` (nil for pass 0).
    ///   - localMask: sliding-window mask for `local`, built before its update.
    ///   - gate: this layer's loop gate (nil for pass 0).
    func callAsFunction(
        _ x: MLXArray,
        loop: Int,
        ropeOffset: RoPEOffset?,
        shared: KVCache,
        sharedMask: MLXFast.ScaledDotProductAttentionMaskMode,
        local: KVCache?,
        localMask: MLXFast.ScaledDotProductAttentionMaskMode,
        gate: SharpAIIquestloopcoderLoopGate?
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))

        var queries = wq(x).reshaped(B, L, numHeads, -1).transposed(0, 2, 1, 3)
        var keys = wk(x).reshaped(B, L, numKVHeads, -1).transposed(0, 2, 1, 3)
        let values = wv(x).reshaped(B, L, numKVHeads, -1).transposed(0, 2, 1, 3)

        queries = applyRotaryPosition(rope, to: queries, offset: ropeOffset)
        keys = applyRotaryPosition(rope, to: keys, offset: ropeOffset)

        let output: MLXArray
        if loop == 0 {
            // Pass 0: ordinary causal attention; the shared cache receives this call's keys/values.
            output = attentionWithCacheUpdate(
                queries: queries, keys: keys, values: values,
                cache: shared, scale: scale, mask: sharedMask)
        } else {
            precondition(local != nil && gate != nil, "iquestloopcoder: loop \(loop) needs a window cache and a gate")

            // Local: sliding-window attention over this pass's own keys/values (cache updated here).
            let localOut = attentionWithCacheUpdate(
                queries: queries, keys: keys, values: values,
                cache: local, scale: scale, mask: localMask)

            // Global: causal attention over all pass-0 keys/values. Pass 0 already appended
            // this call's tokens, so the cache holds `total` positions and the new tokens are
            // the last `L` of them.
            let sharedState = shared.state
            let sharedKeys = sharedState[0]
            let sharedValues = sharedState[1]
            let total = sharedKeys.dim(2)
            let globalMask: MLXFast.ScaledDotProductAttentionMaskMode =
                L > 1 ? .array(createCausalMask(n: L, offset: total - L)) : .none
            let globalOut = attentionWithCacheUpdate(
                queries: queries, keys: sharedKeys, values: sharedValues,
                cache: nil, scale: scale, mask: globalMask)

            output = SharpAIIquestloopcoderLoopGate.mix(
                local: localOut, global: globalOut, gate: gate!(queries))
        }

        return wo(output.transposed(0, 2, 1, 3).reshaped(B, L, -1))
    }
}

final class SharpAIIquestloopcoderMLP: Module {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "down_proj") var down: Linear
    @ModuleInfo(key: "up_proj") var up: Linear

    init(_ c: SharpAIIquestloopcoderConfiguration) {
        self._gate.wrappedValue = Linear(c.hiddenSize, c.intermediateSize, bias: c.mlpBias)
        self._down.wrappedValue = Linear(c.intermediateSize, c.hiddenSize, bias: c.mlpBias)
        self._up.wrappedValue = Linear(c.hiddenSize, c.intermediateSize, bias: c.mlpBias)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        down(silu(gate(x)) * up(x))
    }
}

final class SharpAIIquestloopcoderDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var attention: SharpAIIquestloopcoderAttention
    @ModuleInfo(key: "mlp") var mlp: SharpAIIquestloopcoderMLP

    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    init(_ c: SharpAIIquestloopcoderConfiguration) {
        self._attention.wrappedValue = SharpAIIquestloopcoderAttention(c)
        self._mlp.wrappedValue = SharpAIIquestloopcoderMLP(c)
        self._inputLayerNorm.wrappedValue = RMSNorm(dimensions: c.hiddenSize, eps: c.rmsNormEps)
        self._postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: c.hiddenSize, eps: c.rmsNormEps)
    }

    func callAsFunction(
        _ x: MLXArray,
        loop: Int,
        ropeOffset: RoPEOffset?,
        shared: KVCache,
        sharedMask: MLXFast.ScaledDotProductAttentionMaskMode,
        local: KVCache?,
        localMask: MLXFast.ScaledDotProductAttentionMaskMode,
        gate: SharpAIIquestloopcoderLoopGate?
    ) -> MLXArray {
        let r = attention(
            inputLayerNorm(x), loop: loop, ropeOffset: ropeOffset,
            shared: shared, sharedMask: sharedMask,
            local: local, localMask: localMask, gate: gate)
        let h = x + r
        return h + mlp(postAttentionLayerNorm(h))
    }
}

final class SharpAIIquestloopcoderInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding

    let layers: [SharpAIIquestloopcoderDecoderLayer]
    /// One gate per layer (`model.gate_projections.{i}`). Pass 0 does not use them.
    @ModuleInfo(key: "gate_projections") var gateProjections: [SharpAIIquestloopcoderLoopGate]
    let norm: RMSNorm

    let loopNum: Int
    let loopWindowSize: Int

    init(_ c: SharpAIIquestloopcoderConfiguration) {
        precondition(c.vocabularySize > 0)

        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: c.vocabularySize, dimensions: c.hiddenSize)
        self.layers = (0 ..< c.hiddenLayers).map { _ in SharpAIIquestloopcoderDecoderLayer(c) }
        self._gateProjections.wrappedValue = (0 ..< c.hiddenLayers).map { _ in
            SharpAIIquestloopcoderLoopGate(numHeads: c.attentionHeads, headDim: c.headDim)
        }
        self.norm = RMSNorm(dimensions: c.hiddenSize, eps: c.rmsNormEps)
        self.loopNum = c.loopNum
        self.loopWindowSize = c.loopWindowSize
    }

    /// Cache index of the window cache for pass `loop` (>= 1) and layer `layer`.
    func localCacheIndex(loop: Int, layer: Int) -> Int {
        layers.count + (loop - 1) * layers.count + layer
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]) -> MLXArray {
        var h = embedTokens(inputs)

        // Captured before any pass writes to the caches, so every pass uses the same positions.
        let ropeOffset: RoPEOffset? = cache[0].ropeOffset

        // Pass 0 mask: built from the pre-call shared cache, identical for every layer.
        let sharedMask = createAttentionMask(h: h, cache: cache[0])

        for loop in 0 ..< loopNum {
            // Window masks depend on each pass's own pre-call cache, which is the same across layers.
            let localMask: MLXFast.ScaledDotProductAttentionMaskMode =
                loop == 0
                ? .none
                : createAttentionMask(
                    h: h, cache: cache[localCacheIndex(loop: loop, layer: 0)],
                    windowSize: loopWindowSize)

            for (i, layer) in layers.enumerated() {
                h = layer(
                    h,
                    loop: loop,
                    ropeOffset: ropeOffset,
                    shared: cache[i],
                    sharedMask: sharedMask,
                    local: loop == 0 ? nil : cache[localCacheIndex(loop: loop, layer: i)],
                    localMask: localMask,
                    gate: loop == 0 ? nil : gateProjections[i])
            }
        }

        return norm(h)
    }
}

/// Model for the `iquestloopcoder` model type (IQuestLoopCoderForCausalLM).
public class SharpAIIquestloopcoderModel: Module, LLMModel {

    public let vocabularySize: Int

    let model: SharpAIIquestloopcoderInner
    let configuration: SharpAIIquestloopcoderConfiguration

    @ModuleInfo(key: "lm_head") public var lmHead: Linear?

    public init(_ args: SharpAIIquestloopcoderConfiguration) {
        self.vocabularySize = args.vocabularySize
        self.configuration = args
        self.model = SharpAIIquestloopcoderInner(args)
        if !args.tieWordEmbeddings {
            self._lmHead.wrappedValue = Linear(args.hiddenSize, args.vocabularySize, bias: false)
        }
    }

    /// Layout described at the top of this file. Count is `layers * loop_num`.
    public func newCache(parameters: GenerateParameters?) throws -> [KVCache] {
        makeCaches()
    }

    private func makeCaches() -> [KVCache] {
        let layers = configuration.hiddenLayers
        var caches: [KVCache] = (0 ..< layers).map { _ in KVCacheSimple() }
        for _ in 1 ..< max(configuration.loopNum, 1) {
            caches += (0 ..< layers).map { _ in
                RotatingKVCache(maxSize: configuration.loopWindowSize)
            }
        }
        return caches
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let caches = cache ?? makeCaches()
        let out = model(inputs, cache: caches)
        if let lmHead {
            return lmHead(out)
        } else {
            return model.embedTokens.asLinear(out)
        }
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        SharpAIIquestloopcoderWeights.sanitize(
            weights: weights, tieWordEmbeddings: configuration.tieWordEmbeddings)
    }

    public func messageGenerator(tokenizer: any Tokenizer) -> any MessageGenerator {
        DefaultMessageGenerator()
    }
}

// MARK: - LoRA

extension SharpAIIquestloopcoderModel: LoRAModel {
    public var loraLayers: [Module] {
        model.layers
    }
}
