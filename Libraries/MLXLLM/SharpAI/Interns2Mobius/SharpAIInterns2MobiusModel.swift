// SharpAI fork overlay: interns2_mobius text model. Not part of upstream mlx-swift-lm.
//
// Port of InternS2MobiusTextModel and InternS2MobiusForCausalLM in internlm/Intern-S2-Mobius
// modeling_interns2_mobius.py (Verified against:
// https://huggingface.co/internlm/Intern-S2-Mobius/resolve/main/modeling_interns2_mobius.py).
//
// - linear_attention layers: GatedDeltaNet with separate in_proj_qkv / in_proj_z / in_proj_b / in_proj_a,
//   a causal depthwise conv over qkv, and the gated delta rule (MLXLMCommon `gatedDeltaUpdate`).
// - full_attention layers: q_proj emits a per-head query and an output gate, q/k RMSNorm with (1 + w),
//   partial RoPE (first head_dim * 0.25 channels), output multiplied by sigmoid(gate).
// - Every layer adds a routed MoE bank `meta_mlp[layer_idx % num_blocks]` plus a per-layer shared expert
//   gated by sigmoid(shared_expert_gate(x)).
// - Text-only path. Position ids are the same for the three MRoPE streams, so interleaved MRoPE reduces
//   to plain partial RoPE. The vision tower (`model.visual.*`) and the MTP head (`mtp.*`) are not ported.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - Norms

/// RMSNorm with a (1 + weight) scale and zero-initialized weight, computed in float32 and cast back.
final class SharpAIInterns2MobiusRMSNorm: Module {
    @ParameterInfo(key: "weight") var weight: MLXArray
    let eps: Float

    init(dimensions: Int, eps: Float) {
        self.eps = eps
        self._weight.wrappedValue = MLXArray.zeros([dimensions])
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let x32 = x.asType(.float32)
        let normed = x32 * rsqrt(mean(square(x32), axis: -1, keepDims: true) + MLXArray(eps))
        let scale = MLXArray(Float(1)) + weight.asType(.float32)
        return (normed * scale).asType(x.dtype)
    }
}

/// Gated RMSNorm used after the gated delta rule: weight * rmsnorm(x) * silu(gate), in float32.
final class SharpAIInterns2MobiusRMSNormGated: Module {
    @ParameterInfo(key: "weight") var weight: MLXArray
    let eps: Float

    init(dimensions: Int, eps: Float) {
        self.eps = eps
        self._weight.wrappedValue = MLXArray.ones([dimensions])
        super.init()
    }

    func callAsFunction(_ x: MLXArray, gate: MLXArray) -> MLXArray {
        let inputDType = x.dtype
        let x32 = x.asType(.float32)
        let normed = (x32 * rsqrt(mean(square(x32), axis: -1, keepDims: true) + MLXArray(eps)))
            .asType(inputDType)
        let scaled = weight * normed
        return (scaled.asType(.float32) * silu(gate.asType(.float32))).asType(inputDType)
    }
}

// MARK: - GatedDeltaNet (linear attention)

final class SharpAIInterns2MobiusGatedDeltaNet: Module {
    let numKHeads: Int
    let numVHeads: Int
    let headKDim: Int
    let headVDim: Int
    let keyDim: Int
    let valueDim: Int
    let convKernelSize: Int
    let convDim: Int

    @ModuleInfo(key: "conv1d") var conv1d: Conv1d
    @ModuleInfo(key: "in_proj_qkv") var inProjQKV: Linear
    @ModuleInfo(key: "in_proj_z") var inProjZ: Linear
    @ModuleInfo(key: "in_proj_b") var inProjB: Linear
    @ModuleInfo(key: "in_proj_a") var inProjA: Linear
    @ParameterInfo(key: "dt_bias") var dtBias: MLXArray
    @ParameterInfo(key: "A_log") var aLog: MLXArray
    @ModuleInfo(key: "norm") var norm: SharpAIInterns2MobiusRMSNormGated
    @ModuleInfo(key: "out_proj") var outProj: Linear

    init(_ config: SharpAIInterns2MobiusTextConfiguration) {
        numKHeads = config.linearNumKeyHeads
        numVHeads = config.linearNumValueHeads
        headKDim = config.linearKeyHeadDim
        headVDim = config.linearValueHeadDim
        keyDim = headKDim * numKHeads
        valueDim = headVDim * numVHeads
        convKernelSize = config.linearConvKernelDim
        convDim = keyDim * 2 + valueDim

        _inProjQKV.wrappedValue = Linear(config.hiddenSize, convDim, bias: false)
        _inProjZ.wrappedValue = Linear(config.hiddenSize, valueDim, bias: false)
        _inProjB.wrappedValue = Linear(config.hiddenSize, numVHeads, bias: false)
        _inProjA.wrappedValue = Linear(config.hiddenSize, numVHeads, bias: false)

        _conv1d.wrappedValue = Conv1d(
            inputChannels: convDim,
            outputChannels: convDim,
            kernelSize: convKernelSize,
            stride: 1,
            padding: 0,
            dilation: 1,
            groups: convDim,
            bias: false
        )

        // Placeholders. Checkpoint values replace them at load time.
        _dtBias.wrappedValue = MLXArray.ones([numVHeads])
        _aLog.wrappedValue = MLXArray.zeros([numVHeads])

        _norm.wrappedValue = SharpAIInterns2MobiusRMSNormGated(
            dimensions: headVDim, eps: config.rmsNormEps)
        _outProj.wrappedValue = Linear(valueDim, config.hiddenSize, bias: false)

        super.init()
    }

    func callAsFunction(_ x: MLXArray, cache: MambaCache?) -> MLXArray {
        let B = x.dim(0)
        let S = x.dim(1)
        let zeroConv = MLXArray.zeros([B, convKernelSize - 1, convDim], dtype: x.dtype)
        let (output, newConvState, newRecState) = forward(
            x,
            convState: cache?[0] ?? zeroConv,
            recState: cache?[1])
        if let cache {
            cache[0] = newConvState
            cache[1] = newRecState
            cache.advance(S)
        }
        return output
    }

    func forward(
        _ x: MLXArray, convState: MLXArray, recState: MLXArray?
    ) -> (output: MLXArray, convState: MLXArray, recState: MLXArray) {
        let B = x.dim(0)
        let S = x.dim(1)

        // Causal depthwise conv over qkv with the last K-1 inputs carried in the cache.
        let qkv = inProjQKV(x)
        let convInput = concatenated([convState, qkv], axis: 1)
        let convOut = silu(conv1d(convInput))
        let newConvState = contiguous(convInput[0..., (1 - convKernelSize)..., 0...])

        let parts = MLX.split(convOut, indices: [keyDim, 2 * keyDim], axis: -1)
        var q = parts[0].reshaped(B, S, numKHeads, headKDim)
        var k = parts[1].reshaped(B, S, numKHeads, headKDim)
        let v = parts[2].reshaped(B, S, numVHeads, headVDim)

        // HF: query = l2norm(query) * Dk^-0.5 and key = l2norm(key), with eps (1e-6) added to the sum of squares.
        let scale = pow(Float(headKDim), -0.5)
        q = l2NormalizeLastAxis(q, eps: 1e-6) * MLXArray(scale).asType(x.dtype)
        k = l2NormalizeLastAxis(k, eps: 1e-6)

        if numVHeads > numKHeads {
            let repeatFactor = numVHeads / numKHeads
            q = repeated(q, count: repeatFactor, axis: 2)
            k = repeated(k, count: repeatFactor, axis: 2)
        }

        // beta = sigmoid(b) and decay = exp(-exp(A_log) * softplus(a + dt_bias)) are computed inside
        // gatedDeltaUpdate from the raw a and b projections, as in HF.
        let (y, newRecState) = gatedDeltaUpdate(
            q: q,
            k: k,
            v: v,
            a: inProjA(x),
            b: inProjB(x),
            aLog: aLog,
            dtBias: dtBias,
            state: recState,
            mask: nil,
            useKernel: !training)

        let z = inProjZ(x).reshaped(B, S, numVHeads, headVDim)
        let normed = norm(y.reshaped(B, S, numVHeads, headVDim), gate: z)
        return (outProj(normed.reshaped(B, S, -1)), newConvState, newRecState)
    }

    private func l2NormalizeLastAxis(_ x: MLXArray, eps: Float) -> MLXArray {
        let x32 = x.asType(.float32)
        let inv = rsqrt(sum(x32 * x32, axis: -1, keepDims: true) + MLXArray(eps))
        return (x32 * inv).asType(x.dtype)
    }
}

// MARK: - Gated full attention

final class SharpAIInterns2MobiusAttention: Module {
    let nHeads: Int
    let nKVHeads: Int
    let headDim: Int
    let scale: Float
    let rope: RoPELayer

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: SharpAIInterns2MobiusRMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: SharpAIInterns2MobiusRMSNorm

    init(_ config: SharpAIInterns2MobiusTextConfiguration) {
        nHeads = config.attentionHeads
        nKVHeads = config.kvHeads
        headDim = config.headDim
        scale = pow(Float(config.headDim), -0.5)
        rope = RoPE(
            dimensions: config.rotaryDim, traditional: false, base: config.ropeTheta)

        // q_proj produces a query and an output gate per head (2 * head_dim per head).
        _qProj.wrappedValue = Linear(
            config.hiddenSize, config.attentionHeads * config.headDim * 2,
            bias: config.attentionBias)
        _kProj.wrappedValue = Linear(
            config.hiddenSize, config.kvHeads * config.headDim, bias: config.attentionBias)
        _vProj.wrappedValue = Linear(
            config.hiddenSize, config.kvHeads * config.headDim, bias: config.attentionBias)
        _oProj.wrappedValue = Linear(
            config.attentionHeads * config.headDim, config.hiddenSize, bias: config.attentionBias)
        _qNorm.wrappedValue = SharpAIInterns2MobiusRMSNorm(
            dimensions: config.headDim, eps: config.rmsNormEps)
        _kNorm.wrappedValue = SharpAIInterns2MobiusRMSNorm(
            dimensions: config.headDim, eps: config.rmsNormEps)

        super.init()
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))

        // Per head: first head_dim channels are the query, the next head_dim channels are the gate.
        let qFull = qProj(x).reshaped(B, L, nHeads, 2 * headDim)
        let halves = MLX.split(qFull, indices: [headDim], axis: -1)
        let gate = halves[1].reshaped(B, L, nHeads * headDim)

        var queries = qNorm(halves[0]).transposed(0, 2, 1, 3)
        var keys = kNorm(kProj(x).reshaped(B, L, nKVHeads, headDim)).transposed(0, 2, 1, 3)
        let values = vProj(x).reshaped(B, L, nKVHeads, headDim).transposed(0, 2, 1, 3)

        let offset = cache?.ropeOffset
        queries = applyRotaryPosition(rope, to: queries, offset: offset)
        keys = applyRotaryPosition(rope, to: keys, offset: offset)

        let output = attentionWithCacheUpdate(
            queries: queries,
            keys: keys,
            values: values,
            cache: cache,
            scale: scale,
            mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)

        return oProj(output * sigmoid(gate))
    }
}

// MARK: - MoE

/// Plain gated MLP used by the shared expert.
final class SharpAIInterns2MobiusMLP: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(hiddenSize: Int, intermediateSize: Int) {
        _gateProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        _upProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        _downProj.wrappedValue = Linear(intermediateSize, hiddenSize, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(silu(gateProj(x)) * upProj(x))
    }
}

/// One routed expert bank (`meta_mlp.<block>`): a router and 2560 experts (top-8 by default).
/// The checkpoint stores fused `experts.gate_up_proj` and `experts.down_proj`; sanitize splits them
/// into the `switch_mlp` layout that SwitchGLU expects.
final class SharpAIInterns2MobiusMetaMoE: Module {
    let topK: Int

    @ModuleInfo(key: "gate") var gate: Linear
    @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU

    init(_ config: SharpAIInterns2MobiusTextConfiguration) {
        topK = config.numExpertsPerTok
        _gate.wrappedValue = Linear(config.hiddenSize, config.numExperts, bias: false)
        _switchMLP.wrappedValue = SwitchGLU(
            inputDims: config.hiddenSize,
            hiddenDims: config.moeIntermediateSize,
            numExperts: config.numExperts
        )
        super.init()
    }

    /// `x` is [tokens, hidden]. Router: float32 softmax over all experts, top-k, renormalized over the top-k.
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let probs = MLX.softmax(gate(x).asType(.float32), axis: -1, precise: true)
        let (inds, scores) = moeRouterTopK(probs, k: topK, normalize: true)
        return switchMLP.callAndWeightedReduce(
            x, inds, weights: scores.asType(x.dtype), fuseSortedReduction: true)
    }
}

/// Per-layer shared expert, added to the routed bank output with a sigmoid gate.
final class SharpAIInterns2MobiusSharedExpertBlock: Module {
    @ModuleInfo(key: "shared_expert") var sharedExpert: SharpAIInterns2MobiusMLP
    @ModuleInfo(key: "shared_expert_gate") var sharedExpertGate: Linear

    init(_ config: SharpAIInterns2MobiusTextConfiguration) {
        _sharedExpert.wrappedValue = SharpAIInterns2MobiusMLP(
            hiddenSize: config.hiddenSize,
            intermediateSize: config.sharedExpertIntermediateSize)
        _sharedExpertGate.wrappedValue = Linear(config.hiddenSize, 1, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, expertOutput: MLXArray) -> MLXArray {
        expertOutput + sigmoid(sharedExpertGate(x)) * sharedExpert(x)
    }
}

// MARK: - Decoder layer

final class SharpAIInterns2MobiusDecoderLayer: Module {
    let isLinear: Bool

    @ModuleInfo(key: "linear_attn") var linearAttn: SharpAIInterns2MobiusGatedDeltaNet?
    @ModuleInfo(key: "self_attn") var selfAttn: SharpAIInterns2MobiusAttention?
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: SharpAIInterns2MobiusRMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm:
        SharpAIInterns2MobiusRMSNorm
    @ModuleInfo(key: "mlp") var mlp: SharpAIInterns2MobiusSharedExpertBlock

    init(_ config: SharpAIInterns2MobiusTextConfiguration, isLinear: Bool) {
        self.isLinear = isLinear

        _linearAttn.wrappedValue =
            isLinear ? SharpAIInterns2MobiusGatedDeltaNet(config) : nil
        _selfAttn.wrappedValue =
            isLinear ? nil : SharpAIInterns2MobiusAttention(config)
        _inputLayerNorm.wrappedValue = SharpAIInterns2MobiusRMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = SharpAIInterns2MobiusRMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _mlp.wrappedValue = SharpAIInterns2MobiusSharedExpertBlock(config)

        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        attentionMask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?,
        metaMoE: SharpAIInterns2MobiusMetaMoE
    ) -> MLXArray {
        let B = x.dim(0)
        let S = x.dim(1)

        let mixed: MLXArray
        if let linearAttn {
            mixed = linearAttn(inputLayerNorm(x), cache: cache as? MambaCache)
        } else if let selfAttn {
            mixed = selfAttn(inputLayerNorm(x), mask: attentionMask, cache: cache)
        } else {
            fatalError("interns2_mobius layer has neither an attention nor a linear block")
        }
        let h = x + mixed

        let flat = postAttentionLayerNorm(h).reshaped(B * S, -1)
        let routed = metaMoE(flat)
        let out = mlp(flat, expertOutput: routed).reshaped(B, S, -1)
        return h + out
    }
}

// MARK: - Backbone

final class SharpAIInterns2MobiusBackbone: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "layers") var layers: [SharpAIInterns2MobiusDecoderLayer]
    @ModuleInfo(key: "norm") var norm: SharpAIInterns2MobiusRMSNorm
    @ModuleInfo(key: "meta_mlp") var metaMLP: [SharpAIInterns2MobiusMetaMoE]

    let numBlocks: Int
    let fullAttentionIndex: Int
    let linearAttentionIndex: Int

    init(_ config: SharpAIInterns2MobiusTextConfiguration) {
        _embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize, dimensions: config.hiddenSize)
        _layers.wrappedValue = config.layerTypes.map { layerType in
            SharpAIInterns2MobiusDecoderLayer(config, isLinear: layerType == "linear_attention")
        }
        _norm.wrappedValue = SharpAIInterns2MobiusRMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _metaMLP.wrappedValue = (0 ..< config.numBlocks).map { _ in
            SharpAIInterns2MobiusMetaMoE(config)
        }
        numBlocks = config.numBlocks
        fullAttentionIndex = config.layerTypes.firstIndex(of: "full_attention") ?? 0
        linearAttentionIndex = config.layerTypes.firstIndex(of: "linear_attention") ?? 0

        super.init()
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        var h = embedTokens(inputs)

        let attentionMask = createAttentionMask(h: h, cache: cache?[fullAttentionIndex])

        for (i, layer) in layers.enumerated() {
            // Layer i uses bank i % num_blocks, as in HF `block_idx = layer_idx % num_blocks`.
            h = layer(
                h,
                attentionMask: attentionMask,
                cache: cache?[i],
                metaMoE: metaMLP[i % numBlocks])
        }

        return norm(h)
    }
}

// MARK: - Causal LM

public class SharpAIInterns2MobiusModel: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    let configuration: SharpAIInterns2MobiusTextConfiguration

    @ModuleInfo(key: "model") var model: SharpAIInterns2MobiusBackbone
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public init(_ configuration: SharpAIInterns2MobiusTextConfiguration) {
        self.configuration = configuration
        self.vocabularySize = configuration.vocabularySize
        self.kvHeads = (0 ..< configuration.hiddenLayers).map { _ in configuration.kvHeads }
        _model.wrappedValue = SharpAIInterns2MobiusBackbone(configuration)
        if !configuration.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(
                configuration.hiddenSize, configuration.vocabularySize, bias: false)
        }
        super.init()
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let out = model(inputs, cache: cache)
        if let lmHead {
            return lmHead(out)
        }
        return model.embedTokens.asLinear(out)
    }

    /// Linear layers get a MambaCache. Full-attention layers get the standard attention cache.
    public func newCache(parameters: GenerateParameters?) throws -> [KVCache] {
        var caches = [KVCache]()
        for layerType in configuration.layerTypes {
            if layerType == "linear_attention" {
                caches.append(MambaCache())
            } else {
                caches.append(try makeAttentionKVCache(parameters: parameters))
            }
        }
        return caches
    }

    public func makeCache() -> [KVCache] {
        var caches = [KVCache]()
        for layerType in configuration.layerTypes {
            if layerType == "linear_attention" {
                caches.append(MambaCache())
            } else {
                caches.append(KVCacheSimple())
            }
        }
        return caches
    }

    /// Maps HF checkpoint names to this module tree.
    /// - `model.language_model.X` becomes `model.X`. `lm_head.weight` is unchanged, or dropped
    ///   when `tie_word_embeddings` is true (the embedding is the output projection).
    /// - `meta_mlp.<b>.experts.gate_up_proj` [E, 2I, H] is split into `switch_mlp.gate_proj.weight`
    ///   and `switch_mlp.up_proj.weight` (gate first, as HF `chunk(2)`). `experts.down_proj` becomes
    ///   `switch_mlp.down_proj.weight`. The router `meta_mlp.<b>.gate.weight` is unchanged.
    /// - Conv1d weights [C, 1, K] are moved to MLX's [C, K, 1] layout.
    /// - `mtp.*`, `model.visual.*` and rotary buffers are dropped, since those parts are not ported.
    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        let languagePrefix = "model.language_model."
        var out: [String: MLXArray] = [:]
        let weights = filterLMHeadWeights(
            from: weights, tiedWordEmbeddings: configuration.tieWordEmbeddings)

        for (key, value) in weights {
            if key.hasPrefix("mtp.") || key.hasPrefix("model.visual.")
                || key.contains("rotary_emb.inv_freq")
            {
                continue
            }

            var newKey = key
            if newKey.hasPrefix(languagePrefix) {
                newKey = "model." + String(newKey.dropFirst(languagePrefix.count))
            }

            if newKey.hasSuffix(".experts.gate_up_proj") {
                let base = String(newKey.dropLast("experts.gate_up_proj".count))
                let half = value.dim(1) / 2
                out[base + "switch_mlp.gate_proj.weight"] = value[0..., ..<half, 0...]
                out[base + "switch_mlp.up_proj.weight"] = value[0..., half..., 0...]
                continue
            }
            if newKey.hasSuffix(".experts.down_proj") {
                let base = String(newKey.dropLast("experts.down_proj".count))
                out[base + "switch_mlp.down_proj.weight"] = value
                continue
            }

            if newKey.contains("conv1d.weight") && value.dim(-1) != 1 {
                out[newKey] = value.movedAxis(source: 2, destination: 1)
                continue
            }

            out[newKey] = value
        }
        return out
    }
}

extension SharpAIInterns2MobiusModel: LoRAModel {
    public var loraLayers: [Module] {
        model.layers
    }
}
