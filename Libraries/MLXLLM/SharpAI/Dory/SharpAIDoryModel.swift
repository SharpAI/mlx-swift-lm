// SharpAI fork overlay: Dory text model. Not part of upstream mlx-swift-lm.
//
// Port of the vLLM Dory model (vllm/model_executor/models/dory.py, Apache-2.0), read from the
// upstream file on GitHub for this change. Structure:
//   - no RMSNorm. Every sublayer output is combined with its residual by a slerp on the unit
//     sphere (NGPT). justnorm is x / sqrt(sum(x^2) + 1e-6) over the last axis.
//   - the stack is input layers, then the recurrent layers run n_recurrent_loops times with the
//     input reinjected each loop, then output layers. Each loop gets its own KV cache.
//   - sublayers are attention (sliding or full) or a SiLU MLP. Attention applies RoPE (NeoX
//     style, full head_dim), then qk scaling, then an output gate, then justnorm before o_proj.
//   - logits = lm_head(h) * (logit_scale / initializer_range).
//
// Verified against: https://raw.githubusercontent.com/vllm-project/vllm/b5007e42b09c3990685fe564a31ae6917394a80b/vllm/model_executor/models/dory.py
// (justnorm and slerp residuals, sqk and s_gate factors, suv_gate/suv_up MLP scaling, the
// input/recurrent/output loop with justnorm(state + input_h), per-loop KV caches, and
// logit_scale / initializer_range at the output).
//
// UNVERIFIED: the attention and MLP alpha parameter names (attn_alpha, mlp_alpha) and the
// sqk/s_gate/suv parameter names are taken from the vLLM module tree, not from a checkpoint
// index. No checkpoint was downloaded (the 4B weights are at revision b366b87f...), and no
// reference logits exist in the repo. Parameters the reference leaves uninitialized are set to
// ones here, and a real checkpoint overwrites them. Run the reference-match test before relying
// on this model.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// NGPT normalization: x * rsqrt(sum(x^2) + eps) over the last axis, accumulated in fp32.
func doryJustNorm(_ x: MLXArray) -> MLXArray {
    let xf = x.asType(.float32)
    let y = xf * rsqrt(xf.square().sum(axis: -1, keepDims: true) + MLXArray(Float(1e-6)))
    return y.asType(x.dtype)
}

/// slerp_alpha from the reference: justnorm(ya + lr * (yb - ya)) with lr = |alpha * alphaScale|.
func doryslerpResidual(_ a: MLXArray, _ b: MLXArray, alpha: MLXArray, alphaScale: Float)
    -> MLXArray
{
    let lr = (alpha.asType(.float32) * alphaScale).abs()
    let ya = doryJustNorm(a).asType(.float32)
    let yb = doryJustNorm(b).asType(.float32)
    return doryJustNorm(ya + lr * (yb - ya)).asType(a.dtype)
}

final class SharpAIDoryAttention: Module {
    let numHeads: Int
    let numKVHeads: Int
    let headDim: Int
    let scale: Float
    let sqkFactor: Float
    let sGateFactor: Float
    let rope: RoPELayer

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "g_proj") var gProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ParameterInfo(key: "sqk") var sqk: MLXArray
    @ParameterInfo(key: "s_gate") var sGate: MLXArray
    @ParameterInfo(key: "attn_alpha") var alpha: MLXArray

    init(_ config: SharpAIDoryConfiguration, ropeTheta: Float) {
        let heads = config.attentionHeads
        let kvHeads = config.kvHeads
        let headDim = config.headDim
        let qSize = heads * headDim

        self.numHeads = heads
        self.numKVHeads = kvHeads
        self.headDim = headDim
        self.scale = 1 / Float(headDim).squareRoot()

        // reference: std = initializer_range; sqk factor = sqrt(head_dim) / std;
        // g0 = attention_output_gate_logit_scale_init, or sqrt(hidden) when null;
        // s_gate factor = sqrt(g0) / std.
        let std = config.initializerRange
        self.sqkFactor = Float(headDim).squareRoot() / std
        let g0 = config.attentionOutputGateLogitScaleInit ?? Float(config.hiddenSize).squareRoot()
        self.sGateFactor = g0.squareRoot() / std

        _qProj.wrappedValue = Linear(config.hiddenSize, qSize, bias: config.attentionBias)
        _kProj.wrappedValue = Linear(config.hiddenSize, kvHeads * headDim, bias: config.attentionBias)
        _vProj.wrappedValue = Linear(config.hiddenSize, kvHeads * headDim, bias: config.attentionBias)
        _gProj.wrappedValue = Linear(config.hiddenSize, qSize, bias: false)
        _oProj.wrappedValue = Linear(qSize, config.hiddenSize, bias: config.attentionBias)
        _sqk.wrappedValue = MLXArray.ones([qSize])
        _sGate.wrappedValue = MLXArray.ones([qSize])
        _alpha.wrappedValue = MLXArray.ones([config.hiddenSize])

        self.rope = initializeRope(
            dims: headDim, base: ropeTheta, traditional: false,
            scalingConfig: nil, maxPositionEmbeddings: config.maxPositionEmbeddings)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))

        var q = qProj(x).reshaped(B, L, numHeads, headDim).transposed(0, 2, 1, 3)
        var k = kProj(x).reshaped(B, L, numKVHeads, headDim).transposed(0, 2, 1, 3)
        let v = vProj(x).reshaped(B, L, numKVHeads, headDim).transposed(0, 2, 1, 3)
        let gate = gProj(x)

        let offset = cache?.ropeOffset
        q = applyRotaryPosition(rope, to: q, offset: offset)
        k = applyRotaryPosition(rope, to: k, offset: offset)

        // q = (sqk * sqrt(head_dim) / std)^2 * justnorm(q), per head and channel.
        // k = justnorm(k).
        let sqkSquared = (sqk.asType(.float32) * sqkFactor).square()
            .reshaped(1, numHeads, 1, headDim)
        q = (doryJustNorm(q).asType(.float32) * sqkSquared).asType(q.dtype)
        k = doryJustNorm(k)

        let attended = attentionWithCacheUpdate(
            queries: q, keys: k, values: v, cache: cache, scale: scale, mask: mask)

        // [B, H, L, d] -> [B, L, H*d]
        var out = attended.transposed(0, 2, 1, 3).reshaped(B, L, -1)

        // out = out * sigmoid(gate * (s_gate * factor)^2), then justnorm before o_proj.
        let sg = (sGate.asType(.float32) * sGateFactor).square()
        out = (out.asType(.float32) * sigmoid(gate.asType(.float32) * sg)).asType(out.dtype)
        return oProj(doryJustNorm(out))
    }
}

final class SharpAIDoryMLP: Module {
    let suvFactor: Float

    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear
    @ParameterInfo(key: "suv_gate") var suvGate: MLXArray
    @ParameterInfo(key: "suv_up") var suvUp: MLXArray
    @ParameterInfo(key: "mlp_alpha") var alpha: MLXArray

    init(_ config: SharpAIDoryConfiguration) {
        self.suvFactor = Float(config.hiddenSize).squareRoot()
        _gateProj.wrappedValue = Linear(
            config.hiddenSize, config.intermediateSize, bias: config.mlpBias)
        _upProj.wrappedValue = Linear(
            config.hiddenSize, config.intermediateSize, bias: config.mlpBias)
        _downProj.wrappedValue = Linear(
            config.intermediateSize, config.hiddenSize, bias: config.mlpBias)
        _suvGate.wrappedValue = MLXArray.ones([config.intermediateSize])
        _suvUp.wrappedValue = MLXArray.ones([config.intermediateSize])
        _alpha.wrappedValue = MLXArray.ones([config.hiddenSize])
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // g = gate * (suv_gate * sqrt(hidden)), u = up * (suv_up * sqrt(hidden)),
        // m = justnorm(silu(g) * u), out = down(m).
        // Explicit MLXArray scalar: `MLXArray * Float` does not type-check against the mlx-swift pinned by the SwiftLM umbrella.
        let factor = MLXArray(suvFactor)
        let gateScale: MLXArray = suvGate.asType(.float32) * factor
        let upScale: MLXArray = suvUp.asType(.float32) * factor
        let g: MLXArray = gateProj(x).asType(.float32) * gateScale
        let u: MLXArray = upProj(x).asType(.float32) * upScale
        let m = doryJustNorm((silu(g) * u).asType(x.dtype))
        return downProj(m)
    }
}

final class SharpAIDoryDecoderLayer: Module {
    let isAttention: Bool
    let slidingWindow: Int?
    let alphaScale: Float

    @ModuleInfo(key: "self_attn") var attention: SharpAIDoryAttention?
    @ModuleInfo(key: "mlp") var mlp: SharpAIDoryMLP?

    init(_ config: SharpAIDoryConfiguration, layerType: String, ropeTheta: Float) {
        // reference: alpha_scale = ngpt_alpha_init / initializer_range
        self.alphaScale = config.ngptAlphaInit / config.initializerRange
        if layerType == "mlp" {
            self.isAttention = false
            self.slidingWindow = nil
            _attention.wrappedValue = nil
            _mlp.wrappedValue = SharpAIDoryMLP(config)
        } else {
            self.isAttention = true
            self.slidingWindow =
                layerType == "sliding_attention" ? config.slidingWindow : nil
            _attention.wrappedValue = SharpAIDoryAttention(config, ropeTheta: ropeTheta)
            _mlp.wrappedValue = nil
        }
    }

    func callAsFunction(
        _ h: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        if let attention {
            return doryslerpResidual(
                h, attention(h, mask: mask, cache: cache),
                alpha: attention.alpha, alphaScale: alphaScale)
        }
        if let mlp {
            return doryslerpResidual(h, mlp(h), alpha: mlp.alpha, alphaScale: alphaScale)
        }
        preconditionFailure("SharpAIDoryDecoderLayer has neither attention nor mlp")
    }
}

final class SharpAIDoryBackbone: Module {
    let nInput: Int
    let nRecurrent: Int
    let nLoops: Int

    /// `cacheIndex[layer][loop]` is the position in the cache array for that sublayer and loop.
    /// Input and output sublayers use entry 0 only. MLP sublayers have no entry.
    let cacheIndex: [[Int]]

    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    let layers: [SharpAIDoryDecoderLayer]

    init(_ config: SharpAIDoryConfiguration, slots: [SharpAIDoryCacheSlot]) throws {
        self.nInput = config.nInputLayers
        self.nRecurrent = config.nRecurrentLayers
        self.nLoops = config.nRecurrentLoops

        var table = Array(repeating: [Int](), count: config.hiddenLayers)
        for (index, slot) in slots.enumerated() {
            table[slot.sublayer].append(index)
        }
        self.cacheIndex = table

        _embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize, dimensions: config.hiddenSize)

        self.layers = try config.layerTypes.map { type -> SharpAIDoryDecoderLayer in
            let theta: Float
            if type == "mlp" {
                theta = 0
            } else {
                guard let rope = config.ropeParameters[type] else {
                    throw SharpAIDoryError.invalidConfiguration("missing rope for \(type)")
                }
                theta = rope.ropeTheta
            }
            return SharpAIDoryDecoderLayer(config, layerType: type, ropeTheta: theta)
        }
    }

    /// One sublayer. The cache is looked up by sublayer and loop, so each loop has its own cache.
    private func runLayer(_ i: Int, _ h: MLXArray, loop: Int, cache: [KVCache]?) -> MLXArray {
        let layer = layers[i]
        guard layer.isAttention else {
            return layer(h, mask: .none, cache: nil)
        }

        let isRecurrent = i >= nInput && i < nInput + nRecurrent
        let position = isRecurrent ? loop : 0
        let index = cacheIndex[i][position]
        let layerCache: KVCache? = cache.flatMap { $0.indices.contains(index) ? $0[index] : nil }

        let mask = createAttentionMask(h: h, cache: layerCache, windowSize: layer.slidingWindow)
        return layer(h, mask: mask, cache: layerCache)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        var h = embedTokens(inputs)

        for i in 0 ..< nInput {
            h = runLayer(i, h, loop: 0, cache: cache)
        }
        let inputH = h

        // reference: state = input_h; per loop: h = justnorm(state + input_h), run recurrent
        // layers, state = justnorm(h).
        var state = inputH
        for loop in 0 ..< nLoops {
            h = doryJustNorm(state + inputH)
            for i in nInput ..< (nInput + nRecurrent) {
                h = runLayer(i, h, loop: loop, cache: cache)
            }
            state = doryJustNorm(h)
        }

        h = state
        for i in (nInput + nRecurrent) ..< layers.count {
            h = runLayer(i, h, loop: 0, cache: cache)
        }
        return h
    }
}

/// Output projection. The checkpoint stores `lm_head.weight` and `lm_head.logit_scale`.
final class SharpAIDoryLMHead: Module {
    let outputScale: Float

    @ParameterInfo(key: "weight") var weight: MLXArray
    @ParameterInfo(key: "logit_scale") var logitScale: MLXArray

    init(_ config: SharpAIDoryConfiguration) {
        // reference: logits * (logit_scale / initializer_range)
        self.outputScale = 1 / config.initializerRange
        _weight.wrappedValue =
            MLXRandom.normal([config.vocabularySize, config.hiddenSize]) * Float(0.02)
        _logitScale.wrappedValue = MLXArray.ones([config.vocabularySize])
    }

    func callAsFunction(_ h: MLXArray) -> MLXArray {
        let logits = h.asType(.float32).matmul(weight.T.asType(.float32))
        return logits * (logitScale.asType(.float32) * outputScale)
    }
}

/// Dory text model for `model_type` "dory".
final class SharpAIDoryForCausalLM: Module, LLMModel {
    let configuration: SharpAIDoryConfiguration
    let slots: [SharpAIDoryCacheSlot]
    let model: SharpAIDoryBackbone

    @ModuleInfo(key: "lm_head") var lmHead: SharpAIDoryLMHead

    init(_ configuration: SharpAIDoryConfiguration) throws {
        try configuration.validateModelConfiguration()
        let slots = try SharpAIDoryCachePlan.slots(for: configuration)
        self.configuration = configuration
        self.slots = slots
        self.model = try SharpAIDoryBackbone(configuration, slots: slots)
        _lmHead.wrappedValue = SharpAIDoryLMHead(configuration)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        lmHead(model(inputs, cache: cache))
    }

    /// Drops rotary buffers. No key rename yet: the checkpoint key map is unverified.
    func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        SharpAIDorySanitizer.sanitize(weights)
    }

    /// One cache per slot from SharpAIDoryCachePlan, in the same order as `cacheIndex`.
    /// Sliding sublayers get a window cache. Full sublayers get a full cache.
    func newCache(parameters: GenerateParameters?) throws -> [KVCache] {
        try slots.map { (slot: SharpAIDoryCacheSlot) throws -> KVCache in
            if slot.kind == .sliding {
                guard let window = configuration.slidingWindow else {
                    throw SharpAIDoryError.invalidConfiguration(
                        "sliding_attention layers need sliding_window")
                }
                return try makeSlidingWindowKVCache(parameters: parameters, window: window)
            }
            return try makeAttentionKVCache(parameters: parameters)
        }
    }
}

extension SharpAIDoryForCausalLM: LoRAModel {
    var loraLayers: [Module] {
        model.layers
    }
}
