// SharpAI fork overlay: nemo_speechlm ASR transformer encoder. Not part of upstream mlx-swift-lm.
//
// Verified against: vLLM nemotron3_5_transcribe_audio.py (TransformerEncoder, TransformerBlock,
// MultiHeadAttention, FeatureStacking, RotaryEmbedding) and NeMo transformer_encoder.py
// (FeedForward uses nn.GELU, embed_norm and final_norm are LayerNorm, xscale is sqrt(d_model)).
// Checkpoint names: https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B
// (model.safetensors header, keys perception.encoder.asr_encoder.*).
//
// Scope: the ASR transformer only. The Sortformer speaker branch and the ParallelExpertEncoder
// fuse are not implemented here, so the output of this encoder is NOT the full audio tower
// embedding. See SharpAINemoSpeechlmRegistration.swift for what is still missing.

import Foundation
import MLX
import MLXNN

/// Hyperparameters of the ASR transformer encoder, named as in the checkpoint's
/// `perception.encoder` block (NeMo `TransformerEncoder` arguments).
public struct SharpAINemoSpeechlmEncoderConfiguration: Decodable, Sendable {
    public var featIn: Int
    public var dModel: Int
    public var nHeads: Int
    public var nLayers: Int
    public var ffExpansion: Float
    public var subsamplingFactor: Int
    public var qkvBias: Bool
    public var qkNorm: Bool
    public var preBlockNorm: Bool
    public var xscaling: Bool
    public var ropeBase: Float
    public var rotaryFraction: Float
    public var selfAttentionModel: String
    public var attnMode: String

    enum CodingKeys: String, CodingKey {
        case featIn = "feat_in"
        case dModel = "d_model"
        case nHeads = "n_heads"
        case nLayers = "n_layers"
        case ffExpansion = "ff_expansion"
        case subsamplingFactor = "subsampling_factor"
        case qkvBias = "qkv_bias"
        case qkNorm = "qk_norm"
        case preBlockNorm = "pre_block_norm"
        case xscaling
        case ropeBase = "rope_base"
        case rotaryFraction = "rotary_fraction"
        case selfAttentionModel = "self_attention_model"
        case attnMode = "attn_mode"
    }

    public init(
        featIn: Int, dModel: Int, nHeads: Int, nLayers: Int, ffExpansion: Float,
        subsamplingFactor: Int, qkvBias: Bool, qkNorm: Bool, preBlockNorm: Bool,
        xscaling: Bool = false, ropeBase: Float, rotaryFraction: Float,
        selfAttentionModel: String = "rope", attnMode: String = "full"
    ) {
        self.featIn = featIn
        self.dModel = dModel
        self.nHeads = nHeads
        self.nLayers = nLayers
        self.ffExpansion = ffExpansion
        self.subsamplingFactor = subsamplingFactor
        self.qkvBias = qkvBias
        self.qkNorm = qkNorm
        self.preBlockNorm = preBlockNorm
        self.xscaling = xscaling
        self.ropeBase = ropeBase
        self.rotaryFraction = rotaryFraction
        self.selfAttentionModel = selfAttentionModel
        self.attnMode = attnMode
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        featIn = try c.decode(Int.self, forKey: .featIn)
        dModel = try c.decode(Int.self, forKey: .dModel)
        nHeads = try c.decode(Int.self, forKey: .nHeads)
        nLayers = try c.decode(Int.self, forKey: .nLayers)
        ffExpansion = try c.decode(Float.self, forKey: .ffExpansion)
        subsamplingFactor = try c.decode(Int.self, forKey: .subsamplingFactor)
        qkvBias = try c.decode(Bool.self, forKey: .qkvBias)
        qkNorm = try c.decode(Bool.self, forKey: .qkNorm)
        preBlockNorm = try c.decode(Bool.self, forKey: .preBlockNorm)
        xscaling = try c.decodeIfPresent(Bool.self, forKey: .xscaling) ?? false
        ropeBase = try c.decode(Float.self, forKey: .ropeBase)
        rotaryFraction = try c.decode(Float.self, forKey: .rotaryFraction)
        selfAttentionModel = try c.decode(String.self, forKey: .selfAttentionModel)
        attnMode = try c.decode(String.self, forKey: .attnMode)
    }

    /// Rejects the variants this port does not implement. Only RoPE self-attention with full
    /// (unmasked-by-chunk) attention is ported, as the checkpoint uses.
    public func validate() throws {
        guard selfAttentionModel == "rope" else {
            throw SharpAINemoSpeechlmError.unsupportedEncoder(
                "self_attention_model \(selfAttentionModel) is not implemented (only rope)")
        }
        guard attnMode == "full" else {
            throw SharpAINemoSpeechlmError.unsupportedEncoder(
                "attn_mode \(attnMode) is not implemented (only full)")
        }
        guard nHeads > 0, dModel % nHeads == 0 else {
            throw SharpAINemoSpeechlmError.unsupportedEncoder(
                "d_model \(dModel) must be divisible by n_heads \(nHeads)")
        }
        let rotDim = Int(Float(dModel / nHeads) * rotaryFraction)
        guard rotDim >= 2, rotDim % 2 == 0, rotDim <= dModel / nHeads else {
            throw SharpAINemoSpeechlmError.unsupportedEncoder(
                "rotary dim \(rotDim) must be an even number in 2...head_dim")
        }
        guard subsamplingFactor > 0, featIn > 0, nLayers > 0 else {
            throw SharpAINemoSpeechlmError.unsupportedEncoder(
                "feat_in, subsampling_factor and n_layers must be positive")
        }
    }
}

/// NeMo `RotaryPositionalEncoding`: NeoX rotate-half over the first `rotDim` dims of each head.
/// The angle table is computed in float32 and cast to the activation dtype, as the reference does.
struct SharpAINemoSpeechlmRotary {
    let rotDim: Int
    let base: Float

    init(headDim: Int, rotaryFraction: Float, base: Float) {
        self.rotDim = Int(Float(headDim) * rotaryFraction)
        self.base = base
    }

    /// (length, rotDim) cos and sin tables. Position t uses `freqs = t * inv_freq` with
    /// `inv_freq[i] = base^(-2i / rotDim)`, and the table is `cat(freqs, freqs)`.
    func cosSin(length: Int, dtype: DType) -> (MLXArray, MLXArray) {
        let half = rotDim / 2
        var invFreq = [Float]()
        invFreq.reserveCapacity(half)
        for i in 0 ..< half {
            invFreq.append(Float(pow(Double(base), -Double(2 * i) / Double(rotDim))))
        }
        let positions = MLXArray((0 ..< length).map { Float($0) }, [length, 1])
        let freqs = positions * MLXArray(invFreq, [1, half])
        let emb = concatenated([freqs, freqs], axis: -1)
        return (MLX.cos(emb).asType(dtype), MLX.sin(emb).asType(dtype))
    }

    /// `x` is (batch, heads, time, headDim). Only the first `rotDim` dims are rotated.
    func apply(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
        func rotate(_ t: MLXArray) -> MLXArray {
            let half = t.dim(-1) / 2
            let rotHalf = concatenated(
                [MLX.negative(t[.ellipsis, half...]), t[.ellipsis, ..<half]], axis: -1)
            return t * cos + rotHalf * sin
        }
        if rotDim == x.dim(-1) {
            return rotate(x)
        }
        let head = rotate(x[.ellipsis, ..<rotDim])
        return concatenated([head, x[.ellipsis, rotDim...]], axis: -1)
    }
}

/// NeMo `FeatureStacking` pre-encode: stack `factor` consecutive frames, then project.
/// Checkpoint key: `pre_encode.proj.weight` with shape (d_model, factor * feat_in), no bias.
final class SharpAINemoSpeechlmFeatureStacking: Module {
    let factor: Int

    @ModuleInfo(key: "proj") var proj: Linear

    init(factor: Int, featIn: Int, dModel: Int) {
        self.factor = factor
        self._proj.wrappedValue = Linear(factor * featIn, dModel, bias: false)
        super.init()
    }

    /// (batch, time, featIn) -> (batch, time', dModel) and per-row lengths (ceil(len / factor)).
    func callAsFunction(_ x: MLXArray, lengths: [Int]) -> (MLXArray, [Int]) {
        let (b, t, c) = (x.dim(0), x.dim(1), x.dim(2))
        let pad = (factor - t % factor) % factor
        var padded = x
        if pad > 0 {
            padded = MLX.padded(x, widths: [0, .init((0, pad)), 0])
        }
        let stackedFrames = (t + pad) / factor
        let stacked = padded.reshaped(b, stackedFrames, c * factor)
        return (proj(stacked), lengths.map { ($0 + factor - 1) / factor })
    }
}

/// Two-layer feed-forward with GELU. Checkpoint keys: `ffn.net.0.*` and `ffn.net.3.*`.
/// The index-named layers mirror the NeMo `nn.Sequential` (Linear, GELU, Dropout, Linear).
final class SharpAINemoSpeechlmFeedForwardNet: Module {
    @ModuleInfo(key: "0") var first: Linear
    @ModuleInfo(key: "3") var second: Linear

    init(dModel: Int, hidden: Int) {
        self._first.wrappedValue = Linear(dModel, hidden, bias: true)
        self._second.wrappedValue = Linear(hidden, dModel, bias: true)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        second(gelu(first(x)))
    }
}

final class SharpAINemoSpeechlmFeedForward: Module {
    @ModuleInfo(key: "net") var net: SharpAINemoSpeechlmFeedForwardNet

    init(dModel: Int, hidden: Int) {
        self._net.wrappedValue = SharpAINemoSpeechlmFeedForwardNet(dModel: dModel, hidden: hidden)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        net(x)
    }
}

/// Full (non-causal) multi-head self-attention with RoPE and optional per-head LayerNorm
/// on queries and keys. Checkpoint keys: `attn.w_qkv`, `attn.out_proj`, `attn.q_norm`, `attn.k_norm`.
final class SharpAINemoSpeechlmAttention: Module {
    let nHeads: Int
    let headDim: Int
    let rotary: SharpAINemoSpeechlmRotary

    @ModuleInfo(key: "w_qkv") var wQKV: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: LayerNorm?
    @ModuleInfo(key: "k_norm") var kNorm: LayerNorm?

    init(_ config: SharpAINemoSpeechlmEncoderConfiguration, rotary: SharpAINemoSpeechlmRotary) {
        let hd = config.dModel / config.nHeads
        self.nHeads = config.nHeads
        self.headDim = hd
        self.rotary = rotary
        self._wQKV.wrappedValue = Linear(
            config.dModel, 3 * config.dModel, bias: config.qkvBias)
        self._outProj.wrappedValue = Linear(config.dModel, config.dModel, bias: true)
        if config.qkNorm {
            self._qNorm.wrappedValue = LayerNorm(dimensions: hd)
            self._kNorm.wrappedValue = LayerNorm(dimensions: hd)
        }
        super.init()
    }

    /// `x` is (batch, time, dModel). `keyMask` is (batch, 1, 1, time), true where a key may be attended.
    func callAsFunction(
        _ x: MLXArray, keyMask: MLXArray, cos: MLXArray, sin: MLXArray
    ) -> MLXArray {
        let (b, t) = (x.dim(0), x.dim(1))
        // (3, batch, heads, time, headDim), matching NeMo's view and permute(2, 0, 3, 1, 4).
        let qkv = wQKV(x).reshaped(b, t, 3, nHeads, headDim).transposed(2, 0, 3, 1, 4)
        var q = qkv[0]
        var k = qkv[1]
        let v = qkv[2]
        if let qNorm, let kNorm {
            q = qNorm(q)
            k = kNorm(k)
        }
        q = rotary.apply(q, cos: cos, sin: sin)
        k = rotary.apply(k, cos: cos, sin: sin)

        let out = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v,
            scale: 1.0 / Float(headDim).squareRoot(),
            mask: .array(keyMask))
        return outProj(out.transposed(0, 2, 1, 3).reshaped(b, t, -1))
    }
}

/// Pre-norm transformer block. Checkpoint keys: `norm1`, `attn.*`, `norm2`, `ffn.net.*`.
final class SharpAINemoSpeechlmEncoderLayer: Module {
    @ModuleInfo(key: "norm1") var norm1: LayerNorm
    @ModuleInfo(key: "attn") var attn: SharpAINemoSpeechlmAttention
    @ModuleInfo(key: "norm2") var norm2: LayerNorm
    @ModuleInfo(key: "ffn") var ffn: SharpAINemoSpeechlmFeedForward

    init(_ config: SharpAINemoSpeechlmEncoderConfiguration, rotary: SharpAINemoSpeechlmRotary) {
        self._norm1.wrappedValue = LayerNorm(dimensions: config.dModel)
        self._attn.wrappedValue = SharpAINemoSpeechlmAttention(config, rotary: rotary)
        self._norm2.wrappedValue = LayerNorm(dimensions: config.dModel)
        self._ffn.wrappedValue = SharpAINemoSpeechlmFeedForward(
            dModel: config.dModel, hidden: Int(config.ffExpansion * Float(config.dModel)))
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray, keyMask: MLXArray, cos: MLXArray, sin: MLXArray
    ) -> MLXArray {
        var h = x + attn(norm1(x), keyMask: keyMask, cos: cos, sin: sin)
        h = h + ffn(norm2(h))
        return h
    }
}

/// The NeMo ASR `TransformerEncoder` at inference: feature stacking, optional embedding norm,
/// the transformer blocks, and a final LayerNorm.
///
/// Input is (batch, time, feat_in) log-mel features, time-major. Output is
/// (batch, time', d_model) with per-row lengths `ceil(len / subsampling_factor)`. Padded keys are
/// masked, so padding does not affect valid frames.
public final class SharpAINemoSpeechlmAudioEncoder: Module {
    public let dModel: Int
    public let subsamplingFactor: Int
    let xscale: Float?
    let rotary: SharpAINemoSpeechlmRotary

    @ModuleInfo(key: "pre_encode") var preEncode: SharpAINemoSpeechlmFeatureStacking
    @ModuleInfo(key: "embed_norm") var embedNorm: LayerNorm?
    @ModuleInfo(key: "layers") var layers: [SharpAINemoSpeechlmEncoderLayer]
    @ModuleInfo(key: "final_norm") var finalNorm: LayerNorm

    public init(_ config: SharpAINemoSpeechlmEncoderConfiguration) throws {
        try config.validate()
        let rot = SharpAINemoSpeechlmRotary(
            headDim: config.dModel / config.nHeads,
            rotaryFraction: config.rotaryFraction, base: config.ropeBase)
        self.dModel = config.dModel
        self.subsamplingFactor = config.subsamplingFactor
        self.xscale = config.xscaling ? Float(config.dModel).squareRoot() : nil
        self.rotary = rot

        self._preEncode.wrappedValue = SharpAINemoSpeechlmFeatureStacking(
            factor: config.subsamplingFactor, featIn: config.featIn, dModel: config.dModel)
        if config.preBlockNorm {
            self._embedNorm.wrappedValue = LayerNorm(dimensions: config.dModel)
        }
        self._layers.wrappedValue = (0 ..< config.nLayers).map { _ in
            SharpAINemoSpeechlmEncoderLayer(config, rotary: rot)
        }
        self._finalNorm.wrappedValue = LayerNorm(dimensions: config.dModel)
        super.init()
    }

    /// `lengths` are the valid frame counts of each row before subsampling.
    public func callAsFunction(_ features: MLXArray, lengths: [Int]) -> (MLXArray, [Int]) {
        let stacked = preEncode(features, lengths: lengths)
        var x = stacked.0
        let outLengths = stacked.1
        if let xscale {
            x = x * xscale
        }
        if let embedNorm {
            x = embedNorm(x)
        }

        let (b, t) = (x.dim(0), x.dim(1))
        // A row with no valid frame attends to frame 0 so its output stays finite. It is discarded.
        let positions = MLXArray((0 ..< t).map { Int32($0) }, [1, t])
        let lengthColumn = MLXArray(outLengths.map { Int32($0) }, [b, 1])
        let validKey = MLX.less(positions, lengthColumn)
        let emptyRowFirstKey = MLX.logicalAnd(
            MLX.equal(positions, MLXArray(Int32(0))), MLX.equal(lengthColumn, MLXArray(Int32(0))))
        let keyMask = MLX.logicalOr(validKey, emptyRowFirstKey).reshaped(b, 1, 1, t)

        let (cos, sin) = rotary.cosSin(length: t, dtype: x.dtype)
        for layer in layers {
            x = layer(x, keyMask: keyMask, cos: cos, sin: sin)
        }
        return (finalNorm(x), outLengths)
    }
}
