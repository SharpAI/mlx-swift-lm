// SharpAI fork overlay: qwen3_asr audio tower (transformers Qwen3ASREncoder). Not part of
// upstream mlx-swift-lm.
//
// Input is log-mel features shaped (1, mel, frames). Frames must be a multiple of 2 * n_window.
// Each chunk of 2 * n_window frames goes through three stride-2 convs. Attention runs inside
// windows of n_window_infer frames (after the conv stem), not across the whole utterance.
// Batch size 1 only. Feature extraction is not part of this file.

import Foundation
import MLX
import MLXNN

/// Length helpers that mirror transformers' Qwen3ASR audio code.
public enum SharpAIQwen3AsrAudioMath {

    /// Python-style floor division. Swift's `/` rounds toward zero, which differs for negatives.
    static func floorDiv(_ a: Int, _ b: Int) -> Int {
        let q = a / b
        return (a % b != 0 && ((a < 0) != (b < 0))) ? q - 1 : q
    }

    /// Output length after the three stride-2, kernel-3 convs. Zero stays zero.
    public static func postConvLength(_ frames: Int) -> Int {
        var length = max(frames, 0)
        for _ in 0 ..< 3 {
            length = length > 0 ? (length - 1) / 2 + 1 : 0
        }
        return length
    }

    /// Encoder rows for a whole utterance with `frames` valid mel frames.
    /// Each full chunk gives 13 rows. The partial tail gives `postConvLength` rows.
    public static func audioTokenCount(frames: Int, nWindow: Int) -> Int {
        let chunk = nWindow * 2
        let leave = frames % chunk
        let feat = floorDiv(leave - 1, 2) + 1
        return floorDiv(floorDiv(feat - 1, 2) + 1 - 1, 2) + 1 + (frames / chunk) * 13
    }

    /// Splits `total` encoder rows into attention windows of `windowTokens` rows. The last one
    /// may be shorter.
    public static func windowBounds(total: Int, windowTokens: Int) -> [(Int, Int)] {
        guard total > 0, windowTokens > 0 else { return [] }
        var bounds: [(Int, Int)] = []
        var start = 0
        while start < total {
            let end = min(start + windowTokens, total)
            bounds.append((start, end))
            start = end
        }
        return bounds
    }
}

/// Sinusoid table with sin in the first half of the channels and cos in the second.
/// Matches `SinusoidsPositionEmbedding` in transformers.
func sharpAIQwen3AsrSinusoids(length: Int, channels: Int) -> MLXArray {
    let half = channels / 2
    let increment = log(10_000.0) / Double(half - 1)
    var values = [Float](repeating: 0, count: length * channels)
    for position in 0 ..< length {
        for i in 0 ..< half {
            let angle = Double(position) * exp(-increment * Double(i))
            values[position * channels + i] = Float(sin(angle))
            values[position * channels + half + i] = Float(cos(angle))
        }
    }
    return MLXArray(values, [length, channels])
}

final class SharpAIQwen3AsrAudioAttention: Module {

    let heads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear

    init(dModel: Int, heads: Int) {
        self.heads = heads
        self.headDim = dModel / heads
        self.scale = pow(Float(dModel / heads), -0.5)
        self._qProj.wrappedValue = Linear(dModel, dModel, bias: true)
        self._kProj.wrappedValue = Linear(dModel, dModel, bias: true)
        self._vProj.wrappedValue = Linear(dModel, dModel, bias: true)
        self._outProj.wrappedValue = Linear(dModel, dModel, bias: true)
        super.init()
    }

    /// `x` is (tokens, d_model). Each window attends only to itself.
    func callAsFunction(_ x: MLXArray, windows: [(Int, Int)]) -> MLXArray {
        let n = x.dim(0)
        let q = qProj(x).reshaped(1, n, heads, headDim).transposed(0, 2, 1, 3)
        let k = kProj(x).reshaped(1, n, heads, headDim).transposed(0, 2, 1, 3)
        let v = vProj(x).reshaped(1, n, heads, headDim).transposed(0, 2, 1, 3)

        var outputs: [MLXArray] = []
        for (start, end) in windows {
            outputs.append(
                MLXFast.scaledDotProductAttention(
                    queries: q[0..., 0..., start ..< end, 0...],
                    keys: k[0..., 0..., start ..< end, 0...],
                    values: v[0..., 0..., start ..< end, 0...],
                    scale: scale,
                    mask: MLXFast.ScaledDotProductAttentionMaskMode.none))
        }
        let joined = MLX.concatenated(outputs, axis: 2)
        return outProj(joined.transposed(0, 2, 1, 3).reshaped(n, -1))
    }
}

final class SharpAIQwen3AsrAudioEncoderLayer: Module {

    @ModuleInfo(key: "self_attn") var selfAttn: SharpAIQwen3AsrAudioAttention
    @ModuleInfo(key: "self_attn_layer_norm") var selfAttnLayerNorm: LayerNorm
    @ModuleInfo(key: "fc1") var fc1: Linear
    @ModuleInfo(key: "fc2") var fc2: Linear
    @ModuleInfo(key: "final_layer_norm") var finalLayerNorm: LayerNorm

    init(_ config: SharpAIQwen3AsrAudioConfiguration) {
        self._selfAttn.wrappedValue = SharpAIQwen3AsrAudioAttention(
            dModel: config.dModel, heads: config.encoderAttentionHeads)
        self._selfAttnLayerNorm.wrappedValue = LayerNorm(dimensions: config.dModel, eps: 1e-5)
        self._fc1.wrappedValue = Linear(config.dModel, config.encoderFFNDim, bias: true)
        self._fc2.wrappedValue = Linear(config.encoderFFNDim, config.dModel, bias: true)
        self._finalLayerNorm.wrappedValue = LayerNorm(dimensions: config.dModel, eps: 1e-5)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, windows: [(Int, Int)]) -> MLXArray {
        var h = x + selfAttn(selfAttnLayerNorm(x), windows: windows)
        h = h + fc2(gelu(fc1(finalLayerNorm(h))))
        return h
    }
}

/// Conv stem, transformer encoder, and the two-layer projector (`proj1`, `proj2`).
/// The projector names match the hub checkpoint. Transformers calls them `multi_modal_projector`.
public final class SharpAIQwen3AsrAudioEncoder: Module {

    let config: SharpAIQwen3AsrAudioConfiguration

    @ModuleInfo(key: "conv2d1") var conv2d1: Conv2d
    @ModuleInfo(key: "conv2d2") var conv2d2: Conv2d
    @ModuleInfo(key: "conv2d3") var conv2d3: Conv2d
    @ModuleInfo(key: "conv_out") var convOut: Linear
    @ModuleInfo(key: "layers") var layers: [SharpAIQwen3AsrAudioEncoderLayer]
    @ModuleInfo(key: "ln_post") var lnPost: LayerNorm
    @ModuleInfo(key: "proj1") var proj1: Linear
    @ModuleInfo(key: "proj2") var proj2: Linear

    public init(_ config: SharpAIQwen3AsrAudioConfiguration) {
        self.config = config
        let channels = config.downsampleHiddenSize
        self._conv2d1.wrappedValue = Conv2d(
            inputChannels: 1, outputChannels: channels, kernelSize: IntOrPair(3),
            stride: IntOrPair(2), padding: IntOrPair(1))
        self._conv2d2.wrappedValue = Conv2d(
            inputChannels: channels, outputChannels: channels, kernelSize: IntOrPair(3),
            stride: IntOrPair(2), padding: IntOrPair(1))
        self._conv2d3.wrappedValue = Conv2d(
            inputChannels: channels, outputChannels: channels, kernelSize: IntOrPair(3),
            stride: IntOrPair(2), padding: IntOrPair(1))
        // Frequency bins after three stride-2 convs, e.g. 128 -> 16.
        var freq = config.numMelBins
        for _ in 0 ..< 3 {
            freq = (freq - 1) / 2 + 1
        }
        self._convOut.wrappedValue = Linear(channels * freq, config.dModel, bias: false)
        self._layers.wrappedValue = (0 ..< config.encoderLayers).map { _ in
            SharpAIQwen3AsrAudioEncoderLayer(config)
        }
        self._lnPost.wrappedValue = LayerNorm(dimensions: config.dModel, eps: 1e-5)
        self._proj1.wrappedValue = Linear(config.dModel, config.dModel, bias: true)
        self._proj2.wrappedValue = Linear(config.dModel, config.outputDim, bias: true)
        super.init()
    }

    /// `features` is (1, mel, frames) with frames a multiple of 2 * n_window.
    /// `validFrames` counts the real frames before padding.
    /// Returns (rows, output_dim), one row per audio placeholder token in the prompt.
    public func callAsFunction(_ features: MLXArray, validFrames: Int) throws -> MLXArray {
        let batch = features.dim(0)
        guard batch == 1 else {
            throw SharpAIQwen3AsrError.unsupportedBatch(batch)
        }
        let mel = features.dim(1)
        let frames = features.dim(2)
        let chunk = config.nWindow * 2
        guard frames > 0, frames % chunk == 0 else {
            throw SharpAIQwen3AsrError.invalidInput(
                "audio frames (\(frames)) must be a positive multiple of \(chunk)")
        }
        let numChunks = frames / chunk
        let valid = min(max(validFrames, 0), frames)

        // (1, mel, chunks, chunk) -> (chunks, mel, chunk, 1), channels last.
        let chunked = features.reshaped(1, mel, numChunks, chunk)
            .transposed(0, 2, 1, 3)
            .reshaped(numChunks, mel, chunk, 1)

        var x = gelu(conv2d1(chunked))
        x = gelu(conv2d2(x))
        x = gelu(conv2d3(x))

        // (chunks, freq, time, channels) -> (chunks, time, channels * freq), channels first.
        let (freq, time, channels) = (x.dim(1), x.dim(2), x.dim(3))
        var tokens = convOut(x.transposed(0, 2, 3, 1).reshaped(numChunks, time, channels * freq))
        tokens = tokens + sharpAIQwen3AsrSinusoids(length: time, channels: config.dModel)
            .asType(tokens.dtype)

        // Keep only the valid rows of each chunk, in order.
        var pieces: [MLXArray] = []
        var counts: [Int] = []
        for index in 0 ..< numChunks {
            let chunkFrames = min(max(valid - index * chunk, 0), chunk)
            let count = min(SharpAIQwen3AsrAudioMath.postConvLength(chunkFrames), time)
            counts.append(count)
            if count > 0 {
                pieces.append(tokens[index, 0 ..< count])
            }
        }
        let total = counts.reduce(0, +)
        guard total > 0 else {
            throw SharpAIQwen3AsrError.invalidInput("audio is too short to encode")
        }

        // Windows follow HF get_audio_cu_seqlens for one utterance.
        let perChunkMax = counts.max() ?? 0
        let windowTokens = perChunkMax * (config.nWindowInfer / chunk)
        let windows = SharpAIQwen3AsrAudioMath.windowBounds(
            total: total, windowTokens: windowTokens)

        var hidden = MLX.concatenated(pieces, axis: 0)
        for layer in layers {
            hidden = layer(hidden, windows: windows)
        }
        hidden = lnPost(hidden)
        return proj2(gelu(proj1(hidden)))
    }
}
