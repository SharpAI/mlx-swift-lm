// SharpAI fork overlay: cosmos3_omni generation heads. Not part of upstream mlx-swift-lm.
//
// The small modules around the generation tower: vision latent proj_in/proj_out, the timestep
// embedder, the sound head and the domain-aware action head, plus latent patchify helpers.
// The wrapper model stores this module as the property `gen_heads`, so the parameter names here
// are the checkpoint names without that prefix (18 tensors).
//
// Verified against: huggingface/diffusers src/diffusers/models/transformers/transformer_cosmos3.py
// (main, fetched 2026-11-01):
//   - DomainAwareLinear, lines 197-225 (fc / bias are nn.Embedding; weight viewed as [in, out]).
//   - Cosmos3OmniTransformer.__init__, lines 468-490 (proj_in, proj_out, Timesteps(256,
//     flip_sin_to_cos=True, downscale_freq_shift=0), TimestepEmbedding(256 -> hidden), action and
//     audio heads, modality embeds initialised to zeros).
//   - _apply_timestep_embeds_to_noisy_tokens, lines 435-455 (scatter_add onto noisy-frame tokens).
//   - _patchify_and_pack_latents / _unpatchify_and_unpack_latents, lines 517-582.
//   - forward, lines 744-790 (timestep * timestep_scale, audio / action embedding order).
// huggingface/diffusers src/diffusers/models/embeddings.py: get_timestep_embedding (line 27),
// Timesteps (line 1075), TimestepEmbedding (line 1027; linear_1, SiLU, linear_2).
//
// Scope: single-item tensor helpers (no list packing). The wrapper packs items and builds the
// token indexes.
// UNVERIFIED: the original keeps the time embedder in fp32 (_keep_in_fp32_modules). Here the
// sinusoid is computed in float32 and the linear layers run in the dtype of their weights.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// Head dimensions for the generation heads. Defaults are the nvidia/Cosmos3-Nano values.
public struct SharpAICosmos3OmniHeadsConfiguration: Sendable, Equatable {
    public var hiddenSize: Int
    public var latentChannel: Int
    public var latentPatchSize: Int
    public var soundDim: Int
    public var actionDim: Int
    public var numEmbodimentDomains: Int
    public var timestepScale: Float
    /// Whether the sound head exists (`sound_gen`). Verified against transformer_cosmos3.py:485
    /// (`if sound_gen:`): without it the checkpoint has no `audio_*` keys.
    public var soundGen: Bool
    /// Whether the action head exists (`action_gen`). Verified against transformer_cosmos3.py:479
    /// (`if action_gen:`): without it the checkpoint has no `action_*` keys.
    public var actionGen: Bool
    /// Width of the sinusoidal timestep projection (`Timesteps(num_channels=256)`).
    public var timeProjChannels: Int

    /// `p * p * latent_channel`, 192 for Cosmos3-Nano.
    public var patchLatentDim: Int { latentPatchSize * latentPatchSize * latentChannel }

    public init(
        hiddenSize: Int = 4096,
        latentChannel: Int = 48,
        latentPatchSize: Int = 2,
        soundDim: Int = 64,
        actionDim: Int = 64,
        numEmbodimentDomains: Int = 32,
        timestepScale: Float = 0.001,
        soundGen: Bool = true,
        actionGen: Bool = true,
        timeProjChannels: Int = 256
    ) {
        self.hiddenSize = hiddenSize
        self.latentChannel = latentChannel
        self.latentPatchSize = latentPatchSize
        self.soundDim = soundDim
        self.actionDim = actionDim
        self.numEmbodimentDomains = numEmbodimentDomains
        self.timestepScale = timestepScale
        self.soundGen = soundGen
        self.actionGen = actionGen
        self.timeProjChannels = timeProjChannels
    }
}

extension SharpAICosmos3OmniHeadsConfiguration {
    /// Decodes the head sizes from the merged root / transformer config (same merge as
    /// `SharpAICosmos3OmniGenerationConfiguration.decode`).
    ///
    /// Only the fields the heads need are read and checked, so a config whose tower fields fail the
    /// generation checks (the tiny reference config has an mrope_section that does not sum to
    /// head_dim / 2) still yields correct head shapes. A malformed value or a missing size for an
    /// enabled head throws `ModelFactoryError.invalidConfiguration`. `fallbackHiddenSize` is used
    /// only when the config has no `hidden_size`.
    static func decode(
        rootConfig: Data?, transformerConfig: Data? = nil, fallbackHiddenSize: Int
    ) throws -> Self {
        struct Fields: Decodable {
            let hiddenSize: Int?
            let latentChannel: Int?
            let latentPatchSize: Int?
            let timestepScale: Double?
            let actionGen: Bool?
            let soundGen: Bool?
            let maxActionDim: Int?
            let actionDim: Int?
            let numEmbodimentDomains: Int?
            let soundDim: Int?

            enum CodingKeys: String, CodingKey {
                case hiddenSize = "hidden_size"
                case latentChannel = "latent_channel"
                case latentPatchSize = "latent_patch_size"
                case timestepScale = "timestep_scale"
                case actionGen = "action_gen"
                case soundGen = "sound_gen"
                case maxActionDim = "max_action_dim"
                case actionDim = "action_dim"
                case numEmbodimentDomains = "num_embodiment_domains"
                case soundDim = "sound_dim"
            }
        }
        func fail(_ message: String) -> ModelFactoryError {
            .invalidConfiguration("cosmos3_omni generation heads: \(message)")
        }
        let merged = try SharpAICosmos3OmniGenerationConfiguration.mergedObject(
            rootConfig: rootConfig, transformerConfig: transformerConfig)
        let f = try JSONDecoder().decode(
            Fields.self, from: try JSONSerialization.data(withJSONObject: merged))

        // Defaults are those of the generation configuration (Cosmos3Omni.__init__ keyword defaults).
        var result = SharpAICosmos3OmniHeadsConfiguration(
            hiddenSize: f.hiddenSize ?? fallbackHiddenSize,
            latentChannel: f.latentChannel ?? 48,
            latentPatchSize: f.latentPatchSize ?? 2,
            numEmbodimentDomains: f.numEmbodimentDomains ?? 32,
            timestepScale: Float(f.timestepScale ?? 0.001),
            soundGen: f.soundGen ?? false,
            actionGen: f.actionGen ?? false)
        for (name, value) in [
            ("hidden_size", result.hiddenSize), ("latent_channel", result.latentChannel),
            ("latent_patch_size", result.latentPatchSize),
            ("num_embodiment_domains", result.numEmbodimentDomains),
        ] where value <= 0 {
            throw fail("\(name) must be positive, got \(value)")
        }
        if result.soundGen {
            guard let soundDim = f.soundDim, soundDim > 0 else {
                throw fail("sound_gen requires a positive sound_dim")
            }
            result.soundDim = soundDim
        }
        if result.actionGen {
            // UNVERIFIED: `max_action_dim` is treated as the padded width of `action_dim`.
            guard let actionDim = f.maxActionDim ?? f.actionDim, actionDim > 0 else {
                throw fail("action_gen requires a positive max_action_dim (or action_dim)")
            }
            result.actionDim = actionDim
        }
        return result
    }
}

/// Linear layer with one weight and bias per embodiment domain.
/// Parameters: `fc.weight [domains, in * out]` and `bias.weight [domains, out]`.
public final class SharpAICosmos3OmniDomainAwareLinear: Module {
    public let inputSize: Int
    public let outputSize: Int
    public let numDomains: Int

    @ModuleInfo(key: "fc") var fc: Embedding
    @ModuleInfo(key: "bias") var bias: Embedding

    public init(inputSize: Int, outputSize: Int, numDomains: Int) {
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.numDomains = numDomains
        _fc.wrappedValue = Embedding(embeddingCount: numDomains, dimensions: outputSize * inputSize)
        _bias.wrappedValue = Embedding(embeddingCount: numDomains, dimensions: outputSize)
        super.init()
    }

    /// `x` is `[N, in]` or `[N, L, in]`; `domainIds` holds one id per row `N`.
    /// Row `n` uses `fc[domainIds[n]]` viewed as `[in, out]` plus `bias[domainIds[n]]`.
    public func callAsFunction(_ x: MLXArray, domainIds: MLXArray) -> MLXArray {
        let ids = domainIds.reshaped([-1]).asType(.int32)
        precondition(
            x.dim(0) == ids.dim(0),
            "cosmos3_omni domain ids must match the rows of x: x=\(x.dim(0)), ids=\(ids.dim(0))")
        let idList = ids.asArray(Int32.self)
        precondition(
            idList.allSatisfy { $0 >= 0 && Int($0) < numDomains },
            "cosmos3_omni domain id must be in [0, \(numDomains)), got \(idList)")

        let n = ids.dim(0)
        let weight = fc(ids).reshaped([n, inputSize, outputSize])
        let b = bias(ids).reshaped([n, outputSize])
        switch x.ndim {
        case 2:
            let y = matmul(x.expandedDimensions(axis: 1), weight).squeezed(axis: 1)
            return y + b
        case 3:
            return matmul(x, weight) + b.expandedDimensions(axis: 1)
        default:
            preconditionFailure("cosmos3_omni DomainAwareLinear expects rank 2 or 3, got \(x.shape)")
        }
    }
}

/// `TimestepEmbedding`: linear_1, SiLU, linear_2. Parameters: `linear_1.{weight,bias}`,
/// `linear_2.{weight,bias}`.
public final class SharpAICosmos3OmniTimeEmbedder: Module {
    @ModuleInfo(key: "linear_1") var linear1: Linear
    @ModuleInfo(key: "linear_2") var linear2: Linear

    public init(inputChannels: Int, hiddenSize: Int) {
        _linear1.wrappedValue = Linear(inputChannels, hiddenSize, bias: true)
        _linear2.wrappedValue = Linear(hiddenSize, hiddenSize, bias: true)
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        linear2(silu(linear1(x)))
    }
}

public final class SharpAICosmos3OmniHeads: Module {
    public let config: SharpAICosmos3OmniHeadsConfiguration

    @ModuleInfo(key: "proj_in") var projIn: Linear
    @ModuleInfo(key: "proj_out") var projOut: Linear
    @ModuleInfo(key: "time_embedder") var timeEmbedder: SharpAICosmos3OmniTimeEmbedder
    // The sound and action heads exist only when `sound_gen` / `action_gen` is set (nil otherwise,
    // so the parameter tree has no such keys).
    @ParameterInfo(key: "audio_modality_embed") var audioModalityEmbed: MLXArray?
    @ModuleInfo(key: "audio_proj_in") var audioProjIn: Linear?
    @ModuleInfo(key: "audio_proj_out") var audioProjOut: Linear?
    @ParameterInfo(key: "action_modality_embed") var actionModalityEmbed: MLXArray?
    @ModuleInfo(key: "action_proj_in") var actionProjIn: SharpAICosmos3OmniDomainAwareLinear?
    @ModuleInfo(key: "action_proj_out") var actionProjOut: SharpAICosmos3OmniDomainAwareLinear?

    public init(_ config: SharpAICosmos3OmniHeadsConfiguration) {
        self.config = config
        let hidden = config.hiddenSize
        _projIn.wrappedValue = Linear(config.patchLatentDim, hidden, bias: true)
        _projOut.wrappedValue = Linear(hidden, config.patchLatentDim, bias: true)
        _timeEmbedder.wrappedValue = SharpAICosmos3OmniTimeEmbedder(
            inputChannels: config.timeProjChannels, hiddenSize: hidden)
        if config.soundGen {
            _audioModalityEmbed.wrappedValue = MLXArray.zeros([hidden])
            _audioProjIn.wrappedValue = Linear(config.soundDim, hidden, bias: true)
            _audioProjOut.wrappedValue = Linear(hidden, config.soundDim, bias: true)
        } else {
            _audioModalityEmbed.wrappedValue = nil
            _audioProjIn.wrappedValue = nil
            _audioProjOut.wrappedValue = nil
        }
        if config.actionGen {
            _actionModalityEmbed.wrappedValue = MLXArray.zeros([hidden])
            _actionProjIn.wrappedValue = SharpAICosmos3OmniDomainAwareLinear(
                inputSize: config.actionDim, outputSize: hidden,
                numDomains: config.numEmbodimentDomains)
            _actionProjOut.wrappedValue = SharpAICosmos3OmniDomainAwareLinear(
                inputSize: hidden, outputSize: config.actionDim,
                numDomains: config.numEmbodimentDomains)
        } else {
            _actionModalityEmbed.wrappedValue = nil
            _actionProjIn.wrappedValue = nil
            _actionProjOut.wrappedValue = nil
        }
        super.init()
    }

    // MARK: - Vision

    /// `[N, patch_latent_dim]` packed latent patches to `[N, hidden]` (before timestep embedding).
    public func embedVision(_ patches: MLXArray) -> MLXArray {
        projIn(patches)
    }

    /// Final hidden states `[N, hidden]` to velocity patches `[N, patch_latent_dim]`.
    public func predictVision(_ hidden: MLXArray) -> MLXArray {
        projOut(hidden)
    }

    // MARK: - Timestep

    /// Sinusoidal projection of `Timesteps(num_channels, flip_sin_to_cos: true,
    /// downscale_freq_shift: 0, scale: 1)`, computed in float32. Input `[M]`, output `[M, channels]`.
    /// The result is `[cos(t * f), sin(t * f)]` with `f_i = exp(-ln(10000) * i / (channels / 2))`.
    public static func sinusoidalTimestepProjection(_ timesteps: MLXArray, channels: Int)
        -> MLXArray
    {
        let half = channels / 2
        let logMaxPeriod = MLXArray(Float(-log(10000.0)))
        let halfDim = MLXArray(Float(half))
        let index = MLXArray(0 ..< half).asType(.float32)
        let freqs = exp((logMaxPeriod * index) / halfDim)
        let angles = timesteps.asType(.float32).expandedDimensions(axis: 1)
            * freqs.expandedDimensions(axis: 0)
        // flip_sin_to_cos: the cos half comes first.
        var emb = concatenated([cos(angles), sin(angles)], axis: -1)
        if channels % 2 == 1 {
            emb = MLX.padded(emb, widths: [0, .init((0, 1))])
        }
        return emb
    }

    /// Raw timesteps in `[0, 1000]` (`[M]`) to timestep embeddings `[M, hidden]`.
    /// The original multiplies by `timestep_scale` (0.001) before the sinusoid.
    public func timestepEmbedding(_ rawTimesteps: MLXArray) -> MLXArray {
        let scaled = rawTimesteps.asType(.float32) * MLXArray(config.timestepScale)
        let proj = Self.sinusoidalTimestepProjection(scaled, channels: config.timeProjChannels)
        return timeEmbedder(proj.asType(timeEmbedder.linear1.weight.dtype))
    }

    /// Adds timestep embeddings to the noisy tokens only; conditioning tokens are unchanged.
    /// `tokens` is `[N, hidden]`, `rawTimesteps` holds one value per noisy token (`[M]`), and
    /// `noisyTokenIndexes` (length `M`) are the distinct token positions of those tokens.
    /// Equivalent to the original `scatter_add` for distinct indexes. Implemented as a gather
    /// from the embeddings with an appended zero row so that non-noisy tokens add zero.
    public func addTimestepEmbeddings(
        to tokens: MLXArray, rawTimesteps: MLXArray, noisyTokenIndexes: [Int]
    ) -> MLXArray {
        let m = noisyTokenIndexes.count
        precondition(rawTimesteps.dim(0) == m, "one timestep per noisy token is required")
        if m == 0 { return tokens }
        let n = tokens.dim(0)
        precondition(
            Set(noisyTokenIndexes).count == m && noisyTokenIndexes.allSatisfy { $0 >= 0 && $0 < n },
            "noisy token indexes must be distinct and inside the token range")
        let emb = timestepEmbedding(rawTimesteps).asType(tokens.dtype)
        let table = concatenated([emb, MLXArray.zeros([1, tokens.dim(1)], dtype: tokens.dtype)], axis: 0)
        var source = [Int32](repeating: Int32(m), count: n)
        for (row, tokenIndex) in noisyTokenIndexes.enumerated() { source[tokenIndex] = Int32(row) }
        return tokens + table[MLXArray(source)]
    }

    /// Token positions of the noisy frames inside one item's packed tokens.
    /// Mirrors `_apply_timestep_embeds_to_noisy_tokens`: `frame * spatialTokenCount + s + start`.
    public static func noisyTokenIndexes(
        noisyFrameIndexes: [Int], spatialTokenCount: Int, start: Int = 0
    ) -> [Int] {
        var out: [Int] = []
        out.reserveCapacity(noisyFrameIndexes.count * spatialTokenCount)
        for frame in noisyFrameIndexes {
            for s in 0 ..< spatialTokenCount { out.append(frame * spatialTokenCount + s + start) }
        }
        return out
    }

    // MARK: - Sound

    /// Sound latents `[N, sound_dim]` to `[N, hidden]`: `audio_proj_in(x) + audio_modality_embed`.
    public func embedSound(_ latents: MLXArray) -> MLXArray {
        guard let audioProjIn, let audioModalityEmbed else {
            preconditionFailure("cosmos3_omni: sound head requested but sound_gen is false")
        }
        return audioProjIn(latents) + audioModalityEmbed
    }

    public func predictSound(_ hidden: MLXArray) -> MLXArray {
        guard let audioProjOut else {
            preconditionFailure("cosmos3_omni: sound head requested but sound_gen is false")
        }
        return audioProjOut(hidden)
    }

    // MARK: - Action

    /// Action latents `[N, action_dim]` to `[N, hidden]`:
    /// `action_proj_in(x, domain) + action_modality_embed`. `domainIds` has one id per row.
    public func embedAction(_ latents: MLXArray, domainIds: MLXArray) -> MLXArray {
        guard let actionProjIn, let actionModalityEmbed else {
            preconditionFailure("cosmos3_omni: action head requested but action_gen is false")
        }
        return actionProjIn(latents, domainIds: domainIds) + actionModalityEmbed
    }

    public func predictAction(_ hidden: MLXArray, domainIds: MLXArray) -> MLXArray {
        guard let actionProjOut else {
            preconditionFailure("cosmos3_omni: action head requested but action_gen is false")
        }
        return actionProjOut(hidden, domainIds: domainIds)
    }

    // MARK: - Latent patchify

    /// One latent `[C, T, H, W]` to patches `[T * ceil(H/p) * ceil(W/p), p * p * C]`.
    /// Zero-pads H and W up to a multiple of `p`. Token order is `t, h, w`; the feature order is
    /// `(p_h, p_w, c)`, i.e. `einsum("cthpwq->thwpqc")`.
    public static func patchify(_ latent: MLXArray, patchSize p: Int) -> MLXArray {
        precondition(latent.ndim == 4, "latent must be [C, T, H, W]")
        let c = latent.dim(0)
        let t = latent.dim(1)
        let h = latent.dim(2)
        let w = latent.dim(3)
        let hPad = ((h + p - 1) / p) * p
        let wPad = ((w + p - 1) / p) * p
        var x = latent
        if hPad != h || wPad != w {
            x = MLX.padded(
                x,
                widths: [0, 0, .init((0, hPad - h)), .init((0, wPad - w))])
        }
        x = x.reshaped([c, t, hPad / p, p, wPad / p, p])
        // cthpwq -> thwpqc
        x = x.transposed(1, 2, 4, 3, 5, 0)
        return x.reshaped([-1, p * p * c])
    }

    /// Inverse of `patchify` for the noisy frames. `patches` is
    /// `[noisyFrames.count * hp * wp, p * p * C]`; the result is `[C, totalFrames, H, W]` with
    /// the noisy frames filled in and every other frame zero (as in the original, which writes
    /// predictions into a zero tensor at `noisy_frame_indexes`). `height` and `width` are the
    /// original (unpadded) latent sizes.
    public static func unpatchify(
        _ patches: MLXArray, patchSize p: Int, channels c: Int, height: Int, width: Int,
        totalFrames: Int, noisyFrameIndexes: [Int]
    ) -> MLXArray {
        let hp = (height + p - 1) / p
        let wp = (width + p - 1) / p
        let tn = noisyFrameIndexes.count
        precondition(
            patches.dim(0) == tn * hp * wp && patches.dim(1) == p * p * c,
            "patches must be [\(tn * hp * wp), \(p * p * c)], got \(patches.shape)")
        var frames = patches.reshaped([tn, hp, wp, p, p, c])
        // thwpqc -> cthpwq
        frames = frames.transposed(5, 0, 1, 3, 2, 4)
        frames = frames.reshaped([c, tn, hp * p, wp * p])[0..., 0..., 0 ..< height, 0 ..< width]
        let zero = MLXArray.zeros([c, height, width], dtype: patches.dtype)
        var perFrame: [MLXArray] = []
        for frame in 0 ..< totalFrames {
            if let k = noisyFrameIndexes.firstIndex(of: frame) {
                perFrame.append(frames[0..., k])
            } else {
                perFrame.append(zero)
            }
        }
        return stacked(perFrame, axis: 1)
    }
}
