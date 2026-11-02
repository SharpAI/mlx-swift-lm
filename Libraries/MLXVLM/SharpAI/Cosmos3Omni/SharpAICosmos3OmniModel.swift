// SharpAI fork overlay: cosmos3_omni wrapper model. Not part of upstream mlx-swift-lm.
//
// Holds the upstream `Qwen3VL` (understanding tower) as the child `und`, the generation tower as
// `gen` and the modality heads as `gen_heads`. Parameter paths after sanitize:
//   und.{vision_tower,language_model}.*   gen.layers.N.* / gen.norm.weight   gen_heads.*
// Understanding calls (prepare, callAsFunction, caches, LoRA layers) delegate to the child unchanged.
// `generateVelocity` runs the two-tower joint forward and the vision-latent head: it is the model
// forward only (no sampler, no classifier-free guidance, no VAE, no processor).
//
// Verified against: https://raw.githubusercontent.com/huggingface/diffusers/main/src/diffusers/models/transformers/transformer_cosmos3.py
// (Cosmos3OmniTransformer.forward: proj_in on packed latent patches, timestep embedding added to
// noisy tokens, decoder stack, proj_out on the generation hidden states).
//
// UNVERIFIED: image and video features for the understanding stream enter `generateVelocity` only as
// precomputed `understandingEmbeddings`; the deepstack injection of Qwen3VL is not applied in the
// joint forward (the original Diffusers stack was only checked for the text path).
// UNVERIFIED: the unified 3D mRoPE position ids of the generation tokens are an input of
// `generateVelocity`; building them from fps and frame counts is not implemented here.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// Results of one generation forward.
public struct SharpAICosmos3OmniVelocityOutput {
    /// Velocity patches [B, Ng, patch_latent_dim].
    public let velocity: MLXArray
    /// Generation hidden states after `gen.norm` [B, Ng, hidden].
    public let generationHidden: MLXArray
    /// Understanding hidden states before the final understanding norm [B, Nu, hidden].
    public let understandingHidden: MLXArray
}

public final class SharpAICosmos3OmniModel: Module, VLMModel, KVCacheDimensionProvider {

    /// Key prefix of the understanding child in the parameter tree.
    static let understandingPrefix = "und."

    @ModuleInfo(key: "und") var understanding: Qwen3VL
    @ModuleInfo(key: "gen") var gen: SharpAICosmos3OmniGenTower
    @ModuleInfo(key: "gen_heads") var genHeads: SharpAICosmos3OmniHeads

    public init(_ configuration: Qwen3VLConfiguration, heads: SharpAICosmos3OmniHeadsConfiguration) {
        let text = configuration.textConfiguration
        _understanding.wrappedValue = Qwen3VL(configuration)
        _gen.wrappedValue = SharpAICosmos3OmniGenTower(
            SharpAICosmos3OmniGenTowerConfiguration(
                hiddenSize: text.hiddenSize, intermediateSize: text.intermediateSize,
                numHiddenLayers: text.numHiddenLayers, numAttentionHeads: text.numAttentionHeads,
                numKeyValueHeads: text.numKeyValueHeads, headDim: text.headDim,
                rmsNormEps: Float(text.rmsNormEps)))
        _genHeads.wrappedValue = SharpAICosmos3OmniHeads(heads)
        super.init()
    }

    // MARK: - Understanding (delegated to Qwen3VL)

    public var vocabularySize: Int { understanding.vocabularySize }
    public var kvHeads: [Int] { understanding.kvHeads }
    public var loraLayers: [Module] { understanding.loraLayers }
    public var reasoningConfig: ReasoningConfig? { understanding.reasoningConfig }

    public func prepare(
        _ input: LMInput, cache: [any KVCache], state: LMOutput.State?, prefill: PrefillParameters
    ) throws -> PrepareResult {
        try understanding.prepare(input, cache: cache, state: state, prefill: prefill)
    }

    public func callAsFunction(
        _ input: LMInput.Text, cache: [any KVCache]?, state: LMOutput.State?
    ) -> LMOutput {
        understanding(input, cache: cache, state: state)
    }

    // MARK: - Generation

    /// Embeds understanding token ids with the child's embedding table. The language model is
    /// private to `Qwen3VL`, so the table is found through the module tree.
    func embedUnderstandingTokens(_ inputIds: MLXArray) -> MLXArray {
        guard
            let table = understanding.namedModules().first(where: {
                $0.0.hasSuffix("embed_tokens") && $0.1 is Embedding
            })?.1 as? Embedding
        else {
            preconditionFailure("cosmos3_omni: understanding embed_tokens not found")
        }
        return table(inputIds)
    }

    /// Velocity prediction for vision latent tokens.
    ///
    /// - Parameters:
    ///   - inputIds: understanding token ids [B, Nu] (ignored when `understandingEmbeddings` is given).
    ///   - understandingEmbeddings: optional precomputed understanding embeddings [B, Nu, hidden].
    ///   - latentPatches: noisy patchified latents [B, Ng, patch_latent_dim] (`SharpAICosmos3OmniHeads.patchify`).
    ///   - timesteps: raw flow timesteps in [0, 1000], one per batch item [B]; scaled by `timestep_scale` inside.
    ///   - noisyTokenIndexes: generation token positions that receive the timestep embedding, shared
    ///     by the batch. nil means every generation token is noisy.
    ///   - understandingPositions: mRoPE ids [3, B, Nu]; nil means sequential text positions.
    ///   - generationPositions: mRoPE ids [3, B, Ng].
    public func generateVelocity(
        inputIds: MLXArray,
        understandingEmbeddings: MLXArray? = nil,
        latentPatches: MLXArray,
        timesteps: MLXArray,
        noisyTokenIndexes: [Int]? = nil,
        understandingPositions: MLXArray? = nil,
        generationPositions: MLXArray
    ) -> SharpAICosmos3OmniVelocityOutput {
        let undSeq = understandingEmbeddings ?? embedUnderstandingTokens(inputIds)
        let batch = undSeq.dim(0)
        let undLength = undSeq.dim(1)
        let genLength = latentPatches.dim(1)
        precondition(latentPatches.dim(0) == batch, "latent batch must match understanding batch")
        precondition(timesteps.size == batch, "one timestep per batch item is required")

        var embedded = genHeads.embedVision(latentPatches.asType(undSeq.dtype))
        let noisy = noisyTokenIndexes ?? Array(0 ..< genLength)
        let flatTimesteps = timesteps.reshaped([batch])
        var items = [MLXArray]()
        for b in 0 ..< batch {
            let raw = broadcast(flatTimesteps[b], to: [noisy.count])
            items.append(
                genHeads.addTimestepEmbeddings(
                    to: embedded[b], rawTimesteps: raw, noisyTokenIndexes: noisy))
        }
        embedded = stacked(items, axis: 0)

        let undPositions: MLXArray
        if let understandingPositions {
            undPositions = understandingPositions
        } else {
            let base = MLXArray(0 ..< undLength).asType(.int32).reshaped([1, 1, undLength])
            undPositions = broadcast(base, to: [3, batch, undLength])
        }

        let result = gen.jointForward(
            understanding: understanding, undSeq: undSeq, genSeq: embedded,
            undPositions: undPositions, genPositions: generationPositions)
        return SharpAICosmos3OmniVelocityOutput(
            velocity: genHeads.predictVision(result.gen),
            generationHidden: result.gen,
            understandingHidden: result.und)
    }

    // MARK: - Weights

    /// Checkpoint keys to model keys: Cosmos sanitizer, then Qwen-VL attention names, then
    /// `Qwen3VL.sanitize` for the understanding subset (prefixed with `und.`); `gen.*` and
    /// `gen_heads.*` pass through. Keys that already carry a model prefix (`und.`, `gen.`,
    /// `gen_heads.`) are kept as they are, which lets a checkpoint saved from this model reload.
    public func sanitize(weights: [String: MLXArray]) throws -> [String: MLXArray] {
        var kept = [String: MLXArray]()
        var checkpoint = [String: MLXArray]()
        for (key, value) in weights {
            if key.hasPrefix(Self.understandingPrefix)
                || key.hasPrefix(SharpAICosmos3OmniSanitizer.generationPrefix)
                || key.hasPrefix(SharpAICosmos3OmniSanitizer.generationHeadPrefix)
            {
                kept[key] = value
            } else {
                checkpoint[key] = value
            }
        }

        let config = understanding.config
        let cosmos = try SharpAICosmos3OmniSanitizer.sanitize(
            checkpoint, visionInChannels: config.visionConfiguration.inChannels,
            tieWordEmbeddings: config.textConfiguration.tieWordEmbeddings,
            soundGen: genHeads.config.soundGen, actionGen: genHeads.config.actionGen)
        let named = try SharpAICosmos3OmniSanitizer.qwen3VLAttentionNames(cosmos)
        let (understandingWeights, generationWeights) =
            SharpAICosmos3OmniSanitizer.partitionGeneration(named)

        var result = kept
        for (key, value) in understanding.sanitize(weights: understandingWeights) {
            let prefixed = Self.understandingPrefix + key
            guard result[prefixed] == nil else {
                throw SharpAICosmos3OmniError.duplicateWeightKey(prefixed)
            }
            result[prefixed] = value
        }
        for (key, value) in generationWeights {
            guard result[key] == nil else {
                throw SharpAICosmos3OmniError.duplicateWeightKey(key)
            }
            result[key] = value
        }
        return result
    }
}

extension SharpAICosmos3OmniModel: PreparedInputSplitting {
    public func splitPreparedInput(_ input: LMInput, droppingFirst prefixTokenCount: Int) -> LMInput? {
        understanding.splitPreparedInput(input, droppingFirst: prefixTokenCount)
    }
}
