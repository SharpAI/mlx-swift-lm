// SharpAI fork overlay: pangu_ultra_moe_mtp model. Not part of upstream mlx-swift-lm.
//
// Port of the multi-token-prediction (MTP) head of openPangu-Ultra-MoE-718B as vLLM runs it
// (openpangu_mtp.py: OpenPanguMultiTokenPredictorLayer, which inherits the forward of
// DeepSeekMultiTokenPredictorLayer in deepseek_mtp.py). Verified against:
// https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/model_executor/models/deepseek_mtp.py
// https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/model_executor/models/openpangu_mtp.py
//
// Forward of one MTP step, for the spec layer at index num_hidden_layers + step:
//   1. the input embedding (shared with the main model) is zeroed at absolute position 0;
//   2. enorm(embedding) and hnorm(previous hidden state) are concatenated, embedding first;
//   3. eh_proj maps the 2*hidden concatenation back to hidden;
//   4. mtp_block is one decoder layer of the main model (MoE, since its index is past the dense
//      layers), built with SharpAIPanguUltraMoeDecoderLayer from the base pangu_ultra_moe overlay;
//   5. the hidden state before the final norm is returned; logits are head(shared_head.norm(h)),
//      where head is the caller's lm_head, as vLLM's compute_logits passes the head in.
//
// Checkpoint keys follow SharpAIPanguUltraMoeMtpSanitizer.rewriteKey. The module tree below is
// `model.embed_tokens` and `model.layers.<N>.{enorm,hnorm,eh_proj,shared_head.norm,mtp_block}`.
//
// Not verified: the checkpoint's model.safetensors.index.json was truncated in the fetch, so the
// exact key set for layer 61 (in particular whether shared_head.head is stored) is unconfirmed.
// Shared head weights are dropped on load. No output was compared with the reference Python.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// `shared_head` of an MTP layer: a final RMSNorm. The LM head is not owned here, see the file header.
final class SharpAIPanguUltraMoeMtpSharedHead: Module {
    @ModuleInfo(key: "norm") var norm: RMSNorm

    init(hiddenSize: Int, eps: Float) {
        _norm.wrappedValue = RMSNorm(dimensions: hiddenSize, eps: eps)
        super.init()
    }

    func callAsFunction(_ hidden: MLXArray) -> MLXArray {
        norm(hidden)
    }
}

/// One MTP spec layer, stored under its absolute checkpoint index (`model.layers.<N>`).
final class SharpAIPanguUltraMoeMtpLayer: Module {
    @ModuleInfo(key: "enorm") var enorm: RMSNorm
    @ModuleInfo(key: "hnorm") var hnorm: RMSNorm
    @ModuleInfo(key: "eh_proj") var ehProj: Linear
    @ModuleInfo(key: "shared_head") var sharedHead: SharpAIPanguUltraMoeMtpSharedHead
    @ModuleInfo(key: "mtp_block") var mtpBlock: SharpAIPanguUltraMoeDecoderLayer

    /// `layerIndex` is the absolute index (num_hidden_layers + step). It selects dense or MoE the
    /// same way the main decoder does.
    init(_ config: SharpAIPanguUltraMoeConfiguration, layerIndex: Int) {
        _enorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _hnorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _ehProj.wrappedValue = Linear(config.hiddenSize * 2, config.hiddenSize, bias: false)
        _sharedHead.wrappedValue = SharpAIPanguUltraMoeMtpSharedHead(
            hiddenSize: config.hiddenSize, eps: config.rmsNormEps)
        _mtpBlock.wrappedValue = SharpAIPanguUltraMoeDecoderLayer(config, layer: layerIndex)
        super.init()
    }

    /// Returns the hidden state before the final norm, shape [B, L, hidden].
    ///
    /// `cache` is the cache of this layer's attention, or nil for a stateless call. The absolute
    /// position of the first token is `cache?.offset ?? 0`, and position 0 uses a zero embedding.
    func hiddenStates(
        inputEmbeds: MLXArray, previousHidden: MLXArray, cache: KVCache?
    ) -> MLXArray {
        var embeds = inputEmbeds
        let length = embeds.dim(1)
        if (cache?.offset ?? 0) == 0, length > 0 {
            // vLLM: torch.where(positions == 0, 0, inputs_embeds). Only index 0 of the first call.
            let zeroFirst = MLXArray.zeros(like: embeds[0..., ..<1, 0...])
            if length > 1 {
                embeds = concatenated([zeroFirst, embeds[0..., 1..., 0...]], axis: 1)
            } else {
                embeds = zeroFirst
            }
        }

        let fused = ehProj(
            concatenated([enorm(embeds), hnorm(previousHidden)], axis: -1))
        let mask = createAttentionMask(h: fused, cache: cache)
        // The block returns its full residual output, which is vLLM's `residual + hidden_states`.
        return mtpBlock(fused, mask: mask, cache: cache)
    }
}

/// `model` of the MTP checkpoint namespace: the shared embedding and the spec layers by index.
final class SharpAIPanguUltraMoeMtpInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    /// Keyed by absolute layer index as a string, so `layers.61.*` maps to `layers["61"]`.
    let layers: [String: SharpAIPanguUltraMoeMtpLayer]

    init(_ config: SharpAIPanguUltraMoeConfiguration, layerIndices: [Int]) {
        _embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize, dimensions: config.hiddenSize)
        var built = [String: SharpAIPanguUltraMoeMtpLayer]()
        for index in layerIndices {
            built[String(index)] = SharpAIPanguUltraMoeMtpLayer(config, layerIndex: index)
        }
        self.layers = built
        super.init()
    }
}

/// Multi-token-prediction model for `model_type: pangu_ultra_moe_mtp`.
///
/// The MTP head needs the main model's hidden states, so it is a drafter, not a standalone
/// language model. Use ``draftLogits(inputIds:previousHidden:step:cache:head:)``. The
/// `LanguageModel` entry points cannot produce logits and trap if called.
public final class SharpAIPanguUltraMoeMtpModel: Module, LanguageModel, KVCacheDimensionProvider {
    /// Base pangu_ultra_moe dimensions. The MTP block is built with them.
    public let configuration: SharpAIPanguUltraMoeConfiguration
    /// Absolute index of the first spec layer, num_hidden_layers.
    public let mtpStartLayerIndex: Int
    /// Number of spec layers, num_nextn_predict_layers.
    public let numNextnPredictLayers: Int
    /// One entry per spec layer. Used by the cache constructor.
    public let kvHeads: [Int]

    @ModuleInfo(key: "model") var model: SharpAIPanguUltraMoeMtpInner

    public init(_ configuration: SharpAIPanguUltraMoeConfiguration, numNextnPredictLayers: Int) {
        precondition(numNextnPredictLayers > 0, "num_nextn_predict_layers must be positive")
        self.configuration = configuration
        self.mtpStartLayerIndex = configuration.hiddenLayers
        self.numNextnPredictLayers = numNextnPredictLayers
        self.kvHeads = Array(repeating: configuration.attentionHeads, count: numNextnPredictLayers)
        let indices = (0 ..< numNextnPredictLayers).map { configuration.hiddenLayers + $0 }
        _model.wrappedValue = SharpAIPanguUltraMoeMtpInner(configuration, layerIndices: indices)
        super.init()
    }

    /// Hidden states before the final norm for MTP step `step`, shape [B, L, hidden].
    ///
    /// - Parameters:
    ///   - inputIds: token ids [B, L] that the draft step consumes.
    ///   - previousHidden: main-model (or previous MTP step) hidden states [B, L, hidden].
    ///   - step: which spec layer to run, 0 ..< numNextnPredictLayers.
    ///   - cache: this step's attention cache, or nil for a stateless call from position 0.
    public func draftHidden(
        inputIds: MLXArray, previousHidden: MLXArray, step: Int = 0, cache: KVCache? = nil
    ) -> MLXArray {
        let layer = specLayer(step: step)
        let embeds = model.embedTokens(inputIds)
        return layer.hiddenStates(
            inputEmbeds: embeds, previousHidden: previousHidden, cache: cache)
    }

    /// Draft logits for MTP step `step`: `head(shared_head.norm(hidden))`, shape [B, L, vocab].
    ///
    /// `head` is the main model's LM head, passed in as vLLM's compute_logits does.
    public func draftLogits(
        inputIds: MLXArray,
        previousHidden: MLXArray,
        step: Int = 0,
        cache: KVCache? = nil,
        head: (MLXArray) -> MLXArray
    ) -> MLXArray {
        let hidden = draftHidden(
            inputIds: inputIds, previousHidden: previousHidden, step: step, cache: cache)
        return head(specLayer(step: step).sharedHead(hidden))
    }

    /// Keeps only the MTP namespace and rewrites it to this module's keys. See
    /// ``SharpAIPanguUltraMoeMtpSanitizer/sanitizeMTP(_:numHiddenLayers:numNextnPredictLayers:)``.
    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        SharpAIPanguUltraMoeMtpSanitizer.sanitizeMTP(
            weights,
            numHiddenLayers: mtpStartLayerIndex,
            numNextnPredictLayers: numNextnPredictLayers)
    }

    /// Not a standalone language model: an MTP head has no logits without the main model's hidden
    /// states. Use ``draftLogits(inputIds:previousHidden:step:cache:head:)``.
    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        fatalError(
            "SharpAIPanguUltraMoeMtpModel is an MTP drafter: call draftLogits(inputIds:previousHidden:step:cache:head:)"
        )
    }

    /// Prefill is not defined for an MTP drafter, which needs the main model's hidden states.
    /// Traps, like ``callAsFunction(_:cache:)``.
    public func prepare(
        _ input: LMInput, cache: [KVCache], state: LMOutput.State?, prefill: PrefillParameters
    ) throws -> PrepareResult {
        fatalError(
            "SharpAIPanguUltraMoeMtpModel is an MTP drafter: call draftLogits(inputIds:previousHidden:step:cache:head:)"
        )
    }

    private func specLayer(step: Int) -> SharpAIPanguUltraMoeMtpLayer {
        precondition(
            step >= 0 && step < numNextnPredictLayers,
            "MTP step \(step) out of range 0 ..< \(numNextnPredictLayers)")
        guard let layer = model.layers[String(mtpStartLayerIndex + step)] else {
            fatalError("MTP layer \(mtpStartLayerIndex + step) missing from the module tree")
        }
        return layer
    }
}
