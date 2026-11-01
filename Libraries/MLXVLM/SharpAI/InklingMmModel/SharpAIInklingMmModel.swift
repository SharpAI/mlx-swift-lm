// SharpAI fork overlay: inkling_mm_model model class and checkpoint key sanitization entry point.
// Not part of upstream mlx-swift-lm.
//
// The module tree mirrors the checkpoint: `model.llm` (text), `model.audio` and `model.visual`
// (towers). `model.mtp.*` is not built. The text path is a full-sequence forward. Incremental decoding
// needs KV-cache and short-conv state, which is not implemented, so `newCache` throws instead of
// producing wrong tokens.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// Model tree under the `model` key: text, audio and vision towers.
final class SharpAIInklingMmBackbone: Module {
    @ModuleInfo(key: "llm") var llm: SharpAIInklingMmTextModel
    @ModuleInfo(key: "audio") var audio: SharpAIInklingMmAudioTower
    @ModuleInfo(key: "visual") var visual: SharpAIInklingMmVisionTower

    init(_ config: SharpAIInklingMmModelConfiguration) {
        _llm.wrappedValue = SharpAIInklingMmTextModel(config.text)
        _audio.wrappedValue = SharpAIInklingMmAudioTower(config.audio)
        _visual.wrappedValue = SharpAIInklingMmVisionTower(
            config.vision, textHiddenSize: config.text.hiddenSize)
    }
}

class SharpAIInklingMmModel: Module, VLMModel {
    let vocabularySize: Int
    let kvHeads: [Int]
    let configuration: SharpAIInklingMmModelConfiguration
    @ModuleInfo(key: "model") var model: SharpAIInklingMmBackbone

    init(_ config: SharpAIInklingMmModelConfiguration) {
        self.configuration = config
        self.vocabularySize = config.text.vocabularySize
        let text = config.text
        self.kvHeads = text.layerTypes.map { layerType in
            layerType == .hybridSliding ? text.swaKVHeads : text.kvHeads
        }
        _model.wrappedValue = SharpAIInklingMmBackbone(config)
    }

    /// Text-only forward. `inputs` is `[batch, length]` of token ids. Returns `[batch, length, vocab]`.
    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let text = configuration.text
        let hidden = model.llm.hiddenStates(inputs) / text.logitsMupWidthMultiplier
        var logits = model.llm.unembed(hidden)
        if let unpadded = text.unpaddedVocabularySize, unpadded < logits.dim(-1) {
            logits = logits[.ellipsis, ..<unpadded]
        }
        return logits
    }

    /// Vision features for `pixelValues` (`[num_patches, time, height, width, channels]`), `[num_patches, 1, hidden]`.
    /// Not yet merged into the LM input.
    func imageFeatures(_ pixelValues: MLXArray) -> MLXArray {
        model.visual(pixelValues)
    }

    /// Audio features for dMel ids (`[num_audios, frames, mel_bins]`), `[num_audios, frames, hidden]`.
    /// Not yet merged into the LM input.
    func audioFeatures(_ audioIds: MLXArray) -> MLXArray {
        model.audio(audioIds)
    }

    func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        SharpAIInklingMmModelSanitizer.sanitize(weights)
    }

    func newCache(parameters: GenerateParameters?) throws -> [KVCache] {
        throw SharpAIInklingMmModelError.incrementalDecodeNotImplemented("inkling_mm_model")
    }

    /// Prompt prefill goes through the cache, which this model cannot build yet. Throws the same
    /// error as `newCache` rather than returning logits that a cache-less decode would turn into wrong tokens.
    func prepare(
        _ input: LMInput, cache: [KVCache], state: LMOutput.State?, prefill: PrefillParameters
    ) throws -> PrepareResult {
        throw SharpAIInklingMmModelError.incrementalDecodeNotImplemented("inkling_mm_model")
    }

    /// Decoder layers that LoRA adapters attach to. Same as the other SharpAI overlays.
    var loraLayers: [Module] {
        model.llm.layers
    }
}
