// SharpAI fork overlay: NemotronH_Nano_VL_V2 model. Not part of upstream mlx-swift-lm.
//
// Composition: `language_model` is NemotronHModel, `vision_model` is the RADIO ViT tower and
// `mlp1` is the pixel-shuffle projector, using the checkpoint's own key names.
// The text-only path runs. Merging image features into the text embeddings does not:
// NemotronHModel exposes no inputs_embeds entry point (see forwardWithImages).

import MLX
import MLXLLM
import MLXLMCommon
import MLXNN

public final class SharpAINemotronhNanoVlV2Model: Module {

    public let visionConfiguration: SharpAINemotronhNanoVlV2VisionConfiguration

    @ModuleInfo(key: "language_model") public var languageModel: NemotronHModel
    @ModuleInfo(key: "vision_model") public var visionModel: SharpAINemotronhNanoVlV2VisionModel
    @ModuleInfo(key: "mlp1") public var projector: SharpAINemotronhNanoVlV2Projector

    public init(
        text: NemotronHConfiguration,
        vision: SharpAINemotronhNanoVlV2VisionConfiguration,
        projectorHiddenSize: Int
    ) {
        self.visionConfiguration = vision
        self._languageModel.wrappedValue = NemotronHModel(text)
        self._visionModel.wrappedValue = SharpAINemotronhNanoVlV2VisionModel(vision)
        // The pixel shuffle folds a 2x2 patch block into the channel axis: 4 * embedDim.
        self._projector.wrappedValue = SharpAINemotronhNanoVlV2Projector(
            inputSize: 4 * vision.embedDim,
            hiddenSize: projectorHiddenSize,
            outputSize: text.hiddenSize)
        super.init()
    }

    public func newCache(parameters: GenerateParameters?) throws -> [KVCache] {
        try languageModel.newCache(parameters: parameters)
    }

    /// Text-only forward over token ids.
    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        languageModel(inputs, cache: cache)
    }

    /// NCHW normalized pixels to image embeddings of shape (N, num_image_tokens, text hidden).
    /// num_image_tokens is (H/p / 2) * (W/p / 2), so 512 px at patch 16 gives 256.
    public func imageFeatures(_ pixelValues: MLXArray) -> MLXArray {
        let features = visionModel(pixelValues)
        let n = pixelValues.dim(0)
        let gridH = pixelValues.dim(2) / visionConfiguration.patchSize
        let gridW = pixelValues.dim(3) / visionConfiguration.patchSize
        let grid = features.reshaped(n, gridH, gridW, visionConfiguration.embedDim)
        return projector(sharpAINemotronhNanoVlV2PixelShuffle(grid))
    }

    /// KNOWN GAP: merging image embeddings into the text embeddings needs an inputs_embeds
    /// entry point on NemotronHModel. Its backbone is private in
    /// Libraries/MLXLLM/Models/NemotronH.swift, which is outside this lane's allowed paths.
    /// Once that entry point exists, this should scatter imageFeatures into the rows where
    /// the image context token id appears, then run the language model on the result.
    public func forwardWithImages(_ inputs: MLXArray, pixelValues: MLXArray) throws -> MLXArray {
        throw SharpAINemotronhNanoVlV2Error.mergedForwardNotImplemented
    }
}
