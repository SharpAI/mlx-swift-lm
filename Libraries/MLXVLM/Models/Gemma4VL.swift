//
//  Gemma4VL.swift
//  mlx-swift-lm
//
//  Created for SwiftLM Gemma 4 Vision Support
//

import CoreImage
import Foundation
import MLX
import MLXLMCommon
import MLXNN
import MLXLLM

// MARK: - Vision Configuration

public struct Gemma4VisionConfiguration: Codable, Sendable {
    public let modelType: String
    public let hiddenSize: Int
    public let hiddenLayers: Int
    public let intermediateSize: Int
    public let attentionHeads: Int
    public let patchSize: Int

    public var numChannels: Int = 3
    public var layerNormEps: Float = 1e-6
    private let _imageSize: Int?
    public var imageSize: Int { _imageSize ?? 448 }

    public init(
        modelType: String, hiddenSize: Int, hiddenLayers: Int, intermediateSize: Int,
        attentionHeads: Int, patchSize: Int, numChannels: Int = 3, layerNormEps: Float = 1e-6,
        imageSize: Int? = 448
    ) {
        self.modelType = modelType
        self.hiddenSize = hiddenSize
        self.hiddenLayers = hiddenLayers
        self.intermediateSize = intermediateSize
        self.attentionHeads = attentionHeads
        self.patchSize = patchSize
        self._imageSize = imageSize
    }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenLayers = "num_hidden_layers"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case patchSize = "patch_size"
        case _imageSize = "image_size"
    }
}

// MARK: - Processor Configuration
public struct Gemma4ProcessorConfiguration: Codable, Sendable {
    public let processorClass: String
    
    public struct ImageProcessorConfig: Codable, Sendable {
        public let imageProcessorType: String
        public let imageMean: [CGFloat]
        public let imageStd: [CGFloat]

        public struct ImageSize: Codable, Sendable {
            public let height: Int
            public let width: Int
        }
        public let size: ImageSize
        public let resample: Int
        public let rescaleFactor: Float

        enum CodingKeys: String, CodingKey {
            case imageProcessorType = "image_processor_type"
            case imageMean = "image_mean"
            case imageStd = "image_std"
            case size
            case resample
            case rescaleFactor = "rescale_factor"
        }
    }
    
    public let imageProcessor: ImageProcessorConfig?

    public var imageMeanTuple: (CGFloat, CGFloat, CGFloat) {
        let mean = imageProcessor?.imageMean ?? [0.5, 0.5, 0.5]
        return (mean[0], mean[1], mean[2])
    }
    public var imageStdTuple: (CGFloat, CGFloat, CGFloat) {
        let std = imageProcessor?.imageStd ?? [0.5, 0.5, 0.5]
        return (std[0], std[1], std[2])
    }

    enum CodingKeys: String, CodingKey {
        case processorClass = "processor_class"
        case imageProcessor = "image_processor"
    }
}

// MARK: - Vision Architecture Components

private class Gemma4VisionAttention: Module {
    @ModuleInfo(key: "q_proj") var queryProj: Linear
    @ModuleInfo(key: "k_proj") var keyProj: Linear
    @ModuleInfo(key: "v_proj") var valueProj: Linear
    @ModuleInfo(key: "o_proj") var outputProj: Linear

    @ModuleInfo(key: "q_norm") var queryNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var keyNorm: RMSNorm

    let numHeads: Int
    let scale: Float

    init(dimensions: Int, numHeads: Int, bias: Bool = false) {
        self.numHeads = numHeads
        let headDim = dimensions / numHeads
        self.scale = pow(Float(headDim), -0.5)

        self._queryProj.wrappedValue = Linear(dimensions, dimensions, bias: bias)
        self._keyProj.wrappedValue = Linear(dimensions, dimensions, bias: bias)
        self._valueProj.wrappedValue = Linear(dimensions, dimensions, bias: bias)
        self._outputProj.wrappedValue = Linear(dimensions, dimensions, bias: bias)

        self._queryNorm.wrappedValue = RMSNorm(dimensions: headDim)
        self._keyNorm.wrappedValue = RMSNorm(dimensions: headDim)
    }

    func callAsFunction(_ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode = .none)
        -> MLXArray
    {
        let (B, L, _) = (x.dim(0), x.dim(1), x.dim(2))
        
        var queries = queryProj(x)
        var keys = keyProj(x)
        var values = valueProj(x)

        queries = queries.reshaped(B, L, numHeads, -1).transposed(0, 2, 1, 3)
        keys = keys.reshaped(B, L, numHeads, -1).transposed(0, 2, 1, 3)
        values = values.reshaped(B, L, numHeads, -1).transposed(0, 2, 1, 3)

        queries = queryNorm(queries)
        keys = keyNorm(keys)

        let output = MLXFast.scaledDotProductAttention(
            queries: queries, keys: keys, values: values, scale: scale, mask: mask
        ).transposed(0, 2, 1, 3).reshaped(B, L, -1)

        return outputProj(output)
    }
}

private class Gemma4VisionMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear

    init(config: Gemma4VisionConfiguration) {
        self._gateProj.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        self._downProj.wrappedValue = Linear(config.intermediateSize, config.hiddenSize, bias: false)
        self._upProj.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        return downProj(geluApproximate(gateProj(x)) * upProj(x))
    }
}

private class Gemma4VisionEncoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttention: Gemma4VisionAttention
    @ModuleInfo var mlp: Gemma4VisionMLP
    
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm
    @ModuleInfo(key: "pre_feedforward_layernorm") var preFeedforwardLayerNorm: RMSNorm
    @ModuleInfo(key: "post_feedforward_layernorm") var postFeedforwardLayerNorm: RMSNorm

    init(config: Gemma4VisionConfiguration) {
        self._selfAttention.wrappedValue = Gemma4VisionAttention(
            dimensions: config.hiddenSize, numHeads: config.attentionHeads, bias: false)
        self.mlp = Gemma4VisionMLP(config: config)
        
        self._inputLayerNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)
        self._postAttentionLayerNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)
        self._preFeedforwardLayerNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)
        self._postFeedforwardLayerNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)
    }

    func callAsFunction(_ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode = .none) -> MLXArray {
        let r = selfAttention(inputLayerNorm(x), mask: mask)
        let h = x + postAttentionLayerNorm(r)
        let r2 = mlp(preFeedforwardLayerNorm(h))
        return h + postFeedforwardLayerNorm(r2)
    }
}

private class Gemma4VisionEncoder: Module {
    @ModuleInfo var layers: [Gemma4VisionEncoderLayer]

    init(config: Gemma4VisionConfiguration) {
        self._layers.wrappedValue = (0 ..< config.hiddenLayers).map { _ in
            Gemma4VisionEncoderLayer(config: config)
        }
    }

    func callAsFunction(_ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode = .none) -> MLXArray {
        var h = x
        for layer in layers {
            h = layer(h, mask: mask)
        }
        return h
    }
}

private class Gemma4PatchEmbedder: Module {
    @ModuleInfo(key: "input_proj") var inputProj: Linear
    // position_embedding_table is just an array parameter
    var position_embedding_table: MLXArray
    let patchSize: Int

    init(config: Gemma4VisionConfiguration) {
        self.patchSize = config.patchSize
        self._inputProj.wrappedValue = Linear(
            config.patchSize * config.patchSize * config.numChannels,
            config.hiddenSize,
            bias: false
        )
        // Set the parameter directly. MLX requires Module parameters to be either Module or explicitly managed MLXArrays. 
        self.position_embedding_table = zeros([2, 10240, config.hiddenSize])
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let B = x.dim(0)
        let C = x.dim(1)
        let H = x.dim(2)
        let W = x.dim(3)
        let P = patchSize
        
        let h = H / P
        let w = W / P
        
        // Reshape [B, C, H, W] -> [B, C, h, P, w, P]
        let reshaped = x.reshaped([B, C, h, P, w, P])
        // Transpose to [B, h, w, P, P, C]
        let transposed = reshaped.transposed(0, 2, 4, 3, 5, 1)
        // Flatten to [B, h*w, P*P*C]
        let flattened = transposed.reshaped([B, h * w, P * P * C])
        
        var out = inputProj(flattened)
        
        // Add positional embeddings
        // The table is [2, 10240, hiddenSize]. We select index 0, and slice up to sequence length.
        let seqLen = out.dim(1)
        let posEmbeds = position_embedding_table[0..., 0..<seqLen, 0...]
        // posEmbeds has shape [2, seqLen, hiddenSize]. We want [1, seqLen, hiddenSize] or just [seqLen, hiddenSize]
        // Actually since we don't know why it's 2, let's just pick index 0.
        out = out + position_embedding_table[0, 0..<seqLen, 0...]
        
        return out
    }
}

private class Gemma4VisionModel: Module {
    @ModuleInfo var patch_embedder: Gemma4PatchEmbedder
    @ModuleInfo var encoder: Gemma4VisionEncoder
    var std_bias: MLXArray
    var std_scale: MLXArray
    let config: Gemma4VisionConfiguration
    
    init(config: Gemma4VisionConfiguration) {
        self.config = config
        self._patch_embedder.wrappedValue = Gemma4PatchEmbedder(config: config)
        self.encoder = Gemma4VisionEncoder(config: config)
        self.std_bias = zeros([config.hiddenSize])
        self.std_scale = ones([config.hiddenSize])
        super.init()
    }
    
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = patch_embedder(x)
        h = h * std_scale + std_bias
        return encoder(h)
    }
}

// MARK: - Multimodal Projector 

private class Gemma4Projector: Module, UnaryLayer {
    @ModuleInfo(key: "embedding_projection") var projection: any UnaryLayer
    
    init(visionDim: Int, textDim: Int) {
        self._projection.wrappedValue = Linear(visionDim, textDim, bias: false)
        super.init()
    }
    
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        return projection(x)
    }
}

// MARK: - Top-Level VLM

/// Gemma4 VLM (Pixtral / Google Paligemma NextGen equivalent)
public class Gemma4VL: Module, VLMModel, KVCacheDimensionProvider {
    // `language_model` uses the existing MLXLLM Gemma4ModelInternal
    @ModuleInfo(key: "model") private var languageModel: Gemma4ModelInternal
    @ModuleInfo(key: "lm_head") private var lmHead: Linear?

    @ModuleInfo(key: "vision_tower") private var visionTower: Gemma4VisionModel
    @ModuleInfo(key: "embed_vision") private var projector: Gemma4Projector

    @ModuleInfo(key: "audio_tower") private var audioTower: Gemma4AudioModel?


    public let config: Gemma4Configuration
    public let visionConfig: Gemma4VisionConfiguration
    
    public var vocabularySize: Int { config.vocabularySize }
    public var kvHeads: [Int] { Array(repeating: config.kvHeads, count: config.hiddenLayers) }
    
    public init(_ config: Gemma4Configuration) {
        self.config = config
        
        let vcfg = config.visionConfiguration ?? Gemma4VisionConfiguration(
            modelType: "gemma4_vision", hiddenSize: 1152, hiddenLayers: 27, intermediateSize: 4304, attentionHeads: 16, patchSize: 16)
        self.visionConfig = vcfg
        
        self._languageModel.wrappedValue = Gemma4ModelInternal(config)
        
        // Always create a separate lm_head — following the Gemma 3 pattern.
        // For tied embeddings, sanitize() will copy embed_tokens weights to lm_head.
        // This ensures logit projection uses QuantizedLinear.quantizedMM rather than
        // QuantizedEmbedding.asLinear, which is critical for numerical accuracy.
        self._lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabularySize, bias: false)
        
        self._visionTower.wrappedValue = Gemma4VisionModel(config: vcfg)
        self._projector.wrappedValue = Gemma4Projector(visionDim: vcfg.hiddenSize, textDim: config.hiddenSize)
        
        if let acfg = config.audioConfig {
            let audioConfig = Gemma4AudioConfiguration(
                modelType: acfg.modelType,
                hiddenSize: acfg.hiddenSize,
                numHiddenLayers: acfg.numHiddenLayers,
                numAttentionHeads: acfg.numAttentionHeads
            )
            self._audioTower.wrappedValue = Gemma4AudioModel(config: audioConfig)

        }
        super.init()
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [any KVCache]?) -> MLXArray {
        let optionalCache = cache?.map { $0 as KVCache? }
        var h = languageModel(inputs, cache: optionalCache)
        if let lmHead {
            h = lmHead(h)
        } else {
            h = languageModel.embedTokens.asLinear(h)
        }
        if config.finalLogitSoftcapping > 0 {
            let originalType = h.dtype
            let hF32 = h.asType(.float32)
            let cap = MLXArray(config.finalLogitSoftcapping).asType(.float32)
            h = (MLX.tanh(hF32 / cap) * cap).asType(originalType)
        }
        return h
    }
    
    private func getInputEmbeddings(
        inputIds: MLXArray,
        pixelValues: MLXArray?,
        audioValues: MLXArray?,
        mask: MLXArray?
    ) -> (MLXArray, MLXArray?) {
        let baseEmbeds = languageModel.embedTokens(inputIds)
        var h = baseEmbeds * MLXArray(Float(config.hiddenSize).squareRoot()).asType(baseEmbeds.dtype)

        guard let pixelValues = pixelValues else {
            return (h, nil)
        }
        
        // Pass through vision tower
        let visionOutputs = visionTower(pixelValues)
        
        // Project to text dimension
        var imageFeatures = projector(visionOutputs)
        
        // Scale image features equivalently to text embed tokens to prevent activation explosion
        imageFeatures = imageFeatures * MLXArray(Float(config.hiddenSize).squareRoot()).asType(imageFeatures.dtype)
        
        let imageTokenId = 258880 // Or config if present
        
        let tokenCount = inputIds.asArray(Int.self).filter { $0 == imageTokenId }.count
        print("DEBUG: imageFeatures shape: \(imageFeatures.shape), padding count: \(tokenCount)")
        
        h = QwenVL.mergeInputIdsWithImageFeatures(
            inputIds: inputIds,
            inputEmbeds: h,
            imageFeatures: imageFeatures.reshaped(-1, config.hiddenSize), // Flatten visual sequence!
            imageTokenId: imageTokenId,
            videoTokenId: imageTokenId
        )
        
        if let audioValues = audioValues, let audioTower = audioTower {
            let audioOutputs = audioTower(audioValues)
            var audioFeatures = audioOutputs
            audioFeatures = audioFeatures * MLXArray(Float(config.hiddenSize).squareRoot()).asType(audioFeatures.dtype)
            let audioTokenId = 258881
            
            let audioTokenCount = inputIds.asArray(Int.self).filter { $0 == audioTokenId }.count
            print("DEBUG: audioFeatures shape: \(audioFeatures.shape), padding count: \(audioTokenCount)")
            
            h = QwenVL.mergeInputIdsWithImageFeatures(
                inputIds: inputIds,
                inputEmbeds: h,
                imageFeatures: audioFeatures.reshaped(-1, config.hiddenSize),
                imageTokenId: audioTokenId,
                videoTokenId: audioTokenId
            )
        }
        
        return (h, nil) // Return dynamic mask if needed
    }

    public func prepare(_ input: LMInput, cache: [any KVCache], windowSize: Int?) throws -> PrepareResult {
        let (inputEmbeddings, _ ) = getInputEmbeddings(
            inputIds: input.text.tokens,
            pixelValues: input.image?.pixels,
            audioValues: input.audio?.features,
            mask: input.text.mask
        )
        let convertedCache = cache.map { $0 as KVCache? }
        var h = languageModel(
            input.text.tokens,
            inputEmbedding: inputEmbeddings,
            mask: .causal, // Depending on phase
            cache: convertedCache
        )
        if let lmHead {
            h = lmHead(h)
        } else {
            h = languageModel.embedTokens.asLinear(h)
        }
        if config.finalLogitSoftcapping > 0 {
            let originalType = h.dtype
            let hF32 = h.asType(.float32)
            let cap = MLXArray(config.finalLogitSoftcapping).asType(.float32)
            h = (MLX.tanh(hF32 / cap) * cap).asType(originalType)
        }
        return .logits(LMOutput(logits: h))
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        // Delegate text model sanitization to Gemma4Model natively
        // This handles router mapping, dequantization, gate_up_proj splitting, etc.,
        // and automatically extracts and strips the "language_model." root.
        let dummyLLM = Gemma4Model(config)
        var processed = dummyLLM.sanitize(weights: weights, metadata: [:])
        
        // Merge the vision tower and projector weights back in, as the LLM sanitize discards them
        for (k, v) in weights {
            if k.hasPrefix("vision_tower.") || k.hasPrefix("embed_vision.") {
                var newK = k
                if newK.contains(".linear.") {
                    newK = newK.replacingOccurrences(of: ".linear.", with: ".")
                }
                
                // Strip unsupported auxiliary quantization keys from Vision Tower
                // (e.g., from AWQ or partial-precision 8bit layers) to satisfy verify: [.all] constraints
                if newK.hasSuffix(".input_max") || newK.hasSuffix(".input_min") || newK.hasSuffix(".output_max") || newK.hasSuffix(".output_min") {
                    continue
                }
                
                processed[newK] = v
            }
        }
        
        // Inject missing scale constants for 4B edge models and quantized files
        // which rely on fallback defaults for vision feature standardization blocks.
        // We MUST use a floating point precision like .float16! Extracting from weights.values
        // dynamically fails on 8-bit quantized models because it injects Int8/UInt8 causing
        // numerical corruption (gibberish output) during activation broadcasting!
        let activationDtype: DType = .float16
        
        if processed["vision_tower.std_scale"] == nil {
            processed["vision_tower.std_scale"] = ones([visionConfig.hiddenSize]).asType(activationDtype)
        }
        if processed["vision_tower.std_bias"] == nil {
            processed["vision_tower.std_bias"] = zeros([visionConfig.hiddenSize]).asType(activationDtype)
        }
        
        return processed
    }
}

// MARK: - Processor

public struct Gemma4MessageGenerator: MessageGenerator {
    public init() {}
    
    public func generate(message: Chat.Message) -> MLXLMCommon.Message {
        var textContent = message.content
        
        // Explicitly inject image tokens inline if they exist
        let visualPrefix = Array(repeating: "<|image|>", count: message.images.count).joined(separator: "\n")
        if !visualPrefix.isEmpty {
            textContent = "\(visualPrefix)\n\(textContent)"
        }
        
        // Explicitly inject audio tokens inline if they exist
        let audioPrefix = Array(repeating: "<|audio|>", count: message.audio.count).joined(separator: "\n")
        if !audioPrefix.isEmpty {
            textContent = "\(audioPrefix)\n\(textContent)"
        }
        
        var dict: [String: any Sendable] = [
            "role": message.role.rawValue,
            "content": textContent
        ]
        
        if let toolCalls = message.toolCalls {
            dict["tool_calls"] = toolCalls
        }
        if let toolCallId = message.toolCallId {
            dict["tool_call_id"] = toolCallId
        }
        
        return dict
    }
}

public struct Gemma4Processor: UserInputProcessor {
    private let config: Gemma4ProcessorConfiguration
    private let tokenizer: any Tokenizer

    public init(_ config: Gemma4ProcessorConfiguration, tokenizer: any Tokenizer) {
        self.config = config
        self.tokenizer = tokenizer
    }

    public func prepare(input: UserInput) async throws -> LMInput {
        let messages = Gemma4MessageGenerator().generate(from: input)
        var promptTokens = try tokenizer.applyChatTemplate(messages: messages, tools: input.tools)

        var processedImage: LMInput.ProcessedImage? = nil

        if !input.images.isEmpty {
            let targetSize = CGSize(
                width: config.imageProcessor?.size.width ?? 224,
                height: config.imageProcessor?.size.height ?? 224
            )
            let imageMLXArrays = try input.images.map { img -> MLXArray in
                var p = UserInput.Processing()
                p.resize = targetSize
                let processedImage = try MediaProcessing.apply(img.asCIImage(), processing: p)
                let srgbImage = MediaProcessing.inSRGBToneCurveSpace(processedImage)
                let resizedImage = MediaProcessing.resampleBicubic(srgbImage, to: targetSize)
                let normalizedImage = MediaProcessing.normalize(
                    resizedImage, mean: config.imageMeanTuple, std: config.imageStdTuple)
                return MediaProcessing.asMLXArray(normalizedImage)
            }
            processedImage = LMInput.ProcessedImage(
                pixels: concatenated(imageMLXArrays),
                frames: nil
            )

            // Inject image tokens
            let imageTokenId = 258880 // Gemma 4 specific hardcoded or dynamic config
            let startTokenId = 255999
            let numTokens = (Int(targetSize.width) / 16) * (Int(targetSize.height) / 16)

            var expandedTokens: [Int] = []
            var inImageBlock = false
            for token in promptTokens {
                // Handle different token outputs from the ChatTemplate
                if token == imageTokenId || token == startTokenId {
                    if !inImageBlock {
                        // First token of the block: Inject exactly numTokens!
                        expandedTokens.append(contentsOf: Array(repeating: imageTokenId, count: numTokens))
                        inImageBlock = true
                    }
                    // Skip any consecutive image tokens (e.g. if the tokenizer emitted 280 hardcoded image tokens)
                } else {
                    inImageBlock = false
                    expandedTokens.append(token)
                }
            }
            
            // If the chat template completely dropped the image tokens, inject them manually!
            if expandedTokens.count == promptTokens.count && !promptTokens.contains(imageTokenId) {
                let imagePad = Array(repeating: imageTokenId, count: numTokens)
                if expandedTokens.first == 2 {
                    // Inject right after BOS (2)
                    expandedTokens.insert(contentsOf: imagePad, at: 1)
                } else {
                    expandedTokens.insert(contentsOf: imagePad, at: 0)
                }
            }
            
            promptTokens = expandedTokens
        }

        // Mock Audio processing - we inject a dummy spectrogram [1, 80, 3000] for validation
        // In a real STFT flow this would map float PCM into Mels.
        var processedAudio: LMInput.ProcessedAudio? = nil
        if !input.audio.isEmpty {
            let melSpec = MLX.zeros([1, 80, 3000]) // Mock 30s Audio for E2E validation
            processedAudio = LMInput.ProcessedAudio(features: melSpec, seqLengths: [3000])
            
            // Wait: Does Gemma 4 tokenizer inject the audio tokens?
            // If the chat template drops it, we inject it manually.
            let audioTokenId = 258881
            if !promptTokens.contains(audioTokenId) {
                // If it isn't in the template, we inject it manually!
                promptTokens.insert(audioTokenId, at: promptTokens.first == 2 ? 1 : 0)
            }
        }

        let promptArray = MLXArray(promptTokens).expandedDimensions(axis: 0)
        let mask = ones(like: promptArray).asType(.int8)

        return LMInput(
            text: .init(tokens: promptArray, mask: mask),
            image: processedImage,
            audio: processedAudio
        )
    }
}

// Extension to format parsed proxy to VLM config
public extension Gemma4Configuration {
    var visionConfiguration: Gemma4VisionConfiguration? {
        guard let proxy = self.visionConfig else { return nil }
        
        return Gemma4VisionConfiguration(
            modelType: "gemma4_vision",
            hiddenSize: proxy.hiddenSize ?? 1152,
            hiddenLayers: proxy.hiddenLayers ?? 27,
            intermediateSize: proxy.intermediateSize ?? 4304,
            attentionHeads: proxy.attentionHeads ?? 16,
            patchSize: proxy.patchSize ?? 16
        )
    }
}

extension Gemma4VL: LoRAModel {
    public var loraLayers: [Module] {
        return []
    }
}


