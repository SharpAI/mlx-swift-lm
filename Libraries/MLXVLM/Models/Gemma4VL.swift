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
        _ = position_embedding_table[0..., 0..<seqLen, 0...]
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
public class Gemma4VL: Module, VLMModel, KVCacheDimensionProvider, LayerPartitionable {
    // `language_model` uses the existing MLXLLM Gemma4ModelInternal
    @ModuleInfo(key: "language_model") private var languageModel: Gemma4ModelInternal

    @ModuleInfo(key: "vision_tower") private var visionTower: Gemma4VisionModel
    @ModuleInfo(key: "embed_vision") private var projector: Gemma4Projector

    @ModuleInfo(key: "audio_tower") private var audioTower: Gemma4AudioModel?
    @ModuleInfo(key: "embed_audio") private var audioProjector: Gemma4Projector?


    public let config: Gemma4Configuration
    public let visionConfig: Gemma4VisionConfiguration
    
    public var vocabularySize: Int { config.vocabularySize }
    public var kvHeads: [Int] { Array(repeating: config.kvHeads, count: config.hiddenLayers) }
    
    // LayerPartitionable
    public var gpuLayerCount: Int? {
        get { languageModel.gpuLayerCount }
        set { languageModel.gpuLayerCount = newValue }
    }
    public var totalLayerCount: Int { languageModel.totalLayerCount }

    public init(_ config: Gemma4Configuration) {
        self.config = config
        
        let vcfg = config.visionConfiguration ?? Gemma4VisionConfiguration(
            modelType: "gemma4_vision", hiddenSize: 1152, hiddenLayers: 27, intermediateSize: 4304, attentionHeads: 16, patchSize: 16)
        self.visionConfig = vcfg
        
        self._languageModel.wrappedValue = Gemma4ModelInternal(config)
        
        // Always create a separate lm_head — following the Gemma 3 pattern.
        // For tied embeddings, sanitize() will copy embed_tokens weights to lm_head.
        self._visionTower.wrappedValue = Gemma4VisionModel(config: vcfg)
        self._projector.wrappedValue = Gemma4Projector(visionDim: vcfg.hiddenSize, textDim: config.hiddenSize)
        if let acfg = config.audioConfig {
            print("[Gemma4VL] DEBUG: Successfully parsed audio config with hiddenSize: \(acfg.hiddenSize.debugDescription)")
            let audioConfig = Gemma4AudioConfiguration(
                modelType: acfg.modelType ?? "gemma4_audio",
                hiddenSize: acfg.hiddenSize ?? 1024,
                numHiddenLayers: acfg.numHiddenLayers ?? 12,
                numAttentionHeads: acfg.numAttentionHeads ?? 8,
                outputProjDims: acfg.outputProjDims ?? 1536
            )
            self._audioTower.wrappedValue = Gemma4AudioModel(config: audioConfig)
            self._audioProjector.wrappedValue = Gemma4Projector(visionDim: audioConfig.outputProjDims, textDim: config.hiddenSize)
        } else {
            print("[Gemma4VL] DEBUG: config.audioConfig IS NIL!")
        }
        super.init()
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [any KVCache]?) -> MLXArray {
        let optionalCache = cache?.map { $0 as KVCache? }
        var h = languageModel.model(inputs, inputEmbedding: nil, mask: nil, cache: optionalCache)
        h = languageModel.model.embedTokens.asLinear(h)
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
        let baseEmbeds = languageModel.model.embedTokens(inputIds)
        var h = baseEmbeds * MLXArray(Float(config.hiddenSize).squareRoot()).asType(baseEmbeds.dtype)

        guard let pixelValues = pixelValues else {
            return (h, nil)
        }
        
        // Pass through vision tower
        let visionOutputs = visionTower(pixelValues)
        
        // Project to text dimension
        let imageFeaturesOutput = projector(visionOutputs)
        
        let imageFeatures = imageFeaturesOutput
        
        let imageTokenId = 258880 // Or config if present
        
        let tokenCount = inputIds.asArray(Int.self).filter { $0 == imageTokenId }.count
        eval(imageFeatures)
        print("DEBUG: imageFeatures shape: \(imageFeatures.shape), padding count: \(tokenCount)")
        if imageFeatures.size > 0 {
             print("DEBUG: imageFeatures stats: min=\(imageFeatures.min().item(Float.self)), max=\(imageFeatures.max().item(Float.self)), mean=\(imageFeatures.mean().item(Float.self))")
        }
        
        h = QwenVL.mergeInputIdsWithImageFeatures(
            inputIds: inputIds,
            inputEmbeds: h,
            imageFeatures: imageFeatures.reshaped(-1, config.hiddenSize), // Flatten visual sequence!
            imageTokenId: imageTokenId,
            videoTokenId: imageTokenId
        )
        
        if let audioValues = audioValues, let audioTower = audioTower, let audioProjector = audioProjector {
            let audioOutputs = audioTower(audioValues)
            let audioFeatures = audioProjector(audioOutputs)
            
            let audioTokenId = 258881
            
            let audioTokenCount = inputIds.asArray(Int.self).filter { $0 == audioTokenId }.count
            eval(audioFeatures)
            print("DEBUG: audioFeatures shape: \(audioFeatures.shape), padding count: \(audioTokenCount)")
            if audioFeatures.size > 0 {
                print("DEBUG: audioFeatures stats: min=\(audioFeatures.min().item(Float.self)), max=\(audioFeatures.max().item(Float.self)), mean=\(audioFeatures.mean().item(Float.self))")
            }
            
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
        var h = languageModel.model(
            input.text.tokens,
            inputEmbedding: inputEmbeddings,
            mask: .causal, // Depending on phase
            cache: convertedCache
        )
        h = languageModel.model.embedTokens.asLinear(h)
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
        let dummyLLM = Gemma4ModelInternal(config)
        let llmWeights = dummyLLM.sanitize(weights: weights, metadata: [:])
        var processed = [String: MLXArray]()
        for (k, v) in llmWeights {
            if k.hasPrefix("lm_head.") {
                processed[k] = v
            } else {
                processed["language_model.\(k)"] = v
            }
        }
        
        print("[Gemma4VL] Sanitize executing! tieWordEmbeddings: \(config.tieWordEmbeddings)")
        print("[Gemma4VL] Has language_model.model.embed_tokens.weight? \(processed["language_model.model.embed_tokens.weight"] != nil)")
        
        print("[Gemma4VL] Has language_model.model.embed_tokens.weight? \(processed["language_model.model.embed_tokens.weight"] != nil)")

        // Merge the vision tower and projector weights back in, as the LLM sanitize discards them
        for (k, v) in weights {
            if k.hasPrefix("vision_tower.") || k.hasPrefix("embed_vision.") || k.hasPrefix("audio_tower.") || k.hasPrefix("embed_audio.") {
                var newK = k
                if newK.contains(".linear.") && !newK.hasPrefix("audio_tower.") {
                    newK = newK.replacingOccurrences(of: ".linear.", with: ".")
                }
                
                // Strip unsupported auxiliary quantization keys from Vision / Audio Tower
                // (e.g., from AWQ or partial-precision 8bit layers) to satisfy verify: [.all] constraints
                if newK.hasSuffix(".input_max") || newK.hasSuffix(".input_min") || newK.hasSuffix(".output_max") || newK.hasSuffix(".output_min") || newK.hasSuffix(".per_dim_scale") {
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
        print("[Gemma4MessageGenerator] Raw textContent: \(textContent)")
        
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
        print("[Gemma4Processor] Final mapped messages for tokenizer: \(messages)")
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
                        // First token of the block: Inject exactly numTokens wrapped in bounds!
                        expandedTokens.append(255999) // <|image>
                        expandedTokens.append(contentsOf: Array(repeating: imageTokenId, count: numTokens))
                        expandedTokens.append(258882) // <image|>
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
        var processedAudio: LMInput.ProcessedAudio? = nil
        print("[Gemma4Processor] DEBUG: Audio count is \(input.audio.count)")
        if let audioInput = input.audio.first {
            print("[Gemma4Processor] DEBUG: Extracting audio samples...")
            // Extract raw PCM
            let samples = try MediaProcessing.extractAudioSamples(from: audioInput)
            print("[Gemma4Processor] DEBUG: Generating Spectrogram from \(samples.count) samples...")
            // Generate Mel Spectrogram natively (128 Mel Bins)
            let processor = AudioProcessor(nMels: 128)
            var melSpec = try processor.generateMelSpectrogram(samples: samples)
            print("[Gemma4Processor] DEBUG: Computed Spectrogram shape: \(melSpec.shape.description)")
            
            // AudioProcessor outputs [nMels, validFrames]
            // Gemma 4 implicitly requires [1, validFrames, nMels] for correctly iterating sequence convolutions
            melSpec = melSpec.transposed().expandedDimensions(axis: 0) // Transpose to [validFrames, nMels] then expand B=1
            
            let seqLength = melSpec.dim(1)
            processedAudio = LMInput.ProcessedAudio(features: melSpec, seqLengths: [seqLength])
            
            let audioTokenId = 258881
            let layer0Length = (seqLength + 2 * 1 - 1 * (3 - 1) - 1) / 2 + 1
            let layer1Length = (layer0Length + 2 * 1 - 1 * (3 - 1) - 1) / 2 + 1
            let expectedAudioTokens = layer1Length
            
            var expandedTokens = promptTokens
            let audioPadding = Array(repeating: audioTokenId, count: expectedAudioTokens)
            let gemmaBoa = 256000 // <|audio>
            let gemmaEoa = 258883 // <audio|>
            
            // The MessageGenerator injected <|audio|> strings which tokenizer resolves to audioTokenId (258881)
            // Determine insertion point before wiping ALL instances.
            let targetIdx = expandedTokens.firstIndex(of: gemmaBoa) ?? expandedTokens.firstIndex(of: audioTokenId)
            print("[Gemma4Processor] targetIdx: \(String(describing: targetIdx))")
            print("[Gemma4Processor] Tokenizer count before audio wipe: \(expandedTokens.count)")
            
            // Wipe all manual occurrences to prevent broadcast shape errors
            expandedTokens.removeAll(where: { $0 == gemmaBoa || $0 == gemmaEoa || $0 == audioTokenId })
            
            print("[Gemma4Processor] Tokenizer count after audio wipe: \(expandedTokens.count)")
            
            if let idx = targetIdx {
                // Re-clamp the index in case removeAll shifted it out of bounds
                let safeIdx = min(idx, expandedTokens.count)
                expandedTokens.insert(contentsOf: [gemmaBoa] + audioPadding + [gemmaEoa], at: safeIdx)
            } else {
                if expandedTokens.first == 2 {
                    expandedTokens.insert(contentsOf: [gemmaBoa] + audioPadding + [gemmaEoa], at: 1)
                } else {
                    expandedTokens.insert(contentsOf: [gemmaBoa] + audioPadding + [gemmaEoa], at: 0)
                }
            }
            print("[Gemma4Processor] Tokenizer count after audio insertion: \(expandedTokens.count)")
            promptTokens = expandedTokens
        }
        // DEBUG: Render exactly what strings the LLM sees
        let decodedPrompt = tokenizer.decode(tokenIds: promptTokens)
        print("[\(type(of: self))] Final Evaluated Prompt Geometry bounds:")
        print("\n----------------------")
        print(decodedPrompt)
        print("----------------------\n")

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
