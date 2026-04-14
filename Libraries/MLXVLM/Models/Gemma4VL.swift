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
    public var poolingKernelSize: Int = 3
    private let _imageSize: Int?
    public var imageSize: Int { _imageSize ?? 448 }
    public let ropeParameters: [String: AnyCodable]?

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
        self.poolingKernelSize = 3
        self._imageSize = imageSize
        self.ropeParameters = nil
    }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenLayers = "num_hidden_layers"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case patchSize = "patch_size"
        case poolingKernelSize = "pooling_kernel_size"
        case _imageSize = "image_size"
        case ropeParameters = "rope_parameters"
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
        public let poolingKernelSize: Int?
        public let doNormalize: Bool?

        enum CodingKeys: String, CodingKey {
            case imageProcessorType = "image_processor_type"
            case imageMean = "image_mean"
            case imageStd = "image_std"
            case size
            case resample
            case rescaleFactor = "rescale_factor"
            case poolingKernelSize = "pooling_kernel_size"
            case doNormalize = "do_normalize"
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

    public var doNormalize: Bool {
        imageProcessor?.doNormalize ?? false
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
    let ropeBaseFrequency: Float

    init(dimensions: Int, numHeads: Int, bias: Bool = false, ropeBaseFrequency: Float = 10000.0) {
        self.numHeads = numHeads
        let headDim = dimensions / numHeads
        self.scale = pow(Float(headDim), -0.5)
        self.ropeBaseFrequency = ropeBaseFrequency

        self._queryProj.wrappedValue = Linear(dimensions, dimensions, bias: bias)
        self._keyProj.wrappedValue = Linear(dimensions, dimensions, bias: bias)
        self._valueProj.wrappedValue = Linear(dimensions, dimensions, bias: bias)
        self._outputProj.wrappedValue = Linear(dimensions, dimensions, bias: bias)

        self._queryNorm.wrappedValue = RMSNorm(dimensions: headDim)
        self._keyNorm.wrappedValue = RMSNorm(dimensions: headDim)
    }

    private func applyMultidimensionalRope(_ inputs: MLXArray, _ positions: MLXArray?, baseFrequency: Float = 10000.0) -> MLXArray {
        guard let positions = positions else { return inputs }
        let headDim = inputs.dim(-1)
        let ndim = positions.dim(-1)
        let channelsPerDim = 2 * (headDim / (2 * ndim))
        let halfPerDim = channelsPerDim / 2
        
        var resultParts: [MLXArray] = []
        for d in 0..<ndim {
            let startIdx = d * channelsPerDim
            let endIdx = (d + 1) * channelsPerDim
            let xPart = inputs[0..., 0..., 0..., startIdx..<endIdx]
            
            let exps = (0..<halfPerDim).map { Float($0) }
            let freqExponents = (2.0 / Float(channelsPerDim)) * MLXArray(exps)
            let timescale = pow(MLXArray(baseFrequency), freqExponents)
            let sinusoidInp = positions[0..., 0..., d..<(d+1)].asType(Float.self) / timescale
            
            var cosD = MLX.cos(sinusoidInp)
            var sinD = MLX.sin(sinusoidInp)
            
            cosD = MLX.concatenated([cosD, cosD], axis: -1).asType(inputs.dtype)
            sinD = MLX.concatenated([sinD, sinD], axis: -1).asType(inputs.dtype)
            cosD = cosD.expandedDimensions(axis: 1) // Broadcast over numHeads
            sinD = sinD.expandedDimensions(axis: 1)
            
            let x1 = xPart[0..., 0..., 0..., 0..<halfPerDim]
            let x2 = xPart[0..., 0..., 0..., halfPerDim...]
            let xRotated = MLX.concatenated([-x2, x1], axis: -1)
            
            let yPart = xPart * cosD + xRotated * sinD
            resultParts.append(yPart)
        }
        return MLX.concatenated(resultParts, axis: -1)
    }

    func callAsFunction(_ x: MLXArray, positions: MLXArray? = nil, mask: MLXFast.ScaledDotProductAttentionMaskMode = .none)
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

        queries = applyMultidimensionalRope(queries, positions, baseFrequency: ropeBaseFrequency)
        keys = applyMultidimensionalRope(keys, positions, baseFrequency: ropeBaseFrequency)

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
        let ropeBaseFrequency = (config.ropeParameters?["rope_theta"]?.value as? Double).map { Float($0) } ?? 100.0
        self._selfAttention.wrappedValue = Gemma4VisionAttention(
            dimensions: config.hiddenSize, numHeads: config.attentionHeads, bias: false, ropeBaseFrequency: ropeBaseFrequency)
        self.mlp = Gemma4VisionMLP(config: config)
        
        self._inputLayerNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)
        self._postAttentionLayerNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)
        self._preFeedforwardLayerNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)
        self._postFeedforwardLayerNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)
    }

    func callAsFunction(_ x: MLXArray, positions: MLXArray? = nil, mask: MLXFast.ScaledDotProductAttentionMaskMode = .none) -> MLXArray {
        let r = selfAttention(inputLayerNorm(x), positions: positions, mask: mask)
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

    func callAsFunction(_ x: MLXArray, positions: MLXArray? = nil, mask: MLXFast.ScaledDotProductAttentionMaskMode = .none) -> MLXArray {
        var h = x
        for layer in layers {
            h = layer(h, positions: positions, mask: mask)
        }
        return h
    }
}

private class Gemma4PositionEmbedding: Module {
    @ModuleInfo(key: "weight") var weight: MLXArray

    init(shape: [Int]) {
        self._weight.wrappedValue = zeros(shape)
        super.init()
    }
}

private class Gemma4PatchEmbedder: Module {
    @ModuleInfo(key: "input_proj") var inputProj: Linear
    @ModuleInfo(key: "position_embedding_table") var positionEmbeddingTable: Gemma4PositionEmbedding
    let patchSize: Int

    init(config: Gemma4VisionConfiguration) {
        self.patchSize = config.patchSize
        let inFeat = config.numChannels * config.patchSize * config.patchSize
        self._inputProj.wrappedValue = Linear(inFeat, config.hiddenSize, bias: false)
        self._positionEmbeddingTable.wrappedValue = Gemma4PositionEmbedding(shape: [2, 10240, config.hiddenSize])
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // x is [B, C, H, W]
        let B = x.dim(0)
        let C = x.dim(1)
        let H = x.dim(2)
        let W = x.dim(3)
        let p = patchSize
        
        let pH = H / p
        let pW = W / p
        
        // Reshape: [B, C, pH, p, pW, p] -> transpose to [B, pH, pW, p, p, C] -> reshape to [B, pH * pW, p * p * C]
        var patches = x.reshaped(B, C, pH, p, pW, p)
                       .transposed(0, 2, 4, 3, 5, 1)
                       .reshaped(B, pH * pW, C * p * p)
                       
        // Scale 0...1 to -1...1
        patches = (patches - MLXArray(0.5, dtype: patches.dtype)) * MLXArray(2.0, dtype: patches.dtype)
        
        let hArr = inputProj(patches)
        
        // Add positional embeddings
        // The table is [2, 10240, hiddenSize]. Index 0 is X (columns), Index 1 is Y (rows).
        let table = positionEmbeddingTable.weight
        let colEmbeds = table[0, 0..<pW, 0...] // [pW, hiddenSize]
        let rowEmbeds = table[1, 0..<pH, 0...] // [pH, hiddenSize]
        
        let colExpanded = colEmbeds.expandedDimensions(axis: 0) // [1, pW, hiddenSize]
        let rowExpanded = rowEmbeds.expandedDimensions(axis: 1) // [pH, 1, hiddenSize]
        
        let gridEmbeds = colExpanded + rowExpanded // [pH, pW, hiddenSize]
        let flatGrid = gridEmbeds.reshaped(-1, table.dim(2)) // [pH * pW, hiddenSize]
        
        return hArr + flatGrid.expandedDimensions(axis: 0)
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
        let B = x.dim(0)
        let pH = x.dim(2) / config.patchSize
        let pW = x.dim(3) / config.patchSize
        
        let h = patch_embedder(x)
        
        // Generate X/Y spatial grid for 2D RoPE
        let gY = MLXArray((0..<pH).map { Int32($0) }).reshaped(pH, 1) + zeros([pH, pW], type: Int32.self)
        let gX = MLXArray((0..<pW).map { Int32($0) }).reshaped(1, pW) + zeros([pH, pW], type: Int32.self)
        let flatY = gY.reshaped(1, pH * pW, 1) // [1, L, 1]
        let flatX = gX.reshaped(1, pH * pW, 1) // [1, L, 1]
        
        let positions = MLX.concatenated([flatX, flatY], axis: -1) + zeros([B, pH * pW, 2], type: Int32.self) // [B, L, 2]
        
        var encoded = encoder(h, positions: positions)
        
        // --- Vision Pooler ---
        // Gemma 4 pools patches using config.poolingKernelSize (e.g., 3).
        // Since we ensure dimensions are divisible by poolingKernelSize * patchSize, 
        // we can reshape and take the mean along the spatial blocks.
        let k = config.poolingKernelSize
        encoded = encoded.reshaped([B, pH / k, k, pW / k, k, config.hiddenSize])
                         .mean(axes: [2, 4])
                         .reshaped([B, (pH / k) * (pW / k), config.hiddenSize])
                         
        let rootHiddenSize = MLXArray(Float(config.hiddenSize).squareRoot()).asType(encoded.dtype)
        encoded = encoded * rootHiddenSize
        
        return (encoded - std_bias) * std_scale
    }
}

// MARK: - Multimodal Projector 

private class Gemma4RMSNormNoScale: Module, UnaryLayer {
    let eps: Float
    
    init(eps: Float) {
        self.eps = eps
        super.init()
    }
    
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let meanSq = MLX.mean(MLX.square(x.asType(.float32)), axes: [-1], keepDims: true)
        let inv = MLX.rsqrt(meanSq + MLXArray(eps).asType(.float32))
        return x * inv.asType(x.dtype)
    }
}

private class Gemma4Projector: Module, UnaryLayer {
    @ModuleInfo(key: "embedding_projection") var projection: any UnaryLayer
    @ModuleInfo(key: "embedding_pre_projection_norm") var norm: Gemma4RMSNormNoScale
    
    init(visionDim: Int, textDim: Int, eps: Float) {
        self._projection.wrappedValue = Linear(visionDim, textDim, bias: false)
        self._norm.wrappedValue = Gemma4RMSNormNoScale(eps: eps)
        super.init()
    }
    
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        return projection(norm(x))
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
        self._projector.wrappedValue = Gemma4Projector(visionDim: vcfg.hiddenSize, textDim: config.hiddenSize, eps: vcfg.layerNormEps)
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
            self._audioProjector.wrappedValue = Gemma4Projector(visionDim: audioConfig.outputProjDims, textDim: config.hiddenSize, eps: 1e-6)
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
        
        let imageMaskExpanded = broadcast(MLX.expandedDimensions(inputIds .== imageTokenId, axis: -1), to: h.shape)
        h = gemma4MaskedScatter(
            inputTensor: h,
            mask: imageMaskExpanded,
            source: imageFeatures.reshaped(1, -1, config.hiddenSize)
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
            
            let audioMaskExpanded = broadcast(MLX.expandedDimensions(inputIds .== audioTokenId, axis: -1), to: h.shape)
            h = gemma4MaskedScatter(
                inputTensor: h,
                mask: audioMaskExpanded,
                source: audioFeatures.reshaped(1, -1, config.hiddenSize)
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
        return PrepareResult.logits(LMOutput(logits: h))
    }

    private func gemma4MaskedScatter(
        inputTensor: MLXArray, mask: MLXArray, source: MLXArray
    ) -> MLXArray {
        let flattenedInput = inputTensor.flattened()
        let flattenedMask = mask.flattened().asArray(Bool.self)
        let flattenedSource = source.flattened()

        let targetIndices = flattenedMask.enumerated().compactMap { idx, value in
            value ? Int32(idx) : nil
        }
        guard !targetIndices.isEmpty else {
            return inputTensor
        }

        guard flattenedSource.shape[0] == targetIndices.count else {
            fatalError(
                "Masked scatter shape mismatch. source=\(flattenedSource.shape[0]) mask=\(targetIndices.count)"
            )
        }

        let result = MLX.scatter(flattenedInput, indices: MLXArray(targetIndices), updates: flattenedSource, axes: [0])
        return result.reshaped(inputTensor.shape)
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
                

                if newK == "vision_tower.patch_embedder.position_embedding_table" {
                    processed["vision_tower.patch_embedder.position_embedding_table.weight"] = v
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
            // Gemma 4 requires dimensions divisible by pooling_kernel_size * patch_size (48).
            // Width: 288, Height: 288 perfectly adheres to aspect ratio block mapping, producing EXACTLY 36 soft tokens.
            let targetSize = CGSize(width: 288, height: 288)
            let imageMLXArrays = try input.images.map { img -> MLXArray in
                var p = UserInput.Processing()
                p.resize = targetSize
                let processedImage = try MediaProcessing.apply(img.asCIImage(), processing: p)
                let srgbImage = MediaProcessing.inSRGBToneCurveSpace(processedImage)
                let resizedImage = MediaProcessing.resampleBicubic(srgbImage, to: targetSize)
                let finalImage = config.doNormalize ? MediaProcessing.normalize(
                    resizedImage, mean: config.imageMeanTuple, std: config.imageStdTuple) : resizedImage
                return MediaProcessing.asMLXArray(finalImage)
            }
            processedImage = LMInput.ProcessedImage(
                pixels: concatenated(imageMLXArrays),
                frames: nil
            )

            // Inject image tokens
            let imageTokenId = 258880 // Gemma 4 specific hardcoded or dynamic config
            let startTokenId = 255999
            let patchSize = 16
            let poolingKernel = config.imageProcessor?.poolingKernelSize ?? 3
            let sideMult = patchSize * poolingKernel
            let numTokens = (Int(targetSize.width) / sideMult) * (Int(targetSize.height) / sideMult)
            print("[Gemma4VL] DEBUG: Injecting \(numTokens) pooled image tokens.")

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
                var imagePad = [255999] // <|image>
                imagePad.append(contentsOf: Array(repeating: imageTokenId, count: numTokens))
                imagePad.append(258882) // <image|>
                
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
            let gemmaBoa = 256000 // <|audio>
            let gemmaEoa = 258883 // <audio|>
            
            var audioPadding = [gemmaBoa]
            audioPadding.append(contentsOf: Array(repeating: audioTokenId, count: expectedAudioTokens))
            audioPadding.append(gemmaEoa)
            
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
                expandedTokens.insert(contentsOf: audioPadding, at: safeIdx)
            } else {
                if expandedTokens.first == 2 {
                    expandedTokens.insert(contentsOf: audioPadding, at: 1)
                } else {
                    expandedTokens.insert(contentsOf: audioPadding, at: 0)
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
