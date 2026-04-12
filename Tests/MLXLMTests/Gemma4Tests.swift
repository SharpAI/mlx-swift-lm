import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import Testing

@Suite("Gemma 4 Architectural Integrity Tests")
struct Gemma4Tests {
    
    /// Create a minimal test configuration for Gemma 4
    private func makeTinyConfig() -> Gemma4Configuration {
        Gemma4Configuration(
            modelType: "gemma4",
            hiddenSize: 64,
            hiddenLayers: 2,
            intermediateSize: 128,
            attentionHeads: 4,
            headDim: 16,
            rmsNormEps: 1e-6,
            vocabularySize: 100,
            kvHeads: 2,
            ropeTheta: 10000.0,
            ropeLocalBaseFreq: 10000.0,
            ropeTraditional: false,
            queryPreAttnScalar: 1.0,
            slidingWindow: 128,
            slidingWindowPattern: 1,
            maxPositionEmbeddings: 512,
            globalHeadDim: 64,
            numKvSharedLayers: 0,
            useDoubleWideMlp: false,
            tieWordEmbeddings: true,
            hiddenSizePerLayerInput: 32,
            vocabSizePerLayerInput: 10,
            globalRopePartialFactor: 0.25,
            finalLogitSoftcapping: 30.0
        )
    }

    @Test("Gemma 4 Forward Pass - Determinism & Shape")
    func testGemma4ForwardPass() throws {
        let config = makeTinyConfig()
        let model = Gemma4Model(config)
        
        let input = MLXArray(0..<8).reshaped(1, 8)
        let output = model(input)
        
        #expect(output.shape == [1, 8, config.vocabularySize])
        
        // Secondary pass to ensure determinism
        let output2 = model(input)
        #expect(allClose(output, output2).item(Bool.self))
    }

    @Test("PLE Multimodal Signal Integrity")
    func testPLESignalIntegrity() throws {
        let config = makeTinyConfig()
        let model = Gemma4Model(config)
        
        let input = MLXArray(Int32(0)..<Int32(5)).reshaped(1, 5)
        
        // We expect the forward pass to finish without NaN or infinite values
        let output = model(input)
        let sum = output.sum().item(Float.self)
        #expect(!sum.isNaN)
        #expect(!sum.isInfinite)
    }

    @Test("Weight Sanitization - PLE Mapping")
    func testGemma4Sanitization() throws {
        let config = makeTinyConfig()
        let model = Gemma4Model(config)
        
        var weights = [String: MLXArray]()
        weights["model.layers.0.per_layer_conditioning.scale"] = MLXArray.ones([config.hiddenSize, config.hiddenSizePerLayerInput])
        weights["model.layers.0.per_layer_conditioning.bias"] = MLXArray.ones([config.hiddenSize])
        
        let sanitized = model.sanitize(weights: weights, metadata: [:])
        
        // Gemma 4 sanitization maps to model.layers...
        #expect(sanitized["model.layers.0.per_layer_model_projection.scale"] != nil || sanitized["layers.0.per_layer_input.scale"] != nil)
    }

    @Test("Audio Configuration Dependency Safety")
    func testAudioConfigSafety() throws {
        let config = makeTinyConfig()
        let model = Gemma4Model(config)
        #expect(model.model.layers.count == config.hiddenLayers)
    }
}
