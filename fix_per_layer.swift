import Foundation

let path = "Libraries/MLXLLM/Models/Gemma4.swift"
var file = try! String(contentsOfFile: path)
let target = """
            // Token-based per-layer embeddings
            var tokenEmbeds = embedPerLayer(inputs)
                .reshaped(B, L, nL, D)  // [B, L, numLayers, D]

            // MASK OUT MULTIMODAL TOKENS: Zero out text-space embeddings for visual/audio tokens
            let isTextToken = MLX.logicalOr(MLX.less(inputs, MLXArray(258880)), MLX.greater(inputs, MLXArray(258884)))
            let expandedMask = isTextToken.reshaped([B, L, 1, 1]).asType(tokenEmbeds.dtype)
            tokenEmbeds = tokenEmbeds * expandedMask

            // Model projection (visual/audio/text dense state mapping)
            let modelProjected = modelProj(h).reshaped(B, L, nL, D)
            let modelProjectedNormed = projNorm(modelProjected)

            // Combine and structure to HF standard
            let perLayerInputScale = MLXArray(sqrt(Float(D))).asType(h.dtype)
            perLayerInputs = (tokenEmbeds + modelProjectedNormed) * perLayerInputScale
"""

let replace = """
            // Token-based per-layer embeddings
            var tokenEmbeds = embedPerLayer(inputs)
            tokenEmbeds = tokenEmbeds * MLXArray(sqrt(Float(D))).asType(tokenEmbeds.dtype)
            tokenEmbeds = tokenEmbeds.reshaped(B, L, nL, D)

            // MASK OUT MULTIMODAL TOKENS: Zero out text-space embeddings for visual/audio tokens
            let isTextToken = MLX.logicalOr(MLX.less(inputs, MLXArray(258880)), MLX.greater(inputs, MLXArray(258884)))
            let expandedMask = isTextToken.reshaped([B, L, 1, 1]).asType(tokenEmbeds.dtype)
            tokenEmbeds = tokenEmbeds * expandedMask

            // Model projection (visual/audio/text dense state mapping)
            var modelProjected = modelProj(h)
            modelProjected = modelProjected * MLXArray(1.0 / sqrt(Float(config.hiddenSize))).asType(modelProjected.dtype)
            modelProjected = modelProjected.reshaped(B, L, nL, D)
            let modelProjectedNormed = projNorm(modelProjected)

            // Combine and structure to HF standard
            let combineScale = MLXArray(1.0 / sqrt(2.0)).asType(h.dtype)
            perLayerInputs = (tokenEmbeds + modelProjectedNormed) * combineScale
"""

file = file.replacingOccurrences(of: target, with: replace)
try! file.write(toFile: path, atomically: true, encoding: .utf8)
