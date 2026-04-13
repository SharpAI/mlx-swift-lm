import Foundation

let path = "Libraries/MLXLLM/Models/Gemma4.swift"
var file = try! String(contentsOfFile: path)

let targetBlock1 = """
        // Potential per-layer conditioning
        if let perLayerInput = perLayerInput,
           let perLayerProj = perLayerProjection,
           let perLayerGate = perLayerInputGate,
           let postPerLayerNorm = postPerLayerInputNorm {
            let gate = MLX.sigmoid(perLayerGate(x))
            let gatedInput = perLayerInput * gate
            let conditioned = perLayerProj(gatedInput)
            residual = residual + postPerLayerNorm(conditioned)
        }

        var h = inputLayerNorm(residual)
        h = attention(h, mask: mask ?? .none, cache: cache)
        h = postAttentionLayerNorm(h)
        var out = residual + (h * layerScalar)

        residual = out
        h = preFeedforwardLayerNorm(residual)
        h = mlp(h)
        h = postFeedforwardLayerNorm(h)
        out = residual + (h * layerScalar)

        return out
"""

let replaceBlock1 = """
        // === HuggingFace reference: Gemma4TextDecoderLayer.forward ===
        // 1. Attention block with residual
        var residual = x
        var hiddenStates = inputLayerNorm(residual)
        hiddenStates = attention(hiddenStates, mask: mask ?? .none, cache: cache)
        hiddenStates = postAttentionLayerNorm(hiddenStates)
        hiddenStates = residual + hiddenStates

        // 2. MLP block with residual
        residual = hiddenStates
        hiddenStates = preFeedforwardLayerNorm(hiddenStates)
        hiddenStates = mlp(hiddenStates)
        hiddenStates = postFeedforwardLayerNorm(hiddenStates)
        hiddenStates = residual + hiddenStates

        // 3. Per-layer conditioning (AFTER MLP, per HF reference)
        if let perLayerInput = perLayerInput,
           let perLayerProj = perLayerProjection,
           let perLayerGate = perLayerInputGate {
            residual = hiddenStates
            let gate = geluApproximate(perLayerGate(hiddenStates))
            let gatedInput = gate * perLayerInput
            hiddenStates = perLayerProj(gatedInput)
            if let postNorm = postPerLayerInputNorm {
                hiddenStates = postNorm(hiddenStates)
            }
            hiddenStates = residual + hiddenStates
        }

        // 4. Global layer scalar (applied to ENTIRE residual stream, not individual deltas)
        hiddenStates = hiddenStates * layerScalar
        return hiddenStates
"""

file = file.replacingOccurrences(of: targetBlock1, with: replaceBlock1)

let targetBlock2 = """
            // Token-based per-layer embeddings
            let tokenScale = MLXArray(sqrt(Float(D))).asType(h.dtype)
            var tokenEmbeds = (embedPerLayer(inputs) * tokenScale)
                .reshaped(B, L, nL, D)  // [B, L, numLayers, D]

            // MASK OUT MULTIMODAL TOKENS: Zero out text-space embeddings for visual/audio tokens
            let isTextToken = MLX.logicalOr(MLX.less(inputs, MLXArray(258880)), MLX.greater(inputs, MLXArray(258884)))
            let expandedMask = isTextToken.reshaped([B, L, 1, 1]).asType(tokenEmbeds.dtype)
            tokenEmbeds = tokenEmbeds * expandedMask

            // Model projection (visual/audio/text dense state mapping)
            let modelProjected = modelProj(h)           // [B, L, numLayers * D]
                               .reshaped(B, L, nL, D)

            // Combine and normalize per layer
            let modelProjectedNormed = projNorm(modelProjected)
            let combineScale = MLXArray(1.0 / sqrt(2.0)).asType(h.dtype)
            perLayerInputs = (tokenEmbeds + modelProjectedNormed) * combineScale
"""

let replaceBlock2 = """
            // Token-based per-layer embeddings
            var tokenEmbeds = embedPerLayer(inputs)
                .reshaped(B, L, nL, D)  // [B, L, numLayers, D]

            // MASK OUT MULTIMODAL TOKENS: Zero out text-space embeddings for visual/audio tokens
            let isTextToken = MLX.logicalOr(MLX.less(inputs, MLXArray(258880)), MLX.greater(inputs, MLXArray(258884)))
            let expandedMask = isTextToken.reshaped([B, L, 1, 1]).asType(tokenEmbeds.dtype)
            tokenEmbeds = tokenEmbeds * expandedMask

            // Model projection (visual/audio/text dense state mapping)
            let modelProjected = modelProj(h)           // [B, L, numLayers * D]
                               .reshaped(B, L, nL, D)

            // Combine and scale by sqrt(D)
            let modelProjectedNormed = projNorm(modelProjected)
            let perLayerInputScale = MLXArray(sqrt(Float(D))).asType(h.dtype)
            perLayerInputs = (modelProjectedNormed + tokenEmbeds) * perLayerInputScale
"""

file = file.replacingOccurrences(of: targetBlock2, with: replaceBlock2)

// Also fix RMSNorm back to what MLX Fast expects (`weight + 1.0`) if the initial mean was 10.
// We keep your `self.weight + 1.0` if you reverted it back, let's just make sure it uses exactly what HF Gemma 3 script and Google did!
// Actually, earlier you replaced it to `self.weight + 1.0`. I'll keep it if it's there.

try! file.write(toFile: path, atomically: true, encoding: .utf8)
