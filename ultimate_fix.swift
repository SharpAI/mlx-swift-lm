import Foundation

let path = "Libraries/MLXLLM/Models/Gemma4.swift"
var file = try! String(contentsOfFile: path)

let target1 = """
        var residual = x
        
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
        h = postAttentionNorm(h)
        var out = residual + (h * layerScalar)

        residual = out
        h = preFFWNorm(residual)

        // Evaluate MLP / MoE
        if let expertsBlock = expertsBlock, let router = router {
            // ...
            // We'll replace everything below anyway. Let's just find the entire callAsFunction body!
""" // Instead of matching the middle, I will just rewrite Gemma4TransformerBlock.callAsFunction!

