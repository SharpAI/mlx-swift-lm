import Foundation

let path = "Libraries/MLXLLM/Models/Gemma4.swift"
var file = try! String(contentsOfFile: path)

let target = """
    func callAsFunction(
        _ x: MLXArray, mask: MLXArray? = nil, cache: [KVCache?]? = nil,
        perLayerInput: MLXArray? = nil
    ) -> MLXArray {
        // === HuggingFace reference: Gemma4TextDecoderLayer.forward ===
        var residual = x
        var hiddenStates = inputLayerNorm(residual)
        hiddenStates = selfAttention(hiddenStates, mask: mask, cache: cache)
        hiddenStates = postAttentionLayerNorm(hiddenStates)
        hiddenStates = residual + hiddenStates

        residual = hiddenStates
        
        if isMoe, let expertsBlock = expertsBlock, let preNorm2 = preFeedforwardLayerNorm2, let postNorm1 = postFeedforwardLayerNorm1, let postNorm2 = postFeedforwardLayerNorm2 {
            // MoE path
            var h = preFeedforwardLayerNorm(hiddenStates)
            h = mlp(h)
            h = postNorm1(h)
            
            var m = preNorm2(hiddenStates)
            m = expertsBlock(m)
            m = postNorm2(m)
            
            hiddenStates = residual + h + m
        } else {
            // Dense path
            hiddenStates = preFeedforwardLayerNorm(hiddenStates)
            hiddenStates = mlp(hiddenStates)
            hiddenStates = postFeedforwardLayerNorm(hiddenStates)
            hiddenStates = residual + hiddenStates
        }

        if let perLayerInput = perLayerInput,
           let perLayerProj = perLayerProjectionLayer,
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

        hiddenStates = hiddenStates * layerScalar
        return hiddenStates
    }
"""

let replace = """
    func callAsFunction(
        _ x: MLXArray, mask: MLXArray? = nil, cache: [KVCache?]? = nil,
        perLayerInput: MLXArray? = nil
    ) -> MLXArray {
        var residual = x
        var hiddenStates = inputLayerNorm(residual)
        hiddenStates = selfAttention(hiddenStates, mask: mask, cache: cache)
        hiddenStates = postAttentionLayerNorm(hiddenStates)
        hiddenStates = residual + hiddenStates

        residual = hiddenStates
        
        if isMoe, let expertsBlock = expertsBlock, let preNorm2 = preFeedforwardLayerNorm2, let postNorm1 = postFeedforwardLayerNorm1, let postNorm2 = postFeedforwardLayerNorm2 {
            // MoE path uses 2 mlps according to some Gemma MoE structures natively implemented by HF
            var h = preFeedforwardLayerNorm(hiddenStates)
            h = mlp(h)
            h = postNorm1(h)
            
            var m = preNorm2(hiddenStates)
            m = expertsBlock(m)
            m = postNorm2(m)
            
            hiddenStates = residual + h + m
        } else {
            // Dense path
            hiddenStates = preFeedforwardLayerNorm(hiddenStates)
            hiddenStates = mlp(hiddenStates)
            hiddenStates = postFeedforwardLayerNorm(hiddenStates)
            hiddenStates = residual + hiddenStates
        }

        if let perLayerInput = perLayerInput,
           let perLayerProj = perLayerProjectionLayer,
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

        hiddenStates = hiddenStates * layerScalar
        return hiddenStates
    }
"""

file = file.replacingOccurrences(of: target, with: replace)
try! file.write(toFile: path, atomically: true, encoding: .utf8)
