import Foundation

let path = "Libraries/MLXLLM/Models/Gemma4.swift"
var file = try! String(contentsOfFile: path)

let target = """
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
        //    Reference uses: gate = act_fn(per_layer_input_gate(hidden_states))
        //                    hidden_states = gate * per_layer_input
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

let replacement = """
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
        h = postAttentionLayerNorm(h)
        var out = residual + (h * layerScalar)

        residual = out
        h = preFeedforwardLayerNorm(residual)
        h = mlp(h)
        h = postFeedforwardLayerNorm(h)
        out = residual + (h * layerScalar)

        return out
"""

file = file.replacingOccurrences(of: target, with: replacement)
try! file.write(toFile: path, atomically: true, encoding: .utf8)
