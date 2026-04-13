import Foundation

let path = "Libraries/MLXLLM/Models/Gemma4.swift"
var file = try! String(contentsOfFile: path)
let target = """
        // Remap router keys to be nested under experts block
        var finalWeights = [String: MLXArray]()
        for (k, v) in processedWeights {
            let newK = k.replacingOccurrences(of: ".router.", with: ".experts.router.")
            finalWeights[newK] = v
        }
"""
let replace = """
        // Remap router keys to be nested under experts block
        var finalWeights = [String: MLXArray]()
        for (k, v) in processedWeights {
            var newK = k
            // For Gemma 4 MoE Router projections:
            newK = newK.replacingOccurrences(of: ".router.weight", with: ".experts.router.proj.weight")
            // And remap specific scale and norm bindings
            newK = newK.replacingOccurrences(of: ".router.scale", with: ".experts.router.scale")
            newK = newK.replacingOccurrences(of: ".router.per_expert_scale", with: ".experts.router.per_expert_scale")
            newK = newK.replacingOccurrences(of: ".router.norm.weight", with: ".experts.router.norm.weight")
            newK = newK.replacingOccurrences(of: ".router.", with: ".experts.router.")
            
            finalWeights[newK] = v
        }
"""
file = file.replacingOccurrences(of: target, with: replace)
try! file.write(toFile: path, atomically: true, encoding: .utf8)
