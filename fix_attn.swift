import Foundation

let path = "Libraries/MLXLLM/Models/Gemma4.swift"
var file = try! String(contentsOfFile: path)
let target = "var scores = queries.matmul(k.transposed(0, 1, 3, 2))"
let replacement = """
            var scores = queries.matmul(k.transposed(0, 1, 3, 2))
            
            // Re-adding scaling: 1.0 / sqrt(headDim) is mathematically required 
            // after the dot-product sum, even if q and k are RMSNorm'd to variance=1
            let scaling = Float(queries.dim(-1)).squareRoot()
            scores = scores * MLXArray(1.0 / scaling).asType(scores.dtype)
"""
file = file.replacingOccurrences(of: target, with: replacement)
try! file.write(toFile: path, atomically: true, encoding: .utf8)
