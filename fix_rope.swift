import Foundation

let path = "Libraries/MLXLLM/Models/Gemma4.swift"
var file = try! String(contentsOfFile: path)

let target = """
    public func callAsFunction(_ x: MLXArray, offset: Int) -> MLXArray {
        let xHalf = split(x, parts: 2, axis: -1)
        let q1 = xHalf[0]
        let q2 = xHalf[1]

        let q1_rot = q1[0..., 0..., 0..., 0..<rotatedDims]
        let q1_pass = q1[0..., 0..., 0..., rotatedDims...]

        let q2_rot = q2[0..., 0..., 0..., 0..<rotatedDims]
        let q2_pass = q2[0..., 0..., 0..., rotatedDims...]

        let q1_r = MLXFast.RoPE(q1_rot, dimensions: rotatedDims, traditional: traditional, base: theta, scale: 1.0, offset: offset)
        let q2_r = MLXFast.RoPE(q2_rot, dimensions: rotatedDims, traditional: traditional, base: theta, scale: 1.0, offset: offset)

        let q1_final = concatenated([q1_r, q1_pass], axis: -1)
        let q2_final = concatenated([q2_r, q2_pass], axis: -1)

        return concatenated([q1_final, q2_final], axis: -1)
    }
"""

let replace = """
    public func callAsFunction(_ x: MLXArray, offset: Int) -> MLXArray {
        // Standard 1D Proportional RoPE for Text
        let x_rot = x[0..., 0..., 0..., 0..<rotatedDims]
        let x_pass = x[0..., 0..., 0..., rotatedDims...]

        let x_r = MLXFast.RoPE(x_rot, dimensions: rotatedDims, traditional: traditional, base: theta, scale: 1.0, offset: offset)
        return concatenated([x_r, x_pass], axis: -1)
    }
"""

file = file.replacingOccurrences(of: target, with: replace)
try! file.write(toFile: path, atomically: true, encoding: .utf8)
