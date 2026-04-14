import Foundation

let path = "Libraries/MLXLLM/Models/Gemma4.swift"
var file = try! String(contentsOfFile: path)
let target = "return MLXFast.rmsNorm(x, weight: self.weight, eps: self.eps)"
let replacement = "return MLXFast.rmsNorm(x, weight: self.weight + 1.0, eps: self.eps)"
file = file.replacingOccurrences(of: target, with: replacement)
try! file.write(toFile: path, atomically: true, encoding: .utf8)
