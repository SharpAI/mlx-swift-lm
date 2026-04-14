import Foundation

let path = "Libraries/MLXVLM/Models/Gemma4VL.swift"
var file = try! String(contentsOfFile: path)
let target = "// self._lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabularySize, bias: false)"
let replace = "self._lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabularySize, bias: false)"
file = file.replacingOccurrences(of: target, with: replace)

try! file.write(toFile: path, atomically: true, encoding: .utf8)
