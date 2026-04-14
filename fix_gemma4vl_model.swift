import Foundation

let path = "Libraries/MLXVLM/Models/Gemma4VL.swift"
var file = try! String(contentsOfFile: path)

let target1 = "var h = languageModel(inputs, cache: optionalCache)"
let replace1 = "var h = languageModel.model(inputs, inputEmbedding: nil, mask: nil, cache: optionalCache)"
file = file.replacingOccurrences(of: target1, with: replace1)

let target2 = """
        var h = languageModel(
            input.text.tokens,
            inputEmbedding: inputEmbeddings,
            mask: .causal, // Depending on phase
            cache: convertedCache
        )
"""
let replace2 = """
        var h = languageModel.model(
            input.text.tokens,
            inputEmbedding: inputEmbeddings,
            mask: .causal, // Depending on phase
            cache: convertedCache
        )
"""
file = file.replacingOccurrences(of: target2, with: replace2)

try! file.write(toFile: path, atomically: true, encoding: .utf8)
