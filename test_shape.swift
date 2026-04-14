import Foundation

let path = "Libraries/MLXVLM/Models/Gemma4VL.swift"
var file = try! String(contentsOfFile: path)
let target = """
        if let lmHead {
            h = lmHead(h)
"""
let replace = """
        if let lmHead {
            print("=> SHAPE OF h BEFORE lmHead:", h.shape)
            h = lmHead(h)
"""
file = file.replacingOccurrences(of: target, with: replace)

try! file.write(toFile: path, atomically: true, encoding: .utf8)
