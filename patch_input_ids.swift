import Foundation

let path = "Libraries/MLXLMCommon/LanguageModel.swift"
var file = try! String(contentsOfFile: path)
let target = "let inputTokens = try tokenMemory.applyChatTemplate(messages: prompt)"
let replace = """
            let inputTokens = try tokenMemory.applyChatTemplate(messages: prompt)
            print("🚀 RAW INPUT TOKENS: \\(inputTokens)")
"""
if !file.contains(target) {
    let target2 = "let tokens = try await tokenizer.applyChatTemplate("
    let replace2 = "let tokens = try await tokenizer.applyChatTemplate("
    // we'll just sed it manually
}
