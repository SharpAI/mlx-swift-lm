import Foundation
import MLXLMCommon

@main
struct App {
    static func main() async throws {
        let registry = ModelRegistry.shared
        let url = URL(fileURLWithPath: "/Users/simba/.cache/huggingface/hub/models--mlx-community--gemma-4-e4b-it-4bit/snapshots/d9ab042fedb4e180cc428676230f81d11ca7ad1c/")
        let tokenizerContext = try await TokenizerContext.load(modelURL: url)
        let tokenizer = tokenizerContext.tokenizer
        let messages = [["role": "user", "content": "Hello"]]
        let tokens = try tokenizer.applyChatTemplate(messages: messages)
        print("TOKENS:", tokens)
        for t in tokens {
            print(t, tokenizer.decode(tokens: [t]))
        }
    }
}
