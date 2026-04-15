import Foundation
import MLXVLM
import MLXLMCommon

// Configure bare-minimum Gemma4Processor
let configJSON = """
{
  "processor_class": "Gemma4Processor",
  "image_token_id": 258880,
  "boi_token_id": 255999,
  "eoi_token_id": 258882,
  "audio_token_id": 258881
}
""".data(using: .utf8)!
let decoder = JSONDecoder()
let config = try! decoder.decode(Gemma4ProcessorConfiguration.self, from: configJSON)

// Standin Tokenizer
class MockTokenizer: Tokenizer, @unchecked Sendable {
    func encode(text: String) -> [Int] { return [] }
    func decode(tokens: [Int]) -> String { return "" }
    var bosTokenId: Int? { return nil }
    var eosTokenId: Int? { return nil }
    func applyChatTemplate(messages: [[String : Any]], tools: [ToolSpec]?, additionalContext: [String : Any]?) throws -> [Int] {
        return [256000, 258881, 258883] // Mock what Qwen2VLMessageGenerator produces for <|audio|>
    }
}
let processor = Gemma4Processor(config, tokenizer: MockTokenizer())

// Mock audio
let input = UserInput(audio: [.url(URL(fileURLWithPath: "tests/fixtures/test_audio.wav"))])
print("Done initializing processor.")
