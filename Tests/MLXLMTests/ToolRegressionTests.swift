import XCTest
@testable import MLXLMCommon
import MLX

final class ToolRegressionTests: XCTestCase {
    func testGemma4ToolCallParsing() throws {
        let tools: [[String: any Sendable]] = [
            [
                "type": "function",
                "function": [
                    "name": "system_status",
                    "description": "Check system status",
                    "parameters": [
                        "type": "object",
                        "properties": [String: any Sendable](),
                        "required": [String]()
                    ] as [String: any Sendable]
                ] as [String: any Sendable]
            ]
        ]
        
        let processor = ToolCallProcessor(format: .gemma4, tools: tools)
        
        let chunk1 = "<|tool_call>call:system_status{"
        let chunk2 = "}<tool_call|>"
        let chunk3 = "<turn|>\n"
        
        var output = ""
        output += processor.processChunk(chunk1) ?? ""
        output += processor.processChunk(chunk2) ?? ""
        output += processor.processChunk(chunk3) ?? ""
        
        XCTAssertEqual(processor.toolCalls.count, 1)
        XCTAssertEqual(processor.toolCalls.first?.function.name, "system_status")
    }

    func testGemma4ToolCallParsingWithoutEndTag() throws {
        let tools: [[String: any Sendable]] = [
            [
                "type": "function",
                "function": [
                    "name": "system_status",
                    "description": "Check system status",
                    "parameters": [
                        "type": "object",
                        "properties": [String: any Sendable](),
                        "required": [String]()
                    ] as [String: any Sendable]
                ] as [String: any Sendable]
            ]
        ]
        
        let processor = ToolCallProcessor(format: .gemma4, tools: tools)
        
        let chunk1 = "<|tool_call>call:system_status{"
        let chunk2 = "}<turn|>\n"
        
        var output = ""
        output += processor.processChunk(chunk1) ?? ""
        output += processor.processChunk(chunk2) ?? ""
        
        // At this point, the processor is buffering because it's waiting for <tool_call|>
        XCTAssertEqual(processor.toolCalls.count, 0)
        
        processor.processEOS()
        // If EOS extracts it successfully:
        XCTAssertEqual(processor.toolCalls.count, 1)
        XCTAssertEqual(processor.toolCalls.first?.function.name, "system_status")
    }
}
