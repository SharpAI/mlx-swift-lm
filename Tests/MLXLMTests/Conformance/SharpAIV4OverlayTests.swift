// SL-41: fork-only DeepseekV4 code must not be able to collide with a future upstream DeepseekV4.
//   (a) the overlay's merge is "ours wins" over an upstream-style stand-in for the same key,
//   (b) `deepseek_v4` resolves, through the normal shared registry, to the SharpAI-prefixed model,
//   (c) every model_type that resolved before still resolves,
//   (d) source lints: the upstream-owned factory no longer names V4, and no unprefixed V4 type or
//       helper is left in module MLXLLM.
// upstream-watch: Libraries/MLXLLM/LLMModelFactory.swift (the merge hook at the end of the literal).

import Foundation
import MLX
import MLXLMCommon
import Testing

@testable import MLXLLM

struct SharpAIV4OverlayTests {

    static var repoRoot: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().deletingLastPathComponent()
    }

    static func source(_ rel: String) throws -> String {
        try String(contentsOf: repoRoot.appendingPathComponent(rel), encoding: .utf8)
    }

    struct StandInUpstreamV4: Error {}

    // MARK: (a) ours wins over an upstream-style entry for the same key

    @Test("overlay merge: our deepseek_v4 creator replaces an upstream stand-in")
    func oursWinsOverStandIn() async throws {
        let standIn: SharpAIDeepseekV4Overlay.Creator = { _ in throw StandInUpstreamV4() }
        let merged = SharpAIDeepseekV4Overlay.applying(to: ["deepseek_v4": standIn, "other": standIn])
        let creator = try #require(merged["deepseek_v4"])
        let model = try creator(try ConformanceDeepseekV4.tinyJSON())
        #expect(String(describing: type(of: model)) == "SharpAIDeepseekV4Model")
        // unrelated upstream keys are left alone
        #expect(merged["other"] != nil)
    }

    @Test("overlay merge adds deepseek_v4 to a base that lacks it, and is idempotent")
    func addsWhenAbsentAndIdempotent() throws {
        let once = SharpAIDeepseekV4Overlay.applying(to: [:])
        let twice = SharpAIDeepseekV4Overlay.applying(to: once)
        #expect(once.keys.sorted() == ["deepseek_v4"])
        #expect(twice.keys.sorted() == ["deepseek_v4"])
        let creator = try #require(twice["deepseek_v4"])
        let model = try creator(try ConformanceDeepseekV4.tinyJSON())
        #expect(String(describing: type(of: model)) == "SharpAIDeepseekV4Model")
    }

    // MARK: (b) the shared registry resolves deepseek_v4 to the SharpAI-prefixed model

    @Test("LLMTypeRegistry.shared resolves deepseek_v4 to SharpAIDeepseekV4Model")
    func sharedRegistryResolvesPrefixed() async throws {
        #expect(await LLMTypeRegistry.shared.contains("deepseek_v4"))
        let model = try await LLMTypeRegistry.shared.createModel(
            configuration: try ConformanceDeepseekV4.tinyJSON(), modelType: "deepseek_v4")
        #expect(String(describing: type(of: model)) == "SharpAIDeepseekV4Model")
    }

    // MARK: (c) every previously registered model_type still resolves

    /// Frozen from the LLMTypeRegistry.shared literal at origin/main 8dfc228 (71 keys).
    static let keysBefore: [String] = [
        "mistral", "mixtral", "llama", "phi", "phi3", "phimoe", "gemma", "gemma2", "gemma3",
        "gemma3_text", "gemma3n", "gemma4", "gemma4_unified", "gemma4_text", "gemma4_assistant",
        "qwen2", "qwen3", "qwen3_moe", "qwen3_next", "qwen3_5", "qwen3_5_moe", "qwen3_5_text",
        "qwen3_8", "qwen3_8_moe", "qwen3_8_text", "nanbeige", "minicpm", "starcoder2", "cohere",
        "openelm", "internlm2", "deepseek_v2", "deepseek_v3", "deepseek_v3_2", "glm_moe_dsa",
        "deepseek_v4", "granite", "helium", "granitemoehybrid", "mimo", "mimo_v2_flash", "minimax",
        "glm4", "glm4_moe", "glm4_moe_lite", "acereason", "falcon_h1", "bitnet", "smollm3",
        "ernie4_5", "lfm2", "baichuan_m1", "exaone4", "gpt_oss", "olmoe", "olmo2", "olmo3",
        "mellum", "bailing_moe", "lfm2_moe", "nanochat", "nemotron_h", "nemotron_h_puzzle",
        "afmoe", "jamba_3b", "jamba", "mamba2", "mistral3", "apertus", "hunyuan_v1_dense",
        "nemotron_labs_diffusion",
    ]

    @Test("all 71 model_types registered before SL-41 still resolve")
    func allPreviousKeysResolve() async throws {
        #expect(Self.keysBefore.count == 71)
        var missing: [String] = []
        for key in Self.keysBefore where await !LLMTypeRegistry.shared.contains(key) {
            missing.append(key)
        }
        #expect(missing.isEmpty, "unresolved: \(missing)")
    }

    // MARK: (d) source lints

    @Test("the upstream-owned LLMModelFactory.swift has no deepseek_v4 literal entry")
    func factoryHasNoV4Literal() throws {
        let src = try Self.source("Libraries/MLXLLM/LLMModelFactory.swift")
        let lines = src.split(separator: "\n").filter { !$0.trimmingCharacters(in: .whitespaces).hasPrefix("//") }
        #expect(!lines.contains { $0.contains("\"deepseek_v4\"") }, "deepseek_v4 entry still in the upstream literal")
        #expect(!lines.contains { $0.replacingOccurrences(of: "SharpAIDeepseekV4", with: "").contains("DeepseekV4") }, "unprefixed DeepseekV4 name in the factory")
    }

    @Test("no unprefixed DeepseekV4 file or type name is left under Libraries/MLXLLM")
    func noUnprefixedV4Names() throws {
        let dir = Self.repoRoot.appendingPathComponent("Libraries/MLXLLM")
        let enumerator = try #require(FileManager.default.enumerator(at: dir, includingPropertiesForKeys: nil))
        var badFiles: [String] = []
        var badDecls: [String] = []
        let declPattern = try Regex(#"^\s*(public |internal |final |private |fileprivate )*(class|struct|enum|func|extension|typealias) (DeepseekV4\w*|HCParams|hcSplitSinkhorn|hcPre|hcPost|hcHead)\b"#)
        for case let url as URL in enumerator where url.pathExtension == "swift" {
            let rel = String(url.path.dropFirst(dir.path.count + 1))
            if url.lastPathComponent.hasPrefix("DeepseekV4") { badFiles.append(rel) }
            let text = try String(contentsOf: url, encoding: .utf8)
            for line in text.split(separator: "\n") where line.firstMatch(of: declPattern) != nil {
                badDecls.append("\(rel): \(line.trimmingCharacters(in: .whitespaces))")
            }
        }
        #expect(badFiles.isEmpty, "unprefixed V4 files: \(badFiles)")
        #expect(badDecls.isEmpty, "unprefixed V4 declarations: \(badDecls)")
    }

    @Test("fork-only V4 sources live in Libraries/MLXLLM/SharpAI with SharpAI-prefixed names")
    func v4SourcesInSharpAIDirectory() throws {
        let fm = FileManager.default
        for name in ["", "Compressor", "CompressorState", "Indexer"] {
            let rel = "Libraries/MLXLLM/SharpAI/SharpAIDeepseekV4\(name).swift"
            #expect(fm.fileExists(atPath: Self.repoRoot.appendingPathComponent(rel).path), "missing \(rel)")
        }
    }

    @Test("HF architecture strings and the model_type key are unchanged")
    func externalNamesUnchanged() throws {
        let src = try Self.source("Tests/MLXLMTests/Conformance/ConformanceExpectations.swift")
        for s in ["DeepseekV4ForCausalLM", "DeepseekV4ForConditionalGeneration", "DeepSeekV4MTPModel", "\"deepseek_v4\""] {
            #expect(src.contains(s), "\(s) missing")
        }
    }
}
