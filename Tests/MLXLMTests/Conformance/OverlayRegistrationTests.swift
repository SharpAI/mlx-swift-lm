// Conformance harness: overlay registration helper (SharpAIOverlay.registerIfAbsent).
//
// Rule 4 of scripts/conformance/README.md: overlay registration must check
// ModelTypeRegistry.contains(_:) first so an upstream registration wins. These tests pin the
// helper's behaviour and lint Libraries/ so no other direct registerModelType call appears
// without being reviewed.

import Foundation
import MLXLMCommon
import Testing

struct SharpAIOverlayTests {

    private struct Marker: Sendable { let tag: String }

    private static func make(_ tag: String) -> (Data) throws -> Marker {
        { _ in Marker(tag: tag) }
    }

    @Test("registers when the model_type is absent")
    func registersWhenAbsent() async throws {
        let registry = ModelTypeRegistry<Marker>()
        let registered = await SharpAIOverlay.registerIfAbsent(
            registry: registry, type: "overlay_new", creator: Self.make("overlay"))
        #expect(registered)
        #expect(await registry.contains("overlay_new"))
        let model = try await registry.createModel(configuration: Data(), modelType: "overlay_new")
        #expect(model.tag == "overlay")
    }

    @Test("does not overwrite an existing registration")
    func doesNotOverwrite() async throws {
        let registry = ModelTypeRegistry<Marker>()
        await registry.registerModelType("taken", creator: Self.make("upstream"))
        let registered = await SharpAIOverlay.registerIfAbsent(
            registry: registry, type: "taken", creator: Self.make("overlay"))
        #expect(!registered)
        let model = try await registry.createModel(configuration: Data(), modelType: "taken")
        #expect(model.tag == "upstream")
    }

    @Test("a second overlay registration of the same type is a no-op")
    func secondCallIsNoOp() async throws {
        let registry = ModelTypeRegistry<Marker>()
        let first = await SharpAIOverlay.registerIfAbsent(
            registry: registry, type: "twice", creator: Self.make("first"))
        let second = await SharpAIOverlay.registerIfAbsent(
            registry: registry, type: "twice", creator: Self.make("second"))
        #expect(first)
        #expect(!second)
        let model = try await registry.createModel(configuration: Data(), modelType: "twice")
        #expect(model.tag == "first")
    }

    @Test("concurrent registerIfAbsent for one type: exactly one winner, and it is the one stored")
    func concurrentRegistrationIsAtomic() async throws {
        // Regression test for the check-then-register race (two actor hops). With a non-atomic
        // helper several tasks all see "absent" and all report `true`.
        for round in 0..<200 {
            let registry = ModelTypeRegistry<Marker>()
            let type = "race_\(round)"
            let results = await withTaskGroup(of: (Int, Bool).self) { group in
                for i in 0..<64 {
                    group.addTask(priority: .userInitiated) {
                        let won = await SharpAIOverlay.registerIfAbsent(
                            registry: registry, type: type, creator: Self.make("c\(i)"))
                        return (i, won)
                    }
                }
                var all: [(Int, Bool)] = []
                for await r in group { all.append(r) }
                return all
            }
            let winners = results.filter { $0.1 }.map { $0.0 }
            #expect(winners.count == 1, "round \(round): \(winners.count) tasks reported registering")
            if let w = winners.first, winners.count == 1 {
                let model = try await registry.createModel(configuration: Data(), modelType: type)
                #expect(model.tag == "c\(w)")
            }
        }
    }

    // MARK: - Lint: direct registerModelType calls in Libraries/

    /// Existing direct call sites (file path relative to the repo root -> expected call count, why).
    /// A new call site fails the lint until it either goes through SharpAIOverlay.registerIfAbsent
    /// or is added here with a reason. Counts are exact so a removed call also forces an update.
    static let allowList: [String: (count: Int, why: String)] = [
        "Libraries/MLXVLM/Qwen35VLMMTPRegistration.swift": (
            2, "Qwen3.5 VLM MTP drafters; fork-only, into MTPDrafterTypeRegistry.visionLanguage"
        ),
        "Libraries/MLXLLM/Qwen35TextMTPRegistration.swift": (
            3, "Qwen3.5 text MTP drafters; fork-only, into MTPDrafterTypeRegistry.shared"
        ),
        "Libraries/MLXVLM/Gemma4AssistantRegistration.swift": (
            1, "Gemma4 assistant MTP drafters; fork-only, into MTPDrafterTypeRegistry.shared"
        ),
    ]

    /// The helper's own directory is the only place allowed to call registerModelType freely.
    static let helperDirectory = "Libraries/MLXLMCommon/SharpAIOverlay/"

    static var repoRoot: URL {
        // <root>/Tests/MLXLMTests/Conformance/OverlayRegistrationTests.swift
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().deletingLastPathComponent()
    }

    /// Counts non-comment lines containing a `.registerModelType(` call or a trailing-closure
    /// call, ignoring `func` declarations.
    static func directCallCount(in source: String) -> Int {
        source.split(separator: "\n", omittingEmptySubsequences: false).filter { raw in
            let line = raw.trimmingCharacters(in: .whitespaces)
            if line.hasPrefix("//") { return false }
            if line.contains("func registerModelType") { return false }
            return line.contains("registerModelType(")
        }.count
    }

    @Test("direct registerModelType calls in Libraries/ match the allow-list")
    func lintDirectRegistrationCalls() throws {
        let libraries = Self.repoRoot.appendingPathComponent("Libraries")
        let enumerator = try #require(
            FileManager.default.enumerator(at: libraries, includingPropertiesForKeys: nil))
        var found: [String: Int] = [:]
        for case let url as URL in enumerator where url.pathExtension == "swift" {
            let rel = String(url.path.dropFirst(Self.repoRoot.path.count + 1))
            if rel.hasPrefix(Self.helperDirectory) { continue }
            // ModelTypeRegistry.swift defines the method; its declarations are skipped by the counter.
            let n = Self.directCallCount(in: try String(contentsOf: url, encoding: .utf8))
            if n > 0 { found[rel] = n }
        }
        let expected = Self.allowList.mapValues(\.count)
        #expect(
            found == expected,
            """
            Direct registerModelType calls in Libraries/ changed.
            found:    \(found.sorted { $0.key < $1.key })
            expected: \(expected.sorted { $0.key < $1.key })
            New overlay registrations must use SharpAIOverlay.registerIfAbsent.
            """)
    }

    @Test("lint counter ignores comments and declarations but sees real calls")
    func lintCounterSelfCheck() {
        let src = """
            /// ``ModelTypeRegistry/registerModelType(_:creator:)`` in a doc comment
            // await registry.registerModelType("x") { _ in 1 }
            public func registerModelType(
            await registry.registerModelType(
            await X.shared.registerModelType("y", creator: c)
            """
        #expect(Self.directCallCount(in: src) == 2)
    }
}
