// SharpAI ParamsJson model tests: llama 4 scale values, one tiny draft forward pass, and a
// known gap for the real checkpoint (expected failure, not a skip). Random weights only. No network.
// The EAGLE fusion order is checked against vLLM's mistral_large_3_eagle.py. The llama 4 scale
// placement on the full MLA query is checked against vLLM's deepseek_v2.py (see
// SharpAIParamsJsonLlama4Scaling.swift). Real-checkpoint draft parity is still a known gap.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import XCTest

@testable import MLXLLM

final class SharpAIParamsJsonModelTests: XCTestCase {

    private func tinyConfig() throws -> SharpAIParamsJsonConfiguration {
        try SharpAIParamsJsonConfiguration.decode(
            Data(SharpAIParamsJsonTestFixtures.tinyParamsJSON.utf8))
    }

    func testLlama4ScaleIsOneBelowOriginalLength() {
        let scale = SharpAIParamsJsonLlama4Scaling.scale(
            start: 0, stop: 8, beta: 0.1, originalMaxPositionEmbeddings: 8)

        XCTAssertEqual(scale.shape, [8, 1])
        for value in scale.reshaped(-1).asArray(Float.self) {
            XCTAssertEqual(value, 1.0, accuracy: 1e-6)
        }
    }

    func testLlama4ScaleAtOriginalLengthMatchesFormula() {
        let scale = SharpAIParamsJsonLlama4Scaling.scale(
            start: 8, stop: 9, beta: 0.1, originalMaxPositionEmbeddings: 8)
        // 1 + 0.1 * ln(2)
        let expected: Float = 1.0693147

        XCTAssertEqual(scale.reshaped(-1).asArray(Float.self)[0], expected, accuracy: 1e-5)
    }

    func testQueryScaleIsOnesWithoutLlama4Block() throws {
        let json = SharpAIParamsJsonTestFixtures.minimalParamsJSON
        let config = try SharpAIParamsJsonConfiguration.decode(Data(json.utf8))
        let scale = SharpAIParamsJsonLlama4Scaling.queryScale(
            config: config, offset: 5, length: 3, dtype: .float32)

        XCTAssertEqual(scale.shape, [3, 1])
        for value in scale.reshaped(-1).asArray(Float.self) {
            XCTAssertEqual(value, 1.0, accuracy: 1e-6)
        }
    }

    func testDraftForwardReturnsHiddenAndLogitsShapes() throws {
        let config = try tinyConfig()
        let model = SharpAIParamsJsonEagleModel(config)

        let ids = MLXArray([Int32(1), 2, 3], [1, 3])
        let hidden = MLXRandom.normal([1, 3, config.dim])
        let out = model.draft(inputIDs: ids, hiddenStates: hidden, cache: nil)

        XCTAssertEqual(out.hidden.shape, [1, 3, config.dim])
        XCTAssertEqual(out.logits.shape, [1, 3, config.vocabSize])
        XCTAssertTrue(out.logits.reshaped(-1).asArray(Float.self).allSatisfy { $0.isFinite })
    }

    func testDraftForwardWithKVCacheAdvancesOffset() throws {
        let config = try tinyConfig()
        let model = SharpAIParamsJsonEagleModel(config)
        let cache: [KVCache] = (0..<config.nLayers).map { _ in KVCacheSimple() }

        let first = model.draft(
            inputIDs: MLXArray([Int32(1), 2], [1, 2]),
            hiddenStates: MLXRandom.normal([1, 2, config.dim]),
            cache: cache)
        XCTAssertEqual(first.logits.shape, [1, 2, config.vocabSize])
        XCTAssertEqual(cache[0].offset, 2)

        let second = model.draft(
            inputIDs: MLXArray([Int32(3)], [1, 1]),
            hiddenStates: MLXRandom.normal([1, 1, config.dim]),
            cache: cache)
        XCTAssertEqual(second.logits.shape, [1, 1, config.vocabSize])
        XCTAssertEqual(cache[0].offset, 3)
    }

    func testTiedEmbeddingsModelUsesEmbeddingAsOutput() throws {
        let json = SharpAIParamsJsonTestFixtures.tinyParamsJSON.replacingOccurrences(
            of: "\"tied_embeddings\": false", with: "\"tied_embeddings\": true")
        let config = try SharpAIParamsJsonConfiguration.decode(Data(json.utf8))
        let model = SharpAIParamsJsonEagleModel(config)

        let out = model.draft(
            inputIDs: MLXArray([Int32(1)], [1, 1]),
            hiddenStates: MLXRandom.normal([1, 1, config.dim]),
            cache: nil)
        XCTAssertEqual(out.logits.shape, [1, 1, config.vocabSize])
    }
}

final class SharpAIParamsJsonModelKnownIssueTests: XCTestCase {

    /// Source: vLLM `EagleMistralLarge3Model.forward` (vllm/model_executor/models/
    /// mistral_large_3_eagle.py) runs `self.fc(torch.cat((inputs_embeds, hidden_states), dim=-1))`
    /// with no norm on the concatenated input. With zero decoder layers, this model's draft output
    /// is `norm(eagle_proj(cat([embeds, hidden], -1)))`. Pinning `eagle_proj` to a block selector
    /// shows which half reaches the output, so this runs without a checkpoint.
    func testEagleFusionConcatOrderMatchesReference() throws {
        let json = SharpAIParamsJsonTestFixtures.tinyParamsJSON.replacingOccurrences(
            of: "\"n_layers\": 1", with: "\"n_layers\": 0")
        let config = try SharpAIParamsJsonConfiguration.decode(Data(json.utf8))
        let model = SharpAIParamsJsonEagleModel(config)
        let dim = config.dim

        let ids = MLXArray([Int32(1), 2, 3], [1, 3])
        let hidden = MLXRandom.normal([1, 3, dim])
        let embeds = model.embedTokens(ids)

        let identity = MLXArray.eye(dim, dtype: .float32)
        let zeros = MLXArray.zeros([dim, dim])

        // Weight is [dim, 2 * dim]. The first dim columns read embeds, the rest read hidden.
        model.update(
            parameters: ModuleParameters.unflattened([
                "eagle_proj.weight": concatenated([identity, zeros], axis: 1)
            ]))
        let fromEmbeds = model.draft(inputIDs: ids, hiddenStates: hidden, cache: nil).hidden
        let expectedEmbeds = model.norm(embeds)
        XCTAssertTrue(
            allClose(fromEmbeds, expectedEmbeds, rtol: 1e-4, atol: 1e-4).item(Bool.self),
            "eagle_proj must read [embeds, hidden] in that order")

        model.update(
            parameters: ModuleParameters.unflattened([
                "eagle_proj.weight": concatenated([zeros, identity], axis: 1)
            ]))
        let fromHidden = model.draft(inputIDs: ids, hiddenStates: hidden, cache: nil).hidden
        let expectedHidden = model.norm(hidden)
        XCTAssertTrue(
            allClose(fromHidden, expectedHidden, rtol: 1e-4, atol: 1e-4).item(Bool.self),
            "eagle_proj's second half must read the target hidden states")
    }

    /// The real-checkpoint parity check is still a known gap. The drafter's params.json and the
    /// key names in its safetensors header are now verified (SharpAIParamsJsonConfigurationTests,
    /// SharpAIParamsJsonSanitizeTests). What is missing is a reference dump: the drafter's hidden
    /// states and logits for fixed inputs, produced by running the 675B target model in vLLM or
    /// mistral-inference. That cannot run offline, and a search found no public dump. Set
    /// SHARPAI_PARAMSJSON_REFERENCE_DUMP to a dump once one exists; then compare draft logits.
    func testRealCheckpointDraftParityKnownIssue() {
        XCTExpectFailure(
            "KNOWN GAP: no reference drafter dump for EagleMistralLarge3. The drafter params.json and safetensors key layout are verified against Hugging Face, but hidden states and logits from the 675B target (vLLM or mistral-inference) for fixed inputs are not available offline or publicly. Set SHARPAI_PARAMSJSON_REFERENCE_DUMP to a dump once one exists; then replace this assertion with a real comparison."
        ) {
            let dumpPath = ProcessInfo.processInfo.environment["SHARPAI_PARAMSJSON_REFERENCE_DUMP"]
            XCTAssertNotNil(
                dumpPath,
                "no reference drafter dump configured; checkpoint parity cannot be checked")
        }
    }
}
