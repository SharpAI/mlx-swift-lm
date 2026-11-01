// SharpAI fork overlay tests for the mimo_v2 model. Not part of upstream.
//
// Offline checks only: a tiny synthetic config, random weights, no network.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import XCTest

@testable import MLXLLM

final class SharpAIMimoV2ModelTests: XCTestCase {

    private func tinyModel(
        retainMTP: Bool = false, tied: Bool = false
    ) throws -> SharpAIMimoV2Model {
        let config = try JSONDecoder.json5().decode(
            SharpAIMimoV2Configuration.self,
            from: Data(SharpAIMimoV2TestFixture.tinyJSON(tied: tied).utf8))
        return SharpAIMimoV2Model(config, retainMTP: retainMTP)
    }

    private func parameterKeys(_ model: SharpAIMimoV2Model) -> Set<String> {
        Set(model.parameters().flattened().map(\.0))
    }

    func testForwardPassProducesLogitsShape() throws {
        let model = try tinyModel()
        let inputs = MLXArray([0, 1, 2, 3, 4] as [Int32]).reshaped(1, 5)
        let logits = model(inputs, cache: nil)
        eval(logits)
        XCTAssertEqual(logits.shape, [1, 5, 100])
    }

    func testForwardPassWithHybridCachesProducesLogitsShape() throws {
        let model = try tinyModel()
        let cache = try model.newCache(parameters: nil)
        XCTAssertEqual(cache.count, 4)

        let inputs = MLXArray([0, 1, 2, 3, 4] as [Int32]).reshaped(1, 5)
        let logits = model(inputs, cache: cache)
        eval(logits)
        XCTAssertEqual(logits.shape, [1, 5, 100])
    }

    func testTiedEmbeddingsProduceLogitsShape() throws {
        let model = try tinyModel(tied: true)
        let inputs = MLXArray([0, 1, 2] as [Int32]).reshaped(1, 3)
        let logits = model(inputs, cache: nil)
        eval(logits)
        XCTAssertEqual(logits.shape, [1, 3, 100])
    }

    func testSinkParameterOnlyOnSlidingWindowLayers() throws {
        let keys = parameterKeys(try tinyModel())

        XCTAssertTrue(keys.contains("model.layers.0.self_attn.attention_sink_bias"))
        XCTAssertTrue(keys.contains("model.layers.2.self_attn.attention_sink_bias"))
        XCTAssertFalse(keys.contains("model.layers.1.self_attn.attention_sink_bias"))
    }

    func testDenseAndMoELayerParameterPaths() throws {
        let keys = parameterKeys(try tinyModel())

        XCTAssertTrue(keys.contains("model.layers.0.mlp.gate_proj.weight"))
        XCTAssertFalse(keys.contains("model.layers.0.mlp.gate.weight"))
        XCTAssertTrue(keys.contains("model.layers.1.mlp.gate.weight"))
        XCTAssertTrue(keys.contains("model.layers.1.mlp.gate.e_score_correction_bias"))
        XCTAssertTrue(keys.contains("model.layers.1.mlp.switch_mlp.gate_proj.weight"))
        XCTAssertTrue(keys.contains("model.layers.0.self_attn.q_proj.weight"))
        XCTAssertTrue(keys.contains("model.layers.0.self_attn.v_proj.weight"))
    }

    func testMTPLayerExistsOnlyWhenRetained() throws {
        let off = parameterKeys(try tinyModel(retainMTP: false))
        let on = parameterKeys(try tinyModel(retainMTP: true))

        XCTAssertFalse(off.contains("model.mtp.layers.0.eh_proj.weight"))
        XCTAssertTrue(on.contains("model.mtp.layers.0.eh_proj.weight"))
        XCTAssertTrue(on.contains("model.mtp.layers.0.self_attn.q_proj.weight"))
    }

    func testMTPHiddenStatesHaveModelWidth() throws {
        let model = try tinyModel(retainMTP: true)
        let tokens = MLXArray([1, 2, 3] as [Int32]).reshaped(1, 3)
        let previous = zeros([1, 3, 32])

        let hidden = model.mtpHiddenStates(tokens: tokens, previousHidden: previous, cache: nil)
        eval(hidden)
        XCTAssertEqual(hidden?.shape, [1, 3, 32])
    }

    /// Known gap: Omni towers are absent. The reference is public now: XiaomiMiMo/MiMo-V2.6-Pro-RL
    /// modeling_mimo_v2.py defines MiMoVisionTransformer (28 blocks, SWA and full layers with sinks),
    /// MiMoAudioEncoder and MiMoAudioTokenizer (RVQ codec). None of it is ported here, and the
    /// checkpoint's model.safetensors.index.json is Git LFS only, so the tower key names are unread.
    /// The config is rejected at decode, so the no-throw assertion fails as an expected failure.
    func testOmniTowersAreNotImplemented() throws {
        try XCTExpectFailure(
            "KNOWN GAP: Omni towers not ported. Reference is XiaomiMiMo/MiMo-V2.6-Pro-RL modeling_mimo_v2.py (MiMoVisionTransformer, MiMoAudioEncoder, MiMoAudioTokenizer); Swift port and tower safetensors key names (index is LFS-only, unreadable) are missing"
        ) {
            XCTAssertNoThrow(
                try JSONDecoder.json5().decode(
                    SharpAIMimoV2Configuration.self,
                    from: Data(
                        SharpAIMimoV2TestFixture.tinyJSON(omniTowers: true).utf8))
            )
        }
    }

    /// Known gap: MTP is built and sanitized but not wired into generation. The MTP forward order
    /// matches vLLM mimo_v2_mtp.py (MiMoV2MTPLayer.forward), and the prerequisite (MTP hidden states
    /// on the retained path) runs here. Still missing for MTPSpeculativeTokenIterator: the target
    /// does not emit mtpLastHiddenStatesKey or mtpSharedKVStatesKey, the MTP layer does not conform to
    /// MTPDrafterModel or StatefulMTPDrafterModel, and rollback for the SWA rotating caches is unset.
    func testMTPIsNotWiredIntoGeneration() throws {
        let model = try tinyModel(retainMTP: true)
        let tokens = MLXArray([1, 2, 3] as [Int32]).reshaped(1, 3)
        XCTAssertNotNil(
            model.mtpHiddenStates(tokens: tokens, previousHidden: zeros([1, 3, 32]), cache: nil))

        XCTExpectFailure(
            "KNOWN GAP: mimo_v2 MTP not wired into generation. Missing: target LMOutput state emission (mtpLastHiddenStatesKey, mtpSharedKVStatesKey), MTPDrafterModel/StatefulMTPDrafterModel conformance for the MTP layer, and SWA rotating-cache rollback design"
        ) {
            XCTFail("MTP drafts are not consumed by the generation loop")
        }
    }

    func testMTPHiddenStatesAreNilWhenNotRetained() throws {
        let model = try tinyModel(retainMTP: false)
        let tokens = MLXArray([1, 2, 3] as [Int32]).reshaped(1, 3)

        XCTAssertNil(
            model.mtpHiddenStates(tokens: tokens, previousHidden: zeros([1, 3, 32]), cache: nil))
    }
}
