// SharpAI fork overlay tests for nemotron_h_mtp key sanitization. Not part of upstream.

import Foundation
import MLX
import MLXLLM
import XCTest

final class SharpAINemotronHMtpSanitizeTests: XCTestCase {

    private func scalar() -> MLXArray {
        MLXArray(Float(1.0))
    }

    private func makeConfig() -> NemotronHConfiguration {
        NemotronHConfiguration(
            vocabSize: 100,
            hiddenSize: 64,
            numHiddenLayers: 2,
            numAttentionHeads: 4,
            numKeyValueHeads: 2,
            mambaNumHeads: 4,
            mambaHeadDim: 16,
            ssmStateSize: 16,
            convKernel: 4,
            nGroups: 2,
            intermediateSize: 128,
            moeIntermediateSize: 64,
            moeSharedExpertIntermediateSize: 64,
            nRoutedExperts: 4,
            numExpertsPerTok: 2,
            hybridOverridePattern: "M*",
            layerNormEpsilon: 1e-5,
            nGroup: 2,
            topkGroup: 1
        )
    }

    func testDropsMTPPrefixedKeys() {
        let weights: [String: MLXArray] = [
            "backbone.norm_f.weight": scalar(),
            "mtp.layers.0.mixer.q_proj.weight": scalar(),
            "mtp.norm.weight": scalar(),
        ]
        let sanitized = SharpAINemotronHMtpSanitizer.dropMTPWeights(weights)

        XCTAssertEqual(Set(sanitized.keys), ["backbone.norm_f.weight"])
    }

    func testKeepsKeysThatOnlyContainMTP() {
        let weights: [String: MLXArray] = [
            "backbone.mtp_gate.weight": scalar(),
            "lm_head.weight": scalar(),
        ]
        let sanitized = SharpAINemotronHMtpSanitizer.dropMTPWeights(weights)

        XCTAssertEqual(Set(sanitized.keys), ["backbone.mtp_gate.weight", "lm_head.weight"])
    }

    func testNoOpWhenNoMTPKeys() {
        let weights: [String: MLXArray] = ["backbone.embeddings.weight": scalar()]
        let sanitized = SharpAINemotronHMtpSanitizer.dropMTPWeights(weights)

        XCTAssertEqual(Set(sanitized.keys), ["backbone.embeddings.weight"])
    }

    func testModelSanitizeDropsMTPKeys() {
        let model = SharpAINemotronHMtpModel(makeConfig())
        let weights: [String: MLXArray] = [
            "backbone.norm_f.weight": MLXArray.ones([64]),
            "mtp.layers.0.mixer.q_proj.weight": MLXArray.ones([4, 4]),
        ]
        let sanitized = model.sanitize(weights: weights)

        XCTAssertNotNil(sanitized["backbone.norm_f.weight"])
        XCTAssertNil(sanitized["mtp.layers.0.mixer.q_proj.weight"])
    }

    func testModelExcludesMTPNamespace() {
        let model = SharpAINemotronHMtpModel(makeConfig())

        XCTAssertTrue(model.excludedCheckpointNamespaces.contains("mtp"))
    }
}
