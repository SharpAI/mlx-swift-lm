import Foundation
import MLX
import MLXNN
import XCTest

@testable import MLXLMCommon

// End-to-end reproduction of the SwitchGLU SSD fast path (idx.size <= 32) with a
// synthetic quantized MoE checkpoint on disk. The model is loaded through the real
// `loadWeights`, so ExpertStreamerManager.shared and every QuantizedSwitchLinear's
// tensorName are set up exactly as in production; expert bytes are then paged in
// with MLXFast.preadInto from the safetensors file.

private let numExperts = 32
private let hiddenSize = 128
private let intermediateSize = 64
private let groupSize = 64
private let bits = 4
private let prefix = "model.layers.0.mlp.switch_mlp"

private final class ReproMoEMLP: Module {
    @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU

    override init() {
        _switchMLP.wrappedValue = SwitchGLU(
            inputDims: hiddenSize, hiddenDims: intermediateSize, numExperts: numExperts)
    }
}

private final class ReproMoELayer: Module {
    @ModuleInfo(key: "mlp") var mlp: ReproMoEMLP

    override init() {
        _mlp.wrappedValue = ReproMoEMLP()
    }
}

private final class ReproMoEBackbone: Module {
    @ModuleInfo(key: "layers") var layers: [ReproMoELayer]

    override init() {
        _layers.wrappedValue = [ReproMoELayer()]
    }
}

private final class ReproMoEModel: Module, BaseLanguageModel {
    @ModuleInfo(key: "model") var model: ReproMoEBackbone

    override init() {
        _model.wrappedValue = ReproMoEBackbone()
    }

    var switchGLU: SwitchGLU { model.layers[0].mlp.switchMLP }
}

final class SSDPersistentBufferReproTests: XCTestCase {
    private var directory: URL!
    private var savedManager: ExpertStreamerManager?

    override func setUpWithError() throws {
        savedManager = ExpertStreamerManager.shared
        ExpertStreamingConfig.shared.deactivate()

        directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("ssd-persistent-repro-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)

        MLXRandom.seed(71)
        var arrays = [String: MLXArray]()
        for (name, outDims, inDims) in [
            ("gate_proj", intermediateSize, hiddenSize),
            ("up_proj", intermediateSize, hiddenSize),
            ("down_proj", hiddenSize, intermediateSize),
        ] {
            let w = (MLXRandom.normal([numExperts, outDims, inDims]) * 0.1).asType(.float16)
            let (wq, scales, biases) = MLX.quantized(w, groupSize: groupSize, bits: bits)
            arrays["\(prefix).\(name).weight"] = wq
            arrays["\(prefix).\(name).scales"] = scales
            arrays["\(prefix).\(name).biases"] = biases!
        }
        eval(Array(arrays.values))
        try save(arrays: arrays, url: directory.appendingPathComponent("model.safetensors"))

        let index: [String: Any] = [
            "metadata": [String: String](),
            "weight_map": Dictionary(uniqueKeysWithValues: arrays.keys.map { ($0, "model.safetensors") }),
        ]
        try JSONSerialization.data(withJSONObject: index).write(
            to: directory.appendingPathComponent("model.safetensors.index.json"))
    }

    override func tearDownWithError() throws {
        ExpertStreamingConfig.shared.deactivate()
        ExpertStreamerManager.shared = savedManager
        if let directory { try? FileManager.default.removeItem(at: directory) }
    }

    private func loadModel() throws -> ReproMoEModel {
        let model = ReproMoEModel()
        try loadWeights(
            modelDirectory: directory, model: model,
            quantization: .init(groupSize: groupSize, bits: bits))
        return model
    }

    /// Each call is a list of tokens, each token a list of top_k expert ids.
    private func makeInputs(_ routing: [[UInt32]], seed: UInt64) -> (MLXArray, MLXArray) {
        let tokens = routing.count
        let k = routing[0].count
        let x = MLXRandom.normal([1, tokens, hiddenSize], key: MLXRandom.key(seed)).asType(.float16)
        let idx = MLXArray(routing.flatMap { $0 }, [1, tokens, k])
        return (x, idx)
    }

    /// Runs `calls` through the SSD streaming path and compares every output with the
    /// in-memory reference computed by the same checkpoint without streaming.
    private func runAgainstReference(_ calls: [[[UInt32]]], file: StaticString = #filePath, line: UInt = #line) throws {
        let inputs = calls.enumerated().map { makeInputs($0.element, seed: UInt64(100 + $0.offset)) }

        // Reference: streaming off, experts resident, plain gatherQuantizedMM.
        XCTAssertFalse(ExpertStreamingConfig.shared.isEnabled)
        let reference = try loadModel()
        XCTAssertTrue(reference.switchGLU.gateProj is QuantizedSwitchLinear)
        let expected = inputs.map { (x, idx) -> MLXArray in
            let y = reference.switchGLU(x, idx).asType(.float32)
            eval(y)
            return y
        }

        // Streaming: activate for this directory exactly as SwiftLM's --stream-experts does.
        ExpertStreamingConfig.shared.activate(modelDirectory: directory, useDirectIO: true)
        XCTAssertTrue(ExpertStreamingConfig.shared.isStreaming(modelDirectory: directory))
        let streamed = try loadModel()
        let glu = streamed.switchGLU
        XCTAssertTrue(glu.gateProj is QuantizedSwitchLinear)
        XCTAssertEqual(glu.gateProj.tensorName, "\(prefix).gate_proj.weight")
        XCTAssertEqual(glu.upProj.tensorName, "\(prefix).up_proj.weight")
        XCTAssertEqual(glu.downProj.tensorName, "\(prefix).down_proj.weight")
        XCTAssertNotNil(glu.gateProj.resolveSSDInfo(), "SSD fast path would not be taken")

        // Zero the resident expert weights so only bytes pread from the file can
        // reproduce the reference: proves the SSD path, not the fallback, ran.
        var zeroed = [String: MLXArray]()
        for name in ["gate_proj", "up_proj", "down_proj"] {
            let proj: SwitchLinear =
                name == "gate_proj" ? glu.gateProj : name == "up_proj" ? glu.upProj : glu.downProj
            zeroed["\(prefix).\(name).weight"] = MLXArray.zeros(like: proj.weight)
        }
        try streamed.update(parameters: ModuleParameters.unflattened(zeroed), verify: .none)
        eval(streamed)

        // Negative control: with streaming off, the zeroed model must NOT match.
        ExpertStreamingConfig.shared.deactivate()
        do {
            let (x, idx) = inputs[0]
            let y = glu(x, idx).asType(.float32)
            let diff = abs(y - expected[0]).max().item(Float.self)
            XCTAssertGreaterThan(diff, 1e-2, "zeroing resident experts had no effect", file: file, line: line)
        }
        ExpertStreamingConfig.shared.activate(modelDirectory: directory, useDirectIO: true)

        let latch = SSDStreamingErrorLatch()
        for (i, (x, idx)) in inputs.enumerated() {
            let y = SSDStreamingErrorLatch.withActive(latch) { () -> MLXArray in
                let y = glu(x, idx).asType(.float32)
                eval(y)
                return y
            }
            XCTAssertNil(latch.consume(), "pread error on call \(i)", file: file, line: line)
            XCTAssertEqual(y.shape, expected[i].shape, "call \(i) shape", file: file, line: line)
            let maxDiff = abs(y - expected[i]).max().item(Float.self)
            let scale = abs(expected[i]).max().item(Float.self)
            XCTAssertLessThan(
                maxDiff, max(1e-3, scale * 1e-2),
                "call \(i) (\(calls[i].count) token(s)): SSD output differs from reference, max |diff| = \(maxDiff), max |ref| = \(scale)",
                file: file, line: line)
        }
    }

    /// Decode (1 token x top_k 8) allocates 8 persistent buffers; an MTP / draft verify
    /// step (2 tokens x 8 = 16 slots, 16 distinct experts) then needs slots >= 8.
    /// Before the fix this traps on `_persistentGate![freeSlot]` with freeSlot = 8.
    func testDecodeThenVerifyStepGrowsBuffers() throws {
        try runAgainstReference([
            [[0, 1, 2, 3, 4, 5, 6, 7]],  // cold: 8 buffers
            [[0, 2, 4, 6, 8, 10, 12, 14], [16, 17, 18, 19, 20, 21, 22, 23]],  // 16 slots, 16 distinct
            [[1, 3, 16, 17, 24, 25, 30, 31]],  // back to decode, warm at the grown capacity
            [[5, 9, 13, 17, 21, 25, 29, 31], [0, 4, 8, 12, 16, 20, 24, 28]],  // warm 16-slot, misses land in slots >= 8
            [[0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14, 15],
             [16, 17, 18, 19, 20, 21, 22, 23], [24, 25, 26, 27, 28, 29, 30, 31]],  // 32 slots, grow again
            [[3, 7, 11, 15, 19, 23, 27, 31]],
            [[2, 6, 10, 14, 18, 22, 26, 30], [1, 5, 9, 13, 17, 21, 25, 29]],
        ])
    }

    /// First fast-path call is a verify step (16 buffers), so nothing traps before the
    /// fix. A warm 16-slot call then leaves the slots in hit/miss order, which differs
    /// from `_previousExpertIds` order. Before the fix the next decode call re-derived
    /// maxBuffers = 8, speculatively re-read only slots 0..<8, yet still treated slots
    /// 8..<16 as hits for the previous ids: it reads another expert's weights.
    func testVerifyThenDecodeDoesNotReuseStaleSlots() throws {
        try runAgainstReference([
            [[0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14, 15]],  // cold: slot i = expert i
            [[10, 11, 12, 13, 14, 15, 20, 21], [22, 23, 24, 25, 26, 27, 28, 29]],  // hits 10..15 stay in slots 10..15
            [[24, 25, 26, 27, 28, 29, 22, 23]],  // prev ids map 24..29 -> slots 10..15, which hold experts 10..15
        ])
    }
}
