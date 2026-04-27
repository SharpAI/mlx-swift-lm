import Foundation
@preconcurrency import MLX
import MLXNN

// Port of https://github.com/ml-explore/mlx-examples/blob/main/llms/mlx_lm/models/switch_layers.py

public func gatherSort(x: MLXArray, indices: MLXArray) -> (MLXArray, MLXArray, MLXArray) {
    let m = indices.dim(-1)
    let indices = indices.flattened()
    let order = argSort(indices)
    let inverseOrder = argSort(order)

    return (
        x.flattened(start: 0, end: -3)[order.floorDivide(m)],
        indices[order],
        inverseOrder
    )
}

public func scatterUnsort(x: MLXArray, invOrder: MLXArray, shape: [Int]? = nil) -> MLXArray {
    var x = x[invOrder]
    if let shape {
        x = unflatten(x, axis: 0, shape: shape)
    }
    return x
}


// Shared struct for expert range tracking across projections
public struct ExpertRange: Sendable {
    public let id: Int
    public let start: Int
    public let end: Int
}

// MARK: - SwitchGLU

public class SwitchGLU: Module, @unchecked Sendable {
    @ModuleInfo(key: "gate_proj") public var gateProj: SwitchLinear
    @ModuleInfo(key: "up_proj") public var upProj: SwitchLinear
    @ModuleInfo(key: "down_proj") public var downProj: SwitchLinear

    let inputDims: Int
    let hiddenDims: Int
    let numExperts: Int
    let activation: (MLXArray) -> MLXArray

    // ── Async pipeline state (SSD streaming optimization) ──
    // Persistent buffers: allocated once per layer, reused across tokens.
    // Avoids per-token buffer allocation + eval overhead.
    private var _persistentGate: [MLXArray]?
    private var _persistentUp: [MLXArray]?
    private var _persistentDown: [MLXArray]?
    // Previous token's expert routing per layer for speculative prefetch.
    private var _previousExpertIds: [Int]?

    // ── Cache-slot tunable (env-tunable via `MLX_MOE_CACHE_SLOTS=N`) ──
    // Number of resident expert slots used by SSD-streaming paths that keep
    // experts cached across tokens. Default 16 is a good balance for top-k=8
    // routing on Apple Silicon: enough slack for prev-token spec prefetch +
    // current-token misses without over-pressuring the unified-memory
    // allocator. Larger values trade RAM for hit-rate. Minimum is 6 (must
    // accommodate top-k plus a small eviction margin).
    static let MAX_CACHE_SLOTS: Int = {
        if let v = ProcessInfo.processInfo.environment["MLX_MOE_CACHE_SLOTS"],
           let n = Int(v), n >= 6 {
            return n
        }
        return 16
    }()

    // ── Stacked-buffer fused-matmul fast path (env-gated MLX_MOE_STACKED=1) ──
    // When enabled, allocate a single stacked weight buffer of shape
    // `[CACHE_SLOTS, intermediate, hidden]` per projection (instead of
    // CACHE_SLOTS individual `[1, intermediate, hidden]` buffers) and
    // populate slots via `MLXFast.preadIntoOffset`, which writes one
    // expert into a byte-offset region of the stacked tensor in place.
    //
    // The win is dispatch reduction: `gatherQuantizedMM` runs ONCE per
    // projection per layer (using `rhsIndices = slotPerToken`), instead
    // of `top_k` separate dispatches per projection per layer. On Apple
    // Silicon each Metal dispatch carries ~30 µs of CPU→GPU
    // encode/submit overhead, which dominates per-token compute on
    // SSD-streamed MoE models.
    //
    // Eligible layers: all 3 projections quantized + SSD streaming
    // resolveable + `idx.size <= 32` (single-token generation). Anything
    // else falls through to the existing N-buffer path. The flag is
    // off by default; consumers opt in per launch.
    private static let useStackedBuffers: Bool = {
        let v = ProcessInfo.processInfo.environment["MLX_MOE_STACKED"] ?? ""
        return v == "1" || v.lowercased() == "true"
    }()
    private var _stackedGate: MLXArray?
    private var _stackedUp: MLXArray?
    private var _stackedDown: MLXArray?
    // Per-slot expert occupant; nil means empty.
    private var _slotExpert: [Int?]?
    // Per-slot last-used token counter, used by LRU eviction.
    private var _slotLastUsed: [Int]?
    // Per-layer token counter — incremented per fast-path call.
    private var _tokenCounter: Int = 0
    // Bytes per expert slab in a stacked buffer; computed once on cold init.
    private var _stackedBytesPerExpert: Int = 0
    private var _stackedDownBytesPerExpert: Int = 0

    // ── Fused gate+up SwiGLU mode (env-gated MLX_MOE_FUSE_GATEUP=1) ──
    // SwiGLU MLP is `silu(gate(x)) * up(x)`; gate and up are independent
    // matmuls of identical shape. When enabled (and useStackedBuffers is
    // also enabled), allocate ONE combined buffer of shape
    // `[CACHE_SLOTS, 2 * intermediate, hidden]` and write each expert's
    // gate weights into the first half of its slot and up weights into
    // the second half (offsets `slot * 2 * bpe` and `slot * 2 * bpe + bpe`).
    // A single `gatherQuantizedMM` then produces `[..., 2 * intermediate]`,
    // which is split into the two halves and fed into `silu(g) * u`.
    //
    // Saves ONE projection-level dispatch per layer per token (the gate
    // and up matmuls collapse into one). Requires useStackedBuffers; no
    // effect on the legacy N-buffer or non-stacked paths.
    private static let useFusedGateUp: Bool = {
        let v = ProcessInfo.processInfo.environment["MLX_MOE_FUSE_GATEUP"] ?? ""
        return v == "1" || v.lowercased() == "true"
    }()
    private var _stackedGateUp: MLXArray?
    // Pre-concatenated gate+up scales/biases along the intermediate axis,
    // computed once at cold init so the runtime `MLX.take` is a single op.
    private var _combinedGateUpScales: MLXArray?
    private var _combinedGateUpBiases: MLXArray?
    // Bytes for ONE projection (gate or up) per slot in the combined buffer;
    // total bytes per slot = 2 * _stackedGateUpBytesPerProj.
    private var _stackedGateUpBytesPerProj: Int = 0

    public init(
        inputDims: Int,
        hiddenDims: Int,
        numExperts: Int,
        activation: @escaping (MLXArray) -> MLXArray = MLXNN.silu,
        bias: Bool = false
    ) {
        self.inputDims = inputDims
        self.hiddenDims = hiddenDims
        self.numExperts = numExperts
        self.activation = activation

        self._gateProj.wrappedValue = SwitchLinear(
            inputDims: inputDims, outputDims: hiddenDims, numExperts: numExperts, bias: bias)
        self._upProj.wrappedValue = SwitchLinear(
            inputDims: inputDims, outputDims: hiddenDims, numExperts: numExperts, bias: bias)
        self._downProj.wrappedValue = SwitchLinear(
            inputDims: hiddenDims, outputDims: inputDims, numExperts: numExperts, bias: bias)
        super.init()
    }

    /// Stacked-buffer fused-matmul fast path. Returns nil if the layer is not
    /// eligible (any projection is non-quantized, missing SSD info, idx.size > 32,
    /// or no slot is available); the caller then falls through to the existing
    /// N-buffer path.
    ///
    /// Cold path allocates one `[CACHE_SLOTS, intermediate, hidden]` weight
    /// buffer per projection. Subsequent calls reuse the buffer; an LRU array
    /// rotates slot occupants; misses are written via `MLXFast.preadIntoOffset`
    /// directly into the slot's byte region. Compute issues a single
    /// `gatherQuantizedMM` per projection per layer (vs. `top_k` per projection
    /// in the legacy path).
    private func runStackedFastPath(x: MLXArray, indices: MLXArray) -> MLXArray? {
        var x = MLX.expandedDimensions(x, axes: [-2, -3])
        let doSort = true  // SSD path always sorts so expert ranges are contiguous
        var idx = indices
        var inverseOrder = MLXArray()
        if doSort {
            (x, idx, inverseOrder) = gatherSort(x: x, indices: indices)
        }
        guard idx.size <= 32,
              let qGate = gateProj as? QuantizedSwitchLinear,
              let qUp = upProj as? QuantizedSwitchLinear,
              let qDown = downProj as? QuantizedSwitchLinear,
              let gateSSD = qGate.resolveSSDInfo(),
              let upSSD = qUp.resolveSSDInfo(),
              let downSSD = qDown.resolveSSDInfo() else {
            return nil  // ineligible — fall through to legacy path
        }

        let CACHE_SLOTS = SwitchGLU.MAX_CACHE_SLOTS
        let isFused = SwitchGLU.useFusedGateUp

        // ── Cold-path allocation ──
        if _stackedGate == nil && _stackedGateUp == nil {
            if isFused {
                // Combined gate+up buffer: shape [CACHE_SLOTS, 2*intermediate, hidden].
                _stackedGateUp = MLXArray.zeros(
                    [CACHE_SLOTS, 2 * qGate.weight.dim(1), qGate.weight.dim(2)]
                ).asType(qGate.weight.dtype)
                _stackedDown = MLXArray.zeros(
                    [CACHE_SLOTS, qDown.weight.dim(1), qDown.weight.dim(2)]
                ).asType(qDown.weight.dtype)
                // Pre-concatenate gate+up scales/biases (one-time at cold init).
                _combinedGateUpScales = MLX.concatenated([qGate.scales, qUp.scales], axis: 1)
                if let gb = qGate.biases, let ub = qUp.biases {
                    _combinedGateUpBiases = MLX.concatenated([gb, ub], axis: 1)
                }
                _slotExpert = Array(repeating: nil, count: CACHE_SLOTS)
                _slotLastUsed = Array(repeating: 0, count: CACHE_SLOTS)
                _tokenCounter = 0
                var coldEvalList: [MLXArray] = [idx, _stackedGateUp!, _stackedDown!, _combinedGateUpScales!]
                if let cb = _combinedGateUpBiases { coldEvalList.append(cb) }
                MLX.eval(coldEvalList)
                _stackedGateUpBytesPerProj = _stackedGateUp!.nbytes / CACHE_SLOTS / 2
                _stackedBytesPerExpert = _stackedGateUpBytesPerProj
                _stackedDownBytesPerExpert = _stackedDown!.nbytes / CACHE_SLOTS
            } else {
                _stackedGate = MLXArray.zeros(
                    [CACHE_SLOTS, qGate.weight.dim(1), qGate.weight.dim(2)]
                ).asType(qGate.weight.dtype)
                _stackedUp = MLXArray.zeros(
                    [CACHE_SLOTS, qUp.weight.dim(1), qUp.weight.dim(2)]
                ).asType(qUp.weight.dtype)
                _stackedDown = MLXArray.zeros(
                    [CACHE_SLOTS, qDown.weight.dim(1), qDown.weight.dim(2)]
                ).asType(qDown.weight.dtype)
                _slotExpert = Array(repeating: nil, count: CACHE_SLOTS)
                _slotLastUsed = Array(repeating: 0, count: CACHE_SLOTS)
                _tokenCounter = 0
                MLX.eval([idx, _stackedGate!, _stackedUp!, _stackedDown!])
                _stackedBytesPerExpert = _stackedGate!.nbytes / CACHE_SLOTS
                _stackedDownBytesPerExpert = _stackedDown!.nbytes / CACHE_SLOTS
            }
        } else {
            // Warm path: kick off GPU work asynchronously while we
            // speculatively prefetch the prev-token's experts. The pread
            // overlaps with the GPU-side resolution of `idx`.
            asyncEval(idx)
        }
        _tokenCounter += 1

        // ── Speculative prefetch: pre-load prev-token's experts that are
        //    NOT already cached, evicting LRU slots and pre-claiming them so
        //    the current-token resolution sees them as hits. Token-to-token
        //    expert overlap is high in steady-state generation, so most of
        //    this work pays off on the same call.
        var expertToSlotPre = [Int: Int]()
        for (slot, eid) in _slotExpert!.enumerated() {
            if let eid = eid { expertToSlotPre[eid] = slot }
        }
        func pickPrefetchSlot(excluding claimed: Set<Int>) -> Int {
            for s in 0..<CACHE_SLOTS {
                if claimed.contains(s) { continue }
                if _slotExpert![s] == nil { return s }
            }
            var bestSlot = -1, bestTs = Int.max
            for s in 0..<CACHE_SLOTS {
                if claimed.contains(s) { continue }
                if _slotLastUsed![s] < bestTs {
                    bestTs = _slotLastUsed![s]; bestSlot = s
                }
            }
            return bestSlot
        }
        var specTargets: [(slot: Int, expertId: Int)] = []
        if let prevIds = _previousExpertIds {
            var specClaimed = Set<Int>()
            for eid in prevIds {
                if expertToSlotPre[eid] != nil { continue }  // already cached
                let slot = pickPrefetchSlot(excluding: specClaimed)
                if slot < 0 { break }  // shouldn't happen — CACHE_SLOTS > top_k
                if let old = _slotExpert![slot] { expertToSlotPre.removeValue(forKey: old) }
                _slotExpert![slot] = eid  // claim slot speculatively
                expertToSlotPre[eid] = slot
                specClaimed.insert(slot)
                specTargets.append((slot, eid))
            }
        }
        if !specTargets.isEmpty {
            let bpe = _stackedBytesPerExpert
            let downBpe = _stackedDownBytesPerExpert
            let errState = ThreadSafeError()
            DispatchQueue.concurrentPerform(iterations: specTargets.count * 3) { [specTargets] i in
                errState.catchError {
                let mIdx = i / 3
                let proj = i % 3
                let info = specTargets[mIdx]
                switch proj {
                case 0:
                    if isFused {
                        // Gate -> first half of slot in combined buffer.
                        let off = info.slot * 2 * bpe
                        MLXFast.preadIntoOffset(self._stackedGateUp!, safetensorsPath: gateSSD.path,
                                                tensorName: gateSSD.tensorName, expertIndex: UInt32(info.expertId), dstOffset: off)
                    } else {
                        MLXFast.preadIntoOffset(self._stackedGate!, safetensorsPath: gateSSD.path,
                                                tensorName: gateSSD.tensorName, expertIndex: UInt32(info.expertId), dstOffset: info.slot * bpe)
                    }
                case 1:
                    if isFused {
                        // Up -> second half of slot in combined buffer.
                        let off = info.slot * 2 * bpe + bpe
                        MLXFast.preadIntoOffset(self._stackedGateUp!, safetensorsPath: upSSD.path,
                                                tensorName: upSSD.tensorName, expertIndex: UInt32(info.expertId), dstOffset: off)
                    } else {
                        MLXFast.preadIntoOffset(self._stackedUp!, safetensorsPath: upSSD.path,
                                                tensorName: upSSD.tensorName, expertIndex: UInt32(info.expertId), dstOffset: info.slot * bpe)
                    }
                default:
                    MLXFast.preadIntoOffset(self._stackedDown!, safetensorsPath: downSSD.path,
                                            tensorName: downSSD.tensorName, expertIndex: UInt32(info.expertId), dstOffset: info.slot * downBpe)
                }
                }
            }
            errState.check()
        }

        if idx.size == 0 {
            var outShape = x.shape
            outShape[outShape.count - 1] = qDown.outputDims
            let result = MLXArray.zeros(outShape).asType(.float16)
            if doSort {
                return MLX.squeezed(scatterUnsort(x: result, invOrder: inverseOrder, shape: indices.shape), axis: -2)
            }
            return MLX.squeezed(result, axis: -2)
        }

        // Parse routing — `idx.asArray()` is the actual sync point on GPU.
        // By now, GPU work (current attention + router) is mostly done, AND
        // most of this token's experts are already in cache via spec prefetch.
        let cpuIndices = idx.asArray(UInt32.self)
        var ranges = [ExpertRange]()
        var startIdx = 0
        while startIdx < cpuIndices.count {
            let eid = Int(cpuIndices[startIdx])
            var endIdx = startIdx + 1
            while endIdx < cpuIndices.count && Int(cpuIndices[endIdx]) == eid { endIdx += 1 }
            ranges.append(ExpertRange(id: eid, start: startIdx, end: endIdx))
            startIdx = endIdx
        }

        // ── LRU resolution: route each range to a slot ──
        var expertToSlot = [Int: Int]()
        for (slot, eid) in _slotExpert!.enumerated() {
            if let eid = eid { expertToSlot[eid] = slot }
        }
        func pickEvictionSlot(excluding claimed: Set<Int>) -> Int {
            for s in 0..<CACHE_SLOTS {
                if claimed.contains(s) { continue }
                if _slotExpert![s] == nil { return s }
            }
            var bestSlot = -1, bestTs = Int.max
            for s in 0..<CACHE_SLOTS {
                if claimed.contains(s) { continue }
                if _slotLastUsed![s] < bestTs {
                    bestTs = _slotLastUsed![s]; bestSlot = s
                }
            }
            return bestSlot
        }
        var slotForRange: [Int] = []
        var missesNeedingPread: [(slot: Int, expertId: Int)] = []
        var claimedSlots = Set<Int>()
        for r in ranges {
            if let slot = expertToSlot[r.id], !claimedSlots.contains(slot) {
                slotForRange.append(slot)
                claimedSlots.insert(slot)
                _slotLastUsed![slot] = _tokenCounter
            } else {
                let slot = pickEvictionSlot(excluding: claimedSlots)
                if slot < 0 { return nil }  // no slot available; fall back
                if let old = _slotExpert![slot] { expertToSlot.removeValue(forKey: old) }
                _slotExpert![slot] = r.id
                expertToSlot[r.id] = slot
                _slotLastUsed![slot] = _tokenCounter
                claimedSlots.insert(slot)
                slotForRange.append(slot)
                missesNeedingPread.append((slot, r.id))
            }
        }

        // ── Pread misses into stacked-buffer slots ──
        if !missesNeedingPread.isEmpty {
            let bpe = _stackedBytesPerExpert
            let downBpe = _stackedDownBytesPerExpert
            let errState = ThreadSafeError()
            DispatchQueue.concurrentPerform(iterations: missesNeedingPread.count * 3) { [missesNeedingPread] i in
                errState.catchError {
                let mIdx = i / 3
                let proj = i % 3
                let info = missesNeedingPread[mIdx]
                switch proj {
                case 0:
                    if isFused {
                        let off = info.slot * 2 * bpe
                        MLXFast.preadIntoOffset(self._stackedGateUp!, safetensorsPath: gateSSD.path,
                                                tensorName: gateSSD.tensorName, expertIndex: UInt32(info.expertId), dstOffset: off)
                    } else {
                        MLXFast.preadIntoOffset(self._stackedGate!, safetensorsPath: gateSSD.path,
                                                tensorName: gateSSD.tensorName, expertIndex: UInt32(info.expertId), dstOffset: info.slot * bpe)
                    }
                case 1:
                    if isFused {
                        let off = info.slot * 2 * bpe + bpe
                        MLXFast.preadIntoOffset(self._stackedGateUp!, safetensorsPath: upSSD.path,
                                                tensorName: upSSD.tensorName, expertIndex: UInt32(info.expertId), dstOffset: off)
                    } else {
                        MLXFast.preadIntoOffset(self._stackedUp!, safetensorsPath: upSSD.path,
                                                tensorName: upSSD.tensorName, expertIndex: UInt32(info.expertId), dstOffset: info.slot * bpe)
                    }
                default:
                    MLXFast.preadIntoOffset(self._stackedDown!, safetensorsPath: downSSD.path,
                                            tensorName: downSSD.tensorName, expertIndex: UInt32(info.expertId), dstOffset: info.slot * downBpe)
                }
                }
            }
            errState.check()
        }
        _previousExpertIds = ranges.map { $0.id }

        // ── Build slotPerToken + slotExperts arrays for fused compute ──
        var slotPerTokenArr = [Int32](repeating: 0, count: cpuIndices.count)
        for (rIdx, r) in ranges.enumerated() {
            let s = Int32(slotForRange[rIdx])
            for t in r.start..<r.end { slotPerTokenArr[t] = s }
        }
        let slotPerToken = MLXArray(slotPerTokenArr).asType(.uint32)
        let slotExperts = _slotExpert!.map { Int32($0 ?? 0) }

        // ── Fused compute: ONE gatherQuantizedMM per projection ──
        let intermediate: MLXArray
        if isFused {
            // SINGLE matmul over combined gate+up buffer; split the output into halves.
            let (xGate, xUp) = self.runFusedGateUpMatmul(
                x: x,
                qGate: qGate,
                slotPerToken: slotPerToken,
                slotExperts: slotExperts)
            intermediate = activation(xGate) * xUp
        } else {
            let xGate = qGate.computeExpertsFused(x, stackedBuffer: _stackedGate!,
                                                  slotPerToken: slotPerToken, slotExperts: slotExperts)
            let xUp = qUp.computeExpertsFused(x, stackedBuffer: _stackedUp!,
                                              slotPerToken: slotPerToken, slotExperts: slotExperts)
            intermediate = activation(xGate) * xUp
        }
        x = qDown.computeExpertsFused(intermediate, stackedBuffer: _stackedDown!,
                                      slotPerToken: slotPerToken, slotExperts: slotExperts)

        if doSort {
            return MLX.squeezed(scatterUnsort(x: x, invOrder: inverseOrder, shape: indices.shape), axis: -2)
        }
        return MLX.squeezed(x, axis: -2)
    }

    /// Single fused `gatherQuantizedMM` over the combined `_stackedGateUp`
    /// buffer (shape `[CACHE_SLOTS, 2 * intermediate, hidden]`), then splits
    /// the output `[..., 2 * intermediate]` into `(xGate, xUp)` halves.
    ///
    /// Pre-conditions (guaranteed by `runStackedFastPath` cold init when
    /// `useFusedGateUp` is true):
    ///   - `_stackedGateUp` populated with gate -> first half, up -> second half per slot
    ///   - `_combinedGateUpScales` = `concat(qGate.scales, qUp.scales, axis: 1)`
    ///   - `_combinedGateUpBiases` = `concat(qGate.biases, qUp.biases, axis: 1)` (or nil)
    private func runFusedGateUpMatmul(
        x: MLXArray,
        qGate: QuantizedSwitchLinear,
        slotPerToken: MLXArray,
        slotExperts: [Int32]
    ) -> (MLXArray, MLXArray) {
        let slotExpertsMLX = MLXArray(slotExperts).asType(.uint32)
        // Gather the combined scales/biases for the experts currently in our slots.
        // _combinedGateUpScales is [numExperts, 2 * intermediate, hidden / groupSize].
        let stackedScales = MLX.take(_combinedGateUpScales!, slotExpertsMLX, axis: 0)
        var stackedBiases: MLXArray? = nil
        if let cb = _combinedGateUpBiases {
            stackedBiases = MLX.take(cb, slotExpertsMLX, axis: 0)
        }

        // ONE dispatch instead of two.
        let combined = MLX.gatherQuantizedMM(
            x, _stackedGateUp!,
            scales: stackedScales,
            biases: stackedBiases,
            rhsIndices: slotPerToken,
            transpose: true,
            groupSize: qGate.groupSize, bits: qGate.bits, mode: qGate.mode, sortedIndices: true
        )

        // Split [..., 2 * intermediate] into (xGate, xUp).
        let interDim = qGate.outputDims
        let leadingShape = Array(x.shape.dropLast())
        var combinedReshaped = combined
        let canonicalCombined = leadingShape + [2 * interDim]
        if combinedReshaped.shape != canonicalCombined {
            combinedReshaped = combinedReshaped.reshaped(canonicalCombined)
        }
        let xGate = combinedReshaped[.ellipsis, 0 ..< interDim]
        let xUp = combinedReshaped[.ellipsis, interDim ..< 2 * interDim]
        return (xGate, xUp)
    }

    public func callAsFunction(_ x: MLXArray, _ indices: MLXArray) -> MLXArray {
        // Stacked-buffer fused-matmul fast path (env-gated MLX_MOE_STACKED=1).
        // Early-out into the stacked path when applicable; otherwise fall
        // through to the existing SSD-streaming / legacy code below.
        if SwitchGLU.useStackedBuffers,
           ExpertStreamingConfig.shared.isEnabled,
           let result = self.runStackedFastPath(x: x, indices: indices) {
            return result
        }

        var x = MLX.expandedDimensions(x, axes: [-2, -3])

        // We must force sorting/flattening when SSD streaming is active to properly batch
        // expert kernel dispatches dynamically over contiguous arrays.
        let isSSDStreaming = ExpertStreamingConfig.shared.isEnabled
        // NOTE: indices eval deferred to inside the cross-projection path below,
        // where it's merged with buffer allocation into fewer eval calls.
        let doSort = (indices.size >= 64) || isSSDStreaming

        var idx = indices
        var inverseOrder = MLXArray()

        if doSort {
            (x, idx, inverseOrder) = gatherSort(x: x, indices: indices)
        }

        // ── Cross-projection batched SSD streaming path ──────────────────
        // When all 3 projections are quantized and SSD-streaming is active,
        // orchestrate buffer allocation, pread, and compute across all 3
        // projections to minimize MLX.eval() calls:
        //   - Single-token (fast path): 1 eval merges idx + buffer alloc
        //   - Prompt (large batch): 2 evals (idx, then buffers)
        //   - NO final eval — next layer's eval(idx) forces this layer
        // This reduces from 4 evals/layer (original) to 1 eval/layer.
        if isSSDStreaming,
           let qGate = gateProj as? QuantizedSwitchLinear,
           let qUp = upProj as? QuantizedSwitchLinear,
           let qDown = downProj as? QuantizedSwitchLinear,
           let gateSSD = qGate.resolveSSDInfo(),
           let upSSD = qUp.resolveSSDInfo(),
           let downSSD = qDown.resolveSSDInfo() {

            // ── EVAL REDUCTION STRATEGY ──────────────────────────────────────
            // For single-token generation (idx.size ≤ 32), we merge the sorted-
            // indices eval and buffer-allocation eval into ONE call, cutting from
            // 3 evals/layer to 1.  The final MLX.eval(x) is removed entirely:
            // the NEXT layer's SwitchGLU eval(idx) transitively forces this
            // layer's full output (including KV cache) through the lazy
            // dependency chain.  For the last layer, the generation loop's eval
            // of logits handles it.
            // ─────────────────────────────────────────────────────────────────

            if idx.size <= 32 {
                // ── FAST PATH: single-token generation with async I/O-GPU pipeline ──
                //
                // STRATEGY: Overlap NVMe I/O with GPU compute using asyncEval.
                //
                // Cold path (first token): Allocate persistent buffers, merged eval,
                //   full pread — same as ssd-opt-v1 baseline.
                //
                // Warm path (subsequent tokens): asyncEval(idx) starts GPU work
                //   (prev layer expert compute + current attention/router) while
                //   CPU speculatively preads predicted experts (from previous token's
                //   routing) into persistent buffers. After GPU sync, only ~30% of
                //   experts need on-demand pread (misses). Saves ~60ms/token by
                //   hiding I/O behind GPU compute.
                //
                // Memory cost: ~5GB for persistent buffers across 48 layers
                //   (vs ~13GB for the failed in-memory cache approach).

                let maxBuffers = idx.size  // typically 8 (top_k)

                if _persistentGate == nil {
                    // ── COLD PATH: first token, allocate persistent buffers ──
                    _persistentGate = qGate.allocateExpertBuffers(maxBuffers)
                    _persistentUp = qUp.allocateExpertBuffers(maxBuffers)
                    _persistentDown = qDown.allocateExpertBuffers(maxBuffers)


                    // Merged eval: idx + buffer allocations (same as ssd-opt-v1)
                    var toEval: [MLXArray] = [idx]
                    toEval.append(contentsOf: _persistentGate!)
                    toEval.append(contentsOf: _persistentUp!)
                    toEval.append(contentsOf: _persistentDown!)
                    MLX.eval(toEval)

                    // Handle empty indices
                    if idx.size == 0 {
                        var outShape = x.shape
                        outShape[outShape.count - 1] = qDown.outputDims
                        let result = MLXArray.zeros(outShape).asType(.float16)
                        if doSort {
                            return MLX.squeezed(scatterUnsort(x: result, invOrder: inverseOrder, shape: indices.shape), axis: -2)
                        }
                        return MLX.squeezed(result, axis: -2)
                    }

                    // Parse routing
                    let cpuIndices = idx.asArray(UInt32.self)
                    var ranges = [ExpertRange]()
                    var startIdx = 0
                    while startIdx < cpuIndices.count {
                        let eid = Int(cpuIndices[startIdx])
                        var endIdx = startIdx + 1
                        while endIdx < cpuIndices.count && Int(cpuIndices[endIdx]) == eid { endIdx += 1 }
                        ranges.append(ExpertRange(id: eid, start: startIdx, end: endIdx))
                        startIdx = endIdx
                    }

                    // Full concurrent pread (baseline path)
                    let totalReads = ranges.count * 3
                    let errState = ThreadSafeError()
                    DispatchQueue.concurrentPerform(iterations: totalReads) { [ranges] i in
                        errState.catchError {
                        let expertIdx = i / 3
                        let projIdx = i % 3
                        let r = ranges[expertIdx]
                        switch projIdx {
                        case 0:
                            MLXFast.preadInto(self._persistentGate![expertIdx], safetensorsPath: gateSSD.path,
                                              tensorName: gateSSD.tensorName, expertIndex: UInt32(r.id))
                        case 1:
                            MLXFast.preadInto(self._persistentUp![expertIdx], safetensorsPath: upSSD.path,
                                              tensorName: upSSD.tensorName, expertIndex: UInt32(r.id))
                        default:
                            MLXFast.preadInto(self._persistentDown![expertIdx], safetensorsPath: downSSD.path,
                                              tensorName: downSSD.tensorName, expertIndex: UInt32(r.id))
                        }
                        }
                    }
                    errState.check()

                    // Store routing for next token's predictions
                    _previousExpertIds = ranges.map { $0.id }

                    // Lazy compute
                    let usedGate = Array(_persistentGate![0..<ranges.count])
                    let usedUp = Array(_persistentUp![0..<ranges.count])
                    let usedDown = Array(_persistentDown![0..<ranges.count])
                    let xGate = qGate.computeExperts(x, buffers: usedGate, ranges: ranges)
                    let xUp = qUp.computeExperts(x, buffers: usedUp, ranges: ranges)
                    let intermediate = activation(xGate) * xUp
                    x = qDown.computeExperts(intermediate, buffers: usedDown, ranges: ranges)

                } else {
                    // ── WARM PATH: asyncEval + speculative pread pipeline ──

                    // Start GPU work asynchronously: forces prev layer's expert
                    // compute + current layer's attention + router.
                    // GPU time: ~2.7ms. CPU is free immediately.
                    asyncEval(idx)

                    // Speculative pread during GPU async window.
                    // Load previous token's experts into persistent buffers.
                    // ~70% will match this token's routing (expert stickiness).
                    // The 1.7ms of pread overlaps with 2.7ms of GPU work.
                    if let prevIds = _previousExpertIds {
                        let specCount = min(prevIds.count, maxBuffers)
                        let specReads = specCount * 3
                        let errState = ThreadSafeError()
                        DispatchQueue.concurrentPerform(iterations: specReads) { i in
                            errState.catchError {
                            let slot = i / 3
                            let proj = i % 3
                            let expertId = prevIds[slot]
                            switch proj {
                            case 0:
                                MLXFast.preadInto(self._persistentGate![slot], safetensorsPath: gateSSD.path,
                                                  tensorName: gateSSD.tensorName, expertIndex: UInt32(expertId))
                            case 1:
                                MLXFast.preadInto(self._persistentUp![slot], safetensorsPath: upSSD.path,
                                                  tensorName: upSSD.tensorName, expertIndex: UInt32(expertId))
                            default:
                                MLXFast.preadInto(self._persistentDown![slot], safetensorsPath: downSSD.path,
                                                  tensorName: downSSD.tensorName, expertIndex: UInt32(expertId))
                            }
                            }
                        }
                        errState.check()
                    }

                    // Sync on idx (blocks until GPU finishes attention + router)
                    if idx.size == 0 {
                        var outShape = x.shape
                        outShape[outShape.count - 1] = qDown.outputDims
                        let result = MLXArray.zeros(outShape).asType(.float16)
                        if doSort {
                            return MLX.squeezed(scatterUnsort(x: result, invOrder: inverseOrder, shape: indices.shape), axis: -2)
                        }
                        return MLX.squeezed(result, axis: -2)
                    }

                    // Parse actual routing
                    let cpuIndices = idx.asArray(UInt32.self)
                    var ranges = [ExpertRange]()
                    var startIdx = 0
                    while startIdx < cpuIndices.count {
                        let eid = Int(cpuIndices[startIdx])
                        var endIdx = startIdx + 1
                        while endIdx < cpuIndices.count && Int(cpuIndices[endIdx]) == eid { endIdx += 1 }
                        ranges.append(ExpertRange(id: eid, start: startIdx, end: endIdx))
                        startIdx = endIdx
                    }
                    let actualIds = ranges.map { $0.id }

                    // Map actual experts to persistent buffer slots.
                    // Hits: buffer slot already has correct data from speculative pread.
                    // Misses: assign to a free slot, pread on demand.
                    var usedGate = [MLXArray]()
                    var usedUp = [MLXArray]()
                    var usedDown = [MLXArray]()

                    if let prevIds = _previousExpertIds {
                        var prevSlotMap = [Int: Int]()  // expertId -> buffer slot
                        for (slot, eid) in prevIds.enumerated() {
                            prevSlotMap[eid] = slot
                        }

                        var usedSlots = Set<Int>()
                        var missInfo = [(rangeIdx: Int, expertId: Int, bufferSlot: Int)]()
                        var slotExhausted = false

                        for (ri, r) in ranges.enumerated() {
                            if let slot = prevSlotMap[r.id], !usedSlots.contains(slot) {
                                // HIT: persistent buffer[slot] has correct expert data
                                usedGate.append(_persistentGate![slot])
                                usedUp.append(_persistentUp![slot])
                                usedDown.append(_persistentDown![slot])
                                usedSlots.insert(slot)
                            } else {
                                // MISS: find a free slot
                                guard let freeSlot = (0..<maxBuffers).first(where: { !usedSlots.contains($0) }) else {
                                    // All buffer slots exhausted — fall through to
                                    // full-pread path below (Issue #87)
                                    slotExhausted = true
                                    break
                                }
                                usedGate.append(_persistentGate![freeSlot])
                                usedUp.append(_persistentUp![freeSlot])
                                usedDown.append(_persistentDown![freeSlot])
                                usedSlots.insert(freeSlot)
                                missInfo.append((ri, r.id, freeSlot))
                            }
                        }

                        // Pread only misses (~30% of experts, ~6 reads at QD=6)
                        if !slotExhausted && !missInfo.isEmpty {
                            let totalMissReads = missInfo.count * 3
                            let errState = ThreadSafeError()
                            DispatchQueue.concurrentPerform(iterations: totalMissReads) { [missInfo] i in
                                errState.catchError {
                                let mIdx = i / 3
                                let proj = i % 3
                                let info = missInfo[mIdx]
                                switch proj {
                                case 0:
                                    MLXFast.preadInto(self._persistentGate![info.bufferSlot],
                                                      safetensorsPath: gateSSD.path,
                                                      tensorName: gateSSD.tensorName,
                                                      expertIndex: UInt32(info.expertId))
                                case 1:
                                    MLXFast.preadInto(self._persistentUp![info.bufferSlot],
                                                      safetensorsPath: upSSD.path,
                                                      tensorName: upSSD.tensorName,
                                                      expertIndex: UInt32(info.expertId))
                                default:
                                    MLXFast.preadInto(self._persistentDown![info.bufferSlot],
                                                      safetensorsPath: downSSD.path,
                                                      tensorName: downSSD.tensorName,
                                                      expertIndex: UInt32(info.expertId))
                                }
                                }
                            }
                            errState.check()
                        }
                    }

                    // Slot exhaustion or no predictions — full pread fallback
                    if usedGate.count != ranges.count {
                        usedGate.removeAll()
                        usedUp.removeAll()
                        usedDown.removeAll()
                        for i in 0..<ranges.count {
                            usedGate.append(_persistentGate![i])
                            usedUp.append(_persistentUp![i])
                            usedDown.append(_persistentDown![i])
                        }
                        let totalReads = ranges.count * 3
                        let errState = ThreadSafeError()
                        DispatchQueue.concurrentPerform(iterations: totalReads) { [ranges] i in
                            errState.catchError {
                            let expertIdx = i / 3
                            let projIdx = i % 3
                            let r = ranges[expertIdx]
                            switch projIdx {
                            case 0:
                                MLXFast.preadInto(self._persistentGate![expertIdx], safetensorsPath: gateSSD.path,
                                                  tensorName: gateSSD.tensorName, expertIndex: UInt32(r.id))
                            case 1:
                                MLXFast.preadInto(self._persistentUp![expertIdx], safetensorsPath: upSSD.path,
                                                  tensorName: upSSD.tensorName, expertIndex: UInt32(r.id))
                            default:
                                MLXFast.preadInto(self._persistentDown![expertIdx], safetensorsPath: downSSD.path,
                                                  tensorName: downSSD.tensorName, expertIndex: UInt32(r.id))
                            }
                            }
                        }
                        errState.check()
                    }

                    // Update routing for next token's predictions
                    _previousExpertIds = actualIds

                    // Lazy compute (no eval — next layer forces it)
                    let xGate = qGate.computeExperts(x, buffers: usedGate, ranges: ranges)
                    let xUp = qUp.computeExperts(x, buffers: usedUp, ranges: ranges)
                    let intermediate = activation(xGate) * xUp
                    x = qDown.computeExperts(intermediate, buffers: usedDown, ranges: ranges)
                }

            } else {
                // ── PROMPT PATH: larger batches ──
                // Eval indices first (needed for range count), then allocate exact buffers.
                MLX.eval(idx)

                // Handle empty indices
                if idx.size == 0 {
                    var outShape = x.shape
                    outShape[outShape.count - 1] = qDown.outputDims
                    let result = MLXArray.zeros(outShape).asType(.float16)
                    if doSort {
                        return MLX.squeezed(scatterUnsort(x: result, invOrder: inverseOrder, shape: indices.shape), axis: -2)
                    }
                    return MLX.squeezed(result, axis: -2)
                }

                // Parse expert ranges
                let cpuIndices = idx.asArray(UInt32.self)
                var ranges = [ExpertRange]()
                var startIdx = 0
                while startIdx < cpuIndices.count {
                    let eid = Int(cpuIndices[startIdx])
                    var endIdx = startIdx + 1
                    while endIdx < cpuIndices.count && Int(cpuIndices[endIdx]) == eid { endIdx += 1 }
                    ranges.append(ExpertRange(id: eid, start: startIdx, end: endIdx))
                    startIdx = endIdx
                }

                // Allocate exact buffer count and eval
                let gateBuffers = qGate.allocateExpertBuffers(ranges.count)
                let upBuffers = qUp.allocateExpertBuffers(ranges.count)
                let downBuffers = qDown.allocateExpertBuffers(ranges.count)
                MLX.eval(gateBuffers + upBuffers + downBuffers)

                // Concurrent pread (same as fast path)
                let totalReads = ranges.count * 3
                let errState = ThreadSafeError()
                DispatchQueue.concurrentPerform(iterations: totalReads) { [ranges] i in
                    errState.catchError {
                    let expertIdx = i / 3
                    let projIdx = i % 3
                    let r = ranges[expertIdx]
                    switch projIdx {
                    case 0:
                        MLXFast.preadInto(gateBuffers[expertIdx], safetensorsPath: gateSSD.path,
                                          tensorName: gateSSD.tensorName, expertIndex: UInt32(r.id))
                    case 1:
                        MLXFast.preadInto(upBuffers[expertIdx], safetensorsPath: upSSD.path,
                                          tensorName: upSSD.tensorName, expertIndex: UInt32(r.id))
                    default:
                        MLXFast.preadInto(downBuffers[expertIdx], safetensorsPath: downSSD.path,
                                          tensorName: downSSD.tensorName, expertIndex: UInt32(r.id))
                    }
                    }
                }
                errState.check()

                // Lazy compute (no eval — next layer forces it)
                let xGate = qGate.computeExperts(x, buffers: gateBuffers, ranges: ranges)
                let xUp = qUp.computeExperts(x, buffers: upBuffers, ranges: ranges)
                let intermediate = activation(xGate) * xUp
                x = qDown.computeExperts(intermediate, buffers: downBuffers, ranges: ranges)
            }

            if doSort {
                x = scatterUnsort(x: x, invOrder: inverseOrder, shape: indices.shape)
            }
            return MLX.squeezed(x, axis: -2)
        }

        // ── Fallback: original sequential path (non-SSD or non-quantized) ──
        let xUp = upProj(x, idx, sortedIndices: doSort)
        let xGate = gateProj(x, idx, sortedIndices: doSort)
        x = downProj(
            activation(xGate) * xUp,
            idx,
            sortedIndices: doSort)

        if doSort {
            x = scatterUnsort(x: x, invOrder: inverseOrder, shape: indices.shape)
        }

        return MLX.squeezed(x, axis: -2)
    }
}

public class SwitchLinear: Module, Quantizable {
    @ModuleInfo(key: "weight") public var weight: MLXArray
    @ModuleInfo(key: "bias") public var bias: MLXArray?

    public let inputDims: Int
    public let outputDims: Int
    public let numExperts: Int

    public init(inputDims: Int, outputDims: Int, numExperts: Int, bias: Bool = true) {
        self.inputDims = inputDims
        self.outputDims = outputDims
        self.numExperts = numExperts

        let scale = sqrt(1.0 / Float(inputDims))
        self._weight.wrappedValue = MLXRandom.uniform(
            low: -scale,
            high: scale,
            [numExperts, outputDims, inputDims]
        )

        if bias {
            self._bias.wrappedValue = MLXArray.zeros([numExperts, outputDims])
        }

        super.init()
    }

    /// Initializer meant for subclasses to provide weight and bias arrays directly.
    ///
    /// This is used e.g. by ``QuantizedSwitchLinear`` to provide quantized weights and biases
    /// rather than have ``SwitchLinear`` compute them.
    public init(
        inputDims: Int, outputDims: Int, numExperts: Int,
        weight: MLXArray, bias: MLXArray? = nil
    ) {
        self.inputDims = inputDims
        self.outputDims = outputDims
        self.numExperts = numExperts

        self._weight.wrappedValue = weight
        self._bias.wrappedValue = bias
    }

    public func callAsFunction(
        _ x: MLXArray, _ indices: MLXArray, sortedIndices: Bool = false
    ) -> MLXArray {
        let weightT = self.weight.swappedAxes(-1, -2)
        var result = MLX.gatherMM(x, weightT, rhsIndices: indices, sortedIndices: sortedIndices)

        if let bias = self.bias {
            result = result + MLX.expandedDimensions(bias[indices], axis: -2)
        }

        return result
    }

    public func toQuantized(groupSize: Int = 64, bits: Int = 4, mode: QuantizationMode) -> Module {
        QuantizedSwitchLinear(self, groupSize: groupSize, bits: bits, mode: mode)
    }
}

public class QuantizedSwitchLinear: SwitchLinear, Quantized {
    @ModuleInfo(key: "scales") var scales: MLXArray
    @ModuleInfo(key: "biases") var biases: MLXArray?

    public let groupSize: Int
    public let bits: Int
    public let mode: QuantizationMode
    public var tensorName: String?

    public init(
        _ other: SwitchLinear, groupSize: Int = 64, bits: Int = 4, mode: QuantizationMode = .affine
    ) {
        self.groupSize = groupSize
        self.bits = bits
        self.mode = mode

        let (quantizedWeight, scales, biases) = MLX.quantized(
            other.weight, groupSize: groupSize, bits: bits, mode: mode)

        self._scales.wrappedValue = scales
        self._biases.wrappedValue = biases

        super.init(
            inputDims: other.inputDims, outputDims: other.outputDims, numExperts: other.numExperts,
            weight: quantizedWeight, bias: other.bias)

        self.freeze()
    }

    override public func callAsFunction(
        _ x: MLXArray, _ indices: MLXArray, sortedIndices: Bool = false
    ) -> MLXArray {
        if ExpertStreamingConfig.shared.isEnabled {
            MLX.eval(indices)
            if indices.size == 0 {
                var outShape = x.shape
                outShape[outShape.count - 1] = self.outputDims
                return MLXArray.zeros(outShape).asType(.float16)
            }

            let cpuIndices = indices.asArray(UInt32.self)
            var expertResults = [MLXArray]()
            var startIdx = 0

            // macOS directNVMe: resolve the safetensors shard + tensor offset once.
            // iOS mmapPageCache: ssdInfo = nil → falls through to mmap prefault below.
            let ssdInfo: (path: String, tensorName: String)? = {
                #if os(macOS)
                guard ExpertStreamingConfig.shared.useDirectNVMe,
                      let tName = self.tensorName,
                      let filename = ExpertStreamerManager.shared?.getFile(for: tName),
                      let dir = ExpertStreamingConfig.shared.modelDirectory else { return nil }
                let path = dir.appendingPathComponent(filename).path
                return (path, tName)
                #else
                return nil  // iOS always uses mmap fallback
                #endif
            }()

            // ---- Parse expert ranges ----
            var ranges = [ExpertRange]()
            while startIdx < cpuIndices.count {
                let eid = Int(cpuIndices[startIdx])
                var endIdx = startIdx + 1
                while endIdx < cpuIndices.count && Int(cpuIndices[endIdx]) == eid { endIdx += 1 }
                ranges.append(ExpertRange(id: eid, start: startIdx, end: endIdx))
                startIdx = endIdx
            }

            if let info = ssdInfo {
                // ---- Batch-allocate weight buffers (1 eval for all) ----
                var buffers = [MLXArray]()
                for _ in ranges {
                    buffers.append(MLXArray.zeros([1, self.weight.dim(1), self.weight.dim(2)]).asType(self.weight.dtype))
                }
                MLX.eval(buffers)

                // ---- Sequential pread into each fresh buffer ----
                for (i, r) in ranges.enumerated() {
                    MLXFast.preadInto(
                        buffers[i],
                        safetensorsPath: info.path,
                        tensorName: info.tensorName,
                        expertIndex: UInt32(r.id)
                    )
                }

                // ---- GPU compute for all experts ----
                for (i, r) in ranges.enumerated() {
                    let rangeX = x[r.start ..< r.end]
                    let expertIndices = MLXArray.zeros([rangeX.dim(0)], type: UInt32.self)
                    let expertScales = self.scales[r.id ..< r.id + 1]
                    var expertBiases: MLXArray? = nil
                    if let b = self.biases { expertBiases = b[r.id ..< r.id + 1] }

                    var expertOutput = MLX.gatherQuantizedMM(
                        rangeX, buffers[i],
                        scales: expertScales, biases: expertBiases,
                        rhsIndices: expertIndices, transpose: true,
                        groupSize: self.groupSize, bits: self.bits, mode: mode, sortedIndices: true
                    )
                    if let bias = self.bias {
                        let biasSlice = bias[r.id ..< r.id + 1]
                        expertOutput = expertOutput + MLX.expandedDimensions(biasSlice[expertIndices], axis: -2)
                    }
                    let leadingShape = Array(rangeX.shape.dropLast())
                    let canonicalShape = leadingShape + [self.outputDims]
                    if expertOutput.shape != canonicalShape {
                        expertOutput = expertOutput.reshaped(canonicalShape)
                    }
                    expertResults.append(expertOutput)
                }
            } else {
                // iOS mmap fallback — original sequential path with per-expert eval
                for r in ranges {
                    let rangeX = x[r.start ..< r.end]
                    let expertIndices = MLXArray.zeros([rangeX.dim(0)], type: UInt32.self)
                    let w = self.weight[r.id ..< r.id + 1]
                    MLX.eval(w)
                    MLXFast.prefault(w)
                    let expertScales = self.scales[r.id ..< r.id + 1]
                    var expertBiases: MLXArray? = nil
                    if let b = self.biases { expertBiases = b[r.id ..< r.id + 1] }
                    var expertOutput = MLX.gatherQuantizedMM(
                        rangeX, w,
                        scales: expertScales, biases: expertBiases,
                        rhsIndices: expertIndices, transpose: true,
                        groupSize: self.groupSize, bits: self.bits, mode: mode, sortedIndices: true
                    )
                    if let bias = self.bias {
                        let biasSlice = bias[r.id ..< r.id + 1]
                        expertOutput = expertOutput + MLX.expandedDimensions(biasSlice[expertIndices], axis: -2)
                    }
                    let leadingShape = Array(rangeX.shape.dropLast())
                    let canonicalShape = leadingShape + [self.outputDims]
                    if expertOutput.shape != canonicalShape {
                        expertOutput = expertOutput.reshaped(canonicalShape)
                    }
                    MLX.eval(expertOutput)
                    expertResults.append(expertOutput)
                }
            }

            // Batch eval all expert outputs at once (directNVMe path)
            if let _ = ssdInfo, !expertResults.isEmpty {
                MLX.eval(expertResults)
            }

            if expertResults.isEmpty {
                var outShape = x.shape
                outShape[outShape.count - 1] = self.outputDims
                return MLXArray.zeros(outShape).asType(.float16)
            }
            // PAPPS Heuristic: Prefetch exactly these experts so they are in cache for the N+1 token.
            if let _ = ssdInfo {
                let uniqueIndices = Set(cpuIndices)
                for _ in uniqueIndices {
                    // MLXFast.pappsPrefetch(
                    //     safetensorsPath: info.path,
                    //     tensorName: info.tensorName,
                    //     expertIndex: idx
                    // )
                }
            }

            return MLX.concatenated(expertResults, axis: 0)
        }

        var result = MLX.gatherQuantizedMM(
            x,
            self.weight,
            scales: self.scales,
            biases: self.biases,
            rhsIndices: indices,
            transpose: true,
            groupSize: self.groupSize,
            bits: self.bits,
            mode: mode,
            sortedIndices: sortedIndices
        )

        if let bias = self.bias {
            result = result + MLX.expandedDimensions(bias[indices], axis: -2)
        }

        return result
    }


    // MARK: - Cross-projection batching helpers (SSD streaming)

    /// Resolve the safetensors path and tensor name for SSD streaming.
    public func resolveSSDInfo() -> (path: String, tensorName: String)? {
        #if os(macOS)
        guard ExpertStreamingConfig.shared.useDirectNVMe,
              let tName = self.tensorName,
              let filename = ExpertStreamerManager.shared?.getFile(for: tName),
              let dir = ExpertStreamingConfig.shared.modelDirectory else { return nil }
        let path = dir.appendingPathComponent(filename).path
        return (path, tName)
        #else
        return nil
        #endif
    }

    /// Allocate zero-filled weight buffers for `count` experts (lazy, not yet eval'd).
    public func allocateExpertBuffers(_ count: Int) -> [MLXArray] {
        var buffers = [MLXArray]()
        for _ in 0..<count {
            buffers.append(MLXArray.zeros([1, self.weight.dim(1), self.weight.dim(2)]).asType(self.weight.dtype))
        }
        return buffers
    }

    /// Load expert weights from SSD into pre-allocated (eval'd) buffers.
    public func loadExpertWeights(_ buffers: [MLXArray], ranges: [ExpertRange], ssdInfo: (path: String, tensorName: String)) {
        for (i, r) in ranges.enumerated() {
            MLXFast.preadInto(
                buffers[i],
                safetensorsPath: ssdInfo.path,
                tensorName: ssdInfo.tensorName,
                expertIndex: UInt32(r.id)
            )
        }
    }

    /// Compute expert outputs using pre-loaded weight buffers. Returns LAZY result (no eval).
    public func computeExperts(_ x: MLXArray, buffers: [MLXArray], ranges: [ExpertRange]) -> MLXArray {
        var expertResults = [MLXArray]()
        for (i, r) in ranges.enumerated() {
            let rangeX = x[r.start ..< r.end]
            let expertIndices = MLXArray.zeros([rangeX.dim(0)], type: UInt32.self)
            let expertScales = self.scales[r.id ..< r.id + 1]
            var expertBiases: MLXArray? = nil
            if let b = self.biases { expertBiases = b[r.id ..< r.id + 1] }

            var expertOutput = MLX.gatherQuantizedMM(
                rangeX, buffers[i],
                scales: expertScales, biases: expertBiases,
                rhsIndices: expertIndices, transpose: true,
                groupSize: self.groupSize, bits: self.bits, mode: mode, sortedIndices: true
            )
            if let bias = self.bias {
                let biasSlice = bias[r.id ..< r.id + 1]
                expertOutput = expertOutput + MLX.expandedDimensions(biasSlice[expertIndices], axis: -2)
            }
            let leadingShape = Array(rangeX.shape.dropLast())
            let canonicalShape = leadingShape + [self.outputDims]
            if expertOutput.shape != canonicalShape {
                expertOutput = expertOutput.reshaped(canonicalShape)
            }
            expertResults.append(expertOutput)
        }

        if expertResults.isEmpty {
            var outShape = x.shape
            outShape[outShape.count - 1] = self.outputDims
            return MLXArray.zeros(outShape).asType(.float16)
        }
        return MLX.concatenated(expertResults, axis: 0)
    }

    /// Stacked-buffer fused-matmul variant of `computeExperts`. Replaces the
    /// per-expert `gatherQuantizedMM` loop (one dispatch per expert) with a
    /// single dispatch over the full stacked weight buffer.
    ///
    /// - Parameters:
    ///   - x: input activations, shape `[totalTokens, ..., inputDims]`.
    ///   - stackedBuffer: weight buffer, shape `[CACHE_SLOTS, outputDims, inputDims]`.
    ///       Slots are populated externally via `MLXFast.preadIntoOffset`.
    ///   - slotPerToken: uint32 array mapping each token (along axis 0 of `x`)
    ///       to a slot index in `stackedBuffer`. Built from the routing.
    ///   - slotExperts: per-slot expert IDs (`0..<numExperts`). Used to gather
    ///       per-slot scales/biases from `self.scales` and `self.biases`.
    public func computeExpertsFused(
        _ x: MLXArray,
        stackedBuffer: MLXArray,
        slotPerToken: MLXArray,
        slotExperts: [Int32]
    ) -> MLXArray {
        let slotExpertsMLX = MLXArray(slotExperts).asType(.uint32)
        // Gather scales/biases for the experts currently in our slots.
        // Result shape: [N_slots, outputDims, inputDims / groupSize].
        let stackedScales = MLX.take(self.scales, slotExpertsMLX, axis: 0)
        var stackedBiases: MLXArray? = nil
        if let b = self.biases { stackedBiases = MLX.take(b, slotExpertsMLX, axis: 0) }

        var output = MLX.gatherQuantizedMM(
            x, stackedBuffer,
            scales: stackedScales,
            biases: stackedBiases,
            rhsIndices: slotPerToken,
            transpose: true,
            groupSize: self.groupSize, bits: self.bits, mode: mode, sortedIndices: true
        )

        // Optional per-token bias add (gathered from per-slot bias).
        if let bias = self.bias {
            let stackedBias = MLX.take(bias, slotExpertsMLX, axis: 0)             // [N_slots, outputDims]
            let perTokenBias = MLX.take(stackedBias, slotPerToken, axis: 0)       // [tokens, outputDims]
            output = output + MLX.expandedDimensions(perTokenBias, axis: -2)
        }

        let leadingShape = Array(x.shape.dropLast())
        let canonicalShape = leadingShape + [self.outputDims]
        if output.shape != canonicalShape {
            output = output.reshaped(canonicalShape)
        }
        return output
    }
}

public class ExpertStreamerManager {
    nonisolated(unsafe) public static var shared: ExpertStreamerManager?

    public let weightMap: [String: String]

    public init(modelDirectory: URL) {
        var map = [String: String]()
        let indexUrl = modelDirectory.appendingPathComponent("model.safetensors.index.json")
        if let data = try? Data(contentsOf: indexUrl),
           let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
           let weightMapJson = json["weight_map"] as? [String: String] {
            map = weightMapJson
        }
        self.weightMap = map
    }

    public func getFile(for tensorName: String) -> String? {
        return weightMap[tensorName]
    }
}

public final class SSDStreamMetrics: @unchecked Sendable {
    public static let shared = SSDStreamMetrics()
    private var totalBytes: Int = 0
    private var totalTimeNs: UInt64 = 0
    private var readCount: Int = 0
    private var lastLogTimeNs: UInt64 = DispatchTime.now().uptimeNanoseconds
    private let lock = NSLock()
    
    public func record(bytes: Int, timeNs: UInt64) {
        lock.lock()
        defer { lock.unlock() }
        totalBytes += bytes
        totalTimeNs += timeNs
        readCount += 1
        
        let now = DispatchTime.now().uptimeNanoseconds
        if now - lastLogTimeNs >= 1_000_000_000 {
            let count = readCount
            _ = totalBytes
            _ = totalTimeNs
            
            self.readCount = 0
            self.totalBytes = 0
            self.totalTimeNs = 0
            self.lastLogTimeNs = now
            
            if count > 0 {
                // let mb = Double(bytes) / (1024.0 * 1024.0)
                // let avgMs = (Double(ns) / 1_000_000.0) / Double(count)
                // print(String(format: "[⚡️ SSD Stream] %.1f MB/s over %d chunks | Avg latency per chunk: %.6f ms", mb, count, avgMs))
                // fflush(stdout)
            }
        }
    }
}

