import Foundation

let path = "Libraries/MLXLLM/Models/Gemma4.swift"
var file = try! String(contentsOfFile: path)
let target = """
    public struct Gemma4Router: Module, @unchecked Sendable {
        @ModuleInfo(key: "proj") public var proj: Linear

        public init(dimensions: Int, numExperts: Int) {
            self._proj.wrappedValue = Linear(dimensions, numExperts, bias: false)
            super.init()
        }

        public func callAsFunction(_ x: MLXArray, topK: Int) -> (MLXArray, MLXArray) {
            let logits = proj(x)
            let scores = MLX.softmax(logits, axis: -1)
            return MLX.topk(scores, k: topK, axis: -1)
        }
    }

    public struct Gemma4SparseMoeBlock: Module, @unchecked Sendable {
        let topK: Int

        @ModuleInfo(key: "router") var router: Gemma4Router
        @ModuleInfo(key: "switch_glu") var switchGLU: SwitchGLU

        init(dimensions: Int, numExperts: Int, topK: Int, moeIntermediateSize: Int) {
            self.topK = topK
            self._router.wrappedValue = Gemma4Router(dimensions: dimensions, numExperts: numExperts)
            self._switchGLU.wrappedValue = SwitchGLU(
                inputDims: dimensions,
                hiddenDims: moeIntermediateSize,
                numExperts: numExperts,
                activation: geluApproximate
            )
            super.init()
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            let (scores, inds) = router(x, topK: topK)
            let y = switchGLU(x, inds)
            let combined = (y * scores[.ellipsis, .newAxis]).sum(axis: -2)
            return combined
        }
    }
"""

let replace = """
    public struct Gemma4Router: Module, @unchecked Sendable {
        @ModuleInfo(key: "norm") public var norm: Gemma4RMSNorm
        @ModuleInfo(key: "proj") public var proj: Linear
        @ModuleInfo(key: "scale") public var scale: MLXArray
        @ModuleInfo(key: "per_expert_scale") public var perExpertScale: MLXArray

        let hiddenSize: Int

        public init(dimensions: Int, numExperts: Int, eps: Float) {
            self.hiddenSize = dimensions
            self._norm.wrappedValue = Gemma4RMSNorm(dimensions, eps: eps)
            self._proj.wrappedValue = Linear(dimensions, numExperts, bias: false)
            // Note: We use ones directly; safetensors hydration will overwrite them via MLX array matching.
            self._scale.wrappedValue = MLXArray.ones([dimensions])
            self._perExpertScale.wrappedValue = MLXArray.ones([numExperts])
            super.init()
        }

        public func callAsFunction(_ x: MLXArray, topK: Int) -> (MLXArray, MLXArray, MLXArray) {
            var hiddenStates = norm(x)
            // Scaling matching HF: hidden_states * self.scale * self.scalar_root_size
            let scalarRootSize = MLXArray(pow(Float(hiddenSize), -0.5)).asType(hiddenStates.dtype)
            hiddenStates = hiddenStates * scale * scalarRootSize

            let expertScores = proj(hiddenStates)
            let routerProbabilities = MLX.softmax(expertScores, axis: -1, precise: true).asType(expertScores.dtype)

            let (topKWeightsRaw, topKIndices) = MLX.topk(routerProbabilities, k: topK, axis: -1)

            // Normalize
            var topKWeights = topKWeightsRaw / topKWeightsRaw.sum(axis: -1, keepDims: true)

            // per_expert_scale extraction: perExpertScale[top_k_index]
            // We use take logic
            let expertScaleForTokens = take(perExpertScale, topKIndices)
            topKWeights = topKWeights * expertScaleForTokens

            return (routerProbabilities, topKWeights, topKIndices)
        }
    }

    public struct Gemma4SparseMoeBlock: Module, @unchecked Sendable {
        let topK: Int

        @ModuleInfo(key: "router") var router: Gemma4Router
        @ModuleInfo(key: "switch_glu") var switchGLU: SwitchGLU

        init(dimensions: Int, numExperts: Int, topK: Int, moeIntermediateSize: Int, eps: Float) {
            self.topK = topK
            self._router.wrappedValue = Gemma4Router(dimensions: dimensions, numExperts: numExperts, eps: eps)
            self._switchGLU.wrappedValue = SwitchGLU(
                inputDims: dimensions,
                hiddenDims: moeIntermediateSize,
                numExperts: numExperts,
                activation: geluApproximate
            )
            super.init()
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            let (_, topKWeights, inds) = router(x, topK: topK)
            let y = switchGLU(x, inds)
            let combined = (y * topKWeights[.ellipsis, .newAxis]).sum(axis: -2)
            return combined
        }
    }
"""

file = file.replacingOccurrences(of: target, with: replace)

// Also need to fix the instantiation where eps was not provided.
file = file.replacingOccurrences(of: "Gemma4SparseMoeBlock(\n                dimensions: config.hiddenSize,\n                numExperts: config.numExperts!,\n                topK: config.topKExperts ?? 8,\n                moeIntermediateSize: config.moeIntermediateSize ?? config.intermediateSize\n            )", with: "Gemma4SparseMoeBlock(\n                dimensions: config.hiddenSize,\n                numExperts: config.numExperts!,\n                topK: config.topKExperts ?? 8,\n                moeIntermediateSize: config.moeIntermediateSize ?? config.intermediateSize,\n                eps: config.rmsNormEps\n            )")

try! file.write(toFile: path, atomically: true, encoding: .utf8)
