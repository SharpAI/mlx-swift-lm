// SharpAI fork overlay: deepseek_v32 MTP block. Not part of upstream mlx-swift-lm.
//
// UNVERIFIED: the tensor names (eh_proj, enorm, hnorm, shared_head.norm) and the
// concat order (embedding first, then hidden state) follow the DeepSeek-V3 MTP layout
// as recalled. No deepseek_v32 checkpoint index was available to confirm them.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// Final norm of an MTP block. The lm_head is the main model's.
final class SharpAIDeepseekV32SharedHead: Module {
    @ModuleInfo(key: "norm") var norm: RMSNorm

    init(_ args: DeepseekV32Configuration) {
        self._norm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
    }
}

/// One multi-token-prediction depth: a decoder block with a fusion projection in front.
final class SharpAIDeepseekV32MTPBlock: DeepseekV32DecoderLayer {
    @ModuleInfo(key: "eh_proj") var ehProj: Linear
    @ModuleInfo(key: "enorm") var enorm: RMSNorm
    @ModuleInfo(key: "hnorm") var hnorm: RMSNorm
    @ModuleInfo(key: "shared_head") var sharedHead: SharpAIDeepseekV32SharedHead

    init(mtpConfiguration args: DeepseekV32Configuration, layerIdx: Int) {
        self._ehProj.wrappedValue = Linear(2 * args.hiddenSize, args.hiddenSize, bias: false)
        self._enorm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
        self._hnorm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
        self._sharedHead.wrappedValue = SharpAIDeepseekV32SharedHead(args)
        super.init(args, layerIdx: layerIdx)
    }

    /// `tokenEmbeddings` and `hidden` are both [B, S, D]. Returns the pre-norm hidden state.
    func predict(tokenEmbeddings: MLXArray, hidden: MLXArray, cache: CacheList?) -> MLXArray {
        let fused = ehProj(concatenated([enorm(tokenEmbeddings), hnorm(hidden)], axis: -1))
        let mask = createAttentionMask(h: fused, cache: cache?[0])
        return callAsFunction(fused, mask: mask, cache: cache)
    }
}
