// SharpAI fork overlay: nemotron_h_mtp text model. Not part of upstream.
//
// Subclasses NemotronHModel so the module tree and checkpoint keys match NemotronH.
// The only change is that sanitize drops `mtp.*` tensors first, without relying on the
// base sanitize. UNVERIFIED: the MTP head layers are not built, so the MTP tensors are
// skipped, not loaded. This model runs the text backbone only.

import Foundation
import MLX

public final class SharpAINemotronHMtpModel: NemotronHModel {

    public override init(_ args: NemotronHConfiguration) {
        super.init(args)
    }

    public override func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        super.sanitize(weights: SharpAINemotronHMtpSanitizer.dropMTPWeights(weights))
    }
}
