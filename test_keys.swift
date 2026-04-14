import Foundation
import MLX
import MLXNN
@testable import MLXVLM

MLX.GPU.set(cacheLimit: 10 * 1024 * 1024)
let config = Gemma4AudioConfiguration()
let model = Gemma4AudioModel(config: config)
for (k, _) in model.parameters() {
    if k.contains("norm_out") {
        print(k)
    }
}
