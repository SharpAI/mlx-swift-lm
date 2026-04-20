import Foundation
import MLX
import MLXLMCommon
import MLXNN
import MLXFast

print("Starting reproducer...")

let q = MLXArray.zeros([1, 24, 116, 512])
let k = MLXArray.zeros([1, 1, 116, 512])
let v = MLXArray.zeros([1, 1, 116, 512])

let mask = MLXArray.zeros([1, 1, 116, 116])

print("Attempting to run SDPA...")
do {
    let out = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v, scale: 1.0, mask: .array(mask))
    MLX.eval(out)
    print("SDPA success!")
} catch {
    print("Caught error: \(error)")
}
