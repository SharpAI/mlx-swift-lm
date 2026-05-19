import Foundation
import MLX
MLX.GPU.set(cacheLimit: 10 * 1024 * 1024)

let size: Int = 10
let arr = MLXArray(0 ..< size).asType(.int32)
print(arr)
