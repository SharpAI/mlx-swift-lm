import Foundation
import MLX

MLX.GPU.set(cacheLimit: 10 * 1024 * 1024)

var out = MLXArray.zeros([4, 10])
let rows = MLXArray(0 ..< Int32(4)).reshaped([4, 1])
let cols = MLXArray([1, 2, 0, 4, 3, 5, 2, 9]).reshaped([4, 2])
let vals = MLXArray([10, 20, 30, 40, 50, 60, 70, 80]).reshaped([4, 2])

out[rows, cols] = vals
MLX.eval(out)
print(out)
