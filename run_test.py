import mlx.core as mx
import mlx.core.fast as fast

q = mx.zeros((1, 24, 116, 512))
k = mx.zeros((1, 1, 116, 512))
v = mx.zeros((1, 1, 116, 512))
mask = mx.zeros((1, 1, 116, 116))

try:
    out = fast.scaled_dot_product_attention(q, k, v, scale=1.0, mask=mask)
    mx.eval(out)
    print("Success")
except Exception as e:
    print(f"Failed: {e}")
