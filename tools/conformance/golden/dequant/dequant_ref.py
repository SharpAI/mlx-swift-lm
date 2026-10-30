"""float64 torch-CPU reference for weight dequantisation formats, transcribed from vLLM @ 739e5945d.

Every function states the vLLM file:line it follows. All paths are relative to
vllm/model_executor/layers/quantization/ (Q/) unless stated.

INFERRED points (not literally in the vLLM snapshot; the golden depends on them, so they are listed):
  I1. MXFP4 nibble order: the snapshot's mxfp4 dequant is delegated to amd-quark (Q/utils/mxfp4_utils.py:94-106,
      `mx.dq_mxfp4`), whose source is not in the snapshot. We assume the same packing as NVFP4's break_fp4_bytes
      (Q/utils/nvfp4_emulation_utils.py:27-38): element 2i in the LOW nibble, 2i+1 in the HIGH nibble. MXFP4 and NVFP4
      share the E2M1 value table and the uint8 packing in every other vLLM place (marlin repack), so this is the
      consistent reading, but it is not proven by a vLLM line.
  I2. MXFP4 block size 32: Q/utils/ocp_mx_utils.py:9 (OCP_MX_BLOCK_SIZE = 32) -- stated, not inferred. The E8M0 decode
      2^(byte-127) is stated for mxfp6 (Q/utils/mxfp6_utils.py:78) and mxfp8 (Q/utils/mxfp8_utils.py:118); applying it
      to mxfp4 scales is the inference (marlin_utils_fp4.py:124,249 views mxfp4 scales as torch.float8_e8m0fnu, whose
      value is the same 2^(byte-127)).
  I3. E8M0 byte 255: torch.float8_e8m0fnu defines it NaN; the vLLM FORMULA exp2(byte-127) in float32 overflows to inf.
      Both are recorded in the `e8m0_special` fixture; the numeric matrices never use byte 255.
  I4. ue8m0 FP8 block scales: vLLM's ue8m0 mentions in Q/utils/fp8_utils.py (lines 145, 206, 346) are about DYNAMIC
      activation scales being rounded to a power of two (float scale, no byte storage). A weight scale stored as an
      E8M0 byte and then used as the block scale is the composition  scaled_dequantize(w, 2^(byte-127), (bs, bs))
      (Q/utils/quant_utils.py:333-343 with the E8M0 decode above). The Swift overlay documents the same composition.
  I5. NVFP4 global scale: the emulation multiplies the decoded block scale by `global_scale`
      (nvfp4_emulation_utils.py:72). In the checkpoint-loading path the stored per-tensor value is inverted first
      (Q/compressed_tensors/schemes/compressed_tensors_w4a16_nvfp4.py:88-90 `1.0 / weight_global_scale.max()`), so the
      reference takes the already-inverted multiplier, as the emulation does.
  I6. Non-divisible FP8 blocks: vLLM asserts divisibility (Q/utils/quant_utils.py:297-298 / scaled_dequantize group
      asserts); the zero-pad-and-crop behaviour of the Swift code is NOT covered by a vLLM line, so no golden case uses it.
  I7. compressed-tensors pack-quantized int4: weight_packed int32 [N, K/8], nibble i of a word = bits [4i, 4i+4)
      (Q/utils/quant_utils.py:426-444 unpack_quantized_values_into_int32; pack at :404-423). Symmetric storage is
      uint4b8 = signed + 8 (compressed_tensors_wNa16.py:35,75; Q/utils/quant_utils.py:826-848 subtracts 8). The
      asymmetric zero point is a uint4 subtracted before the scale (Q/utils/quant_utils.py:604 `(w_q - zp) * w_s`);
      its packing along the OUTPUT dim (dim 0) is by the same unpack function with packed_dim=0 and
      packed_factor=pack_factor (compressed_tensors_wNa16.py:170-171 and :182-183 PackedvLLMParameter /
      PackedColumnParameter; the zero-point tensor shape is [out // pack_factor, groups] at :158). Stated in the
      snapshot, not inferred; verified by grep against 739e5945d.
"""

import numpy as np
import torch

torch.set_default_dtype(torch.float64)

# Q/utils/nvfp4_emulation_utils.py:18-21  kE2M1ToFloat_handle.val (magnitudes, code bits 0-2)
E2M1_MAG = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


def fp4_e2m1_value(code: int) -> float:
    """Nibble -> value. sign = bit 3, magnitude index = bits 0-2 (nvfp4_emulation_utils.py:33-38)."""
    sign = -1.0 if (code & 0x08) else 1.0  # :33 signs = combined & 0x08
    return E2M1_MAG[code & 0x07] * sign  # :34,38 abs_vals = combined & 0x07; values = kE2M1[abs_vals] * where(signs,-1,1)


def break_fp4_bytes(packed: torch.Tensor) -> torch.Tensor:
    """uint8 [m, n] -> float64 [m, 2n]; low nibble first (nvfp4_emulation_utils.py:23-40)."""
    assert packed.dtype == torch.uint8
    m, n = packed.shape
    flat = packed.flatten().to(torch.long)
    high = (flat & 0xF0) >> 4  # :28
    low = flat & 0x0F  # :29
    combined = torch.stack((low, high), dim=1).flatten()  # :31 low first (I1 for mxfp4)
    vals = torch.tensor([fp4_e2m1_value(int(c)) for c in combined], dtype=torch.float64)
    return vals.reshape(m, n * 2)  # :40


def e8m0_value(byte: int) -> float:
    """2^(byte-127) in float64 (mxfp6_utils.py:78, mxfp8_utils.py:118). Byte 255 -> see I3 (float64 does not overflow)."""
    return float(2.0 ** (byte - 127))


def mxfp4_dequant(packed: torch.Tensor, scales: torch.Tensor, block: int = 32) -> torch.Tensor:
    """packed uint8 [R, C/2], scales uint8 [R, C/32] -> float64 [R, C].

    value = e2m1(nibble) * 2^(scale-127), one scale per `block`=32 consecutive elements (ocp_mx_utils.py:9; I1, I2).
    """
    vals = break_fp4_bytes(packed)  # [R, C]
    R, C = vals.shape
    assert C % block == 0 and scales.shape == (R, C // block)
    descale = torch.tensor([[e8m0_value(int(b)) for b in row] for row in scales], dtype=torch.float64)
    # mxfp8_utils.py:114-122: x_blocked * descale.unsqueeze(-1)
    return (vals.reshape(R, C // block, block) * descale.unsqueeze(-1)).reshape(R, C)


def e4m3fn_value(byte: int) -> float:
    """float8_e4m3fn: S.EEEE.MMM, bias 7, no inf, NaN = S.1111.111, subnormal e==0: m/8 * 2^-6.

    Format definition (OCP FP8 spec / torch.float8_e4m3fn); vLLM uses the torch dtype directly
    (nvfp4_emulation_utils.py:68 `tensor_sf.view(torch.float8_e4m3fn)`). Cross-checked against torch in dequant_crosscheck.py.
    """
    s = -1.0 if byte & 0x80 else 1.0
    e = (byte >> 3) & 0xF
    m = byte & 0x7
    if e == 0xF and m == 0x7:
        return float("nan")
    if e == 0:
        return s * (m / 8.0) * 2.0 ** -6
    return s * (1.0 + m / 8.0) * 2.0 ** (e - 7)


def e5m2_value(byte: int) -> float:
    """float8_e5m2: S.EEEEE.MM, bias 15, inf = S.11111.00, NaN = S.11111.{01,10,11}, subnormal m/4 * 2^-14."""
    s = -1.0 if byte & 0x80 else 1.0
    e = (byte >> 2) & 0x1F
    m = byte & 0x3
    if e == 0x1F:
        return s * float("inf") if m == 0 else float("nan")
    if e == 0:
        return s * (m / 4.0) * 2.0 ** -14
    return s * (1.0 + m / 4.0) * 2.0 ** (e - 15)


def nvfp4_dequant(packed: torch.Tensor, scale_bytes: torch.Tensor, global_scale: float, block: int = 16) -> torch.Tensor:
    """packed uint8 [R, K/2], fp8-e4m3 scale bytes [R, K/16], multiplier -> float64 [R, K].

    nvfp4_emulation_utils.py:53-76 dequantize_to_dtype(swizzle=False):
      tensor_f32 = break_fp4_bytes(...).reshape(m, k//bs, bs)          :66-67
      tensor_sf  = tensor_sf.view(float8_e4m3fn)                        :68
      tensor_sf_dtype = tensor_sf.to(float32) * global_scale            :72
      out = (tensor_f32 * tensor_sf_dtype.unsqueeze(-1)).reshape(m,k)   :75
    """
    vals = break_fp4_bytes(packed)
    R, K = vals.shape
    assert scale_bytes.shape == (R, K // block)
    sf = torch.tensor([[e4m3fn_value(int(b)) for b in row] for row in scale_bytes], dtype=torch.float64) * global_scale
    return (vals.reshape(R, K // block, block) * sf.unsqueeze(-1)).reshape(R, K)


def fp8_channel_dequant(w_bytes: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Per-output-channel: scale [N, 1] (or [N]) -> w * scale. quant_utils.py:333-343 scaled_dequantize with group_broadcast
    (:207-219) expanding [N,1] over K. Compressed-tensors channel strategy stores weight_scale as [out, 1]
    (compressed_tensors_w8a8_fp8.py:124-131, fp8_utils.py:1064-1079 passes it through unchanged)."""
    N, K = w_bytes.shape
    wf = torch.tensor([[e4m3fn_value(int(b)) for b in row] for row in w_bytes], dtype=torch.float64)
    s = scale.reshape(N, 1).to(torch.float64)
    return wf * s


def fp8_block_dequant(w_bytes: torch.Tensor, scale: torch.Tensor, bs: int) -> torch.Tensor:
    """Block (bs x bs) scale [N/bs, K/bs] (`weight_scale_inv`): quant_utils.py:333-343 with group_shape=(bs, bs), the
    divisibility asserts at :336-339 (I6), and group_broadcast :207-219 (repeat each scale bs times on both dims)."""
    N, K = w_bytes.shape
    assert N % bs == 0 and K % bs == 0 and scale.shape == (N // bs, K // bs)
    wf = torch.tensor([[e4m3fn_value(int(b)) for b in row] for row in w_bytes], dtype=torch.float64)
    s = scale.to(torch.float64).repeat_interleave(bs, dim=0).repeat_interleave(bs, dim=1)
    return wf * s


def unpack_int4_words(words: torch.Tensor) -> torch.Tensor:
    """int32 [..., W] -> int [..., 8W]; nibble i = (word >> 4i) & 0xF  (quant_utils.py:426-444, packed dim last)."""
    w = words.to(torch.int64) & 0xFFFFFFFF
    out = torch.zeros(*words.shape[:-1], words.shape[-1] * 8, dtype=torch.int64)
    for i in range(8):
        out[..., i::8] = (w >> (4 * i)) & 0xF  # :441-442
    return out


def int4_pack_quantized_dequant(
    weight_packed: torch.Tensor, scale: torch.Tensor, group: int, zero_packed: torch.Tensor | None = None
) -> torch.Tensor:
    """weight_packed int32 [N, K/8]; scale float [N, K/group]; symmetric (zero_packed None): (nibble - 8) * s;
    asymmetric: (nibble - zp) * s with zp = unpack(zero_packed [N/8, K/group], packed_dim=0)  (I7).
    Returns float64 [N, K]."""
    q = unpack_int4_words(weight_packed)  # [N, K]
    N, K = q.shape
    assert scale.shape == (N, K // group)
    s = scale.to(torch.float64).repeat_interleave(group, dim=1)
    if zero_packed is None:
        return (q - 8).to(torch.float64) * s  # uint4b8: compressed_tensors_wNa16.py:35; quant_utils.py:826-848
    zp = unpack_int4_words(zero_packed.t().contiguous()).t().contiguous()  # packed along dim 0 (I7) -> [N, K/group]
    zp = zp.repeat_interleave(group, dim=1)
    return (q - zp).to(torch.float64) * s  # quant_utils.py:604


def awq_dequant(qweight: torch.Tensor, qzeros: torch.Tensor, scales: torch.Tensor, group: int) -> torch.Tensor:
    """AWQ int4 as produced by awq_pack (quant_utils.py:777-796): qweight int32 [K, N/8], qzeros int32 [K/g, N/8],
    scales [K/g, N]. Columns are stored in the interleave order [0,2,4,6,1,3,5,7] inside each 8-group; word nibble i is
    at bits [4i, 4i+4) (pack_cols, :711-734). Returns float64 [K, N] = (q - zero) * scale (zero per group per column)."""
    interleave = [0, 2, 4, 6, 1, 3, 5, 7]  # quant_utils.py:786

    def unpack(words):
        raw = unpack_int4_words(words)  # nibble j of word = bits [4j,4j+4), position j in the interleaved order
        K, Nn = raw.shape
        raw = raw.reshape(K, Nn // 8, 8)
        out = torch.zeros_like(raw)
        # awq_pack: packed[:, p] holds original column interleave[p] of the group  (q_w[:, interleave])
        for p, c in enumerate(interleave):
            out[:, :, c] = raw[:, :, p]
        return out.reshape(K, Nn)

    q = unpack(qweight)
    z = unpack(qzeros)
    s = scales.to(torch.float64).repeat_interleave(group, dim=0)
    zz = z.repeat_interleave(group, dim=0)
    return (q - zz).to(torch.float64) * s
