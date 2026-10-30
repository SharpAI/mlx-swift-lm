// SL-31 K1: gated-delta recurrence with a PER-KEY-DIMENSION gate (KDA, Kimi-Linear / Kimi K3; probably
// MuseSpark GDN2 -- INFERRED, see the SL-31 kernel plan). New file, SharpAI-only; never edits upstream
// GatedDelta.swift. The structure of the Metal source is copied from `gatedDeltaKernel` there; the only
// maths difference is the decay: `state[dk] *= g[b, t, hv, dk]` instead of one scalar `g[b, t, hv]`.
//
// upstream-watch: Libraries/MLXLMCommon/GatedDelta.swift. If upstream gains a per-key-gate path, compare
// against this file (oracle: Tests/.../GapReference/GdnReference.swift) and retire it.

import Foundation
import MLX

public enum SharpAIOverlayGatedDeltaKDA {

    private static func makeKernel(hasMask: Bool) -> MLXFast.MLXFastKernel {
        let maskSource = hasMask ? "mask[b_idx * T + t]" : "true"
        let source = """
            auto n = thread_position_in_grid.z;
            auto b_idx = n / Hv;
            auto hv_idx = n % Hv;
            auto hk_idx = hv_idx / (Hv / Hk);
            constexpr int n_per_t = Dk / 32;

            // q, k: [B, T, Hk, Dk]
            auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk;
            auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;

            // v, y: [B, T, Hv, Dv]
            auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;
            y += b_idx * T * Hv * Dv + hv_idx * Dv;

            auto dk_idx = thread_position_in_threadgroup.x;
            auto dv_idx = thread_position_in_grid.y;

            // g: [B, T, Hv, Dk] per-key decay multiplier; beta: [B, T, Hv]
            auto g_ = g + (b_idx * T * Hv + hv_idx) * Dk;
            auto beta_ = beta + b_idx * T * Hv;

            // state_in, state_out: [B, Hv, Dv, Dk]
            auto i_state = state_in + (n * Dv + dv_idx) * Dk;
            auto o_state = state_out + (n * Dv + dv_idx) * Dk;

            float state[n_per_t];
            for (int i = 0; i < n_per_t; ++i) {
              auto s_idx = n_per_t * dk_idx + i;
              state[i] = static_cast<float>(i_state[s_idx]);
            }

            for (int t = 0; t < T; ++t) {
              if (\(maskSource)) {
                float kv_mem = 0.0f;
                {
                  // Preserve Kahan summation under Metal's default fast math.
                  #pragma clang fp reassociate(off)
                  #pragma clang fp contract(off)
                  float kv_compensation = 0.0f;
                  for (int i = 0; i < n_per_t; ++i) {
                    auto s_idx = n_per_t * dk_idx + i;
                    state[i] = state[i] * g_[s_idx];
                    auto product = state[i] * k_[s_idx];
                    auto corrected = product - kv_compensation;
                    auto next_sum = kv_mem + corrected;
                    kv_compensation = (next_sum - kv_mem) - corrected;
                    kv_mem = next_sum;
                  }
                }
                kv_mem = simd_sum(kv_mem);

                auto delta = (v_[dv_idx] - kv_mem) * beta_[hv_idx];

                float out = 0.0f;
                for (int i = 0; i < n_per_t; ++i) {
                  auto s_idx = n_per_t * dk_idx + i;
                  state[i] = state[i] + k_[s_idx] * delta;
                  out += state[i] * q_[s_idx];
                }
                out = simd_sum(out);
                if (thread_index_in_simdgroup == 0) {
                  y[dv_idx] = static_cast<InT>(out);
                }
              } else {
                y[dv_idx] = static_cast<InT>(0);
              }
              q_ += Hk * Dk;
              k_ += Hk * Dk;
              v_ += Hv * Dv;
              y += Hv * Dv;
              g_ += Hv * Dk;
              beta_ += Hv;
            }
            for (int i = 0; i < n_per_t; ++i) {
              auto s_idx = n_per_t * dk_idx + i;
              o_state[s_idx] = static_cast<StT>(state[i]);
            }
        """
        var inputNames = ["q", "k", "v", "g", "beta", "state_in", "T"]
        if hasMask { inputNames.append("mask") }
        return MLXFast.metalKernel(
            name: "sharpai_gated_delta_kda_step\(hasMask ? "_mask" : "")",
            inputNames: inputNames,
            outputNames: ["y", "state_out"],
            source: source
        )
    }

    private final class Kernels: Sendable {
        static let shared = Kernels()
        let plain = SharpAIOverlayGatedDeltaKDA.makeKernel(hasMask: false)
        let masked = SharpAIOverlayGatedDeltaKDA.makeKernel(hasMask: true)
    }

    /// Gated delta rule with a per-key-dimension decay.
    ///
    /// - q: `[B, T, Hk, Dk]` (already L2-normed and scaled), k: `[B, T, Hk, Dk]` (L2-normed),
    ///   v: `[B, T, Hv, Dv]`, g: `[B, T, Hv, Dk]` decay MULTIPLIER (`exp(logDecay)`, rank 4 required),
    ///   beta: `[B, T, Hv]`, state: `[B, Hv, Dv, Dk]` (promoted to float32; nil means zeros),
    ///   mask: `[B, T]` bool (masked step: state unchanged, y = 0).
    /// - Returns: y `[B, T, Hv, Dv]` in q's dtype and the float32 state.
    ///
    /// Falls back to the op composition (`gatedDeltaOps`, which accepts a rank-4 gate) when the kernel
    /// cannot run: CPU stream, `useKernel == false` (training; the kernel has no gradient),
    /// `Dk % 32 != 0` (the kernel splits Dk over 32 lanes and would truncate), or `T == 0`.
    public static func update(
        q: MLXArray, k: MLXArray, v: MLXArray, g: MLXArray, beta: MLXArray,
        state: MLXArray? = nil, mask: MLXArray? = nil, useKernel: Bool = true
    ) -> (y: MLXArray, state: MLXArray) {
        precondition(g.ndim == 4, "KDA gate must be [B, T, Hv, Dk]; use gatedDeltaUpdate for a scalar gate")
        let (B, T, Hk, Dk) = (k.dim(0), k.dim(1), k.dim(2), k.dim(3))
        let (Hv, Dv) = (v.dim(2), v.dim(3))
        precondition(Hk > 0 && Hv % Hk == 0, "Hv must be a multiple of Hk")

        var state = state ?? MLXArray.zeros([B, Hv, Dv, Dk], dtype: .float32)
        if state.dtype != .float32 { state = state.asType(.float32) }
        if T == 0 { return (MLXArray.zeros([B, 0, Hv, Dv], dtype: q.dtype), state) }

        let g = g.asType(.float32), beta = beta.asType(.float32)
        let isCPU = Device.defaultDevice().deviceType == .cpu
        guard !isCPU, useKernel, Dk % 32 == 0 else {
            return gatedDeltaOps(q: q, k: k, v: v, g: g, beta: beta, state: state, mask: mask)
        }

        var inputs: [MLXArray] = [q, k, v, g, beta, state, MLXArray(T)]
        let kernel: MLXFast.MLXFastKernel
        if let mask {
            kernel = Kernels.shared.masked
            inputs.append(mask)
        } else {
            kernel = Kernels.shared.plain
        }
        let out = kernel(
            inputs,
            template: [
                ("InT", q.dtype), ("StT", DType.float32), ("Dk", Dk), ("Dv", Dv), ("Hk", Hk), ("Hv", Hv),
            ],
            grid: (32, Dv, B * Hv),
            threadGroup: (32, 4, 1),
            outputShapes: [[B, T, Hv, Dv], state.shape],
            outputDTypes: [q.dtype, .float32]
        )
        return (out[0], out[1])
    }
}
