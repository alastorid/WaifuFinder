
      const uint lane = thread_index_in_simdgroup;
      const uint row = thread_position_in_grid.x / 32;
      const uint head = row % H;
      const uint token = (row / H) % 196;
      const uint window = row / (H * 196);
      const uint input = (window * 196 + token) * (3 * H * 32) + head * 32 + lane;
      const float q = float(packed[input]);
      const float k = float(packed[input + H * 32]);
      const float qnorm = metal::max(metal::sqrt(simd_sum(q*q)), 1e-12f);
      const float knorm = metal::max(metal::sqrt(simd_sum(k*k)), 1e-12f);
      const uint output = ((window * H + head) * 196 + token) * 32 + lane;
      // Explicit intermediate rounding matches the preceding MLX probe.
      Q[output] = half(half(q/qnorm) * scale[head]);
      K[output] = half(k/knorm);
      V[output] = packed[input + 2 * H * 32];
