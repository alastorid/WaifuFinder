
      const uint lane = thread_index_in_simdgroup;
      const uint simd = simdgroup_index_in_threadgroup;
      const uint tid = thread_position_in_threadgroup.x;
      const uint row = threadgroup_position_in_grid.x;
      const uint batch = row / S;
      const uint src = IDENTITY_ROUTE ? row : batch * S + route[row % S];
      threadgroup float reductions[8];
      // At most four values per thread for the model's 128..1024 channels.
      // Retain FP32 inputs across both statistics passes and the epilogue.
      // Preserve the original reduction and FP16 affine/residual boundaries.
      constexpr uint VALUES = (C + NT - 1) / NT;
      float values[VALUES];
      float total = 0.0f;
      #pragma unroll
      for (uint i=0; i<VALUES; ++i) {
        const uint c=tid+i*NT;
        values[i] = c<C ? float(x[src*C+c]) : 0.0f;
        total += values[i];
      }
      total = simd_sum(total);
      if (lane==0) reductions[simd]=total;
      threadgroup_barrier(mem_flags::mem_threadgroup);
      float mean=0.0f;
      for (uint i=0;i<NT/32;++i) mean += reductions[i];
      mean /= float(C);
      float square=0.0f;
      #pragma unroll
      for (uint i=0; i<VALUES; ++i) {
        if (tid+i*NT<C) { float d=values[i]-mean; square += d*d; }
      }
      square=simd_sum(square);
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (lane==0) reductions[simd]=square;
      threadgroup_barrier(mem_flags::mem_threadgroup);
      float variance=0.0f;
      for (uint i=0;i<NT/32;++i) variance += reductions[i];
      const float inv=metal::rsqrt(variance/float(C)+1e-5f);
      #pragma unroll
      for (uint i=0; i<VALUES; ++i) {
        const uint c=tid+i*NT;
        if (c<C) {
          const half normalized=half((values[i]-mean)*inv*float(weight[c])+float(bias[c]));
          out[row*C+c] = ADD_RESIDUAL ? half(normalized+residual[src*C+c]) : normalized;
        }
      }
