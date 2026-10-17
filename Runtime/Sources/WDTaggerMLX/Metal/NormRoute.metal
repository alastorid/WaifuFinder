
      const uint lane = thread_index_in_simdgroup;
      const uint simd = simdgroup_index_in_threadgroup;
      const uint tid = thread_position_in_threadgroup.x;
      const uint row = threadgroup_position_in_grid.x;
      const uint batch = row / S;
      const uint src = batch * S + route[row % S];
      threadgroup float reductions[8];
      float total = 0.0f;
      for (uint c=tid; c<C; c+=NT) total += float(x[src*C+c]);
      total = simd_sum(total);
      if (lane==0) reductions[simd]=total;
      threadgroup_barrier(mem_flags::mem_threadgroup);
      float mean=0.0f;
      for (uint i=0;i<NT/32;++i) mean += reductions[i];
      mean /= float(C);
      float square=0.0f;
      for (uint c=tid;c<C;c+=NT) { float d=float(x[src*C+c])-mean; square += d*d; }
      square=simd_sum(square);
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (lane==0) reductions[simd]=square;
      threadgroup_barrier(mem_flags::mem_threadgroup);
      float variance=0.0f;
      for (uint i=0;i<NT/32;++i) variance += reductions[i];
      const float inv=metal::rsqrt(variance/float(C)+1e-5f);
      for (uint c=tid;c<C;c+=NT) {
        const half normalized=half((float(x[src*C+c])-mean)*inv*float(weight[c])+float(bias[c]));
        out[row*C+c] = ADD_RESIDUAL ? half(normalized+residual[src*C+c]) : normalized;
      }
