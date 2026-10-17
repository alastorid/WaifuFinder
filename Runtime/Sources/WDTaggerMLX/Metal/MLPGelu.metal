
      constexpr int BM=TBM, BN=TBN, BK=TBK, WM=TWM, WN=TWN;
      using Kernel = GEMMKernel<half,half,BM,BN,BK,WM,WN,false,true,true,true,float,WaifuBiasActivation<ACT>,RELAXED>;
      const uint lane=thread_index_in_simdgroup;
      const uint simd=simdgroup_index_in_threadgroup;
      const uint3 tile=threadgroup_position_in_grid;
      const int row=tile.y*BM, col=tile.x*BN;
      const short rows=(FAST_FULL_TILES && M_ALIGNED) ? BM : metal::min(BM,int(x_shape[0])-row);
      if (rows<=0) return;
      threadgroup half As[Kernel::tgp_mem_size_a];
      threadgroup half Bs[Kernel::tgp_mem_size_b];
      typename Kernel::loader_a_t a(x+row*K,K,As,simd,lane);
      typename Kernel::loader_b_t b(weight+col*K,K,Bs,simd,lane);
      typename Kernel::mma_t mma(simd,lane);
      const short cols=BN, leftover=0;
      if constexpr (FAST_FULL_TILES && M_ALIGNED) {
        Kernel::template gemm_loop<true,true,true>(As,Bs,K/BK,a,b,mma,rows,cols,leftover);
        threadgroup_barrier(mem_flags::mem_none);
        mma.store_result(out+row*N+col,N,bias+col,0,1,WaifuBiasActivation<ACT>{});
      } else if (FAST_FULL_TILES && rows == BM) {
        Kernel::template gemm_loop<true,true,true>(As,Bs,K/BK,a,b,mma,rows,cols,leftover);
        threadgroup_barrier(mem_flags::mem_none);
        mma.store_result(out+row*N+col,N,bias+col,0,1,WaifuBiasActivation<ACT>{});
      } else {
        Kernel::template gemm_loop<false,true,true>(As,Bs,K/BK,a,b,mma,rows,cols,leftover);
        threadgroup_barrier(mem_flags::mem_none);
        mma.store_result_safe(out+row*N+col,N,bias+col,0,1,short2(cols,rows),WaifuBiasActivation<ACT>{});
      }
