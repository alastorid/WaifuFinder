// Each pair of N SIMD groups owns complete 32-channel heads. Generalize
// the register epilogue with the MMA geometry; never stage [M, 3C].
constexpr int C = H * 32, BM = TBM, BN = TBN, BK = TBK;
constexpr int WM = TWM, ROW_STEP = 8 * WM;
constexpr int ROW_FRAGS = BM / ROW_STEP, HEADS_PER_TILE = BN / 32;
constexpr int COL_FRAGS = BN / 16;
static_assert(BM % (8 * WM) == 0 && C % BK == 0, "Invalid row or K geometry");
static_assert(BN % 32 == 0 && C % BN == 0, "Tiles must contain complete heads and Q/K/V segments");
using Kernel = GEMMKernel<half, half, BM, BN, BK, WM, 2, false, true, true, true,
                          float, WaifuBiasActivation<false>, RELAXED>;
const uint lane = thread_index_in_simdgroup;
const uint simd = simdgroup_index_in_threadgroup;
const uint3 tile = threadgroup_position_in_grid;
const int row = tile.y * BM, col = tile.x * BN;
const short rows = M_ALIGNED ? BM : metal::min(BM, int(x_shape[0]) - row);
threadgroup half As[Kernel::tgp_mem_size_a];
threadgroup half Bs[Kernel::tgp_mem_size_b];
threadgroup float partial[BM * HEADS_PER_TILE * 2];
typename Kernel::loader_a_t a(x + row * C, C, As, simd, lane);
typename Kernel::loader_b_t b(weight + col * C, C, Bs, simd, lane);
typename Kernel::mma_t mma(simd, lane);
const short cols = BN, leftover = 0;
if (rows == BM) {
  Kernel::template gemm_loop<true, true, true>(As, Bs, C / BK, a, b, mma, rows, cols, leftover);
} else {
  Kernel::template gemm_loop<false, true, true>(As, Bs, C / BK, a, b, mma, rows, cols, leftover);
}
const int segment = col / C, firstHead = (col % C) / 32;
half values[ROW_FRAGS * COL_FRAGS * 2];
#pragma unroll
for (short i = 0; i < ROW_FRAGS; ++i) {
  #pragma unroll
  for (short j = 0; j < COL_FRAGS; ++j) {
    #pragma unroll
    for (short e = 0; e < 2; ++e) {
      values[(i * COL_FRAGS + j) * 2 + e] = half(mma.Ctile.frag_at(i, j)[e] + float(bias[col + j * 16 + mma.sn + e]));
    }
  }
}
if (segment != 2) {
  #pragma unroll
  for (short i = 0; i < ROW_FRAGS; ++i) {
    #pragma unroll
    for (short h = 0; h < HEADS_PER_TILE; ++h) {
      float square = 0.0f;
      #pragma unroll
      for (short j = 0; j < 2; ++j) {
        #pragma unroll
        for (short e = 0; e < 2; ++e) {
          const float value = float(values[(i * COL_FRAGS + h * 2 + j) * 2 + e]);
          square += value * value;
        }
      }
      square += simd_shuffle_xor(square, 1);
      square += simd_shuffle_xor(square, 8);
      if (mma.sn % 8 == 0) partial[(mma.sm + i * ROW_STEP) * (HEADS_PER_TILE * 2) + h * 2 + simd % 2] = square;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
}
#pragma unroll
for (short i = 0; i < ROW_FRAGS; ++i) {
  const int tokenRow = row + mma.sm + i * ROW_STEP;
  if (M_ALIGNED || tokenRow < int(x_shape[0])) {
    #pragma unroll
    for (short h = 0; h < HEADS_PER_TILE; ++h) {
      const int head = firstHead + h;
      float norm = 1.0f;
      if (segment != 2) {
        const int index = (mma.sm + i * ROW_STEP) * (HEADS_PER_TILE * 2) + h * 2;
        norm = metal::max(metal::sqrt(partial[index] + partial[index + 1]), 1e-12f);
      }
      const int base = (((tokenRow / 196) * H + head) * 196 + tokenRow % 196) * 32;
      #pragma unroll
      for (short j = 0; j < 2; ++j) {
        #pragma unroll
        for (short e = 0; e < 2; ++e) {
          const int index = base + j * 16 + mma.sn + e;
          const float value = float(values[(i * COL_FRAGS + h * 2 + j) * 2 + e]);
          if (segment == 0) Q[index] = half(half(value / norm) * scale[head]);
          else if (segment == 1) K[index] = half(value / norm);
          else V[index] = half(value);
        }
      }
    }
  }
}
