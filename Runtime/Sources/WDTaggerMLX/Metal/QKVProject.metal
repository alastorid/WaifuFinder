// Preserve the tuned 32x64 projection tile. Normalize two heads in the epilogue;
// only 128 FP32 partial sums cross SIMD groups, never the projected tensor.
constexpr int C = H * 32, BM = OCCUPANCY ? 64 : 32, BN = 64, BK = OCCUPANCY ? 16 : 32;
constexpr int WM = OCCUPANCY ? 4 : 2, ROW_STEP = 8 * WM;
using Kernel = GEMMKernel<half, half, BM, BN, BK, WM, 2, false, true, true, true,
                          float, WaifuBiasActivation<false>>;
const uint lane = thread_index_in_simdgroup;
const uint simd = simdgroup_index_in_threadgroup;
const uint3 tile = threadgroup_position_in_grid;
const int row = tile.y * BM, col = tile.x * BN;
const short rows = metal::min(BM, int(x_shape[0]) - row);
threadgroup half As[Kernel::tgp_mem_size_a];
threadgroup half Bs[Kernel::tgp_mem_size_b];
threadgroup float partial[BM * 4];
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
half values[16];
#pragma unroll
for (short i = 0; i < 2; ++i) {
  #pragma unroll
  for (short j = 0; j < 4; ++j) {
    #pragma unroll
    for (short e = 0; e < 2; ++e) {
      values[(i * 4 + j) * 2 + e] = half(mma.Ctile.frag_at(i, j)[e] + float(bias[col + j * 16 + mma.sn + e]));
    }
  }
}
if (segment != 2) {
  #pragma unroll
  for (short i = 0; i < 2; ++i) {
    #pragma unroll
    for (short h = 0; h < 2; ++h) {
      float square = 0.0f;
      #pragma unroll
      for (short j = 0; j < 2; ++j) {
        #pragma unroll
        for (short e = 0; e < 2; ++e) {
          const float value = float(values[(i * 4 + h * 2 + j) * 2 + e]);
          square += value * value;
        }
      }
      square += simd_shuffle_xor(square, 1);
      square += simd_shuffle_xor(square, 8);
      if (mma.sn % 8 == 0) partial[(mma.sm + i * ROW_STEP) * 4 + h * 2 + simd % 2] = square;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
}
#pragma unroll
for (short i = 0; i < 2; ++i) {
  const int tokenRow = row + mma.sm + i * ROW_STEP;
  if (tokenRow < int(x_shape[0])) {
    #pragma unroll
    for (short h = 0; h < 2; ++h) {
      const int head = firstHead + h;
      float norm = 1.0f;
      if (segment != 2) {
        const int index = (mma.sm + i * ROW_STEP) * 4 + h * 2;
        norm = metal::max(metal::sqrt(partial[index] + partial[index + 1]), 1e-12f);
      }
      const int base = (((tokenRow / 196) * H + head) * 196 + tokenRow % 196) * 32;
      #pragma unroll
      for (short j = 0; j < 2; ++j) {
        #pragma unroll
        for (short e = 0; e < 2; ++e) {
          const int index = base + j * 16 + mma.sn + e;
          const float value = float(values[(i * 4 + h * 2 + j) * 2 + e]);
          if (segment == 0) Q[index] = half(half(value / norm) * scale[head]);
          else if (segment == 1) K[index] = half(value / norm);
          else V[index] = half(value);
        }
      }
    }
  }
}
