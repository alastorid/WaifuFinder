// Project one 32-channel head per column tile. Each SIMD group owns eight rows
// and all 32 head lanes, so cosine normalization stays in registers.
constexpr int C = H * 32, BM = 32, BN = 32, BK = 32;
using Kernel = GEMMKernel<half, half, BM, BN, BK, 4, 1, false, true, true, true,
                          float, WaifuBiasActivation<false>>;
const uint lane = thread_index_in_simdgroup;
const uint simd = simdgroup_index_in_threadgroup;
const uint3 tile = threadgroup_position_in_grid;
const int row = tile.y * BM, col = tile.x * BN;
const short rows = metal::min(BM, int(x_shape[0]) - row);
threadgroup half As[Kernel::tgp_mem_size_a];
threadgroup half Bs[Kernel::tgp_mem_size_b];
typename Kernel::loader_a_t a(x + row * C, C, As, simd, lane);
typename Kernel::loader_b_t b(weight + col * C, C, Bs, simd, lane);
typename Kernel::mma_t mma(simd, lane);
const short cols = BN, leftover = 0;
if (rows == BM) {
  Kernel::template gemm_loop<true, true, true>(As, Bs, C / BK, a, b, mma, rows, cols, leftover);
} else {
  Kernel::template gemm_loop<false, true, true>(As, Bs, C / BK, a, b, mma, rows, cols, leftover);
}
const int tokenRow = row + mma.sm;
if (tokenRow >= int(x_shape[0])) return;
const int segment = col / C, head = (col % C) / 32;
half values[8];
float square = 0.0f;
#pragma unroll
for (short j = 0; j < 4; ++j) {
  #pragma unroll
  for (short e = 0; e < 2; ++e) {
    const half value = half(mma.Ctile.frag_at(0, j)[e] + float(bias[col + j * 8 + mma.sn + e]));
    values[j * 2 + e] = value;
    if (segment != 2) square += float(value) * float(value);
  }
}
// Matrix fragment rows occupy lane quads {lane, lane^1, lane^8, lane^9}.
float norm = 1.0f;
if (segment != 2) {
  square += simd_shuffle_xor(square, 1);
  square += simd_shuffle_xor(square, 8);
  norm = metal::max(metal::sqrt(square), 1e-12f);
}
const int base = (((tokenRow / 196) * H + head) * 196 + tokenRow % 196) * 32;
#pragma unroll
for (short j = 0; j < 4; ++j) {
  #pragma unroll
  for (short e = 0; e < 2; ++e) {
    const int index = base + j * 8 + mma.sn + e;
    const float value = float(values[j * 2 + e]);
    if (segment == 0) Q[index] = half(half(value / norm) * scale[head]);
    else if (segment == 1) K[index] = half(value / norm);
    else V[index] = half(value);
  }
}
