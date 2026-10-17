// One SIMD group owns a row. Keep input values in registers across both stable
// reduction passes; independent rows share a threadgroup without shared memory.
const uint lane = thread_index_in_simdgroup;
const uint row = thread_position_in_grid.x / 32;
if (row >= uint(x_shape[0])) return;
const uint src = (row / S) * S + route[row % S];
float values[C / 32];
float total = 0.0f;
#pragma unroll
for (uint i = 0; i < C / 32; ++i) {
  values[i] = float(x[src * C + i * 32 + lane]);
  total += values[i];
}
const float mean = simd_sum(total) / float(C);
float square = 0.0f;
#pragma unroll
for (uint i = 0; i < C / 32; ++i) {
  const float d = values[i] - mean;
  square += d * d;
}
const float inv = metal::rsqrt(simd_sum(square) / float(C) + 1e-5f);
#pragma unroll
for (uint i = 0; i < C / 32; ++i) {
  const uint c = i * 32 + lane;
  const half normalized = half((values[i] - mean) * inv * float(weight[c]) + float(bias[c]));
  out[row * C + c] = ADD_RESIDUAL ? half(normalized + residual[src * C + c]) : normalized;
}
