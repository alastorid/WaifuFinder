#include <metal_stdlib>
using namespace metal;
kernel void smoke(device float *values [[buffer(0)]], uint index [[thread_position_in_grid]]) {
    values[index] *= 2.0f;
}
