// Fixed 4x4 stride-4 RGB patch embedding preparation. Window order is produced
// directly, so neither a general convolution im2col nor a separate spatial
// permutation is needed. Input pixels are the exact retained native RGB values.
const uint i = thread_position_in_grid.x;
const uint channel = i % 3;
const uint pixel = (i / 3) % 16;
const uint row = i / 48;
const uint batch = row / (112 * 112);
const uint window = (row % (112 * 112)) / 196;
const uint token = row % 196;
const uint h = ((window / 8) * 14 + token / 14) * 4 + pixel / 4;
const uint w = ((window % 8) * 14 + token % 14) * 4 + pixel % 4;
const uint source = RGBA ? ((batch * 448 + h) * 448 + w) * 4 + channel
                         : (batch * 3 + channel) * 448 * 448 + h * 448 + w;
out[i] = half(float(image[source]) * (2.0f / 255.0f) - 1.0f);
