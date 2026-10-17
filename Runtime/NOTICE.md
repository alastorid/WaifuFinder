The SeedVR2MLX module derives from xocialize/seedvr2-mlx-swift, snapshot recorded in
UPSTREAM_REVISION. Its source files carry MIT SPDX notices; the upstream README
identifies the port as MIT. The original model and weights are Apache-2.0, ByteDance Seed.
MLX and MLX Swift are MIT, Apple / ml-explore. Their license files remain in the dependency
checkout; the build copies them into the app's third-party notices.

WaifuFinder adds multi-frame latents, causal VAE caches and slicing, correct first-frame
upsampling, shared attention modules, loading of existing 4-bit prepared weights,
explicit evaluation boundaries, native PNG IO, temporal context planning, spatial tiling,
shared global noise, a physical-memory monitor and the native worker/app integration.

Reference implementations:
https://github.com/xocialize/seedvr2-mlx-swift
https://github.com/lpalbou/mlx-gen
https://github.com/ByteDance-Seed/SeedVR
https://github.com/ml-explore/mlx-swift
