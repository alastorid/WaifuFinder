# FlashVSR 1.1 Tiny Streaming

The native FlashVSR graph and attention semantics derive from OpenImagingLab/FlashVSR,
Apache-2.0, revision cf910c61a60733e610e9c6e8b607f80c3a6c202b.
https://github.com/OpenImagingLab/FlashVSR

Official model source: JunhaoZhuang/FlashVSR-v1.1, Apache-2.0,
revision 27561b186ded3402d7c975f4fd722e2885b6135f.
https://huggingface.co/JunhaoZhuang/FlashVSR-v1.1

WaifuFinder's modifications implement Swift/MLX execution on Apple Metal, complete checkpoint
shape/coverage verification, BF16 layout conversion, causal projection memories,
streaming KV retention, geometry-dependent LCSA eligibility and global block threshold,
a GPU sparse online-softmax kernel, recurrent Tiny Conditional Decoder, bounded memory,
native PNG IO, visible-frame trimming, and the shared Uncover app integration.

The CUDA attention dispatch is replaced with equivalent sparse Metal execution, not
unrestricted attention. Offline validation uses the official graph and equivalent
masked PyTorch SDPA on CPU. Neither Python nor PyTorch is a production runtime dependency.
FLASHVSR-LICENSE contains the upstream Apache-2.0 license.
