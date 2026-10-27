# WaifuFinder app notices

This folder is included in `WaifuFinder.app/Contents/Resources/ThirdPartyLicenses/`.
It contains the license texts and attribution notices for the app and native workers.

- `cuOpt-LICENSE` and `cuOpt-NOTICE` cover the Apache-2.0 Swift/Metal port of
  cuOpt's routing 2-opt search, specialized for an open photo-similarity path.

- `WaifuFinder-LICENSE` and `WaifuFinder-Licensing.md` cover WaifuFinder's source and
  explain the separate terms that apply to model downloads.
- `Runtime-NOTICE.md`, `FlashVSR-NOTICE.md`, and `RealESRGAN-NOTICE.md` identify the
  relevant upstream projects and describe our changes.
- `Apache-2.0-LICENSE` applies to the SeedVR2 positional embedding included with the app;
  it also accompanies the relevant downloaded model packages.
- `MLX-Swift-LICENSE`, `MLX-LICENSE`, `MLX-C-LICENSE`, `Nlohmann-JSON-LICENSE`,
  `fmt-LICENSE`, `Metal-CPP-LICENSE`, and `Swift-Numerics-LICENSE` are the full license
  texts for the native runtime dependencies.
- `INFERKIT-LICENSE`, `FlashVSR-LICENSE`, and `RealESRGAN-LICENSE` are the applicable
  component license texts. `THIRD_PARTY_LICENSES.md` records the InferKit code attribution.

Real-ESRGAN x2plus and x4plus pretrained models are the work of Xintao Wang
and collaborators, with BasicSR contributors supplying the reference architecture.
See `RealESRGAN-NOTICE.md` for the paper, provenance and our conversion changes;
BSD-3-Clause and Apache-2.0 notices accompany the model packages.

Full model packages are downloaded separately and include their own licenses and notices.
See `WaifuFinder-Licensing.md` for the current inventory and the unresolved PGTFormer
weights license question.
