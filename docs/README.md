# WaifuFinder documentation

Start with the [app overview and installation guide](../README.md). The guides below
describe current workflows. Dated reports and experiment notes preserve the
configuration and results of a specific investigation.

## App workflows

| Task | Guide |
|---|---|
| Browse media and work with Photos albums | [Photos workspace](../README.md#photos-workspace), [preview loading](Photos-preview.md) |
| Find the best version of one photograph | [PhotoArena](PhotoArena.md), [selection, reconstruction and limitations](PhotoArena-2.0.md) |
| Inspect photographic alterations | [TruFor](TruFor.md), [SAFIRE](ImageLineage.md) |
| Find visually similar frames or inspect candidates | [Vector Search](WaifuVectorSearch.md), [Crystal Orb](WaifuCrystalOrb.md) |
| Filter and organize tags | [Tag groups](Danbooru-tag-groups.md), [tag summaries](Tag-summary.md) |
| Restore or upscale images and video | [Uncover](Uncover.md), [FlashVSR](FlashVSR.md), [RealESRGAN preview](../README.md#realesrgan-photo-preview) |
| Use another Mac for inference | [Remote inference](Remote-inference.md) |

## Development and distribution

| Task | Reference |
|---|---|
| Set up a clean Mac and build | [Development setup](Development-setup.md), [build commands](../README.md#build-and-run), [native workers](../Runtime/README.md) |
| Understand engine and UI boundaries | [Layer API](API.md) |
| Run regression checks | [Test harness](../Tests/README.md), [repository instructions](../AGENTS.md) |
| Find pinned models and runtime packages | [Model inventory](../models/README.md) |
| Publish a release or update the showcase | [Release procedure](RELEASING.md), [DMG readme](DMG-Read-Me.html) |
| Check dependency and distribution terms | [Licensing](Licensing.md), [dependency audit](Dependency-audit.md), [credits](Credits.md) |

## Engineering evidence

Hardware results apply to the configurations and fixtures in each report. Keep the
associated JSON and examples beside their reports so old links remain usable.

- **PhotoArena:** [v1 architecture](PhotoArena-architecture.md),
  [deterministic forensics](PhotoArena-forensics.md), [v1 delivery](PhotoArena-delivery.md),
  [native worker](PhotoArena-model-runtime.md), [paired inference optimization](PhotoArena-optimization.md),
  and [multi-source model results](photoarena-group-review.json).
- **Native model comparisons:** [dated WD Tagger, TruFor and SAFIRE snapshot](Native-model-status.md).
- **WD Tagger:** [model card](WDTagger-native-model-card.md), [batch measurements](WDTagger-optimization.md),
  [native optimization](WDTagger-native-optimization.md), [kernel review](WDTagger-native-kernel-review.md),
  [speed tuning](WDTagger-native-tuning.md), [memory and input](WDTagger-native-memory.md),
  [pipeline overlap](WDTagger-native-pipeline.md), [M3 input overlap](WDTagger-M3-input-overlap.md),
  [M4 confirmation](WDTagger-native-m4.md), and [ANE/Core ML experiments](WDTagger-ANE.md).
- **TruFor:** [M4 experiment notes](TruFor-M4-experiment-notes.md),
  [direct ANE experiment](TruFor-direct-ANE.md), and [October implementation review](TruFor-review-2026-10-25.md).
- **Restoration and upscaling:** [PGTFormer native audit](Face-restoration-native-audit.md),
  [RealESRGAN x4 integration](RealESRGAN-x4-integration.md), [RealESRGAN ANE experiment](RealESRGAN-ANE.md),
  [FlashVSR model card](FlashVSR-model-card.md), and [SeedVR2 model card](SeedVR2-model-card.md).
- **Photos, tags and grouping:** [loading profile](Photos-loading-profile-2026-11-01.md),
  [memory profile](Photos-memory-profile-2026-11-01.md), [tag result memory](TagResultMemory.md),
  [inverse tag filtering](inverse-tag-performance.md), [visual-distance ordering](MVD-optimization-2026-11-01.md),
  [Metal routing](CuOptMetal.md), and [profiling summary](Profiling-findings.md).
- **Historical records:** [development archive](development-archive/), [private history archive](Private-history-archive.md),
  and [earlier repository review](Review-dc734db.md).
