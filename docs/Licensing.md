# Licensing

This is a factual record of what licence governs what in WaifuFinder, what each one
obliges us to do, and the one place where they do not combine. It is written to be
checked, not to be reassuring.

**This is not legal advice.** The summaries below are read off the licence texts and the
upstream model cards. Whether a particular use is commercial, and whether a distribution
is lawful in your jurisdiction, are questions for a lawyer. The two open questions in
[What needs a human](#what-needs-a-human) are exactly the ones a build flag cannot answer.

## The incompatibility, stated plainly

WaifuFinder's own code is **MIT**, which permits commercial use, sublicensing and resale.
Almost everything it drives is permissively licensed too. Two models are not:

> **PGTFormer may only be redistributed and used for non-commercial purposes.**
> Commercial use requires permission from Xidian University or the contributors.
>
> **TruFor may be used, reproduced and modified only for informational and nonprofit
> purposes**, must retain all original notices, and must credit the original authors
> whenever results made with it are made public.

MIT and "non-commercial only" cannot be reconciled. Attribution does not help, and neither
does the fact that the files are separable — what the licence restricts is the *use*, not
the packaging.

This matters more than a normal dependency conflict, because **the app fetches the model on
the user's behalf**. There is no bundle to omit: the user taps Download, and an MIT-licensed
app hands them a non-commercial model. If that user is commercial, they are infringing, and
an app that ships that button is arguably contributing. Declining at download time would not
fix it — the offer is the problem.

So the restriction is enforced at build time instead. See
[Distribution modes](#distribution-modes).

## What governs what

| Component | Licence | Commercial use | Source |
|---|---|---|---|
| WaifuFinder application code | MIT | Yes | [`LICENSE`](../LICENSE) |
| cuOpt routing search adapted to Swift/Metal | Apache-2.0 | Yes | [Pinned source audit](CuOptMetal.md), [license and notice](../ThirdParty/cuopt/) |
| `ds_store` / `mac_alias` (DMG packaging tools only) | MIT | Yes | [Vendored versions and license texts](../tools/vendor/dmg_layout/README.md) |
| SeedVR2 3B / 7B weights | Apache-2.0 | Yes | [ByteDance-Seed/SeedVR2-3B](https://huggingface.co/ByteDance-Seed/SeedVR2-3B) |
| SeedVR2 4-bit conversion | Apache-2.0 | Yes | [AbstractFramework/seedvr2-3b-4bit](https://huggingface.co/AbstractFramework/seedvr2-3b-4bit) |
| SeedVR2 text embedding | Apache-2.0 | Yes | [mlx-community/SeedVR2-3B-mlx](https://huggingface.co/mlx-community/SeedVR2-3B-mlx) |
| `seedvr2-mlx-swift` port | MIT | Yes | [xocialize/seedvr2-mlx-swift](https://github.com/xocialize/seedvr2-mlx-swift) |
| FlashVSR 1.1 Tiny weights | Apache-2.0 | Yes | [JunhaoZhuang/FlashVSR-v1.1](https://huggingface.co/JunhaoZhuang/FlashVSR-v1.1) |
| Real-ESRGAN x2plus weights | BSD-3-Clause | Yes | [xinntao/Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) |
| BasicSR reference architecture | Apache-2.0 | Yes | [XPixelGroup/BasicSR](https://github.com/XPixelGroup/BasicSR) |
| MLX, MLX Swift | MIT | Yes | [ml-explore](https://github.com/ml-explore) |
| MLX C, `{fmt}`, Metal-cpp (MLX dependencies) | MIT / Apache-2.0 | Yes | [ml-explore/mlx-swift](https://github.com/ml-explore/mlx-swift) |
| nlohmann/json (vendored in mlx-c) | MIT | Yes | [nlohmann/json](https://github.com/nlohmann/json) |
| Swift Numerics | Apache-2.0 | Yes | [apple/swift-numerics](https://github.com/apple/swift-numerics) |
| InferKit (CodeFormer formulas) | MIT | Yes | `Runtime/INFERKIT-LICENSE` |
| **PGTFormer network** | **BSD-style, non-commercial** | **No** | [kepengxu/PGTFormer](https://github.com/kepengxu/PGTFormer) |
| **PGTFormer weights** | **Undeclared** | **Unknown** | [kepeng/pgtformer-base](https://huggingface.co/kepeng/pgtformer-base) |
| **TruFor network and weights** | **GRIP-UNINA licence: informational/nonprofit, credit required for public results** | **No** | [grip-unina/TruFor](https://github.com/grip-unina/TruFor) |

Note on the last row: the weights repository declares no licence in its card metadata and
ships no `LICENSE` file. The only terms available anywhere are the non-commercial `LICENSE`
in the author's GitHub repository. See [What needs a human](#what-needs-a-human).

Apache-2.0 is the only licence here carrying an express patent grant and a
"must include NOTICE" obligation for derivative works. MIT and BSD-3-Clause carry no
patent grant. That difference matters if you intend to patent anything derived from these
components; it does not otherwise obstruct combining them.

## Distribution modes

The build decides, and it decides at compile time rather than at download time.

| Build | Command | PGTFormer offered? | Product is usable commercially? |
|---|---|---|---|
| Default | `./build.sh` | Yes | **No** |
| Commercial | `WAIFU_COMMERCIAL_DISTRIBUTION=1 ./build.sh` | No | Yes |
| Public GitHub Release | Release workflow (default PGTFormer-enabled mode) | Yes | **No** |

Daily and manually dispatched GitHub releases use the default non-commercial build, include
the pinned PGTFormer catalog and label the release as a non-commercial edition. Settings
also marks PGTFormer as non-commercial use only. The commercial build remains available
with `WAIFU_COMMERCIAL_DISTRIBUTION=1`; it omits PGTFormer.

A commercial build:

- compiles `WAIFU_COMMERCIAL_DISTRIBUTION`, which makes `ManagedModel.offered` omit PGTFormer;
- omits `tools/pgtformer-release.json` from the bundle, so there is no pinned pointer to it;
- hides it from the Settings → Models list and the Uncover algorithm selector;
- keeps `UncoverAlgorithm.allCases` complete, so a session saved while PGTFormer was
  available still decodes and reports itself unavailable instead of failing to load;
- refuses with an explicit licence message if anything reaches for the missing descriptor,
  rather than reporting a corrupt catalogue.

The descriptor directory is cleared before it is repopulated. Building the default mode and
then the commercial mode into the same `dist/` tree would otherwise leave the PGTFormer
descriptor in the commercial bundle — the exact failure the gate exists to prevent.

## What needs a human

Two questions are not code decisions.

**1. The PGTFormer weights have no licence of their own.** `kepeng/pgtformer-base` declares
none and ships no `LICENSE`. The author's GitHub repository has a non-commercial `LICENSE`,
and he published the weights himself, so the reasonable reading is that he intended the same
terms to cover them. That is an inference, not a grant. Without an explicit grant, default
copyright applies and nobody has a documented right to redistribute the weights. We are
redistributing them, on the Hub and to users. **Ask the author.** Until he confirms, treat
the weights as unlicensed, which is a weaker position than "non-commercial".

**2. Whether a use is commercial.** The default build offers PGTFormer and TruFor and must
be kept to non-commercial use under their upstream terms. The commercial build excludes
both offers and is intended for commercial use under the remaining listed licences. A
build flag cannot change a model's licence or grant rights to unlicensed weights; the
PGTFormer weights question above still needs an explicit answer before those weights are
offered again. TruFor's licence is explicit but its terms still require a human judgement
about what counts as industrial or profit-oriented use in a given deployment.

A third, smaller point: the default build is non-commercial, but the repository `LICENSE`
says MIT, which normally implies the code may be used commercially. The code may; the
default *build* may not, because it offers PGTFormer. Anyone vendoring this should read this
document rather than the `LICENSE` header alone.

## Obligations we meet

- **Attribution.** Every hosted bundle ships the upstream `LICENSE`, a `NOTICE` naming the
  authors and revisions, and the model card linking the upstream source. The PGTFormer
  bundle ships the author's non-commercial licence verbatim.
- **App bundle notices.** The app's `Contents/Resources/ThirdPartyLicenses/` directory
  contains the project licence and licensing inventory, the full text of bundled runtime
  dependency licences (MLX Swift, MLX, MLX C, nlohmann/json, fmt, Metal-cpp and Swift
  Numerics), the InferKit licence, and the SeedVR2, FlashVSR and Real-ESRGAN notices and
  licences. Model packages downloaded later carry their own notices alongside the weights.
- **NOTICE propagation (Apache-2.0).** Relevant upstream notices are included in the app
  bundle and in the hosted model packages; the Apache-2.0 license text accompanies the
  bundled SeedVR2 positional embedding and applicable downloads.
- **Modification marking.** The BSD-3-Clause Real-ESRGAN licence asks that the copyright
  notice and disclaimer travel with copies. The hosted bundle's `NOTICE` records that our
  Core ML conversion is a modified work and describes what changed.
- **Non-commercial marking.** The PGTFormer card declares `license: other` with
  `license_name: pgtformer-bsd-style-non-commercial`, its `NOTICE` restates the restriction
  in full, and the app's model row and documentation both say non-commercial.
- **Full model packages are not bundled in the app.** The app includes the small SeedVR2
  positional embedding, whose Apache-2.0 terms and attribution ship in the bundle. The
  other model packages are downloaded at runtime and verified by SHA-256 against pinned
  manifests; their license texts and notices travel with those packages.

## Media handling

Video loading, frame decoding and export use Apple's AVFoundation and CoreMedia
frameworks. The app bundle contains no FFmpeg libraries or GPL/LGPL media components.
Tests, benchmarks and CI use Apple-native media fixtures and require no external
media codec executable. [Credits](Credits.md) records the projects we acknowledge.

## Provenance

Model weights are downloaded, never committed. Each hosted manifest pins the SHA-256 of the
upstream checkpoint it was converted from, so the conversion can be checked against the
original: the RealESRGAN manifest records `weightsSHA256` and `architectureSHA256`, and the
PGTFormer manifest records `weightsSHA256` plus per-file fingerprints of both `.mlpackage`
directories, which the native worker re-verifies on every inference job.

## SAFIRE

The native SAFIRE architecture adaptation retains the upstream SAFIRE/SAM
Apache-2.0 notices. Converted pretrained weights remain CC BY 4.0, with upstream
attribution and source/checkpoint identity; they are not relicensed as MIT.
Fixed-point resampling and clustering adaptations retain their BSD-3-Clause and
MIT notices. Independently authored WaifuFinder source remains MIT. See
`Runtime/SAFIRE-NOTICE.md`, the accompanying full license texts, and
[SAFIRE for Photos](ImageLineage.md). No additional third-party production package
is introduced; the worker uses the existing pinned MLX stack and Apple frameworks.

## TruFor

TruFor's own licence is explicit, unlike PGTFormer's undeclared weights: it allows
use, reproduction and modification free of charge **for informational and nonprofit
purposes only**, forbids industrial or profit-oriented use, requires retaining all
original notices, and requires crediting the original authors whenever results
obtained with the software are made public. The redistributed package carries the
full licence (`LICENSE.txt`), the CMX MIT notice (`LICENSE_CMX.txt`), and a `NOTICE`
crediting Guillaro, Cozzolino, Sud, Dufour and Verdoliva (CVPR 2023). The converted
weights are not relicensed as MIT; independently authored WaifuFinder source remains
MIT. TruFor is offered only in the non-commercial build and is otherwise excluded the
same way PGTFormer is. See [TruFor for Photos](TruFor.md), `Runtime/TRUFOR-LICENSE.txt`
and `Runtime/TRUFOR-CMX-LICENSE.txt`.
