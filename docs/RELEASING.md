# Releases and showcase site

Push a version tag such as `v1.2.0` to build and publish a macOS release:

```sh
git tag v1.2.0
git push origin v1.2.0
```

GitHub Actions runs the regression suite, builds the app and its native Apple silicon
workers on a macOS 15 runner, creates a compressed DMG and publishes it as a GitHub
Release. Each release includes a versioned disk image, a SHA-256 checksum and the stable
`WaifuFinder-macos-arm64.dmg` download name used by the showcase site. The build targets
macOS 15 and Apple silicon. The DMG contains `WaifuFinder.app` and an Applications
shortcut.

Published DMGs use the default non-commercial build, including the pinned PGTFormer
download offer, and are labeled as non-commercial editions. For a build that omits
PGTFormer, use `WAIFU_COMMERCIAL_DISTRIBUTION=1 ./build.sh`. See
[`Licensing.md`](Licensing.md) for the model terms and remaining limitations.

Model weights remain on the pinned Hugging Face repositories referenced by the release
descriptors; they are downloaded separately by the app. GitHub release assets contain
only the app DMGs and their checksum. Test data is generated locally from source scripts;
do not attach test media, fixtures, model checkpoints or working-folder archives.

The showcase is published from `site/` to GitHub Pages when that directory changes on
`main`. The repository's Pages publishing source must be **GitHub Actions**. Its download
button follows the latest release asset; the release page is
`https://github.com/alastorid/WaifuFinder/releases`.

`build.sh` ad-hoc signs the app, so the DMG does not carry a Developer ID signature or
notarization ticket. A macOS user may need to approve the downloaded app in Privacy &
Security. Developer ID signing and notarization can be added once release credentials are
available.
