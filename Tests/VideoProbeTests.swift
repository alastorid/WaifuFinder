import Foundation

// Probing a dropped file for the metadata the sampler needs.
//
// `VideoProbe.probe` is the first thing the engine does with a clip, and it
// decides two things that are expensive to get wrong: what the file *is*
// (duration, dimensions, frame rate) and whether it is analysable at all. The
// value half of that is pure decoding of a real container, so the interesting
// assertions here are the rejections — the probe's job is to say "no" quickly
// and with the right reason, before a progress bar is shown for a file that
// will never produce a frame.
//
// The rejections are pinned by *identity*, not merely by "something threw":
// `.notAFile` must be distinguishable from `.noVideoTrack` from
// `.zeroOrInvalidDuration`, because the UI tells the user which it is and the
// sampling layer keys off `hasVideo`. `checkThrowsAsync` returns the thrown
// error for exactly this reason.
//
// Real MP4 cases use the Apple-native VideoFixture encoder. URL rejection cases
// need no media fixture.
//
// Two production behaviours are deliberately asserted rather than assumed, and
// both are recorded in `VideoProbe`'s own documentation:
//   * `estimatedFrameCount` is `round(duration × effectiveFrameRate)`, where a
//     zero `nominalFrameRate` (variable-frame-rate content) is replaced with
//     30 — never a division by zero. The formula case pins the positive case.
//   * A near-zero-duration clip may still probe to a tiny positive duration
//     rather than being refused; the zero-duration case accepts either outcome
//     and only requires that the probe does not crash or return a non-positive
//     duration.

func registerVideoProbeTests() {
    let suite = "video probe"

    // MARK: - The happy path (needs an encoder)

    Registry.shared.add(suite: suite, TestCase(name: "Probe a generated MP4 returns correct VideoInfo",
        knownBug: nil) {
        // A 2 s 320×240 @30 clip, probed as a user's drop would be. The
        // tolerances are loose on purpose: the container's duration is a
        // sum of sample durations and the nominal rate is a hint, so an exact
        // match would be asserting Apple-native fixture generation's rounding rather than the probe's.
        let directory = try Temporary.directory("probe-happy")
        let outputURL = try VideoFixture.makeMP4(in: directory, durationSeconds: 2,
                                                 width: 320, height: 240, frameRate: 30)

        let info = try await VideoProbe.probe(outputURL)

        checkEqual(info.url, outputURL, "the probe reports the file it was handed")
        checkClose(info.durationSeconds, 2.0, tolerance: 0.1, "a 2 s clip is ~2 s")
        checkEqual(info.width, 320, "the natural width")
        checkEqual(info.height, 240, "the natural height")
        checkClose(info.frameRate, 30.0, tolerance: 1.0, "the nominal frame rate is ~30 fps")
        check(info.estimatedFrameCount > 0, "a decodable clip estimates a positive frame count")
        check(info.hasVideo, "a file with a video track hasVideo")
    })

    Registry.shared.add(suite: suite, TestCase(name: "Probe a longer generated MP4 (10s) has correct frame count estimate",
        knownBug: nil) {
        // 10 s at 25 fps is 250 frames; the estimate is what bounds progress and
        // the sampling cap, so it has to track duration × rate rather than be
        // pinned to a container's `nb_frames` (which MP4 does not reliably carry).
        let directory = try Temporary.directory("probe-long")
        let outputURL = try VideoFixture.makeMP4(in: directory, durationSeconds: 10,
                                                 width: 640, height: 480, frameRate: 25)

        let info = try await VideoProbe.probe(outputURL)

        checkClose(info.durationSeconds, 10.0, tolerance: 0.2, "a 10 s clip is ~10 s")
        checkEqual(info.width, 640, "the natural width")
        checkEqual(info.height, 480, "the natural height")
        checkClose(info.frameRate, 25.0, tolerance: 1.0, "the nominal frame rate is ~25 fps")
        check(abs(info.estimatedFrameCount - 250) < 10,
              "10 s × 25 fps estimates ~250 frames, got \(info.estimatedFrameCount)")
    })

    // MARK: - Rejection: nothing to read

    Registry.shared.add(suite: suite, TestCase(name: "Probe passes HTTP URLs to AVFoundation",
        knownBug: nil) {
        // A local closed port fails promptly, but it must pass URL validation and
        // reach AVFoundation instead of being rejected for not being a file URL.
        let url = URL(string: "http://127.0.0.1:1/video.mp4")!
        let thrown = await checkThrowsAsync("unavailable HTTP media fails to probe") {
            try await VideoProbe.probe(url)
        }
        check((thrown as? VideoProbeError) != .notAFile, "HTTP URLs are not rejected as non-file URLs")
    })

    Registry.shared.add(suite: suite, TestCase(name: "Probe rejects non-file URL (ftp)",
        knownBug: nil) {
        // The scheme check is `isFileURL`, not a list of allowed schemes, so an
        // exotic one lands in the same bucket — pinned so a future "support
        // more schemes" edit cannot quietly let a network URL through.
        let url = URL(string: "ftp://example.com/video.mp4")!
        let thrown = await checkThrowsAsync("an ftp URL is not a probeable file") {
            try await VideoProbe.probe(url)
        }
        checkEqual(thrown as? VideoProbeError, .notAFile, "ftp is rejected as notAFile")
    })

    Registry.shared.add(suite: suite, TestCase(name: "Probe rejects non-existent file",
        knownBug: nil) {
        // `file://` but absent: a quick `FileManager` check gives `.notAFile`
        // instantly instead of waiting on AVFoundation's opaque "cannot open".
        let directory = try Temporary.directory("probe-missing")
        let url = directory.appendingPathComponent("does-not-exist.mp4")
        let thrown = await checkThrowsAsync("a file:// URL to nothing is not a probeable file") {
            try await VideoProbe.probe(url)
        }
        checkEqual(thrown as? VideoProbeError, .notAFile, "a missing path is rejected as notAFile")
    })

    Registry.shared.add(suite: suite, TestCase(name: "Probe rejects directory",
        knownBug: nil) {
        // A directory *exists* and is `file://`, so only the directory check
        // catches it — the same quick check that keeps AVAsset from being handed
        // a folder. Rejected as `.notAFile`, like the rest of the "no bytes to
        // read" family.
        let directory = try Temporary.directory("probe-dir")
        let subdirectory = directory.appendingPathComponent("subdir", isDirectory: true)
        try FileManager.default.createDirectory(at: subdirectory, withIntermediateDirectories: true)
        let thrown = await checkThrowsAsync("a directory is not a probeable file") {
            try await VideoProbe.probe(subdirectory)
        }
        checkEqual(thrown as? VideoProbeError, .notAFile, "a directory is rejected as notAFile")
    })

    Registry.shared.add(suite: suite, TestCase(name: "Probe rejects audio-only file (no video track)",
        knownBug: nil) {
        // A valid, decodable container with no video track is the case the probe
        // exists to name precisely: the file is fine, it just cannot be sampled,
        // and `.noVideoTrack` is what lets the UI say so. A plain "something
        // threw" would not distinguish it from corruption.
        let directory = try Temporary.directory("probe-audio")
        let outputURL = try VideoFixture.makeAudioOnly(in: directory, durationSeconds: 2)

        let thrown = await checkThrowsAsync("an audio-only file has no frames to sample") {
            try await VideoProbe.probe(outputURL)
        }
        checkEqual(thrown as? VideoProbeError, .noVideoTrack, "an audio-only file is rejected as noVideoTrack")
    })

    // MARK: - Rejection: nothing usable to sample

    Registry.shared.add(suite: suite, TestCase(name: "Probe rejects zero-duration video",
        knownBug: nil) {
        // A sub-frame clip is at the boundary the probe guards: a zero duration
        // would divide by zero in the sampling plan. But an encoder may still
        // give the single frame a tiny positive duration, so the honest contract
        // is "either refused as zero-or-invalid, or probed with a positive
        // duration" — never a crash and never a non-positive duration.
        let directory = try Temporary.directory("probe-zero")
        let outputURL = try VideoFixture.makeMP4(in: directory, durationSeconds: 0.001,
                                                 width: 320, height: 240, frameRate: 30)

        var refusal: Error?
        do {
            let info = try await VideoProbe.probe(outputURL)
            check(info.durationSeconds > 0,
                  "if a 1 ms clip probes at all, its duration is positive, got \(info.durationSeconds)")
        } catch {
            refusal = error
        }
        if let refusal {
            checkEqual(refusal as? VideoProbeError, .zeroOrInvalidDuration,
                       "a sub-frame clip that is refused is refused as zero-or-invalid duration")
        }
    })

    Registry.shared.add(suite: suite, TestCase(name: "Probe handles corrupted MP4 gracefully",
        knownBug: nil) {
        // Text with an `.mp4` name. The probe must not believe the extension:
        // whatever AVFoundation reports (an empty track list, or a load failure)
        // has to surface as a thrown error rather than a plausible `VideoInfo`
        // built from zeros. No fixture needed — the corruption is written inline.
        let directory = try Temporary.directory("probe-corrupt")
        let corruptURL = directory.appendingPathComponent("corrupt.mp4")
        try "not a video file".write(to: corruptURL, atomically: true, encoding: .utf8)

        let thrown = await checkThrowsAsync("a text file named .mp4 cannot be probed as video") {
            try await VideoProbe.probe(corruptURL)
        }
        checkNotNil(thrown, "corruption surfaces as a thrown error, not a VideoInfo")
    })

    // MARK: - Edge cases

    Registry.shared.add(suite: suite, TestCase(name: "Probe handles file:// URL with spaces in path",
        knownBug: nil) {
        // The probe takes a `URL`, not a path string, so a space in a component
        // must survive as an encoded file URL. The fixture is placed in a
        // labelled temp directory so the *whole* path contains a space, not just
        // the filename — the failure mode being guarded is a hand-built
        // `URL(string:)` (which would mangle the space), not a filename quirk.
        let directory = try Temporary.directory("probe with spaces")
        let outputURL = try VideoFixture.makeMP4(in: directory, durationSeconds: 1,
                                                 width: 320, height: 240, frameRate: 30)

        let info = try await VideoProbe.probe(outputURL)

        check(info.hasVideo, "a clip at a spaced path still probes with a video track")
        check(info.durationSeconds > 0, "and a positive duration, got \(info.durationSeconds)")
    })

    Registry.shared.add(suite: suite, TestCase(name: "Probe returns correct estimatedFrameCount = round(duration * frameRate)",
        knownBug: nil) {
        // The exact formula, not a tolerance: `estimatedFrameCount` is defined
        // as `Int((duration × effectiveFrameRate).rounded())` and is asserted
        // against the duration and rate the probe itself reported. For a real
        // 24 fps clip the nominal rate is positive, so `effectiveFrameRate` is
        // that rate and the identity must hold exactly — a rounded estimate an
        // off-by-one away would mean the sampler's progress and cap are sized
        // from a number the probe did not intend to publish.
        let directory = try Temporary.directory("probe-formula")
        let outputURL = try VideoFixture.makeMP4(in: directory, durationSeconds: 5,
                                                 width: 320, height: 240, frameRate: 24)

        let info = try await VideoProbe.probe(outputURL)

        let expected = Int((info.durationSeconds * info.frameRate).rounded())
        checkEqual(info.estimatedFrameCount, expected,
                   "estimatedFrameCount is round(duration × frameRate) for the reported values")
    })
}
