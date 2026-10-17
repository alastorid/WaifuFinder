import Foundation

// Frame export: file naming, overwrite suffixes, and one real decode-and-write.
//
// The export layer is the only place a user's selection becomes something on
// disk, and its whole contract is two pure decisions plus one I/O path:
//
//   * `ExportRequest.fileName(for:sourceStem:)` decides what a frame is called.
//     The name is *derived*, never asked for — a user exporting forty frames
//     wants them to sort into selection order and say which video they came
//     from, so the counter is zero-padded to six digits and the source stem is
//     part of the name. These cases pin that exact shape, because a name is the
//     one artefact the user keeps.
//   * `FrameExporter.uniqueDestinationURL(base:)` decides what to do when that
//     name is already taken. It is `internal` rather than `private` precisely so
//     this decision can be seen directly: the alternative is writing 10,001
//     files into a directory to reach the suffix cap, and a full export can only
//     ever show the first suffix.
//   * `export(_:sourceURL:)` is an `AsyncThrowingStream` where a *source* error
//     is thrown at the stream level (no reader can even be opened) while a
//     *per-frame* failure is yielded into the stream. The two integration cases
//     below separate those paths, and the round trip proves the decoder →
//     `CGImageDestination` → PNG/JPEG pipeline against real bytes rather than a
//     mock, since that is the only way the magic-number checks mean anything.
//
// The round-trip cases build their clip with the Apple-native shared
// `VideoFixture`; the naming and suffix cases are pure and run everywhere.

func registerFrameExporterTests() {
    let suite = "frame export"

    // MARK: - Filename derivation

    Registry.shared.add(suite: suite, TestCase(name: "fileName produces zero-padded 6-digit counter", knownBug: nil) {
        let tempDir = try Temporary.directory("export-naming")
        let request = ExportRequest(
            frameIDs: [FrameID(sampleIndex: 0), FrameID(sampleIndex: 1), FrameID(sampleIndex: 123)],
            destinationDirectory: tempDir,
            format: .png
        )

        // Six digits is what makes a directory listing read in selection order up
        // to a million frames, which is the whole reason the pad exists.
        checkEqual(request.fileName(for: FrameID(sampleIndex: 0), sourceStem: "testvideo"),
                   "testvideo-000000.png", "first frame")
        checkEqual(request.fileName(for: FrameID(sampleIndex: 1), sourceStem: "testvideo"),
                   "testvideo-000001.png", "second frame")
        checkEqual(request.fileName(for: FrameID(sampleIndex: 123), sourceStem: "testvideo"),
                   "testvideo-000123.png", "frame 123 is padded, not bare")
    })

    Registry.shared.add(suite: suite, TestCase(name: "fileName uses the JPEG extension for JPEG format", knownBug: nil) {
        let tempDir = try Temporary.directory("export-jpeg")
        let request = ExportRequest(
            frameIDs: [FrameID(sampleIndex: 42)],
            destinationDirectory: tempDir,
            format: .jpeg
        )

        // The format is `.jpeg` but the on-disk extension is `.jpg`; pinning the
        // sRGB-era spelling is deliberate — `ExportFormat.fileExtension` maps it,
        // and a preview panel that looked for `.jpeg` would miss the file.
        checkEqual(request.fileName(for: FrameID(sampleIndex: 42), sourceStem: "testvideo"),
                   "testvideo-000042.jpg", "JPEG carries the .jpg extension")
    })

    Registry.shared.add(suite: suite, TestCase(name: "fileName includes the source stem to avoid collisions", knownBug: nil) {
        let tempDir = try Temporary.directory("export-stem")
        let request = ExportRequest(
            frameIDs: [FrameID(sampleIndex: 5)],
            destinationDirectory: tempDir,
            format: .png
        )

        // Two videos exported into one folder with an identical counter would
        // collide; the stem is what keeps them apart.
        checkEqual(request.fileName(for: FrameID(sampleIndex: 5), sourceStem: "vacation"),
                   "vacation-000005.png", "stem comes from the first source")
        checkEqual(request.fileName(for: FrameID(sampleIndex: 5), sourceStem: "birthday"),
                   "birthday-000005.png", "and changes with the source")
    })

    // MARK: - Overwrite suffixes

    Registry.shared.add(suite: suite, TestCase(name: "uniqueDestinationURL appends -1 then -2 as names fill", knownBug: nil) {
        let tempDir = try Temporary.directory("export-suffix")
        let exporter = FrameExporter()
        let baseURL = tempDir.appendingPathComponent("testvideo-000001.png")

        // An empty slot is used as itself: suffixing only exists to avoid a clash.
        let first = exporter.uniqueDestinationURL(base: baseURL)
        checkEqual(first, baseURL, "no existing file means the base is returned unchanged")

        try "content".write(to: baseURL, atomically: true, encoding: .utf8)

        let second = exporter.uniqueDestinationURL(base: baseURL)
        checkEqual(second.lastPathComponent, "testvideo-000001-1.png", "first occupied name gets -1")

        try "content".write(to: second, atomically: true, encoding: .utf8)

        let third = exporter.uniqueDestinationURL(base: baseURL)
        checkEqual(third.lastPathComponent, "testvideo-000001-2.png", "second occupied name gets -2")
    })

    Registry.shared.add(suite: suite, TestCase(name: "uniqueDestinationURL preserves the extension", knownBug: nil) {
        let tempDir = try Temporary.directory("export-ext")
        let exporter = FrameExporter()
        let baseURL = tempDir.appendingPathComponent("photo.jpg")

        try "content".write(to: baseURL, atomically: true, encoding: .utf8)
        let next = exporter.uniqueDestinationURL(base: baseURL)

        // The suffix is inserted *before* the extension, so the file stays
        // recognisable as a JPEG — otherwise a viewer would stop opening it.
        checkEqual(next.pathExtension, "jpg", "the extension survives the suffix")
        checkEqual(next.lastPathComponent, "photo-1.jpg", "and the suffix lands on the stem")
    })

    Registry.shared.add(suite: suite, TestCase(name: "uniqueDestinationURL still suffixes a no-extension name", knownBug: nil) {
        let tempDir = try Temporary.directory("export-noext")
        let exporter = FrameExporter()
        let baseURL = tempDir.appendingPathComponent("README")

        try "content".write(to: baseURL, atomically: true, encoding: .utf8)
        let next = exporter.uniqueDestinationURL(base: baseURL)

        // The suffix goes on the stem and the (absent) extension is not joined
        // with a stray dot: the result is `README-1`, not `README-1.`. Production
        // used to interpolate the extension unconditionally, which produced the
        // trailing dot; that is fixed in `uniqueDestinationURL`, and this pins the
        // fixed behaviour rather than the defect.
        checkEqual(next.lastPathComponent, "README-1", "a name with no extension still gets -1")
        checkEqual(next.deletingLastPathComponent(), baseURL.deletingLastPathComponent(),
                   "the suffixed name stays in the same directory")
    })

    // MARK: - Real decode and write

    Registry.shared.add(suite: suite, TestCase(name: "round-trip export writes real PNG and JPEG bytes", knownBug: nil) {

        let tempDir = try Temporary.directory("export-roundtrip")
        let videoURL = try VideoFixture.makeMP4(
            in: tempDir, durationSeconds: 2, width: 160, height: 120, frameRate: 30
        )
        let destDir = tempDir.appendingPathComponent("export")
        try FileManager.default.createDirectory(at: destDir, withIntermediateDirectories: true)

        let frameID = FrameID(sampleIndex: 0)
        let exporter = FrameExporter()

        // PNG: one frame in, one file out, and the file is a PNG by its bytes.
        let pngRequest = ExportRequest(
            frameIDs: [frameID],
            destinationDirectory: destDir,
            format: .png,
            overwrite: true
        )
        var pngResults: [ExportResult] = []
        for try await result in exporter.export(pngRequest, sourceURL: videoURL) {
            pngResults.append(result)
        }
        checkEqual(pngResults.count, 1, "one frame requested, one result yielded")
        guard let pngResult = pngResults.first else { return }
        checkEqual(pngResult.frameID, frameID, "the result names the frame it exported")
        check(FileManager.default.fileExists(atPath: pngResult.url.path),
              "the PNG exists at \(pngResult.url.path)")
        check(pngResult.byteCount > 0, "the PNG is non-empty")
        checkEqual(pngResult.url.pathExtension, "png", "the PNG keeps its extension")

        let pngData = try Data(contentsOf: pngResult.url)
        check(pngData.starts(with: [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]),
              "the bytes are an actual PNG signature, not merely a .png name")

        // JPEG: same path, different destination format and quality.
        let jpegRequest = ExportRequest(
            frameIDs: [frameID],
            destinationDirectory: destDir,
            format: .jpeg,
            jpegQuality: 0.8,
            overwrite: true
        )
        var jpegResults: [ExportResult] = []
        for try await result in exporter.export(jpegRequest, sourceURL: videoURL) {
            jpegResults.append(result)
        }
        checkEqual(jpegResults.count, 1, "one frame requested, one JPEG result yielded")
        guard let jpegResult = jpegResults.first else { return }
        checkEqual(jpegResult.frameID, frameID, "the JPEG result names the frame it exported")
        check(FileManager.default.fileExists(atPath: jpegResult.url.path),
              "the JPEG exists at \(jpegResult.url.path)")
        check(jpegResult.byteCount > 0, "the JPEG is non-empty")
        checkEqual(jpegResult.url.pathExtension, "jpg", "the JPEG keeps its extension")

        let jpegData = try Data(contentsOf: jpegResult.url)
        check(jpegData.starts(with: [0xFF, 0xD8, 0xFF]),
              "the bytes are an actual JPEG signature")

        // Both files share one directory, so the two formats do not overwrite
        // each other by name — the extension is what keeps them apart.
        check(pngResult.url != jpegResult.url, "PNG and JPEG land on distinct URLs")
    })

    Registry.shared.add(suite: suite, TestCase(name: "export throws when the source video cannot be read", knownBug: nil) {
        let tempDir = try Temporary.directory("export-missing-source")
        let destDir = tempDir.appendingPathComponent("export")
        let nonExistentURL = tempDir.appendingPathComponent("nonexistent.mp4")

        let request = ExportRequest(
            frameIDs: [FrameID(sampleIndex: 0)],
            destinationDirectory: destDir,
            format: .png,
            overwrite: true
        )
        let exporter = FrameExporter()

        // A source failure happens before any frame is decoded, so the stream
        // itself throws rather than yielding a per-frame failure. `for try await`
        // is where that surfaces — the error is not swallowed by iterating.
        let thrown = await checkThrowsAsync("exporting a missing source throws at the stream level") {
            for try await _ in exporter.export(request, sourceURL: nonExistentURL) {
                // No frame should ever arrive for an unreadable source.
            }
        }
        checkNotNil(thrown, "the stream surfaced an error for the unreadable source")
    })

    Registry.shared.add(suite: suite, TestCase(name: "thumbnail export without a store surfaces a stream error, not a file", knownBug: nil) {
        let tempDir = try Temporary.directory("export-no-store")
        let destDir = tempDir.appendingPathComponent("export")
        try FileManager.default.createDirectory(at: destDir, withIntermediateDirectories: true)

        let request = ExportRequest(
            frameIDs: [FrameID(sampleIndex: 0)],
            destinationDirectory: destDir,
            format: .png,
            useThumbnails: true,
            overwrite: true
        )
        // No thumbnail store: every frame fails, and `AsyncThrowingStream` turns
        // the first yielded failure into a throw on the next iteration.
        let exporter = FrameExporter(thumbnailStore: nil)

        let thrown = await checkThrowsAsync("thumbnail export without a store surfaces an error") {
            for try await _ in exporter.export(request, sourceURL: tempDir.appendingPathComponent("dummy.mp4")) {
                // The store is missing, so no frame is ever produced.
            }
        }

        guard let error = checkNotNil(thrown, "an error came out of the stream") else { return }
        guard let exportError = error as? ExportError else {
            check(false, "expected an ExportError, got \(error)")
            return
        }
        if case .writeFailed = exportError {
            // The per-frame failure is wrapped as a write failure, which is what
            // the UI shows next to the frame the user selected.
        } else {
            check(false, "expected .writeFailed for a missing thumbnail, got \(exportError)")
        }

        // Crucially, the failure must not have left a truncated or bogus file
        // behind: a stream error is not a silent empty export.
        let expected = destDir.appendingPathComponent("dummy-000000.png")
        check(!FileManager.default.fileExists(atPath: expected.path),
              "no file is written for a frame that could not be exported")
    })
}
