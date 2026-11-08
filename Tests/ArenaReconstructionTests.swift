import CoreGraphics
import CoreImage
import Foundation
import ImageIO
import UniformTypeIdentifiers

func registerArenaReconstructionTests() {
    Registry.shared.add(suite: "PhotoArena reconstruction", TestCase(name: "complementary source pixels export with exact provenance and reversible overrides", knownBug: nil) {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("arena-reconstruction-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let width = 24, height = 12
        let original = try arenaReconstructionFixture(width: width, height: height, damagedStrip: nil, directory: directory, name: "original")
        let sources = try (0..<3).map { index -> ArenaAnalyzedSource in
            let url = try arenaReconstructionFixture(width: width, height: height, damagedStrip: index, directory: directory, name: "source-\(index)")
            return ArenaAnalyzedSource(id: "s\(index)", url: url, width: width, height: height,
                transform: ArenaTransform(), alignmentConfidence: 1, accepted: true, issues: [])
        }
        let regions = (0..<3).map { strip in
            ArenaRegion(id: "r\(strip)", rect: ArenaPixelRect(CGRect(x: strip*8, y: 0, width: 8, height: height)),
                measurements: sources.enumerated().map { index, source in
                    ArenaRegionMeasurement(sourceID: source.id, difference: index == strip ? 0.5 : 0,
                        sharpness: 0.1, clipping: 0, validFraction: 1, textCandidateFraction: 0,
                        evidence: [index == strip ? "Known damaged test strip" : "Matches held-out fixture"])
                }, unresolved: true, evidence: [])
        }
        let analysis = ArenaAnalysis(referenceID: "s0", width: width, height: height,
            sources: sources, regions: regions, warnings: [], elapsedSeconds: 0)
        let selections = ArenaReconstruction.select(analysis)
        checkEqual(selections.count, 3, "all regions selected")
        check(selections.allSatisfy(\.unresolved), "cross-source agreement does not establish authenticity")
        for (index, selection) in selections.enumerated() {
            check(selection.sourceID != "s\(index)", "damaged strip avoided")
        }
        let output = directory.appendingPathComponent("export")
        let manifest = try ArenaReconstruction.export(analysis, selections: selections, to: output)
        checkEqual(manifest.sourceSHA256.count, 3, "all source byte identities recorded")
        check(manifest.sourceSHA256.values.allSatisfy { $0.count == 64 }, "SHA256 identifiers")
        for file in manifest.sourceFiles.values {
            check(FileManager.default.fileExists(atPath: output.appendingPathComponent(file).path), "source bytes archived portably")
        }
        checkEqual(manifest.width, width, "native width")
        checkEqual(manifest.height, height, "native height")
        let reconstructed = try arenaReconstructionRead(output.appendingPathComponent(manifest.imageFile))
        let expected = try arenaReconstructionRead(original)
        checkEqual(arenaReconstructionPixels(reconstructed), arenaReconstructionPixels(expected), "reconstruction exactly matches known original RGB samples")
        let sourceMap = try arenaReconstructionRead(output.appendingPathComponent(manifest.sourceMapFile))
        checkEqual(sourceMap.width, width, "native provenance map width")
        checkEqual(sourceMap.height, height, "native provenance map height")
        let mapBytes = Array((sourceMap.dataProvider!.data! as Data))
        for (index, selection) in selections.enumerated() {
            let offset = (height/2)*sourceMap.bytesPerRow + index*8+4
            checkEqual(Int(mapBytes[offset]), manifest.sourceMapIndices[selection.sourceID], "source index encoded without interpolation")
        }
        let decoded = try JSONDecoder().decode(ArenaExportManifest.self, from: Data(contentsOf: output.appendingPathComponent("manifest.json")))
        checkEqual(decoded.selections.map(\.sourceID), selections.map(\.sourceID), "provenance survives JSON")
        let overridden = ArenaReconstruction.select(analysis, overrides: ["r0": "s2"])
        checkEqual(overridden[0].sourceID, "s2", "manual choice applied")
        check(overridden[0].manual, "override explicitly recorded")
        checkEqual(ArenaReconstruction.select(analysis)[0].sourceID, selections[0].sourceID, "removing override reverses the choice")
        do { _ = try ArenaReconstruction.export(analysis, selections: selections, to: output); check(false, "existing evidence must not be overwritten") } catch { }
    })
    Registry.shared.add(suite: "PhotoArena reconstruction", TestCase(name: "invalid alignment override is rejected and shared corruption remains unresolved", knownBug: nil) {
        let sources = [ArenaAnalyzedSource(id: "a", url: URL(fileURLWithPath: "/a"), width: 10, height: 10,
            transform: ArenaTransform(), alignmentConfidence: 1, accepted: true, issues: []),
            ArenaAnalyzedSource(id: "b", url: URL(fileURLWithPath: "/b"), width: 10, height: 10,
                transform: ArenaTransform(), alignmentConfidence: 0, accepted: false, issues: ["Bad alignment"])]
        let region = ArenaRegion(id: "shared", rect: ArenaPixelRect(CGRect(x: 0, y: 0, width: 10, height: 10)),
            measurements: sources.map { ArenaRegionMeasurement(sourceID: $0.id, difference: 0, sharpness: 1,
                clipping: 0, validFraction: 1, textCandidateFraction: 0.5, evidence: ["Shared overlay cannot be recovered"]) },
            unresolved: true, evidence: [])
        let analysis = ArenaAnalysis(referenceID: "a", width: 10, height: 10, sources: sources,
            regions: [region], warnings: [], elapsedSeconds: 0)
        let selected = ArenaReconstruction.select(analysis, overrides: ["shared": "b"])
        checkEqual(selected.first?.sourceID, "a", "rejected source cannot be forced into reconstruction")
        checkEqual(selected.first?.manual, false, "failed override not misrepresented as applied")
        checkEqual(selected.first?.unresolved, true, "shared damage unresolved")
    })
    Registry.shared.add(suite: "PhotoArena reconstruction", TestCase(name: "real NASA photograph complementary patch recovery", knownBug: nil) {
        let fixture = URL(fileURLWithPath: #filePath).deletingLastPathComponent().appendingPathComponent("Fixtures/RealESRGAN/nasa-collins-896x1134.jpg")
        let temporary = FileManager.default.temporaryDirectory.appendingPathComponent("arena-nasa-\(UUID().uuidString)")
        let retained = ProcessInfo.processInfo.environment["WAIFU_ARENA_EXAMPLE"].map { URL(fileURLWithPath: $0) }
        let directory = retained ?? temporary
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { if retained == nil { try? FileManager.default.removeItem(at: temporary) } }
        let original = try arenaReconstructionRead(fixture)
        let width = original.width, height = original.height
        let sourceSpace = CGColorSpace(name: CGColorSpace.sRGB)!
        var inputs: [ArenaSource] = []
        for index in 0..<3 {
            let url = directory.appendingPathComponent("variant-\(index).tiff")
            guard let context = CGContext(data: nil, width: width, height: height, bitsPerComponent: 8, bytesPerRow: width*4,
                space: sourceSpace, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { throw CocoaError(.fileWriteUnknown) }
            context.draw(original, in: CGRect(x: 0, y: 0, width: width, height: height))
            context.setFillColor(red: 1, green: 0, blue: 1, alpha: 0.8)
            let damage = CGRect(x: (index % 2)*448+32, y: (index / 2)*448+32, width: 160, height: 100)
            context.fill(damage)
            guard let pixels = context.makeImage(), let destination = CGImageDestinationCreateWithURL(url as CFURL, UTType.tiff.identifier as CFString, 1, nil) else { throw CocoaError(.fileWriteUnknown) }
            CGImageDestinationAddImage(destination, pixels, nil)
            guard CGImageDestinationFinalize(destination) else { throw CocoaError(.fileWriteUnknown) }
            inputs.append(ArenaSource(id: "variant-\(index)", url: url, seconds: 0))
        }
        let analysis = try ArenaForensics.analyzeFiles(inputs, tileSize: 448)
        checkEqual(analysis.sources.filter(\.accepted).count, 3, "all complementary real sources align")
        let selections = ArenaReconstruction.select(analysis)
        let rankingEncoder = JSONEncoder(); rankingEncoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try rankingEncoder.encode(ArenaRanking.technical(analysis)).write(to: directory.appendingPathComponent("ranking.json"))
        let exportDirectory = directory.appendingPathComponent("recovery")
        let manifest = try ArenaReconstruction.export(analysis, selections: selections, to: exportDirectory)
        let result = try arenaReconstructionRead(exportDirectory.appendingPathComponent(manifest.imageFile))
        let expected = arenaReconstructionPixels(original), actual = arenaReconstructionPixels(result)
        let absoluteError = zip(expected, actual).reduce(0.0) { $0 + abs(Double($1.0)-Double($1.1)) } / Double(expected.count)
        check(absoluteError < 1.5, "real source reconstruction preserves original within color rounding (MAE \(absoluteError))")
        let report: [String: Any] = ["fixture": "Tests/Fixtures/RealESRGAN/nasa-collins-896x1134.jpg",
            "attribution": "NASA; original repository fixture attribution in Tests/Fixtures/RealESRGAN/README.md",
            "modification": "Three conventional translucent magenta rectangle overlays; no real AI edits are claimed",
            "width": width, "height": height, "meanAbsoluteRGBA8Error": absoluteError,
            "acceptedSources": analysis.sources.filter(\.accepted).count,
            "regions": analysis.regions.count, "unresolvedRegions": selections.filter(\.unresolved).count,
            "analysisSeconds": analysis.elapsedSeconds, "exportSeconds": manifest.elapsedSeconds,
            "limitations": "Small real photograph and conventional overlays only. No real AI-edit localization accuracy, authenticity verification, or ANE placement is established."]
        try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys]).write(to: directory.appendingPathComponent("evaluation.json"))
    })
    Registry.shared.add(suite: "PhotoArena reconstruction", TestCase(name: "40 megapixel native export benchmark (opt-in)", knownBug: nil) {
        guard ProcessInfo.processInfo.environment["WAIFU_ARENA_40MP"] == "1" else { return }
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("arena-40mp-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let width = 8000, height = 5000
        let url = try arenaReconstructionFixture(width: width, height: height, damagedStrip: nil, directory: directory, name: "40mp")
        let source = ArenaAnalyzedSource(id: "native", url: url, width: width, height: height,
            transform: ArenaTransform(), alignmentConfidence: 1, accepted: true, issues: [])
        let region = ArenaRegion(id: "whole", rect: ArenaPixelRect(CGRect(x: 0, y: 0, width: width, height: height)),
            measurements: [ArenaRegionMeasurement(sourceID: source.id, difference: 0, sharpness: 1,
                clipping: 0, validFraction: 1, textCandidateFraction: 0, evidence: [])], unresolved: true, evidence: [])
        let analysis = ArenaAnalysis(referenceID: source.id, width: width, height: height, sources: [source],
            regions: [region], warnings: [], elapsedSeconds: 0)
        let sampler = RealESRGANMemorySampler()
        let manifest = try ArenaReconstruction.export(analysis, selections: ArenaReconstruction.select(analysis), to: directory.appendingPathComponent("output"))
        let decoded = try arenaReconstructionRead(directory.appendingPathComponent("output/reconstruction.tiff"))
        checkEqual(decoded.width*decoded.height, 40_000_000, "all 40MP retained")
        let report: [String: Any] = ["width": decoded.width, "height": decoded.height,
            "exportSeconds": manifest.elapsedSeconds, "peakPhysicalFootprintBytes": sampler.peakBytes,
            "computeBackend": "Apple Core Image and ImageIO; ANE not requested or verified",
            "scope": "single-source native reconstruction/export; not an end-to-end 40MP analysis benchmark"]
        if let path = ProcessInfo.processInfo.environment["WAIFU_ARENA_40MP_REPORT"] {
            try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys]).write(to: URL(fileURLWithPath: path))
        }
        print("PhotoArena 40MP export: \(manifest.elapsedSeconds)s; peak process footprint \(sampler.peakBytes) bytes")
    })
}

/// Apple-native, deterministic fixture. Three strips have distinct real samples;
/// variants corrupt one strip each. No synthetic AI fixture is mislabelled as AI.
private func arenaReconstructionFixture(width: Int, height: Int, damagedStrip: Int?, directory: URL, name: String) throws -> URL {
    try autoreleasepool {
        let space = CGColorSpace(name: CGColorSpace.sRGB)!
        guard let context = CGContext(data: nil, width: width, height: height, bitsPerComponent: 8, bytesPerRow: width*4,
            space: space, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { throw CocoaError(.fileWriteUnknown) }
        let colors: [(Double,Double,Double)] = [(0.2,0.4,0.6),(0.6,0.3,0.2),(0.1,0.7,0.3)]
        for index in 0..<3 {
            let color = damagedStrip == index ? (0.9,0.1,0.9) : colors[index]
            context.setFillColor(red: color.0, green: color.1, blue: color.2, alpha: 1)
            let x0 = index*width/3, x1 = (index+1)*width/3
            context.fill(CGRect(x: x0, y: 0, width: x1-x0, height: height))
        }
        let url = directory.appendingPathComponent(name+".tiff")
        guard let image = context.makeImage(), let destination = CGImageDestinationCreateWithURL(url as CFURL, UTType.tiff.identifier as CFString, 1, nil) else { throw CocoaError(.fileWriteUnknown) }
        CGImageDestinationAddImage(destination, image, nil)
        guard CGImageDestinationFinalize(destination) else { throw CocoaError(.fileWriteUnknown) }
        return url
    }
}
private func arenaReconstructionRead(_ url: URL) throws -> CGImage {
    guard let source = CGImageSourceCreateWithURL(url as CFURL, nil), let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else { throw CocoaError(.fileReadCorruptFile) }
    return image
}
private func arenaReconstructionPixels(_ image: CGImage) -> [UInt8] {
    let context = CIContext(options: [.cacheIntermediates: false])
    var pixels = [UInt8](repeating: 0, count: image.width*image.height*4)
    context.render(CIImage(cgImage: image), toBitmap: &pixels, rowBytes: image.width*4,
        bounds: CGRect(x: 0, y: 0, width: image.width, height: image.height), format: .RGBA8, colorSpace: CGColorSpace(name: CGColorSpace.sRGB)!)
    return pixels
}
