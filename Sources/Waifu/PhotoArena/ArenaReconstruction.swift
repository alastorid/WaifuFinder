import CoreGraphics
import CoreImage
import CryptoKit
import Foundation
import ImageIO
import UniformTypeIdentifiers

struct ArenaSourceSelection: Codable, Identifiable, Sendable {
    var id: String { regionID }
    let regionID: String
    let sourceID: String
    let rect: ArenaPixelRect
    let unresolved: Bool
    let reasons: [String]
    let manual: Bool
}
struct ArenaExportManifest: Codable, Sendable {
    var schemaVersion = 1
    let width: Int
    let height: Int
    let imageFile: String
    let sourceMapFile: String
    let sourceMapIndices: [String: Int]
    let sourceSHA256: [String: String]
    let sourceFiles: [String: String]
    let analysis: ArenaAnalysis
    let selections: [ArenaSourceSelection]
    let warnings: [String]
    let elapsedSeconds: Double
}

/// Copies real, aligned source samples. No generative processing, sharpening,
/// exposure adjustment, or unrecorded blending is performed.
enum ArenaReconstruction {
    static func select(_ analysis: ArenaAnalysis, overrides: [String: String] = [:]) -> [ArenaSourceSelection] {
        let accepted = Set(analysis.sources.filter(\.accepted).map(\.id))
        return analysis.regions.compactMap { region in
            let candidates = region.measurements.filter {
                accepted.contains($0.sourceID) && $0.validFraction >= 0.995
                    && $0.difference.isFinite && $0.sharpness.isFinite
            }
            guard !candidates.isEmpty else { return nil }
            let requested = overrides[region.id]
            let manual = requested.flatMap { id in candidates.first { $0.sourceID == id } }
            // Consensus disagreement dominates texture. Sharpness only resolves
            // close scores; it does not establish that fine detail is original.
            let best = manual ?? candidates.enumerated().min { lhs, rhs in
                let l = lhs.element.difference + lhs.element.textCandidateFraction * 0.15 + lhs.element.clipping * 0.05
                let r = rhs.element.difference + rhs.element.textCandidateFraction * 0.15 + rhs.element.clipping * 0.05
                if abs(l-r) > 0.01 { return l < r }
                if abs(lhs.element.sharpness-rhs.element.sharpness) > 0.001 {
                    return lhs.element.sharpness > rhs.element.sharpness
                }
                return lhs.offset < rhs.offset
            }!.element
            var reasons = best.evidence
            reasons.append(manual == nil ? "Selected by measured agreement, text candidates and technical detail; these signals do not prove originality." : "Explicit human source override; originality remains unverified.")
            if requested != nil && manual == nil { reasons.append("Requested source rejected because it lacks reliable aligned coverage.") }
            return ArenaSourceSelection(regionID: region.id, sourceID: best.sourceID, rect: region.rect,
                unresolved: true, reasons: reasons, manual: manual != nil)
        }
    }

    static func preview(_ analysis: ArenaAnalysis, selections: [ArenaSourceSelection], maxPixelSize: Int = 1600) throws -> CGImage {
        guard maxPixelSize > 0 else { throw failure("Invalid preview size") }
        let scale = min(1, Double(maxPixelSize) / Double(max(analysis.width, analysis.height)))
        return try render(analysis, selections: selections, scale: scale).image
    }

    /// The directory is new to prevent accidental overwrite of source/evidence files.
    /// One RGBA output and one byte-per-pixel source map are allocated. Sources
    /// decode lazily and are rendered sequentially in bounded 512px tiles.
    static func export(_ analysis: ArenaAnalysis, selections: [ArenaSourceSelection], to directory: URL) throws -> ArenaExportManifest {
        let start = Date()
        guard !FileManager.default.fileExists(atPath: directory.path) else {
            throw failure("Choose a new export folder; existing files will not be overwritten.")
        }
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        do {
            let sourceDirectory = directory.appendingPathComponent("sources")
            try FileManager.default.createDirectory(at: sourceDirectory, withIntermediateDirectories: true)
            var archivedFiles: [String: String] = [:]
            var hashes: [String: String] = [:]
            for (index, source) in analysis.sources.enumerated() {
                let fileName = String(format: "%03d", index+1) + "." + source.url.pathExtension
                let target = sourceDirectory.appendingPathComponent(fileName)
                try FileManager.default.copyItem(at: source.url, to: target)
                archivedFiles[source.id] = "sources/" + fileName
                hashes[source.id] = try hash(target)
            }
            var archivedAnalysis = analysis
            archivedAnalysis.sources = analysis.sources.map { source in
                ArenaAnalyzedSource(id: source.id, url: directory.appendingPathComponent(archivedFiles[source.id]!),
                    width: source.width, height: source.height, transform: source.transform,
                    alignmentConfidence: source.alignmentConfidence, accepted: source.accepted, issues: source.issues)
            }
            let rendered = try render(archivedAnalysis, selections: selections, scale: 1)
            try write(rendered.image, to: directory.appendingPathComponent("reconstruction.tiff"))
            try write(rendered.map, to: directory.appendingPathComponent("source-map.tiff"))
            let manifest = ArenaExportManifest(width: analysis.width, height: analysis.height,
                imageFile: "reconstruction.tiff", sourceMapFile: "source-map.tiff",
                sourceMapIndices: Dictionary(uniqueKeysWithValues: analysis.sources.enumerated().map { ($0.element.id, $0.offset+1) }),
                sourceSHA256: hashes, sourceFiles: archivedFiles,
                analysis: analysis, selections: selections,
                warnings: analysis.warnings + ["Authenticity is unresolved. Agreement between sources does not prove original pixels.",
                    "Hard region seams may be visible. No blending or color corrections were applied.",
                    "Output is 8-bit sRGB; high-bit-depth and wide-gamut inputs are converted.",
                    "Source map values are one-based manifest source indices; zero denotes uncovered pixels.",
                    "Original source bytes are archived in sources/ with SHA256 identities; analysis records original file URLs.",
                    "Consensus may favor correlated edited copies; it is evidence of agreement, not independent provenance."],
                elapsedSeconds: Date().timeIntervalSince(start))
            let encoder = JSONEncoder(); encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            try encoder.encode(manifest).write(to: directory.appendingPathComponent("manifest.json"), options: .atomic)
            return manifest
        } catch {
            try? FileManager.default.removeItem(at: directory)
            throw error
        }
    }

    private static func render(_ analysis: ArenaAnalysis, selections: [ArenaSourceSelection], scale: Double) throws -> (image: CGImage, map: CGImage) {
        guard analysis.width > 0, analysis.height > 0, analysis.width <= 100_000, analysis.height <= 100_000,
              analysis.sources.count <= 254, Set(analysis.sources.map(\.id)).count == analysis.sources.count,
              Int64(analysis.width) * Int64(analysis.height) <= 200_000_000 else {
            throw failure("Invalid reconstruction dimensions or source count (maximum 200MP and 254 sources).")
        }
        let width = max(1, Int((Double(analysis.width)*scale).rounded())), height = max(1, Int((Double(analysis.height)*scale).rounded()))
        let colorSpace = CGColorSpace(name: CGColorSpace.sRGB)!
        guard let output = CGContext(data: nil, width: width, height: height, bitsPerComponent: 8, bytesPerRow: width*4,
            space: colorSpace, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue),
              let map = CGContext(data: nil, width: width, height: height, bitsPerComponent: 8, bytesPerRow: width,
                space: CGColorSpaceCreateDeviceGray(), bitmapInfo: CGImageAlphaInfo.none.rawValue) else {
            throw failure("Cannot allocate reconstruction buffers")
        }
        let context = CIContext(options: [.cacheIntermediates: false, .workingColorSpace: colorSpace])
        let canvas = CGRect(x: 0, y: 0, width: width, height: height)
        output.setBlendMode(.copy); output.interpolationQuality = .none
        for (sourceIndex, source) in analysis.sources.enumerated() {
            let regions = selections.filter { $0.sourceID == source.id }
            guard !regions.isEmpty else { continue }
            guard source.accepted else { throw failure("Selection references a rejected alignment") }
            try autoreleasepool {
                let image = try ArenaForensics.alignedImage(source)
                    .transformed(by: CGAffineTransform(scaleX: scale, y: scale))
                for region in regions {
                    guard region.rect.x.isFinite, region.rect.y.isFinite,
                          region.rect.width.isFinite, region.rect.height.isFinite else { throw failure("Nonfinite selection geometry") }
                    let rect = region.rect.cgRect.applying(CGAffineTransform(scaleX: scale, y: scale)).integral.intersection(canvas)
                    guard !rect.isNull, rect.width > 0, rect.height > 0 else { continue }
                    for y in stride(from: Int(rect.minY), to: Int(rect.maxY), by: 512) {
                        for x in stride(from: Int(rect.minX), to: Int(rect.maxX), by: 512) {
                            try autoreleasepool {
                                let tile = CGRect(x: x, y: y, width: min(512, Int(rect.maxX)-x), height: min(512, Int(rect.maxY)-y))
                                guard image.extent.contains(tile), let pixels = context.createCGImage(image, from: tile, format: .RGBA8, colorSpace: colorSpace) else {
                                    throw failure("Selected region extends outside source coverage")
                                }
                                output.draw(pixels, in: tile)
                            }
                        }
                    }
                    map.setFillColor(gray: Double(sourceIndex+1)/255, alpha: 1)
                    map.fill(rect)
                }
                context.clearCaches()
            }
        }
        guard selections.allSatisfy({ selection in analysis.sources.contains { $0.id == selection.sourceID } }),
              let image = output.makeImage(), let sourceMap = map.makeImage() else { throw failure("Invalid source selection") }
        return (image, sourceMap)
    }
    private static func hash(_ url: URL) throws -> String {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        var digest = SHA256()
        while let chunk = try handle.read(upToCount: 1024*1024), !chunk.isEmpty { digest.update(data: chunk) }
        return digest.finalize().map { String(format: "%02x", $0) }.joined()
    }
    private static func write(_ image: CGImage, to url: URL) throws {
        guard let destination = CGImageDestinationCreateWithURL(url as CFURL, UTType.tiff.identifier as CFString, 1, nil) else {
            throw failure("Cannot create TIFF export")
        }
        CGImageDestinationAddImage(destination, image, nil)
        guard CGImageDestinationFinalize(destination) else { throw failure("Cannot finish TIFF export") }
    }
    private static func failure(_ reason: String) -> NSError {
        NSError(domain: "PhotoArena.Reconstruction", code: 1, userInfo: [NSLocalizedDescriptionKey: reason])
    }
}
