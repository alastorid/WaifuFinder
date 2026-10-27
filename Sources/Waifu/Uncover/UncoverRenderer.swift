import AVFoundation
import CoreImage
import Foundation

struct UncoverRenderedFrame: @unchecked Sendable {
    let original: CGImage
    let reconstructed: CGImage
    let index: Int
}

actor UncoverRenderer {
    private let context = CIContext(options: [.cacheIntermediates: false, .useSoftwareRenderer: true])

    func preview(source: URL, seconds: Double) async throws -> CGImage {
        try await PhotosMedia.shared.image(source, seconds: seconds)
    }

    func render(_ result: UncoverResult, index: Int, feather: Double, strength: Double) throws -> UncoverRenderedFrame {
        let manifest = result.prepared.manifest
        guard manifest.frames.indices.contains(index) else { throw UncoverError.invalid("Frame is outside the processed interval.") }
        let original = try RegionImages.read(RegionImages.frameURL(result.directory, "original", index))
        let b = try RegionImages.read(RegionImages.frameURL(result.directory, "reconstructed", index))
        let rect = CGRect(x: manifest.regionX, y: manifest.regionY, width: manifest.regionWidth, height: manifest.regionHeight)
        return UncoverRenderedFrame(original: original,
            reconstructed: try RegionImages.composite(original: original, region: b, rect: rect,
                feather: feather, strength: strength, context: context, outputScale: result.outputScale),
            index: index)
    }
}
