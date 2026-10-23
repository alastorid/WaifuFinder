import CoreGraphics
import Foundation

/// ImageIO and Core Image rendering are synchronous native work. Serial actors
/// bound those operations even when rapid replacements cancel tasks faster than
/// an in-flight native call can return. Cancelled queued work never decodes.
actor CrystalOrbQueryDecoder {
    static let shared = CrystalOrbQueryDecoder()

    func read(url: URL?, data: Data?, maxPixelSize: Int? = nil, id: UUID = UUID()) throws -> CrystalOrbQuery {
        try Task.checkCancellation()
        let result: CrystalOrbQuery
        if let url { result = try CrystalOrbImageSource.read(url: url, maxPixelSize: maxPixelSize, id: id) }
        else if let data { result = try CrystalOrbImageSource.read(data: data, maxPixelSize: maxPixelSize, id: id) }
        else { throw CrystalOrbImageSource.failure }
        try Task.checkCancellation()
        return result
    }
}

actor CrystalOrbPairRenderer {
    static let shared = CrystalOrbPairRenderer()

    func render(target: CGImage, candidate: CGImage, alignment: CrystalOrbAlignment? = nil, settings: CrystalOrbDifferenceSettings = .init()) throws -> CrystalOrbDifference {
        try Task.checkCancellation()
        let result = try CrystalOrbDifferenceRenderer.render(target: target, candidate: candidate, alignment: alignment, settings: settings)
        try Task.checkCancellation()
        return result
    }
    func remap(_ difference: CrystalOrbDifference, settings: CrystalOrbDifferenceSettings) throws -> CrystalOrbDifference {
        try Task.checkCancellation()
        return try CrystalOrbDifferenceRenderer.remap(difference, settings: settings)
    }

}
