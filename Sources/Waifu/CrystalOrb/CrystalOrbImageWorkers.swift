import CoreGraphics
import Foundation

/// ImageIO and Core Image rendering are synchronous native work. Serial actors
/// bound those operations even when rapid replacements cancel tasks faster than
/// an in-flight native call can return. Cancelled queued work never decodes.
actor CrystalOrbQueryDecoder {
    static let shared = CrystalOrbQueryDecoder()

    func read(url: URL?, data: Data?) throws -> CrystalOrbQuery {
        try Task.checkCancellation()
        let result: CrystalOrbQuery
        if let url { result = try CrystalOrbImageSource.read(url: url) }
        else if let data { result = try CrystalOrbImageSource.read(data: data) }
        else { throw CrystalOrbImageSource.failure }
        try Task.checkCancellation()
        return result
    }
}

actor CrystalOrbPairRenderer {
    static let shared = CrystalOrbPairRenderer()

    func render(target: CGImage, candidate: CGImage) throws -> CrystalOrbDifference {
        try Task.checkCancellation()
        let result = try CrystalOrbDifferenceRenderer.render(target: target, candidate: candidate)
        try Task.checkCancellation()
        return result
    }
}
