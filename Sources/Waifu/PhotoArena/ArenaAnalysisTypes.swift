import CoreGraphics
import Foundation

/// All geometry uses EXIF-oriented native pixel edges with a bottom-left origin.
struct ArenaPixelRect: Codable, Hashable, Sendable {
    var x: Double; var y: Double; var width: Double; var height: Double
    init(_ rect: CGRect) { x = rect.minX; y = rect.minY; width = rect.width; height = rect.height }
    var cgRect: CGRect { CGRect(x: x, y: y, width: width, height: height) }
}
struct ArenaTransform: Codable, Hashable, Sendable {
    var a: Double = 1; var b: Double = 0; var c: Double = 0; var d: Double = 1
    var tx: Double = 0; var ty: Double = 0
    var perspectiveX: Double = 0; var perspectiveY: Double = 0
    func point(_ p: CGPoint) -> CGPoint {
        let denominator = perspectiveX * p.x + perspectiveY * p.y + 1
        return CGPoint(x: (a*p.x+c*p.y+tx)/denominator, y: (b*p.x+d*p.y+ty)/denominator)
    }
    init(_ value: CGAffineTransform = .identity) { a = value.a; b = value.b; c = value.c; d = value.d; tx = value.tx; ty = value.ty }
    var cgAffine: CGAffineTransform { CGAffineTransform(a: a, b: b, c: c, d: d, tx: tx, ty: ty) }
}
struct ArenaAnalyzedSource: Codable, Identifiable, Sendable {
    let id: String
    let url: URL
    let width: Int
    let height: Int
    var transform: ArenaTransform
    var alignmentConfidence: Double
    var accepted: Bool
    var issues: [String]
}
struct ArenaRegionMeasurement: Codable, Sendable {
    let sourceID: String
    var difference: Double
    var sharpness: Double
    var clipping: Double
    var validFraction: Double
    var textCandidateFraction: Double
    var evidence: [String]
}
struct ArenaRegion: Codable, Identifiable, Sendable {
    let id: String
    let rect: ArenaPixelRect
    var measurements: [ArenaRegionMeasurement]
    var unresolved: Bool
    var evidence: [String]
}
struct ArenaAnalysis: Codable, Sendable {
    var schemaVersion = 1
    let referenceID: String
    let width: Int
    let height: Int
    var sources: [ArenaAnalyzedSource]
    var regions: [ArenaRegion]
    var warnings: [String]
    var elapsedSeconds: Double
}
