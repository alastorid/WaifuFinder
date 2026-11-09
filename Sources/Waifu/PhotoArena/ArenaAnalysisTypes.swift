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
    var fileSHA256: String? = nil
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
    /// Native reference-pixel runs. Nil means this older analysis has no mask.
    var changeMask: ArenaPixelMask? = nil
    var maskDeferred: Bool? = nil
}

struct ArenaMaskRun: Codable, Hashable, Sendable {
    let y: Int
    let x: Int
    let length: Int
}

/// Sparse, lossless binary mask in bottom-left reference coordinates.
struct ArenaPixelMask: Codable, Sendable {
    var runs: [ArenaMaskRun]
    var pixelCount: Int { runs.reduce(0) { $0 + $1.length } }
    func contains(x: Int, y: Int) -> Bool {
        runs.contains { $0.y == y && x >= $0.x && x < $0.x + $0.length }
    }
    var bounds: CGRect {
        guard let first = runs.first else { return .zero }
        var x0 = first.x, x1 = first.x + first.length, y0 = first.y, y1 = first.y + 1
        for run in runs { x0 = min(x0, run.x); x1 = max(x1, run.x + run.length); y0 = min(y0, run.y); y1 = max(y1, run.y + 1) }
        return CGRect(x: x0, y: y0, width: x1-x0, height: y1-y0)
    }
    /// Four-connected components; isolated pixels remain visible in the full mask.
    var components: [ArenaPixelMask] {
        guard !runs.isEmpty else { return [] }
        var parent = Array(runs.indices)
        func root(_ index: Int) -> Int {
            var i = index
            while parent[i] != i { i = parent[i] }
            return i
        }
        var previous: [Int] = [], current: [Int] = [], row = runs[0].y
        for i in runs.indices {
            let run = runs[i]
            if run.y != row {
                previous = run.y == row + 1 ? current : []; current = []; row = run.y
            }
            for j in previous where runs[j].x < run.x + run.length && run.x < runs[j].x + runs[j].length {
                let a = root(i), b = root(j)
                if a != b { parent[a] = b }
            }
            current.append(i)
        }
        var groups: [Int: [ArenaMaskRun]] = [:]
        for i in runs.indices { groups[root(i), default: []].append(runs[i]) }
        return groups.values.map { ArenaPixelMask(runs: $0) }.sorted {
            $0.bounds.minY == $1.bounds.minY ? $0.bounds.minX < $1.bounds.minX : $0.bounds.minY < $1.bounds.minY
        }
    }
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
