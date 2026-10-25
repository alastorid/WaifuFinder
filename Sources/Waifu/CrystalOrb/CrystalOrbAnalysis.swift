import CoreImage
import Foundation

struct CrystalOrbStatistics: Sendable {
    var overlap = 0
    var changed = 0
    var covered = 0
    var sum: Double = 0
    var maximum: Double = 0
    var mean: Double { overlap == 0 ? 0 : sum / Double(overlap) }
    var percent: Double { overlap == 0 ? 0 : 100 * Double(changed) / Double(overlap) }
    var coverage: Double { covered == 0 ? 0 : 100 * Double(overlap) / Double(covered) }
}
struct CrystalOrbHotspot: Identifiable, Sendable { let id: Int; let rect: CGRect; let count: Int }
struct CrystalOrbMeasurements: Sendable {
    let differenceID: UUID
    let full: CrystalOrbStatistics
    let region: CrystalOrbStatistics?
    let hotspots: [CrystalOrbHotspot]
}
struct CrystalOrbPixel: Sendable {
    let point: CGPoint
    let targetPoint: CGPoint?
    let candidatePoint: CGPoint?
    let a: [UInt8]
    let b: [UInt8]
    let delta: [UInt8]
    let luma: Double
    let significant: Bool
    let targetGrid: CGImage
    let candidateGrid: CGImage
}

actor CrystalOrbAnalysis {
    static let shared = CrystalOrbAnalysis()
    private var cached: (UUID, Double, CrystalOrbDifferenceSettings.Channel, CGRect?, CrystalOrbMeasurements)?
    static func bytes(_ image: CIImage, rect: CGRect) -> [UInt8] {
        let width = Int(rect.width), height = Int(rect.height)
        var pixels = [UInt8](repeating: 0, count: width * height * 4)
        pixels.withUnsafeMutableBytes {
            CrystalOrbDifferenceRenderer.context.value.render(image, toBitmap: $0.baseAddress!, rowBytes: width * 4,
                bounds: rect, format: .RGBA8, colorSpace: CrystalOrbDifferenceRenderer.color)
        }
        // CIContext writes bitmap rows top-to-bottom even though bounds are in
        // bottom-left image coordinates. All analysis arrays use bottom-up rows.
        let rowBytes = width * 4
        for y in 0..<(height / 2) {
            let opposite = height - 1 - y
            for x in 0..<rowBytes { pixels.swapAt(y*rowBytes+x, opposite*rowBytes+x) }
        }
        return pixels
    }
    func measure(_ difference: CrystalOrbDifference, roi: CGRect?) throws -> CrystalOrbMeasurements {
        if let cached, cached.0 == difference.measurementID, cached.1 == difference.settings.tolerance,
           cached.2 == difference.settings.channel, cached.3 == roi {
            return CrystalOrbMeasurements(differenceID: difference.id, full: cached.4.full, region: cached.4.region, hotspots: cached.4.hotspots)
        }
        let extent = difference.alignment.extent
        let width = Int(extent.width), height = Int(extent.height)
        var full = CrystalOrbStatistics(), region = CrystalOrbStatistics()
        let cell = max(1, Int(ceil(max(extent.width, extent.height) / 128)))
        let cols = (width + cell - 1) / cell, rows = (height + cell - 1) / cell
        var counts = [Int](repeating: 0, count: cols * rows)
        for start in stride(from: 0, to: height, by: 64) {
            try Task.checkCancellation()
            let rect = CGRect(x: 0, y: start, width: width, height: min(64, height - start))
            let a = Self.bytes(difference.target, rect: rect)
            let b = Self.bytes(difference.candidate, rect: rect)
            let overlap = Self.bytes(difference.overlap, rect: rect)
            let union = Self.bytes(difference.union, rect: rect)
            for y in 0..<Int(rect.height) {
                for x in 0..<width {
                    let offset = (y * width + x) * 4
                    let p = CGPoint(x: CGFloat(x) + 0.5, y: CGFloat(y + start) + 0.5)
                    let inROI = roi?.contains(p) == true
                    if union[offset] > 127 { full.covered += 1; if inROI { region.covered += 1 } }
                    guard overlap[offset] > 254 else { continue }
                    let dr = Int(a[offset]) - Int(b[offset])
                    let dg = Int(a[offset+1]) - Int(b[offset+1])
                    let db = Int(a[offset+2]) - Int(b[offset+2])
                    let units: Double
                    switch difference.settings.channel {
                    case .rgb: units = Double(max(abs(dr),abs(dg),abs(db)))
                    case .red: units = Double(abs(dr))
                    case .green: units = Double(abs(dg))
                    case .blue: units = Double(abs(db))
                    case .luminance: units = abs(Double(dr)*0.2126+Double(dg)*0.7152+Double(db)*0.0722).rounded()
                    }
                    let value = units / 255
                    let changed = units > difference.settings.tolerance
                    func add(_ stats: inout CrystalOrbStatistics) {
                        stats.overlap += 1; stats.sum += value; stats.maximum = max(stats.maximum, value)
                        if changed { stats.changed += 1 }
                    }
                    add(&full); if inROI { add(&region) }
                    if changed { counts[((y + start) / cell) * cols + x / cell] += 1 }
                }
            }
        }
        // Connected occupied cells bound memory independently of image resolution.
        // Counts remain exact; boxes are conservatively expanded to cell boundaries.
        var visited = Set<Int>(), hotspots: [CrystalOrbHotspot] = []
        for index in counts.indices where counts[index] > 0 && !visited.contains(index) {
            var queue = [index], cursor = 0, count = 0
            var box = CGRect.null
            visited.insert(index)
            while cursor < queue.count {
                let k = queue[cursor]; cursor += 1; count += counts[k]
                let x = k % cols, y = k / cols
                box = box.union(CGRect(x: x * cell, y: y * cell, width: cell, height: cell))
                for (nx, ny) in [(x-1,y),(x+1,y),(x,y-1),(x,y+1)] where nx >= 0 && ny >= 0 && nx < cols && ny < rows {
                    let next = ny * cols + nx
                    if counts[next] > 0 && visited.insert(next).inserted { queue.append(next) }
                }
            }
            if count >= max(4, full.overlap / 100_000) { hotspots.append(.init(id: index, rect: box.intersection(extent), count: count)) }
        }
        let result = CrystalOrbMeasurements(differenceID: difference.id, full: full, region: roi == nil ? nil : region,
            hotspots: Array(hotspots.sorted { $0.count > $1.count }.prefix(12)))
        cached = (difference.measurementID, difference.settings.tolerance, difference.settings.channel, roi, result)
        return result
    }
    nonisolated static func sample(_ difference: CrystalOrbDifference, point: CGPoint) throws -> CrystalOrbPixel {
        try Task.checkCancellation()
        let p = CGPoint(x: floor(point.x) + 0.5, y: floor(point.y) + 0.5)
        let rect = CGRect(x: floor(p.x) - 5, y: floor(p.y) - 5, width: 11, height: 11)
        let a = Self.bytes(difference.target, rect: rect), b = Self.bytes(difference.candidate, rect: rect)
        let offset = (5 * 11 + 5) * 4
        let av = Array(a[offset..<offset+3]), bv = Array(b[offset..<offset+3])
        let delta = zip(av, bv).map { UInt8(abs(Int($0) - Int($1))) }
        let luma = abs(zip(av, bv).enumerated().reduce(0.0) { $0 + (Double($1.element.0) - Double($1.element.1)) * [0.2126, 0.7152, 0.0722][$1.offset] }) / 255
        let value: Double
        switch difference.settings.channel {
        case .rgb: value = Double(delta.max() ?? 0)
        case .red: value = Double(delta[0])
        case .green: value = Double(delta[1])
        case .blue: value = Double(delta[2])
        case .luminance: value = (luma * 255).rounded()
        }
        guard let ag = CrystalOrbDifferenceRenderer.context.value.createCGImage(difference.target, from: rect, format: .RGBA8, colorSpace: CrystalOrbDifferenceRenderer.color),
              let bg = CrystalOrbDifferenceRenderer.context.value.createCGImage(difference.candidate, from: rect, format: .RGBA8, colorSpace: CrystalOrbDifferenceRenderer.color) else { throw CrystalOrbImageSource.failure }
        return CrystalOrbPixel(point: p, targetPoint: difference.alignment.source(p, candidate: false), candidatePoint: difference.alignment.source(p, candidate: true),
            a: av, b: bv, delta: delta, luma: luma, significant: value > difference.settings.tolerance, targetGrid: ag, candidateGrid: bg)
    }
}

struct CrystalOrbRegistrationResult: Sendable { let registration: CrystalOrbRegistration; let correlation: Double }
actor CrystalOrbRegistrar {
    static let shared = CrystalOrbRegistrar()
    private struct Key: Equatable { let query: UUID; let frame: FrameID; let generation: UUID?; let aWidth: Int; let bWidth: Int }
    private var cache: [(Key, CrystalOrbRegistrationResult)] = []
    func align(target: CGImage, candidate: CGImage, geometry: CrystalOrbAlignment, query: UUID, frame: FrameID, generation: UUID? = nil) throws -> CrystalOrbRegistrationResult? {
        let key = Key(query: query, frame: frame, generation: generation, aWidth: target.width, bWidth: candidate.width)
        if let cached = cache.first(where: { $0.0 == key }) { return cached.1 }
        let base = CrystalOrbAlignment(target: geometry.targetSize, candidate: geometry.candidateSize)
        let a = CrystalOrbDifferenceRenderer.source(target, alignment: base, candidate: false).0
        let b = CrystalOrbDifferenceRenderer.source(candidate, alignment: base, candidate: true).0
        var dx = 0, dy = 0, lastScale: CGFloat = 1, quality = -1.0
        var fractionalX: CGFloat = 0, fractionalY: CGFloat = 0
        for limit in [128, 384, 1024] {
            try Task.checkCancellation()
            let scale = min(1, CGFloat(limit) / max(base.size.width, base.size.height))
            let w = max(1, Int(base.size.width * scale)), h = max(1, Int(base.size.height * scale))
            let rect = CGRect(x: 0, y: 0, width: w, height: h)
            let transform = CGAffineTransform(scaleX: scale, y: scale)
            func gray(_ image: CIImage) -> [Double] {
                let bytes = CrystalOrbAnalysis.bytes(image.transformed(by: transform), rect: rect)
                return stride(from: 0, to: bytes.count, by: 4).map { Double(bytes[$0]) * 0.2126 + Double(bytes[$0+1]) * 0.7152 + Double(bytes[$0+2]) * 0.0722 }
            }
            let ap = gray(a), bp = gray(b)
            let cx = Int((CGFloat(dx) * scale / lastScale).rounded()), cy = Int((CGFloat(dy) * scale / lastScale).rounded())
            let radius = limit == 128 ? min(16, min(w,h) / 5) : 4
            let strideSize = max(1, max(w,h) / 48)
            func score(_ x: Int, _ y: Int) -> Double {
                var sa = 0.0, sb = 0.0, aa = 0.0, bb = 0.0, ab = 0.0, n = 0.0
                let minX = max(0, x, Int(ceil(base.target.minX*scale-0.5)), Int(ceil(base.candidate.minX*scale-0.5))+x)
                let maxX = min(w, w+x, Int(ceil(base.target.maxX*scale-0.5)), Int(ceil(base.candidate.maxX*scale-0.5))+x)
                let minY = max(0, y, Int(ceil(base.target.minY*scale-0.5)), Int(ceil(base.candidate.minY*scale-0.5))+y)
                let maxY = min(h, h+y, Int(ceil(base.target.maxY*scale-0.5)), Int(ceil(base.candidate.maxY*scale-0.5))+y)
                guard maxX > minX, maxY > minY else { return -1 }
                for py in stride(from: minY, to: maxY, by: strideSize) {
                    for px in stride(from: minX, to: maxX, by: strideSize) {
                        let av = ap[py*w+px], bv = bp[(py-y)*w+px-x]
                        sa += av; sb += bv; aa += av*av; bb += bv*bv; ab += av*bv; n += 1
                    }
                }
                guard n > 64 else { return -1 }
                let va = aa-sa*sa/n, vb = bb-sb*sb/n
                guard va/n > 4, vb/n > 4 else { return -1 } // textureless inputs cannot register reliably
                return (ab-sa*sb/n) / sqrt(va*vb)
            }
            var best = -1.0, second = -1.0
            for y in (cy-radius)...(cy+radius) {
                try Task.checkCancellation()
                for x in (cx-radius)...(cx+radius) {
                    let value = score(x,y)
                    if value > best { second = best; best = value; dx = x; dy = y }
                    else { second = max(second,value) }
                }
            }
            if limit == 1024 && scale < 1 {
                func refine(_ low: Double, _ high: Double) -> CGFloat {
                    let curvature = low - 2*best + high
                    guard curvature < -0.000001 else { return 0 }
                    return CGFloat(max(-0.5, min(0.5, 0.5 * (low-high) / curvature)))
                }
                fractionalX = refine(score(dx-1,dy),score(dx+1,dy))
                fractionalY = refine(score(dx,dy-1),score(dx,dy+1))
            }
            quality = best; lastScale = scale
            // An ambiguous coarse peak is not enough to apply a transform.
            if limit == 128 && (best < 0.55 || best - second < 0.00001) { return nil }
        }
        guard quality >= 0.8 else { return nil }
        let result = CrystalOrbRegistrationResult(registration: .init(x: (CGFloat(dx)+fractionalX)/lastScale, y: (CGFloat(dy)+fractionalY)/lastScale), correlation: quality)
        cache.append((key,result)); if cache.count > 12 { cache.removeFirst() }
        return result
    }
}

actor CrystalOrbPixelReader {
    static let shared = CrystalOrbPixelReader()
    func sample(_ difference: CrystalOrbDifference, point: CGPoint) throws -> CrystalOrbPixel {
        try CrystalOrbAnalysis.sample(difference, point: point)
    }
}
