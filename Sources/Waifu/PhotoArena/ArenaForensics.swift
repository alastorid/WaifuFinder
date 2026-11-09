import CoreGraphics
import CoreImage
import CryptoKit
import Foundation
import ImageIO
import Vision
import simd

/// Deterministic pixel evidence. It deliberately does not classify AI edits from appearance.
enum ArenaForensics {
    private static let color = CGColorSpace(name: CGColorSpace.sRGB)!
    // Apple documents CIContext as immutable and safe to share across threads.
    // Older SDKs omit its Sendable annotation; keep that compatibility assertion
    // narrowly scoped instead of disabling checking for the CoreImage module.
    private final class SharedContext: @unchecked Sendable {
        let value = CIContext(options: [.cacheIntermediates: false])
    }
    private static let sharedContext = SharedContext()
    private static var context: CIContext { sharedContext.value }
    private static func failure(_ message: String) -> NSError {
        NSError(domain: "PhotoArena.Analysis", code: 1, userInfo: [NSLocalizedDescriptionKey: message])
    }
    static func image(_ source: ArenaAnalyzedSource) throws -> CIImage {
        guard source.url.isFileURL, let image = CIImage(contentsOf: source.url, options: [.applyOrientationProperty: true]) else {
            throw failure("PhotoArena analysis requires readable local image files: \(source.url.lastPathComponent)")
        }
        return image.transformed(by: .init(translationX: -image.extent.minX, y: -image.extent.minY))
    }
    static func alignedImage(_ source: ArenaAnalyzedSource) throws -> CIImage {
        let input = try image(source), t = source.transform
        if t.perspectiveX == 0 && t.perspectiveY == 0 { return input.transformed(by: t.cgAffine) }
        let w = Double(source.width), h = Double(source.height)
        return input.applyingFilter("CIPerspectiveTransform", parameters: [
            "inputBottomLeft": CIVector(cgPoint: t.point(.zero)),
            "inputBottomRight": CIVector(cgPoint: t.point(CGPoint(x: w, y: 0))),
            "inputTopLeft": CIVector(cgPoint: t.point(CGPoint(x: 0, y: h))),
            "inputTopRight": CIVector(cgPoint: t.point(CGPoint(x: w, y: h)))])
    }
    /// White denotes covered source pixels; black denotes invalid geometry/alpha.
    static func validityMask(_ source: ArenaAnalyzedSource, rect: CGRect) throws -> CGImage {
        let alpha = try alignedImage(source).applyingFilter("CIColorMatrix", parameters: [
            "inputRVector": CIVector(x: 0,y: 0,z: 0,w: 1),
            "inputGVector": CIVector(x: 0,y: 0,z: 0,w: 1),
            "inputBVector": CIVector(x: 0,y: 0,z: 0,w: 1),
            "inputAVector": CIVector(x: 0,y: 0,z: 0,w: 0),
            "inputBiasVector": CIVector(x: 0,y: 0,z: 0,w: 1)])
        guard let result = context.createCGImage(alpha, from: rect, format: .RGBA8, colorSpace: color) else { throw failure("Cannot render validity mask") }
        return result
    }
    /// Localized changed-pixel mask, not a learned manipulation/watermark classifier.
    static func differenceMask(_ source: ArenaAnalyzedSource, other: ArenaAnalyzedSource, rect: CGRect, tolerance: UInt8 = 12) throws -> CGImage {
        let a = bytes(try alignedImage(source), rect: rect), b = bytes(try alignedImage(other), rect: rect)
        var output = [UInt8](repeating: 255, count: a.count)
        let width = Int(rect.width), height = Int(rect.height)
        for y in 0..<height {
            for x in 0..<width {
                let offset = (y*width+x)*4
                let changed = a[offset+3] > 250 && b[offset+3] > 250 && (0..<3).contains { abs(Int(a[offset+$0])-Int(b[offset+$0])) > Int(tolerance) }
                let destination = ((height-1-y)*width+x)*4
                for c in 0..<3 { output[destination+c] = changed ? 255 : 0 }
            }
        }
        guard let provider = CGDataProvider(data: Data(output) as CFData), let result = CGImage(width: width,height: height,bitsPerComponent: 8,bitsPerPixel: 32,bytesPerRow: width*4,space: color,bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.last.rawValue),provider: provider,decode: nil,shouldInterpolate: false,intent: .defaultIntent) else { throw failure("Cannot render difference mask") }
        return result
    }
    static func crop(_ source: ArenaAnalyzedSource, rect: CGRect) throws -> CGImage {
        try crop(alignedImage(source), rect: rect)
    }
    static func crop(_ image: CIImage, rect: CGRect) throws -> CGImage {
        guard let output = context.createCGImage(image, from: rect, format: .RGBA8, colorSpace: color) else {
            throw failure("Cannot render aligned PhotoArena crop")
        }
        return output
    }
    static func pixelBytes(_ image: CGImage) -> [UInt8] {
        bytes(CIImage(cgImage: image), rect: CGRect(x: 0,y: 0,width: image.width,height: image.height))
    }
    private static func bytes(_ image: CIImage, rect: CGRect) -> [UInt8] {
        let width = Int(rect.width), height = Int(rect.height)
        var result = [UInt8](repeating: 0, count: width * height * 4)
        result.withUnsafeMutableBytes { raw in
            context.render(image, toBitmap: raw.baseAddress!, rowBytes: width * 4, bounds: rect, format: .RGBA8, colorSpace: color)
        }
        // CI bitmap rows are top-down. Public geometry and analysis arrays are bottom-up.
        for y in 0..<(height / 2) {
            for x in 0..<(width * 4) { result.swapAt(y*width*4+x, (height-1-y)*width*4+x) }
        }
        return result
    }
    private struct Match { let correlation: Double; let difference: Double; let overlap: Double }
    private static func match(_ a: [UInt8], _ b: [UInt8]) -> Match {
        var sa=0.0, sb=0.0, aa=0.0, bb=0.0, ab=0.0, delta=0.0, n=0.0
        for i in stride(from: 0, to: a.count, by: 16) where a[i+3] > 250 && b[i+3] > 250 {
            let x = Double(a[i])*0.2126+Double(a[i+1])*0.7152+Double(a[i+2])*0.0722
            let y = Double(b[i])*0.2126+Double(b[i+1])*0.7152+Double(b[i+2])*0.0722
            sa += x; sb += y; aa += x*x; bb += y*y; ab += x*y; delta += abs(x-y); n += 1
        }
        guard n > 64 else { return Match(correlation: 0, difference: 1, overlap: 0) }
        let va=aa-sa*sa/n, vb=bb-sb*sb/n
        let corr = va/n > 9 && vb/n > 9 ? (ab-sa*sb/n)/sqrt(va*vb) : 0
        return Match(correlation: max(0, min(1, corr)), difference: delta/n/255, overlap: n / Double((a.count+15)/16))
    }
    private static func matrix(_ transform: CGAffineTransform) -> simd_double3x3 {
        simd_double3x3(columns: (SIMD3(transform.a, transform.b, 0), SIMD3(transform.c, transform.d, 0), SIMD3(transform.tx, transform.ty, 1)))
    }
    private static func record(_ m: simd_double3x3) -> ArenaTransform? {
        let z = m[2][2]
        guard z.isFinite, abs(z) > 0.000001 else { return nil }
        let m = m * (1 / z)
        guard (0..<3).allSatisfy({ c in (0..<3).allSatisfy({ r in m[c][r].isFinite }) }) else { return nil }
        var result = ArenaTransform(CGAffineTransform(a: m[0][0], b: m[0][1], c: m[1][0], d: m[1][1], tx: m[2][0], ty: m[2][1]))
        result.perspectiveX = m[0][2]; result.perspectiveY = m[1][2]
        return result
    }
    private static func register(_ source: ArenaAnalyzedSource, reference: ArenaAnalyzedSource) throws -> ArenaAnalyzedSource {
        let scale = min(1, 768 / Double(max(reference.width, reference.height)))
        let bounds = CGRect(x: 0, y: 0, width: max(1, Int(Double(reference.width)*scale)), height: max(1, Int(Double(reference.height)*scale)))
        let s = CGAffineTransform(scaleX: scale, y: scale)
        let ref = try image(reference).transformed(by: s)
        let refBytes = bytes(ref, rect: bounds)
        var candidate = source
        let fit = min(Double(reference.width)/Double(source.width), Double(reference.height)/Double(source.height))
        var base = CGAffineTransform(scaleX: fit, y: fit).concatenating(.init(translationX: (Double(reference.width)-Double(source.width)*fit)/2, y: (Double(reference.height)-Double(source.height)*fit)/2))
        candidate.transform = ArenaTransform(base)
        var best = match(refBytes, bytes(try alignedImage(candidate).transformed(by: s), rect: bounds))
        // A smaller image may be a native crop rather than a resized full scene.
        // Evaluate both geometries before asking Vision for a projective correction.
        if source.width <= reference.width && source.height <= reference.height {
            let nativeBase = CGAffineTransform(translationX: Double(reference.width-source.width)/2, y: Double(reference.height-source.height)/2)
            var native = candidate; native.transform = ArenaTransform(nativeBase)
            let nativeScore = match(refBytes, bytes(try alignedImage(native).transformed(by: s), rect: bounds))
            if nativeScore.overlap >= 0.45 && nativeScore.correlation > best.correlation + 0.002 {
                candidate = native; best = nativeScore; base = nativeBase
            }
        }
        // Identical geometry does not require a costly learned registration request.
        if best.correlation < 0.998 || best.difference > 0.003 {
            let moving = try alignedImage(candidate).transformed(by: s)
            if let a = context.createCGImage(ref, from: bounds), let b = context.createCGImage(moving, from: bounds) {
                let request = VNHomographicImageRegistrationRequest(targetedCGImage: b, options: [:])
                try? VNImageRequestHandler(cgImage: a, options: [:]).perform([request])
                if let warp = request.results?.first?.warpTransform {
                    let h = simd_double3x3(columns: (SIMD3(Double(warp[0][0]),Double(warp[0][1]),Double(warp[0][2])), SIMD3(Double(warp[1][0]),Double(warp[1][1]),Double(warp[1][2])), SIMD3(Double(warp[2][0]),Double(warp[2][1]),Double(warp[2][2]))))
                    // Evaluate both documented pixel orientation and flipped image convention.
                    let flip = matrix(CGAffineTransform(a: 1,b: 0,c: 0,d: -1,tx: 0,ty: bounds.height))
                    for correction in [h, flip*h*flip] {
                        guard abs(simd_determinant(correction)) > 0.000001,
                              let t = record(matrix(s.inverted())*correction*matrix(s)*matrix(base)) else { continue }
                        var trial = candidate; trial.transform = t
                        let corners = [CGPoint.zero, CGPoint(x: source.width,y: 0), CGPoint(x: 0,y: source.height), CGPoint(x: source.width,y: source.height)].map(t.point)
                        guard corners.allSatisfy({ $0.x.isFinite && $0.y.isFinite && abs($0.x) < Double(reference.width)*4 && abs($0.y) < Double(reference.height)*4 }) else { continue }
                        let score = match(refBytes, bytes(try alignedImage(trial).transformed(by: s), rect: bounds))
                        if score.overlap >= 0.45 && score.correlation > best.correlation + 0.002 {
                            candidate = trial; best = score
                        }
                    }
                }
            }
        }
        candidate.alignmentConfidence = best.correlation
        candidate.accepted = best.correlation >= 0.90 && best.difference <= 0.18 && best.overlap >= 0.45
        candidate.issues = candidate.accepted ? ["Global alignment only; local distortions remain unverified."] : ["Registration rejected: insufficient textured correspondence, overlap or photometric agreement."]
        if candidate.transform.a*candidate.transform.d-candidate.transform.b*candidate.transform.c > 1.002 { candidate.issues.append("Lower-resolution source requires conventional geometric resampling; no additional detail is recovered.") }
        return candidate
    }
    private static func textBoxes(_ source: ArenaAnalyzedSource) -> [CGRect] {
        guard let input = try? alignedImage(source) else { return [] }
        let bounds = input.extent
        let scale = min(1, 1280 / max(bounds.width, bounds.height))
        let reduced = input.transformed(by: .init(scaleX: scale, y: scale))
        guard let image = context.createCGImage(reduced, from: reduced.extent) else { return [] }
        let request = VNRecognizeTextRequest(); request.recognitionLevel = .fast; request.usesLanguageCorrection = false
        guard (try? VNImageRequestHandler(cgImage: image).perform([request])) != nil else { return [] }
        return (request.results ?? []).map { observation in
            let r = observation.boundingBox
            return CGRect(x: bounds.minX+r.minX*bounds.width, y: bounds.minY+r.minY*bounds.height, width: r.width*bounds.width, height: r.height*bounds.height)
        }
    }
    static func analyze(_ inputs: [ArenaSource], tileSize: Int = 448) async throws -> ArenaAnalysis {
        let task = Task.detached(priority: .userInitiated) { try analyzeFiles(inputs, tileSize: tileSize) }
        return try await withTaskCancellationHandler { try await task.value } onCancel: { task.cancel() }
    }
    static func analyzeFiles(_ inputs: [ArenaSource], tileSize: Int = 448) throws -> ArenaAnalysis {
        let start = Date()
        guard inputs.count >= 2, inputs.count <= 16, Set(inputs.map(\.id)).count == inputs.count, tileSize >= 32, tileSize <= 1024 else {
            throw failure("Select 2–16 distinct photographs; analysis tiles must be 32–1024 pixels.")
        }
        var sources: [ArenaAnalyzedSource] = try inputs.map { source in
            guard source.url.isFileURL, let container = CGImageSourceCreateWithURL(source.url as CFURL, [kCGImageSourceShouldCache: false] as CFDictionary),
                  let properties = CGImageSourceCopyPropertiesAtIndex(container, 0, nil) as? [CFString: Any],
                  let w = properties[kCGImagePropertyPixelWidth] as? Int, let h = properties[kCGImagePropertyPixelHeight] as? Int,
                  w > 0, h > 0, Double(w)*Double(h) <= 160_000_000 else { throw failure("Unreadable or unsupported image: \(source.url.lastPathComponent)") }
            let rotated = (5...8).contains((properties[kCGImagePropertyOrientation] as? Int) ?? 1)
            let handle = try FileHandle(forReadingFrom: source.url); defer { try? handle.close() }
            var digest = SHA256()
            while let data = try handle.read(upToCount: 1024*1024), !data.isEmpty { try Task.checkCancellation(); digest.update(data: data) }
            return ArenaAnalyzedSource(id: source.id,url: source.url,width: rotated ? h : w,height: rotated ? w : h,
                transform: ArenaTransform(),alignmentConfidence: 1,accepted: true,issues: [],
                fileSHA256: digest.finalize().map { String(format: "%02x", $0) }.joined())
        }
        let reference = sources.max { $0.width*$0.height < $1.width*$1.height }!
        var registered: [String: ArenaAnalyzedSource] = [:]
        if let hash = reference.fileSHA256 { registered[hash] = reference }
        for i in sources.indices where sources[i].id != reference.id {
            try Task.checkCancellation()
            if let hash = sources[i].fileSHA256, let identical = registered[hash] {
                sources[i].transform = identical.transform; sources[i].accepted = identical.accepted
                sources[i].alignmentConfidence = identical.alignmentConfidence
                sources[i].issues = identical.issues + ["Byte-identical source; registration reused."]
            } else {
                sources[i] = try register(sources[i], reference: reference)
                if let hash = sources[i].fileSHA256 { registered[hash] = sources[i] }
            }
        }
        let accepted = sources.filter(\.accepted)
        let images = try accepted.map(alignedImage)
        let text = accepted.map(textBoxes)
        var seen: Set<String> = []
        let distinct = accepted.map { seen.insert($0.fileSHA256 ?? $0.id).inserted }
        var regions: [ArenaRegion] = []
        var retainedRuns = 0
        var deferredMasks = 0
        for y in stride(from: 0, to: reference.height, by: tileSize) {
            for x in stride(from: 0, to: reference.width, by: tileSize) {
                try Task.checkCancellation()
                let rect = CGRect(x: x,y: y,width: min(tileSize,reference.width-x),height: min(tileSize,reference.height-y))
                let buffers = images.map { bytes($0, rect: rect) }
                let count = Int(rect.width*rect.height), width = Int(rect.width)
                var differences = [Double](repeating: 0,count: accepted.count), sharpness = differences, clipped = differences, valid = differences
                var available = [Int](); available.reserveCapacity(accepted.count)
                var channelValues = [Double](); channelValues.reserveCapacity(accepted.count)
                var median = [Double](repeating: 0, count: 3)
                var changed = [Bool](repeating: false, count: count)
                for pixel in 0..<count {
                    let offset = pixel*4
                    available.removeAll(keepingCapacity: true)
                    for index in buffers.indices where buffers[index][offset+3] > 250 { available.append(index) }
                    guard !available.isEmpty else { continue }
                    for c in 0..<3 {
                        channelValues.removeAll(keepingCapacity: true)
                        for index in available where distinct[index] { channelValues.append(Double(buffers[index][offset+c])) }
                        channelValues.sort()
                        if channelValues.count >= 2 && channelValues.last! - channelValues.first! > 12 { changed[pixel] = true }
                        let middle = channelValues.count/2
                        median[c] = channelValues.count%2 == 0 ? (channelValues[middle-1]+channelValues[middle])/2 : channelValues[middle]
                    }
                    for index in available {
                        let b = buffers[index]
                        valid[index] += 1
                        differences[index] += (0..<3).reduce(0.0) { $0+abs(Double(b[offset+$1])-median[$1])/765 }
                        if b[offset] < 2 || b[offset] > 253 || b[offset+1] < 2 || b[offset+1] > 253 || b[offset+2] < 2 || b[offset+2] > 253 { clipped[index] += 1 }
                        if pixel%width > 0 && pixel >= width {
                            sharpness[index] += (abs(Double(b[offset])-Double(b[offset-4]))+abs(Double(b[offset])-Double(b[offset-width*4])))/510
                        }
                    }
                }
                let measurements = accepted.indices.map { i in
                    let textArea = min(1, text[i].reduce(0.0) { $0 + $1.intersection(rect).standardized.areaIfNonempty } / Double(count))
                    return ArenaRegionMeasurement(sourceID: accepted[i].id,difference: differences[i]/max(1,valid[i]),sharpness: sharpness[i]/max(1,valid[i]),
                        clipping: clipped[i]/max(1,valid[i]),validFraction: valid[i]/Double(count),textCandidateFraction: textArea,
                        evidence: ["Residual is RGB deviation from cross-source pixel median; agreement is not proof of originality.","Sharpness includes noise and sharpening; it is not authenticated detail."] + (textArea > 0 ? ["OCR text candidate: may be scene text, not a confirmed watermark."] : []))
                }
                var runs: [ArenaMaskRun] = []
                for row in 0..<Int(rect.height) {
                    var column = 0
                    while column < width {
                        guard changed[row*width+column] else { column += 1; continue }
                        let start = column
                        while column < width && changed[row*width+column] { column += 1 }
                        runs.append(ArenaMaskRun(y: y+row, x: x+start, length: column-start))
                    }
                }
                let keepMask = retainedRuns + runs.count <= 500_000
                if keepMask { retainedRuns += runs.count } else { deferredMasks += 1 }
                regions.append(ArenaRegion(id: "region_\(x)_\(y)_\(Int(rect.width))_\(Int(rect.height))",rect: ArenaPixelRect(rect),measurements: measurements,unresolved: true,
                    evidence: [accepted.count < 2 ? "Fewer than two registered sources cover this analysis." : "Pixel evidence cannot establish original provenance; shared edits remain possible."] +
                        (keepMask ? [] : ["Precise mask deferred: the 500,000-run memory budget was reached; this region is not identical."]),
                    changeMask: keepMask ? ArenaPixelMask(runs: runs) : nil, maskDeferred: keepMask ? nil : true))
            }
        }
        return ArenaAnalysis(referenceID: reference.id,width: reference.width,height: reference.height,sources: sources,regions: regions,
            warnings: ["Authenticity and watermark classification remain unverified. OCR and residuals are candidate evidence only.","Global homography cannot validate parallax or local deformation. Tile boundaries group exact changed-pixel runs; they are not object segmentation."] +
                (deferredMasks == 0 ? [] : ["\(deferredMasks) precise masks deferred by the memory budget; measurements remain available and these regions cannot be automatically patched."]),elapsedSeconds: Date().timeIntervalSince(start))
    }
    static func differenceOverlay(_ analysis: ArenaAnalysis, sourceID: String, maxPixelSize: Int = 1600) throws -> CGImage {
        guard let source = analysis.sources.first(where: { $0.id == sourceID && $0.accepted }), let reference = analysis.sources.first(where: { $0.id == analysis.referenceID }) else { throw failure("Registered source unavailable") }
        let rect = CGRect(x: 0,y: 0,width: analysis.width,height: analysis.height)
        let diff = try alignedImage(source).applyingFilter("CIDifferenceBlendMode", parameters: [kCIInputBackgroundImageKey: alignedImage(reference)]).cropped(to: rect)
        let scale = min(1,Double(maxPixelSize)/Double(max(analysis.width,analysis.height)))
        guard let result = context.createCGImage(diff.transformed(by: .init(scaleX: scale,y: scale)),from: rect.applying(.init(scaleX: scale,y: scale))) else { throw failure("Cannot render difference overlay") }
        return result
    }
}
private extension CGRect {
    var areaIfNonempty: Double { isNull || isEmpty ? 0 : width*height }
}
