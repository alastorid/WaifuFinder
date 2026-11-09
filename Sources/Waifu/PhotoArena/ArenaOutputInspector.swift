import CoreGraphics
import CoreImage
import Foundation

struct ArenaOutputDifference: Codable, Sendable {
    let sourceID: String
    let comparedPixels: Int
    let changedPixels: Int
    let selectedFromOtherSourcePixels: Int
    let meanAbsoluteError: Double
    let maximumChannelError: Int
    let uncoveredPixels: Int
}

/// Replays the actual source selections in reference coordinates. Output/source masks
/// are always recomputed, never borrowed from original source-to-source masks.
enum ArenaOutputInspector {
    static func detectedOverlay(_ analysis: ArenaAnalysis, tolerance: Int = 12, opacity: Double = 0.55, maxPixelSize: Int = 1600) throws -> CGImage {
        let scale = min(1,Double(maxPixelSize)/Double(max(analysis.width,analysis.height)))
        let width = max(1,Int((Double(analysis.width)*scale).rounded())), height = max(1,Int((Double(analysis.height)*scale).rounded()))
        guard let overlay = CGContext(data:nil,width:width,height:height,bitsPerComponent:8,bytesPerRow:width*4,
            space:CGColorSpace(name:CGColorSpace.sRGB)!,bitmapInfo:CGImageAlphaInfo.premultipliedLast.rawValue) else { throw ModelInstaller.fail("Cannot allocate detected differences") }
        overlay.scaleBy(x:scale,y:scale); overlay.setFillColor(red:1,green:0,blue:0,alpha:opacity)
        for region in analysis.regions {
            try Task.checkCancellation()
            if tolerance == 12, let mask = region.changeMask {
                let spans = mask.runs.map { CGRect(x:$0.x,y:$0.y,width:$0.length,height:1) }
                overlay.fill(spans); continue
            }
            try autoreleasepool {
                let rect = region.rect.cgRect, w = Int(rect.width), h = Int(rect.height)
                let buffers = try analysis.sources.filter(\.accepted).map { pixels(try ArenaForensics.crop($0,rect:rect)) }
                for y in 0..<h {
                    var start: Int?
                    for x in 0...w {
                        var changed = false
                        if x < w {
                            let offset = (y*w+x)*4
                            var low = SIMD3<Int>(repeating: 255), high = SIMD3<Int>(repeating: 0), count = 0
                            for buffer in buffers where buffer[offset+3] > 250 {
                                count += 1
                                for c in 0..<3 { let value = Int(buffer[offset+c]); low[c] = min(low[c],value); high[c] = max(high[c],value) }
                            }
                            changed = count > 1 && (high.x-low.x > tolerance || high.y-low.y > tolerance || high.z-low.z > tolerance)
                        }
                        if changed && start == nil { start = x }
                        if !changed, let first = start {
                            overlay.fill(CGRect(x:rect.minX+Double(first),y:rect.minY+Double(y),width:Double(x-first),height:1)); start = nil
                        }
                    }
                }
            }
        }
        guard let image = overlay.makeImage() else { throw ModelInstaller.fail("Cannot finish detected differences") }
        return image
    }
    static func tile(_ analysis: ArenaAnalysis, selections: [ArenaSourceSelection], rect: CGRect) throws -> CGImage {
        let images = try Dictionary(uniqueKeysWithValues: analysis.sources.filter { source in selections.contains { $0.sourceID == source.id } }
            .map { ($0.id, try ArenaForensics.alignedImage($0)) })
        return try tile(images: images, selections: selections, rect: rect)
    }
    private static func tile(images: [String:CIImage], selections: [ArenaSourceSelection], rect: CGRect) throws -> CGImage {
        let color = CGColorSpace(name: CGColorSpace.sRGB)!
        guard let canvas = CGContext(data: nil, width: Int(rect.width), height: Int(rect.height), bitsPerComponent: 8,
            bytesPerRow: Int(rect.width)*4, space: color, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else {
            throw ModelInstaller.fail("Cannot allocate PhotoArena inspection tile")
        }
        canvas.translateBy(x: -rect.minX, y: -rect.minY); canvas.setBlendMode(.copy); canvas.interpolationQuality = .none
        for selection in selections {
            let bounds = selection.rect.cgRect.intersection(rect)
            guard !bounds.isNull, !bounds.isEmpty,
                  let source = images[selection.sourceID] else { continue }
            canvas.saveGState()
            if let mask = selection.mask {
                let spans = mask.runs.filter { Double($0.y) >= bounds.minY && Double($0.y) < bounds.maxY }
                    .map { CGRect(x: $0.x, y: $0.y, width: $0.length, height: 1) }
                canvas.addRects(spans); canvas.clip()
            }
            canvas.draw(try ArenaForensics.crop(source, rect: bounds), in: bounds)
            canvas.restoreGState()
        }
        guard let image = canvas.makeImage() else { throw ModelInstaller.fail("Cannot render PhotoArena inspection tile") }
        return image
    }
    static func pixels(_ image: CGImage) -> [UInt8] {
        ArenaForensics.pixelBytes(image)
    }
    static func compare(_ analysis: ArenaAnalysis, selections: [ArenaSourceSelection], sourceID: String,
                        tolerance: Int = 12, opacity: Double = 0.55, maxPixelSize: Int = 1600,
                        heatmap: Bool = false,
                        renderedOutput: CIImage? = nil) throws -> (ArenaOutputDifference, CGImage) {
        guard let source = analysis.sources.first(where: { $0.id == sourceID && $0.accepted }), maxPixelSize > 0 else {
            throw ModelInstaller.fail("A reliable registration is required for output inspection")
        }
        let scale = min(1, Double(maxPixelSize)/Double(max(analysis.width, analysis.height)))
        let width = max(1, Int((Double(analysis.width)*scale).rounded())), height = max(1, Int((Double(analysis.height)*scale).rounded()))
        let color = CGColorSpace(name: CGColorSpace.sRGB)!
        guard let overlay = CGContext(data: nil, width: width, height: height, bitsPerComponent: 8, bytesPerRow: width*4,
            space: color, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { throw ModelInstaller.fail("Cannot allocate PhotoArena overlay") }
        overlay.scaleBy(x: scale, y: scale)
        var compared = 0, changed = 0, selected = 0, uncovered = 0, total = 0.0, maximum = 0
        let threshold = max(0, min(255, tolerance)), alpha = UInt8(max(0, min(1, opacity))*255)
        let comparedImage = try ArenaForensics.alignedImage(source)
        let images: [String:CIImage] = renderedOutput != nil ? [:] : try Dictionary(uniqueKeysWithValues: analysis.sources.filter { candidate in
            selections.contains { $0.sourceID == candidate.id }
        }.map { ($0.id,try ArenaForensics.alignedImage($0)) })
        for y in stride(from: 0, to: analysis.height, by: 256) {
            for x in stride(from: 0, to: analysis.width, by: 256) {
                try Task.checkCancellation()
                try autoreleasepool {
                    let rect = CGRect(x: x, y: y, width: min(256, analysis.width-x), height: min(256, analysis.height-y))
                    let output: CGImage
                    if let renderedOutput { output = try ArenaForensics.crop(renderedOutput,rect:rect) }
                    else { output = try tile(images:images,selections:selections,rect:rect) }
                    let a = pixels(output)
                    let b = pixels(try ArenaForensics.crop(comparedImage, rect: rect))
                    let w = Int(rect.width), h = Int(rect.height)
                    var red = [UInt8](repeating: 0, count: w*h*4)
                    // Replay owner masks once per tile, instead of searching every run for every pixel.
                    var owner = [Bool](repeating: false, count: w*h)
                    for selection in selections {
                        let overlap = selection.rect.cgRect.intersection(rect)
                        guard !overlap.isNull, !overlap.isEmpty else { continue }
                        if let mask = selection.mask {
                            for run in mask.runs where run.y >= y && run.y < y+h {
                                for px in max(x,run.x)..<max(max(x,run.x),min(x+w,run.x+run.length)) {
                                    owner[(run.y-y)*w+px-x] = selection.sourceID != sourceID
                                }
                            }
                        } else {
                            for py in Int(overlap.minY)..<Int(overlap.maxY) {
                                for px in Int(overlap.minX)..<Int(overlap.maxX) { owner[(py-y)*w+px-x] = selection.sourceID != sourceID }
                            }
                        }
                    }
                    for p in 0..<w*h {
                        let offset = p*4
                        guard a[offset+3] > 250 && b[offset+3] > 250 else { uncovered += 1; continue }
                        compared += 1
                        var delta = 0
                        for c in 0..<3 {
                            let d = abs(Int(a[offset+c])-Int(b[offset+c])); delta = max(delta,d); total += Double(d)
                        }
                        maximum = max(maximum,delta)
                        if delta > threshold {
                            changed += 1
                            if owner[p] { selected += 1 }
                            let index = ((h-1-p/w)*w+p%w)*4
                            let intensity = heatmap ? UInt8(min(Double(alpha),Double(alpha)*Double(delta)/96+20)) : alpha
                            red[index] = intensity; red[index+3] = intensity
                        }
                    }
                    let provider = CGDataProvider(data: Data(red) as CFData)!
                    let cg = CGImage(width: w, height: h, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: w*4, space: color,
                        bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue), provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent)!
                    overlay.draw(cg, in: rect)
                }
            }
        }
        guard let image = overlay.makeImage() else { throw ModelInstaller.fail("Cannot finish output differences") }
        return (ArenaOutputDifference(sourceID: sourceID, comparedPixels: compared, changedPixels: changed,
            selectedFromOtherSourcePixels: selected, meanAbsoluteError: total/Double(max(1,compared)*3), maximumChannelError: maximum,
            uncoveredPixels: uncovered), image)
    }
}
