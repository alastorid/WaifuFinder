import CoreGraphics
import CoreImage
import CoreText
import CryptoKit
import Foundation

/// Two bounded montages carry ALL candidates in a shared order: context and native detail.
/// No pairwise tournament. The 256px analysis tile fits without resizing into a detail cell.
enum ArenaGroupEvidence {
    static func prepare(_ analysis: ArenaAnalysis, job: ArenaReviewJob) throws -> ArenaGroupPayload {
        let sources = analysis.sources
        let columns = min(4, sources.count), rows = (sources.count+columns-1)/columns
        let cell = 280, labelHeight = 24
        let color = CGColorSpace(name: CGColorSpace.sRGB)!
        let region = analysis.regions.first { $0.id == job.id }
        let detailRect = (job.detailRect ?? job.rect).cgRect
        func montage(detail: Bool) throws -> CGImage {
            guard let canvas = CGContext(data: nil, width: columns*cell, height: rows*(cell+labelHeight), bitsPerComponent: 8,
                bytesPerRow: columns*cell*4, space: color, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else {
                throw ModelInstaller.fail("Cannot allocate PhotoArena evidence montage")
            }
            canvas.setFillColor(gray: 0.12, alpha: 1); canvas.fill(CGRect(x: 0, y: 0, width: canvas.width, height: canvas.height))
            let context = CIContext(options: [.cacheIntermediates: false])
            for (index, source) in sources.enumerated() {
                try Task.checkCancellation()
                try autoreleasepool {
                    let x = (index%columns)*cell, y = (rows-1-index/columns)*(cell+labelHeight)
                    let label = "Source \(index)" + (source.accepted ? "" : " · unaligned")
                    let attrs: [NSAttributedString.Key: Any] = [
                        NSAttributedString.Key(kCTFontAttributeName as String): CTFontCreateWithName("Helvetica" as CFString, 14, nil),
                        NSAttributedString.Key(kCTForegroundColorAttributeName as String): CGColor(gray: 1, alpha: 1)]
                    canvas.textPosition = CGPoint(x: x+6, y: y+cell+5)
                    CTLineDraw(CTLineCreateWithAttributedString(NSAttributedString(string: label, attributes: attrs)), canvas)
                    let input: CIImage
                    let rect: CGRect
                    if detail && !job.overview && source.accepted {
                        input = try ArenaForensics.alignedImage(source); rect = detailRect
                    } else if source.accepted {
                        input = try ArenaForensics.alignedImage(source)
                        rect = CGRect(x: 0,y: 0,width: analysis.width,height: analysis.height)
                    } else {
                        input = try ArenaForensics.image(source); rect = input.extent
                    }
                    let scale = min(detail && !job.overview && source.accepted ? 3 : Double.greatestFiniteMagnitude,
                        Double(cell-12)/max(rect.width, rect.height))
                    let reduced = input.cropped(to: rect).transformed(by: CGAffineTransform(translationX: -rect.minX, y: -rect.minY))
                        .transformed(by: CGAffineTransform(scaleX: scale, y: scale))
                    guard let cg = context.createCGImage(reduced, from: reduced.extent, format: .RGBA8, colorSpace: color) else {
                        throw ModelInstaller.fail("Cannot render PhotoArena evidence")
                    }
                    canvas.interpolationQuality = .none
                    let destinationX = CGFloat(x) + (CGFloat(cell)-CGFloat(cg.width))/2
                    let destinationY = CGFloat(y) + (CGFloat(cell)-CGFloat(cg.height))/2
                    canvas.draw(cg, in: CGRect(x: destinationX, y: destinationY, width: CGFloat(cg.width), height: CGFloat(cg.height)))
                }
                context.clearCaches()
            }
            guard let image = canvas.makeImage() else { throw ModelInstaller.fail("Cannot finish PhotoArena montage") }
            return image
        }
        let measures = (region?.measurements ?? []).compactMap { m -> String? in
            guard let i = sources.firstIndex(where: { $0.id == m.sourceID }) else { return nil }
            return String(format: "Source %d: RGB median residual %.4f, texture %.4f, clipping %.4f, coverage %.4f, OCR area %.4f (scene text possible).", i, m.difference, m.sharpness, m.clipping, m.validFraction, m.textCandidateFraction)
        }
        let geometry = sources.enumerated().map { index, source in
            let equivalent = source.fileSHA256.flatMap { hash in sources.firstIndex { $0.fileSHA256 == hash } }
            let shared = equivalent.flatMap { $0 != index ? " Byte-identical to Source \($0); same photographic evidence." : nil } ?? ""
            return "Source \(index): native \(source.width)x\(source.height), aligned \(source.accepted), correlation \(String(format: "%.3f", source.alignmentConfidence))." + shared
        }
        var observations: [String] = []
        if let mask = region?.changeMask, mask.pixelCount > 0 {
            let width = Int(detailRect.width), x0 = Int(detailRect.minX), y0 = Int(detailRect.minY)
            for (i,source) in sources.enumerated() where source.accepted {
                try Task.checkCancellation()
                let pixels = ArenaForensics.pixelBytes(try ArenaForensics.crop(source,rect:detailRect))
                var count = 0, sums = SIMD3<Double>.zero, squared = SIMD3<Double>.zero
                for run in mask.runs {
                    for x in run.x..<(run.x+run.length) {
                        let offset = ((run.y-y0)*width+x-x0)*4
                        guard offset >= 0, offset+3 < pixels.count, pixels[offset+3] > 250 else { continue }
                        let rgb = SIMD3(Double(pixels[offset]),Double(pixels[offset+1]),Double(pixels[offset+2]))
                        count += 1; sums += rgb; squared += rgb*rgb
                    }
                }
                guard count > 0 else { continue }
                let mean = sums/Double(count), variance = squared/Double(count)-mean*mean
                let contrast = sqrt(max(0,(variance.x+variance.y+variance.z)/3))
                observations.append(String(format:"Source %d inside the exact change mask: %d pixels, average RGB (%.0f,%.0f,%.0f), channel variation %.1f out of 255.",i,count,mean.x,mean.y,mean.z,contrast))
            }
        }
        let marked = region?.changeMask?.pixelCount ?? 0
        let total = job.overview ? analysis.regions.reduce(0) { $0+($1.changeMask?.pixelCount ?? 0) } : 0
        let countEvidence = marked > 0 ? "This region has \(marked) measured changed pixels." :
            "Exact mask count unavailable; inspect the source crops and measured residuals."
        let location = job.overview ? "Whole-image overview. At least \(total) pixels differ across the registered candidates. Shared defects can exist even when images agree." :
            "\(countEvidence) The second montage shows the corresponding focused source crops at reference x=\(Int(detailRect.minX)), y=\(Int(detailRect.minY)), \(Int(detailRect.width))×\(Int(detailRect.height)); crops may be magnified for readability. No guides or masks are drawn over the photos. Unaligned sources show overview only and must get low confidence and usable=false in regional review."
        let context = ([location] + observations + measures + geometry).joined(separator: "\n")
        var images = [try ArenaJudge.png(montage(detail:false))]
        if !job.overview { images.append(try ArenaJudge.png(montage(detail:true))) }
        return ArenaGroupPayload(revision:ArenaGroupPayload.revision,sourceIDs:sources.map(\.id),images:images,context:context)
    }
    static func key(_ payload: ArenaGroupPayload, job: ArenaReviewJob) throws -> String {
        let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys]
        var digest = SHA256()
        func add(_ bytes: Data) {
            var length = UInt64(bytes.count).bigEndian
            withUnsafeBytes(of:&length) { digest.update(bufferPointer:$0) }
            digest.update(data:bytes)
        }
        add(Data(ArenaGroupPayload.revision.utf8)); add(Data(ArenaJudgeConfiguration.revision.utf8))
        add(try encoder.encode(job)); add(try encoder.encode(payload.sourceIDs)); add(Data(payload.context.utf8))
        for image in payload.images { add(image) }
        return digest.finalize().map { String(format: "%02x", $0) }.joined()
    }
}
