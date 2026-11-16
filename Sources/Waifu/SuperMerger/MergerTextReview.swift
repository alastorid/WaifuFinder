import CoreGraphics
import CoreImage
import CryptoKit
import Foundation
import Vision

/// Fast regional heuristic. Only differences explained by recognized text can
/// bypass model review; the ordinary source-coverage and seam verifier still runs.
enum MergerTextReview {
    /// Localized recovery remains useful when compression or resampling makes
    /// the global difference mask unsuitable for automatic reconstruction.
    static func localizedAnalysis(_ input: ArenaAnalysis) throws -> ArenaAnalysis? {
        let sources = input.sources.filter(\.accepted)
        guard sources.count >= 2 else { return nil }
        let canvas = CGRect(x: 0, y: 0, width: input.width, height: input.height)
        let scale = min(1, 2048 / max(canvas.width, canvas.height))
        var bounds: [CGRect] = []
        for source in sources {
            try Task.checkCancellation()
            let image = try ArenaForensics.alignedImage(source).cropped(to: canvas)
                .transformed(by: CGAffineTransform(scaleX: scale, y: scale))
            let crop = try ArenaForensics.crop(image, rect: image.extent)
            let request = VNRecognizeTextRequest()
            request.recognitionLevel = .accurate
            request.usesLanguageCorrection = false
            request.minimumTextHeight = 0
            guard (try? VNImageRequestHandler(cgImage: crop).perform([request])) != nil else { continue }
            for observation in request.results ?? [] {
                guard let text = observation.topCandidates(1).first, text.confidence >= 0.5 else { continue }
                let box = observation.boundingBox
                var rect = CGRect(x: box.minX * canvas.width, y: box.minY * canvas.height,
                                  width: box.width * canvas.width, height: box.height * canvas.height)
                    .insetBy(dx: -4, dy: -4).integral.intersection(canvas)
                // Consolidate corresponding detections across sources, including
                // shadows and antialiased glyph edges in the replacement patch.
                var index = 0
                while index < bounds.count {
                    if bounds[index].intersects(rect) {
                        rect = rect.union(bounds.remove(at: index))
                        index = 0
                    } else { index += 1 }
                }
                bounds.append(rect)
            }
        }
        bounds.sort { $0.minY == $1.minY ? $0.minX < $1.minX : $0.minY < $1.minY }
        var result = input
        result.sources = sources
        result.regions = []
        for (index, rect) in bounds.enumerated() {
            try Task.checkCancellation()
            var measurements: [ArenaRegionMeasurement] = []
            for source in sources {
                let pixels = ArenaForensics.pixelBytes(try ArenaForensics.crop(source, rect: rect))
                let covered = stride(from: 3, to: pixels.count, by: 4).filter { pixels[$0] > 250 }.count
                measurements.append(ArenaRegionMeasurement(sourceID: source.id, difference: 0,
                    sharpness: 0, clipping: 0, validFraction: Double(covered) / Double(max(1, pixels.count / 4)),
                    textCandidateFraction: 0, evidence: ["Coverage measured on the complete OCR patch."]))
            }
            let runs = (Int(rect.minY)..<Int(rect.maxY)).map {
                ArenaMaskRun(y: $0, x: Int(rect.minX), length: Int(rect.width))
            }
            result.regions.append(ArenaRegion(id: "text-region-\(index)", rect: ArenaPixelRect(rect),
                measurements: measurements, unresolved: true,
                evidence: ["OCR recovery region including a four-pixel context margin."], changeMask: ArenaPixelMask(runs: runs), textRecoveryRegion: true))
        }
        // Keep only regions for which cross-source OCR actually identifies a donor.
        // Shared scene text and shared watermarks therefore remain unresolved.
        let jobs = ArenaRecoveryPlanner.jobs(result).filter { !$0.overview }
        var recoverable = Set<String>()
        for job in jobs {
            if try review(result, job: job) != nil { recoverable.insert(job.id) }
        }
        result.regions.removeAll { !recoverable.contains($0.id) }
        guard !result.regions.isEmpty else { return nil }
        let coverage = sources.map { source in
            ArenaRegionMeasurement(sourceID: source.id, difference: 0, sharpness: 0, clipping: 0,
                validFraction: input.regions.compactMap { region in
                    region.measurements.first { $0.sourceID == source.id }?.validFraction
                }.min() ?? 0, textCandidateFraction: 0, evidence: ["Whole-canvas coverage from the original analysis."])
        }
        result.regions.append(ArenaRegion(id: "unresolved-global-differences", rect: ArenaPixelRect(canvas),
            measurements: coverage, unresolved: true,
            evidence: ["Global differences remain unresolved. Only separately verified OCR patches may be replaced."],
            maskDeferred: true))
        return result
    }

    static func review(_ analysis: ArenaAnalysis, job: ArenaReviewJob) throws -> ArenaReviewRecord? {
        guard !job.overview,
              let region = analysis.regions.first(where: { $0.id == job.id }),
              let mask = region.changeMask, mask.pixelCount > 0 else { return nil }
        let bounds = (job.detailRect ?? job.rect).cgRect
        var boxes = [[CGRect]](repeating: [], count: analysis.sources.count)
        var inspected = Set<Int>()
        for (index, source) in analysis.sources.enumerated() where source.accepted {
            try Task.checkCancellation()
            guard region.measurements.contains(where: {
                $0.sourceID == source.id && $0.validFraction >= 0.9999
            }) else { continue }
            guard let image = try? ArenaForensics.crop(source, rect: bounds) else { continue }
            let request = VNRecognizeTextRequest()
            request.recognitionLevel = .accurate
            request.usesLanguageCorrection = false
            request.minimumTextHeight = 0
            // OCR failure is inconclusive; the caller falls back to model review.
            guard (try? VNImageRequestHandler(cgImage: image).perform([request])) != nil else { continue }
            inspected.insert(index)
            boxes[index] = (request.results ?? []).compactMap { observation in
                guard let text = observation.topCandidates(1).first,
                      text.confidence >= 0.5, !text.string.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return nil }
                let rect = observation.boundingBox
                return CGRect(x: bounds.minX + rect.minX * bounds.width,
                              y: bounds.minY + rect.minY * bounds.height,
                              width: rect.width * bounds.width, height: rect.height * bounds.height)
                    .insetBy(dx: -3, dy: -3).intersection(bounds)
            }
        }
        let textBoxes = boxes.flatMap { $0 }.filter { $0.intersects(mask.bounds) }
        guard !textBoxes.isEmpty else { return nil }
        // A mixed region (text plus a changed object, for example) needs model review.
        for run in mask.runs where region.textRecoveryRegion != true {
            try Task.checkCancellation()
            let intervals = textBoxes.filter { Double(run.y) >= $0.minY && Double(run.y) < $0.maxY }
                .sorted { $0.minX < $1.minX }
            var coveredTo = Double(run.x)
            for interval in intervals where interval.minX <= coveredTo {
                coveredTo = max(coveredTo, interval.maxX)
            }
            guard coveredTo >= Double(run.x + run.length) else { return nil }
        }
        let marked = Set(inspected.filter { index in
            boxes[index].contains { box in mask.runs.contains { run in
                Double(run.y) >= box.minY && Double(run.y) < box.maxY
                    && Double(run.x) < box.maxX && Double(run.x + run.length) > box.minX
            } }
        })
        let donors = inspected.filter { boxes[$0].isEmpty }
        guard !marked.isEmpty, let donor = donors.sorted(by: {
            let a = analysis.sources[$0], b = analysis.sources[$1]
            let areaA = Double(a.width) * Double(a.height), areaB = Double(b.width) * Double(b.height)
            return areaA == areaB ? a.id < b.id : areaA > areaB
        }).first else { return nil }
        let findings = analysis.sources.indices.map { index in
            let clean = donors.contains(index)
            return ArenaCandidateFinding(source: index, originality: 50, cleanliness: clean ? 100 : 0,
                detail: 50, alteration: marked.contains(index) ? 100 : 0, enhancement: 0,
                degradation: 0, consistency: 50, usable: clean, confidence: inspected.contains(index) ? .medium : .low,
                defects: marked.contains(index) ? [.watermark, .text] : clean ? [.natural] : [.unknown],
                evidence: marked.contains(index)
                    ? "Apple Vision recognized text overlapping the difference mask; treated as a watermark by the text-removal heuristic."
                    : clean ? "Apple Vision found no text in the corresponding fully covered crop. Clean-donor status is heuristic; source pixels and patch boundaries still require verification."
                    : "OCR or coverage was inconclusive; excluded from automatic text replacement.")
        }
        let assessment = ArenaGroupAssessment(sources: findings, selected_source: donor, unresolved: false,
            reason: "All segmented differences lie within recognized text bounds; select a covered source without detected text.")
        let result = ArenaJudgeResult(judgment: ArenaJudgment(watermark: .unknown, authenticity: .unknown,
            detail: .unknown, quality: .unknown, consistency: .unknown, recommended_source: .unknown,
            uncertain: true, evidence: ["Apple Vision text-removal heuristic; no model inference or authenticity determination."]),
            tokens: 0, prefillSeconds: 0, decodeSeconds: 0, physicalFootprintBytes: 0, compressedBytes: 0, review: assessment)
        try result.validate()
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        var digest = SHA256()
        digest.update(data: try encoder.encode(region))
        digest.update(data: try encoder.encode(analysis.sources))
        let key = "vision-text-v1-" + digest.finalize().map { String(format: "%02x", $0) }.joined()
        return ArenaReviewRecord(job: job, sourceIDs: analysis.sources.map(\.id), cacheKey: key, result: result)
    }
}
