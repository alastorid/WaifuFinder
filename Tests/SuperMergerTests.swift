import CoreGraphics
import Foundation
import ImageIO
import UniformTypeIdentifiers

func registerSuperMergerTests() {
    Registry.shared.add(suite: "Super Merger", TestCase(name: "native preprocessing handles sparse edits compression duplicates and eight edits", knownBug: nil) {
        let root = try Temporary.directory("super-merger-photographic")
        let fixture = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
            .appendingPathComponent("Fixtures/RealESRGAN/nasa-collins-896x1134.jpg")
        let container = CGImageSourceCreateWithURL(fixture as CFURL, nil)!
        let image = CGImageSourceCreateImageAtIndex(container, 0, nil)!
        func source(_ name: String, boxes: [CGRect], jpeg: Bool = false) throws -> ArenaSource {
            let context = CGContext(data: nil, width: image.width, height: image.height, bitsPerComponent: 8,
                bytesPerRow: image.width * 4, space: CGColorSpace(name: CGColorSpace.sRGB)!,
                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
            context.draw(image, in: CGRect(x: 0, y: 0, width: image.width, height: image.height))
            context.setFillColor(red: 1, green: 0, blue: 0, alpha: 0.9)
            for box in boxes { context.fill(box) }
            let url = root.appendingPathComponent(name + (jpeg ? ".jpg" : ".png"))
            let destination = CGImageDestinationCreateWithURL(url as CFURL,
                (jpeg ? UTType.jpeg : UTType.png).identifier as CFString, 1, nil)!
            CGImageDestinationAddImage(destination, context.makeImage()!,
                [kCGImageDestinationLossyCompressionQuality: 0.88] as CFDictionary)
            guard CGImageDestinationFinalize(destination) else { throw CocoaError(.fileWriteUnknown) }
            return ArenaSource(id: name, url: url, seconds: 0)
        }
        let clean = try source("clean", boxes: [])
        let a = try source("upper", boxes: [CGRect(x: 200, y: 240, width: 100, height: 70)])
        let b = try source("lower", boxes: [CGRect(x: 500, y: 700, width: 100, height: 70)])
        let sparse = try MergerPreprocessor.prepare([clean, a, b])
        check(sparse.admitted, "same-root complementary edits admitted")
        checkEqual(sparse.analysis.regions.filter(ArenaRecoveryPlanner.needsReview).count, 2, "tile-spanning edits become two regions")
        let identical = try MergerPreprocessor.prepare([clean, try source("copy", boxes: [])])
        check(identical.admitted, "identical files admitted")
        checkEqual(identical.analysis.regions.filter(ArenaRecoveryPlanner.needsReview).count, 0, "duplicates introduce no regions")
        let compressed = try MergerPreprocessor.prepare([clean, try source("compressed", boxes: [], jpeg: true)])
        check(compressed.analysis.regions.filter(ArenaRecoveryPlanner.needsReview).count <= 7 || !compressed.admitted,
              "compression cannot dispatch excessive review jobs")
        let boxes = (0..<8).map { index in
            CGRect(x: 60 + (index % 4) * 210, y: 200 + (index / 4) * 550, width: 30, height: 30)
        }
        let complex = try MergerPreprocessor.prepare([clean, try source("eight-edits", boxes: boxes)])
        check(!complex.admitted, "eight genuine alterations stop admission")
        checkEqual(complex.analysis.regions.filter(ArenaRecoveryPlanner.needsReview).count, 8, "all eight edits remain inspectable")

        guard ProcessInfo.processInfo.environment["WAIFU_SUPER_MERGER_LIVE"] == "1" else { return }
        let repository = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
        let started = Date()
        let records = try await ArenaJudge.review(analysis: sparse.analysis, jobs: ArenaRecoveryPlanner.jobs(sparse.analysis),
            previous: [], directory: repository.appendingPathComponent("models/photoarena"),
            executable: repository.appendingPathComponent(".native-runtime/waifu-photoarena"), onResult: { _ in })
        let plan = ArenaRecoveryPlanner.plan(sparse.analysis, reviews: records)
        struct Report: Encodable {
            let elapsedSeconds: Double
            let selectedCleanSource: Bool
            let records: [ArenaReviewRecord]
            let plan: ArenaRecoveryPlan
        }
        let report = Report(elapsedSeconds: Date().timeIntervalSince(started),
                            selectedCleanSource: plan.bestSourceID == clean.id, records: records, plan: plan)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let reportURL = repository.appendingPathComponent(".build/super-merger-live-review.json")
        try encoder.encode(report).write(to: reportURL)
        print("      Live review selected known clean source: \(report.selectedCleanSource), \(report.elapsedSeconds)s")
    })
}
