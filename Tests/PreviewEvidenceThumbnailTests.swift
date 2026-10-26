import AppKit
import Metal
import SwiftUI

func registerPreviewEvidenceThumbnailTests() {
    let suite = "Preview evidence thumbnails"
    Registry.shared.add(suite: suite, TestCase(name: "rasters stay bounded and preserve portrait and landscape aspect ratios", knownBug: nil) {
        for (width, height) in [(2048, 1024), (1024, 2048), (40, 32), (1, 2000)] {
            let image = try ImageFixture.cgImage(width: width, height: height)
            guard let thumbnail = EvidenceThumbnailRaster.make(image) else {
                check(false, "thumbnail allocation succeeds"); continue
            }
            check(max(thumbnail.width, thumbnail.height) <= 310, "bounded raster memory")
            check(thumbnail.width <= width && thumbnail.height <= height, "small images are never enlarged")
            let scale = min(1, 310.0 / Double(max(width, height)))
            checkEqual(thumbnail.width, max(1, Int((Double(width) * scale).rounded())), "aspect-fit width")
            checkEqual(thumbnail.height, max(1, Int((Double(height) * scale).rounded())), "aspect-fit height")
        }
    })
    Registry.shared.add(suite: suite, TestCase(name: "all thumbnails contain their actual model composites", knownBug: nil) {
        guard MTLCreateSystemDefaultDevice() != nil else { throw SkipTest(reason: "Metal unavailable") }
        let width = 64, height = 32
        let image = try ImageFixture.cgImage(width: width, height: height)
        let evidence = TruForSpatial(width: width, height: height, score: 0.7,
            anomaly: (0..<(width * height)).map { $0 % width < 32 ? 0.9 : 0.1 },
            reliability: Array(repeating: 0.8, count: width * height),
            noiseprint: Array(repeating: 0.05, count: width * height))
        let thumbnails = try await TruForEvidenceRenderer.shared.thumbnails(original: image, evidence: evidence)
        checkEqual(thumbnails.count, TruForDisplay.allCases.count, "every output is present")
        for mode in TruForDisplay.allCases {
            let rendered = try await TruForEvidenceRenderer.shared.render(original: image, evidence: evidence, mode: mode)
            let expected = EvidenceThumbnailRaster.make(rendered)!
            checkEqual(thumbnails[mode]?.dataProvider?.data as Data?, expected.dataProvider?.data as Data?, "thumbnail shows the actual " + mode.rawValue + " output")
        }
    })
    Registry.shared.add(suite: suite, TestCase(name: "long labels and errors retain bounded offscreen layout", knownBug: nil) {
        await MainActor.run {
            let longText = String(repeating: "Very long photo evidence description ", count: 500)
            let tile = NSHostingView(rootView: EvidenceThumbnail(image: nil, label: longText,
                detail: longText, selected: true, action: {}))
            checkEqual(tile.fittingSize.width, 155, "thumbnail width ignores intrinsic label width")
            checkEqual(tile.fittingSize.height, 116, "thumbnail height ignores long labels")
            let metadata = NSHostingView(rootView: PhotoMetadataInspectorView(metadata: nil,
                isLoading: false, errorMessage: longText, scrollsDetails: false))
            checkEqual(metadata.fittingSize.width, 350, "metadata errors cannot widen the inspector")
            check(metadata.fittingSize.height < 150, "long errors remain one bounded row")
        }
    })
}
