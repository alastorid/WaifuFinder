import AppKit
import AVFoundation
import Foundation

// Test fixtures shared by more than one suite.
//
// Everything here writes into a throwaway directory under the system temporary
// directory and nothing here reads or writes the real cache at
// `~/Library/Application Support/Waifu`, the user's `settings.json`, or the sample
// video. Synthetic clips use only Apple system frameworks and temporary files.

// MARK: - Temporary directories

/// Directories handed out by `Temporary.directory(label:)`, removed at the end of
/// the run. Cases never clean up inline: a case that fails midway would otherwise
/// leave its fixture behind, and the interesting output is exactly the failing run.
enum Temporary {
    nonisolated(unsafe) private static var created: [URL] = []

    static func directory(_ label: String) throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("waifu-tests-\(label)-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        created.append(url)
        return url
    }

    static func removeAll() {
        for url in created { try? FileManager.default.removeItem(at: url) }
        created = []
    }
}

// MARK: - Images

enum ImageFixture {
    /// A solid-colour `NSImage`, the smallest thing the thumbnail writer accepts.
    static func solid(width: Int, height: Int, color: NSColor = .systemBlue) -> NSImage {
        let image = NSImage(size: NSSize(width: width, height: height))
        image.lockFocus()
        color.setFill()
        NSRect(x: 0, y: 0, width: width, height: height).fill()
        image.unlockFocus()
        return image
    }

    /// A solid-colour `CGImage` — what a `SampledFrame` carries from the sampler to
    /// the scorer. `gray` lets a caller make otherwise-identical frames distinct.
    static func cgImage(width: Int, height: Int, gray: CGFloat = 0.5) throws -> CGImage {
        guard let context = CGContext(
            data: nil, width: width, height: height,
            bitsPerComponent: 8, bytesPerRow: 0,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else {
            throw SkipTest(reason: "could not create a CGContext for a frame fixture")
        }
        context.setFillColor(gray: gray, alpha: 1)
        context.fill(CGRect(x: 0, y: 0, width: width, height: height))
        guard let image = context.makeImage() else {
            throw SkipTest(reason: "could not rasterise a frame fixture")
        }
        return image
    }
}

// MARK: - Progress

/// Collects the ticks a `@Sendable` progress callback reports. The callback can be
/// invoked from any thread, so the recorder is lock-guarded; the order is the order
/// the calls happened in, which is what progress has to be checked against.
final class TickLog: @unchecked Sendable {
    private let lock = NSLock()
    private var ticks: [Int] = []

    func record(_ value: Int) { lock.withLock { ticks.append(value) } }
    var all: [Int] { lock.withLock { ticks } }
}

// MARK: - Frames
enum FrameFixture {
    /// An `AnalyzedFrame` with everything but the fields a grouping/ranking case
    /// cares about left at the neutral value. `sampleIndex` is taken from the
    /// six-digit suffix of `id`, so a case can name its frames the way the
    /// production identifier does.
    static func make(
        id: String,
        time: Double = 0,
        artScore: Float = 0.5,
        faceScore: Float? = nil,
        faceAreaFraction: Float? = nil,
        featurePrint: Data? = nil,
        groupID: FrameGroupID? = nil
    ) -> AnalyzedFrame {
        AnalyzedFrame(
            sampleIndex: sampleIndex(of: id),
            timeSeconds: time,
            artScore: artScore,
            faceScore: faceScore,
            faceCount: faceScore != nil ? 1 : 0,
            faceAreaFraction: faceAreaFraction,
            featurePrint: featurePrint,
            thumbnailKey: "thumb-\(id)",
            groupID: groupID
        )
    }

    /// `"f000123"` → `123`; anything unparseable → `0`.
    static func sampleIndex(of id: String) -> Int {
        Int(id.drop(while: { !$0.isNumber })) ?? 0
    }
}
