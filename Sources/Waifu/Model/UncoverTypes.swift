import CoreGraphics
import CoreMedia
import Foundation

enum UncoverError: Error, LocalizedError {
    case invalid(String)
    case backend(String)
    var errorDescription: String? {
        switch self {
        case .invalid(let message), .backend(let message): return message
        }
    }
}

/// Top-left coordinates in the oriented video, independent of display points.
struct UncoverROI: Codable, Equatable, Sendable {
    var x: Double
    var y: Double
    var width: Double
    var height: Double
    static let initial = UncoverROI(x: 0.25, y: 0.25, width: 0.5, height: 0.5)

    func pixels(in size: CGSize) throws -> CGRect {
        guard [x, y, width, height, size.width, size.height].allSatisfy({ $0.isFinite }),
              width > 0, height > 0, size.width > 0, size.height > 0 else {
            throw UncoverError.invalid("Select a nonempty region within the video.")
        }
        // Normalizing an integer rectangle and mapping it back can differ by an
        // ulp. Do not grow the selection on every redraw/drag round trip.
        func stable(_ value: Double) -> Double {
            abs(value - value.rounded()) < 1e-7 ? value.rounded() : value
        }
        let left = max(0, min(size.width, floor(stable(x * size.width))))
        let top = max(0, min(size.height, floor(stable(y * size.height))))
        let right = max(0, min(size.width, ceil(stable((x + width) * size.width))))
        let bottom = max(0, min(size.height, ceil(stable((y + height) * size.height))))
        guard right > left, bottom > top else { throw UncoverError.invalid("The region is outside the video.") }
        return CGRect(x: left, y: top, width: right - left, height: bottom - top)
    }

    static func from(pixels: CGRect, in size: CGSize) -> Self {
        Self(x: pixels.minX / size.width, y: pixels.minY / size.height,
             width: pixels.width / size.width, height: pixels.height / size.height)
    }

    static func aspectFit(video: CGSize, in viewport: CGSize) -> CGRect {
        guard video.width > 0, video.height > 0 else { return .zero }
        let scale = min(viewport.width / video.width, viewport.height / video.height)
        let size = CGSize(width: video.width * scale, height: video.height * scale)
        return CGRect(x: (viewport.width - size.width) / 2, y: (viewport.height - size.height) / 2,
                      width: size.width, height: size.height)
    }
}

enum UncoverScale: Int, CaseIterable, Codable, Sendable {
    case one = 1, two = 2, four = 4, eight = 8, sixteen = 16, thirtyTwo = 32, sixtyFour = 64
    var label: String { self == .one ? "1× Original (No Downsampling)" : "\(rawValue)×" }
    /// 1× has no enlargement stages; other scales use real learned 2× stages.
    var stages: Int { rawValue.trailingZeroBitCount }
    func degradedSize(_ size: CGSize) -> CGSize {
        CGSize(width: max(1, ceil(size.width / Double(rawValue))),
               height: max(1, ceil(size.height / Double(rawValue))))
    }
}

enum UncoverViewScaleMode: String, CaseIterable, Sendable {
    case original, fitWidth, fitHeight
}

/// Preserve rational source presentation times; nominal FPS never addresses a frame.
struct UncoverTime: Codable, Equatable, Sendable {
    let value: Int64
    let timescale: Int32
    init(_ time: CMTime) { value = time.value; timescale = time.timescale }
    var time: CMTime { CMTime(value: value, timescale: timescale) }
    var seconds: Double { time.seconds }
}

struct UncoverFrame: Codable, Sendable {
    let index: Int
    let timestamp: UncoverTime
    let duration: UncoverTime
}

struct UncoverManifest: Codable, Sendable {
    let source: URL
    let sourceBytes: Int64
    let sourceModified: Double
    let roi: UncoverROI
    let width: Int
    let height: Int
    let regionX: Int
    let regionY: Int
    let regionWidth: Int
    let regionHeight: Int
    let scale: UncoverScale
    let frames: [UncoverFrame]
    var realESRGANImage: RealESRGANImageRequest? = nil
}

struct UncoverPrepared: Sendable {
    let directory: URL
    let manifest: UncoverManifest
}

struct UncoverResult: Sendable {
    let prepared: UncoverPrepared
    let model: String
    let seconds: Double
    let outputScale: Int

    init(prepared: UncoverPrepared, model: String, seconds: Double, outputScale: Int = 1) {
        self.prepared = prepared
        self.model = model
        self.seconds = seconds
        self.outputScale = outputScale
    }

    var directory: URL { prepared.directory }
    var outputWidth: Int { prepared.manifest.width * outputScale }
    var outputHeight: Int { prepared.manifest.height * outputScale }
}

enum UncoverOutputLimits {
    static let maximumDimension = 16_384
    static let maximumPixels = 100_000_000

    static func accepts(width: Int, height: Int, scale: Int) -> Bool {
        guard width > 0, height > 0, scale > 0,
              width <= maximumDimension / scale, height <= maximumDimension / scale else { return false }
        let outputWidth = width * scale, outputHeight = height * scale
        return outputHeight > 0 && outputWidth <= maximumPixels / outputHeight
    }
}

struct UpscalerCapabilities: Sendable {
    let available: Bool
    let message: String
    let scales: [UncoverScale]
    let minimumFrames: Int
    let requiresUnsafeMemoryOptIn: Bool
    var maximumPaddedArea: Int? = nil
    var maximumOutputPixels: Int? = nil
}

protocol TemporalVideoUpscaler: Sendable {
    var cacheIdentity: String { get }
    func capabilities() async -> UpscalerCapabilities
    func reconstruct(_ prepared: UncoverPrepared, allowUnsafeMemory: Bool,
                     progress: @escaping @Sendable (String) -> Void) async throws -> UncoverResult
}

extension TemporalVideoUpscaler {
    var cacheIdentity: String { String(reflecting: Self.self) }
}

enum UncoverAlgorithm: String, CaseIterable, Sendable {
    case seedVR2, seedVR2SevenB, flashVSR11, pgtFormer, realESRGAN, realESRGANFour
    var realESRGANVariant: RealESRGANVariant? { self == .realESRGAN ? .x2plus : (self == .realESRGANFour ? .x4plus : nil) }
    var supportsStillPhotos: Bool { self == .realESRGAN || self == .realESRGANFour }
    var label: String {
        switch self {
        case .seedVR2: return "SeedVR2 3B"
        case .seedVR2SevenB: return "SeedVR2 7B"
        case .flashVSR11: return "FlashVSR 1.1"
        case .pgtFormer: return "PGTFormer"
        case .realESRGAN: return "RealESRGAN x2"
        case .realESRGANFour: return "RealESRGAN x4"
        }
    }
    /// The algorithms this build is licensed to offer.
    ///
    /// A commercial build has no PGTFormer to select, because it ships no
    /// PGTFormer to download. `allCases` stays complete so a saved session naming
    /// PGTFormer still decodes and reports itself unavailable rather than failing.
    static var offered: [UncoverAlgorithm] {
        allCases.filter { algorithm in
            ManagedModel.offered.contains { $0.algorithm == algorithm }
        }
    }
    func makeBackend() -> any TemporalVideoUpscaler {
        switch self {
        case .seedVR2: return SeedVR2Backend()
        case .seedVR2SevenB: return SeedVR2Backend(variant: "seedvr2-7b")
        case .flashVSR11: return FlashVSRBackend()
        case .pgtFormer: return PGTFormerBackend()
        case .realESRGAN: return RealESRGANBackend()
        case .realESRGANFour: return RealESRGANBackend(variant: .x4plus)
        }
    }
}
