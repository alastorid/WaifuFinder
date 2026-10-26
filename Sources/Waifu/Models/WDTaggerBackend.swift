import CoreGraphics
import CoreImage
import CoreML
import Foundation
import ImageIO

struct WDTaggerConfiguration: Sendable {
    let directory: URL
    var modelURL: URL { directory.appendingPathComponent("model.safetensors") }
    var labelsURL: URL { directory.appendingPathComponent("wd-tagger-labels.csv") }

    static var configurationURL: URL {
        FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("Waifu/Uncover/wd-tagger-model.txt")
    }
    static var configuredDirectory: URL? {
        guard let text = try? String(contentsOf: configurationURL, encoding: .utf8)
            .trimmingCharacters(in: .whitespacesAndNewlines), !text.isEmpty else { return nil }
        return URL(fileURLWithPath: text, isDirectory: true)
    }
    static var availableDirectory: URL? {
        let storage = UserDefaults.standard.string(forKey: "modelStoragePath")
            .map { URL(fileURLWithPath: $0, isDirectory: true) }
            ?? FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
                .appendingPathComponent("Waifu/Models", isDirectory: true)
        return resolveDirectory(configured: configuredDirectory, storage: storage,
                                bundled: Bundle.main.resourceURL?.appendingPathComponent("WDTagger", isDirectory: true))
    }
    static func resolveDirectory(configured: URL?, storage: URL, bundled: URL?) -> URL? {
        [configured, storage.appendingPathComponent("wd-tagger-native-v2", isDirectory: true),
         storage.appendingPathComponent("wd-tagger", isDirectory: true), bundled]
            .compactMap { $0 }.first { directory in
                let config = Self(directory: directory)
                return FileManager.default.fileExists(atPath: config.modelURL.path)
                    && FileManager.default.fileExists(atPath: config.labelsURL.path)
            }
    }
    static func load() -> Self? { availableDirectory.map { Self(directory: $0) } }
}

/// Actor isolation keeps model prediction and the reusable eight-frame input
/// buffer serial, including when multiple callers enqueue work simultaneously.
actor WDTaggerBackend: TaggerBackend {
    static let batchSize = 8
    static let imageSize = 448

    /// Video tagging consumes cached JPEGs. Photos stills arrive directly from
    /// PhotoKit at model size. Bound decoding for legacy and video thumbnails.
    static func thumbnailImage(data: Data) throws -> CGImage {
        guard let source = CGImageSourceCreateWithData(data as CFData, nil),
              let image = CGImageSourceCreateThumbnailAtIndex(source, 0, [
                kCGImageSourceCreateThumbnailFromImageAlways: true,
                kCGImageSourceThumbnailMaxPixelSize: PhotosMedia.analysisPixelSize,
                kCGImageSourceCreateThumbnailWithTransform: true
              ] as CFDictionary) else {
            throw ModelInstaller.fail("A cached frame could not be loaded for tagging.")
        }
        return image
    }
    private let worker: WDTaggerWorker
    private let inference: WDTaggerInference
    private let vocabulary: TagVocabulary
    private let generalThreshold: Float
    private let context: CIContext

    static func isAvailable(directory: URL) -> Bool {
        let config = WDTaggerConfiguration(directory: directory)
        return FileManager.default.fileExists(atPath: config.modelURL.path)
            && (try? TagVocabulary.load(from: config.labelsURL)) != nil
    }

    init(directory: URL, generalThreshold: Float = 0.35, executableURL: URL? = nil) throws {
        guard generalThreshold.isFinite, (0...1).contains(generalThreshold) else { throw TaggerError.invalidProbabilities }
        let config = WDTaggerConfiguration(directory: directory)
        vocabulary = try TagVocabulary.load(from: config.labelsURL)
        self.generalThreshold = generalThreshold
        guard FileManager.default.fileExists(atPath: config.modelURL.path) else {
            throw TaggerError.invalidModel("Download the optimized WD Tagger from Settings → Models.")
        }
        guard let executable = executableURL ?? WDTaggerWorker.executableURL,
              FileManager.default.isExecutableFile(atPath: executable.path) else {
            throw TaggerError.invalidModel("The native WD Tagger runtime requires the Apple Silicon app build.")
        }
        context = CIContext(options: [.workingColorSpace: NSNull(), .cacheIntermediates: true])
        let native = try WDTaggerWorker(weights: config.modelURL, executable: executable)
        worker = native
        inference = WDTaggerInference(worker: native)
    }

    func tag(batch: [CGImage]) async throws -> [TagResult] {
        if batch.isEmpty { return [] }
        guard batch.count <= Self.batchSize else { throw TaggerError.invalidBatch }
        return try await withTaskCancellationHandler {
            // Staging belongs to this request. While the serial inference actor
            // waits for its worker, this actor can render the next batch without
            // overwriting the worker's live shared input mapping.
            let input = try preparedInput(batch: batch)
            let rows = try await inference.quantized(input: input, count: batch.count)
            try Task.checkCancellation()
            return try autoreleasepool {
                try rows.map { try $0.result(vocabulary: vocabulary, threshold: generalThreshold) }
            }
        } onCancel: { worker.cancel() }
    }

    /// Numerical validation shares the same staging, worker and tail padding.
    func probabilityRows(batch: [CGImage]) async throws -> [[Float]] {
        if batch.isEmpty { return [] }
        guard batch.count <= Self.batchSize else { throw TaggerError.invalidBatch }
        return try await withTaskCancellationHandler {
            let input = try preparedInput(batch: batch)
            let rows = try await inference.probabilities(input: input, count: batch.count)
            try Task.checkCancellation()
            return rows
        } onCancel: { worker.cancel() }
    }

    private func preparedInput(batch: [CGImage]) throws -> Data {
        try Task.checkCancellation()
        let renderContext = context
        var input = Data(count: WDTaggerWorker.inputBytes)
        try input.withUnsafeMutableBytes { storage in
            let pointer = storage.baseAddress!
            let stride = Self.imageSize * Self.imageSize * 4
            for (index, image) in batch.enumerated() {
                try Task.checkCancellation()
                try autoreleasepool {
                    try Self.renderRGBA(image, context: renderContext, to: pointer.advanced(by: index * stride))
                }
            }
            // Duplicate only the last real row; never invent an output frame.
            for index in batch.count..<Self.batchSize {
                pointer.advanced(by: index * stride).copyMemory(
                    from: pointer.advanced(by: (batch.count - 1) * stride), byteCount: stride)
            }
        }
        return input
    }

    /// Core ML may pad each row of its output buffer. Physical strides belong to
    /// pointer access, whereas MLMultiArray's single-index subscript is logical.
    /// Bulk copying contiguous columns also avoids one NSNumber per output head.
    static func probabilities(from output: MLMultiArray, row: Int) throws -> [Float] {
        guard output.dataType == .float32,
              output.shape.map(\.intValue) == [batchSize, TagVocabulary.headCount],
              (0..<batchSize).contains(row) else {
            throw TaggerError.invalidModel("unexpected probability output")
        }
        let pointer = output.dataPointer.assumingMemoryBound(to: Float.self)
        let rowStride = output.strides[0].intValue, columnStride = output.strides[1].intValue
        let start = pointer.advanced(by: row * rowStride)
        if columnStride == 1 {
            return Array(UnsafeBufferPointer(start: start, count: TagVocabulary.headCount))
        }
        return (0..<TagVocabulary.headCount).map { start[$0 * columnStride] }
    }

    /// Returns interleaved RGBA bytes, with RGB in 0...255. BGR swapping and
    /// normalization are already fused into the exported model weights/graph.
    func preprocess(_ image: CGImage) throws -> [UInt8] {
        try Self.preprocessRGBA(image, context: context)
    }

    static func preprocessRGBA(_ image: CGImage) throws -> [UInt8] {
        try preprocessRGBA(image, context: CIContext(options: [.workingColorSpace: NSNull()]))
    }

    private static func preprocessRGBA(_ image: CGImage, context: CIContext) throws -> [UInt8] {
        var bytes = [UInt8](repeating: 0, count: Self.imageSize * Self.imageSize * 4)
        try renderRGBA(image, context: context, into: &bytes)
        return bytes
    }

    static func renderRGBA(_ image: CGImage, context: CIContext, into bytes: inout [UInt8]) throws {
        try bytes.withUnsafeMutableBytes { try renderRGBA(image, context: context, to: $0.baseAddress!) }
    }

    static func renderRGBA(_ image: CGImage, context: CIContext, to pointer: UnsafeMutableRawPointer) throws {
        let colorSpace = CGColorSpace(name: CGColorSpace.sRGB)!
        guard image.width > 0, image.height > 0 else { throw TaggerError.preprocessing }
        let originalBounds = CGRect(x: 0, y: 0, width: image.width, height: image.height)
        let originalWhite = CIImage(color: CIColor(red: 1, green: 1, blue: 1)).cropped(to: originalBounds)
        var source = CIImage(cgImage: image).composited(over: originalWhite).cropped(to: originalBounds)
        let longest = max(image.width, image.height)
        let scale = min(1, 1024.0 / Double(longest))
        let width = max(1, Int(Double(image.width) * scale))
        let height = max(1, Int(Double(image.height) * scale))
        if scale < 1 {
            source = source.clampedToExtent().applyingFilter("CILanczosScaleTransform", parameters: [
                kCIInputScaleKey: Double(height) / Double(image.height),
                kCIInputAspectRatioKey: (Double(width) / Double(image.width)) / (Double(height) / Double(image.height))
            ]).cropped(to: CGRect(x: 0, y: 0, width: width, height: height))
        }
        let side = max(width, height)
        let square = CGRect(x: 0, y: 0, width: side, height: side)
        let centered = source.transformed(by: CGAffineTransform(translationX: CGFloat((side - width) / 2), y: CGFloat(side - height - (side - height) / 2)))
        let white = CIImage(color: CIColor(red: 1, green: 1, blue: 1)).cropped(to: square)
        let letterboxed = centered.composited(over: white).cropped(to: square)
        let resized: CIImage
        if side == Self.imageSize {
            // A model-sized PhotoKit rendition needs only white padding.
            resized = letterboxed
        } else {
            resized = letterboxed.clampedToExtent().applyingFilter("CIBicubicScaleTransform", parameters: [
                kCIInputScaleKey: Double(Self.imageSize) / Double(side),
                kCIInputAspectRatioKey: 1.0, "inputB": 0.0, "inputC": 0.5
            ])
        }
        context.render(resized, toBitmap: pointer, rowBytes: Self.imageSize * 4,
                           bounds: CGRect(x: 0, y: 0, width: Self.imageSize, height: Self.imageSize),
                       format: .RGBA8, colorSpace: colorSpace)
    }

    static func writeRGB(_ rgba: [UInt8], to input: MLMultiArray, batchIndex: Int) {
        let plane = imageSize * imageSize
        precondition(rgba.count == plane * 4 && (0..<batchSize).contains(batchIndex))
        let pointer = input.dataPointer.bindMemory(to: Float.self, capacity: input.count)
        let base = batchIndex * plane * 3
        for pixel in 0..<plane {
            pointer[base + pixel] = Float(rgba[pixel * 4])
            pointer[base + plane + pixel] = Float(rgba[pixel * 4 + 1])
            pointer[base + 2 * plane + pixel] = Float(rgba[pixel * 4 + 2])
        }
    }
}

/// Exactly one inference owns the mapped input at a time. The separate actor
/// allows the backend to prepare another request during blocking worker IPC.
private actor WDTaggerInference {
    let worker: WDTaggerWorker
    init(worker: WDTaggerWorker) { self.worker = worker }
    private func upload(_ input: Data) throws {
        try Task.checkCancellation()
        guard input.count == WDTaggerWorker.inputBytes else { throw TaggerError.invalidBatch }
        let destination = worker.pointer
        input.withUnsafeBytes { destination.copyMemory(from: $0.baseAddress!, byteCount: $0.count) }
    }
    func quantized(input: Data, count: Int) throws -> [QuantizedTagRow] {
        try upload(input)
        return try worker.predictQuantized(count: count)
    }
    func probabilities(input: Data, count: Int) throws -> [[Float]] {
        try upload(input)
        return try worker.predict(count: count)
    }
}
