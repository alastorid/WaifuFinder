import AppKit
import Foundation
import ImageIO
import Photos
import UniformTypeIdentifiers

/// Metadata read from the original still-photo resource in Photos. Library
/// timestamps stay separate from dates embedded in the image file.
struct PhotoMetadata: Sendable, Equatable {
    enum ExtractionStatus: Sendable, Equatable {
        case loadingOriginal
        case complete
        case originalUnavailableLocally
        case originalResourceMissing
        case unreadableOriginal
    }

    enum CaptureDateSource: String, Sendable, Equatable {
        case exifOriginal = "EXIF capture date"
        case exifDigitized = "EXIF digitized date"
    }

    var filename: String
    var fileType: String
    var width: Int
    var height: Int
    var fileSizeBytes: Int64?
    var captureDate: Date?
    var captureDateSource: CaptureDateSource?
    var photosCreationDate: Date?
    var photosModificationDate: Date?
    var colorProfile: String?
    var orientation: String?
    var cameraMake: String?
    var cameraModel: String?
    var lensMake: String?
    var lensModel: String?
    var focalLengthMM: Double?
    var focalLength35MM: Int?
    var aperture: Double?
    var exposureTime: Double?
    var iso: Int?
    var exposureCompensationEV: Double?
    var exposureProgram: String?
    var exposureMode: String?
    var meteringMode: String?
    var flash: String?
    var whiteBalance: String?
    var embeddedKeywords: [String]
    var title: String?
    var description: String?
    var extractionStatus: ExtractionStatus

    var dimensionsText: String { "\(width) × \(height)" }
    var megapixelsText: String {
        guard width > 0, height > 0 else { return "" }
        return String(format: "%.1f MP", Double(width) * Double(height) / 1_000_000)
    }

    var hasCameraDetails: Bool {
        cameraMake != nil || cameraModel != nil || lensMake != nil || lensModel != nil
            || focalLengthMM != nil || focalLength35MM != nil
    }

    var hasExposureDetails: Bool {
        aperture != nil || exposureTime != nil || iso != nil || exposureCompensationEV != nil
            || exposureProgram != nil || exposureMode != nil || meteringMode != nil
            || flash != nil || whiteBalance != nil
    }

    var hasDescriptionDetails: Bool { !embeddedKeywords.isEmpty || title != nil || description != nil }

    var summaryKeywordText: String {
        if !embeddedKeywords.isEmpty { return embeddedKeywords.joined(separator: ", ") }
        switch extractionStatus {
        case .loadingOriginal: return "Reading embedded details…"
        case .complete: return "No embedded keywords"
        case .originalUnavailableLocally: return "Unavailable locally"
        case .originalResourceMissing, .unreadableOriginal: return "Could not read embedded keywords"
        }
    }

    var extractionMessage: String? {
        switch extractionStatus {
        case .loadingOriginal: return nil
        case .complete: return nil
        case .originalUnavailableLocally:
            return "The original could not be read from local Photos storage. If it is iCloud-only, embedded details and file size remain unavailable; no download was started."
        case .originalResourceMissing:
            return "Photos did not provide an original still-image resource for this asset."
        case .unreadableOriginal:
            return "The original resource was available, but ImageIO could not read its metadata."
        }
    }

    func copyableText() -> String {
        var lines = [filename, fileType, "\(dimensionsText) pixels (\(megapixelsText))"]
        if let fileSizeBytes { lines.append("File size: \(PhotoMetadataFormatter.fileSize(fileSizeBytes))") }
        if let captureDate { lines.append("\(captureDateSource?.rawValue ?? "Capture date"): \(PhotoMetadataFormatter.dateTime(captureDate))") }
        if let photosCreationDate { lines.append("Photos asset creation date: \(PhotoMetadataFormatter.dateTime(photosCreationDate))") }
        if let photosModificationDate { lines.append("Photos asset modified: \(PhotoMetadataFormatter.dateTime(photosModificationDate))") }
        append("Color profile", colorProfile, to: &lines)
        append("Orientation", orientation, to: &lines)
        append("Camera make", cameraMake, to: &lines)
        append("Camera model", cameraModel, to: &lines)
        append("Lens make", lensMake, to: &lines)
        append("Lens model", lensModel, to: &lines)
        if let focalLengthMM { lines.append("Focal length: \(PhotoMetadataFormatter.millimeters(focalLengthMM))") }
        if let focalLength35MM { lines.append("35mm equivalent: \(focalLength35MM) mm") }
        if let aperture { lines.append("Aperture: \(PhotoMetadataFormatter.aperture(aperture))") }
        if let exposureTime { lines.append("Shutter speed: \(PhotoMetadataFormatter.shutterSpeed(exposureTime))") }
        if let iso { lines.append("ISO: \(iso)") }
        if let exposureCompensationEV { lines.append("Exposure compensation: \(PhotoMetadataFormatter.ev(exposureCompensationEV))") }
        append("Exposure program", exposureProgram, to: &lines)
        append("Exposure mode", exposureMode, to: &lines)
        append("Metering mode", meteringMode, to: &lines)
        append("Flash", flash, to: &lines)
        append("White balance", whiteBalance, to: &lines)
        if !embeddedKeywords.isEmpty { lines.append("Embedded keywords: \(embeddedKeywords.joined(separator: ", "))") }
        append("Title", title, to: &lines)
        append("Description", description, to: &lines)
        lines.append("Photos.app library keywords: unavailable through public PhotoKit APIs")
        return lines.joined(separator: "\n")
    }

    private func append(_ label: String, _ value: String?, to lines: inout [String]) {
        if let value, !value.isEmpty { lines.append("\(label): \(value)") }
    }
}

enum PhotoMetadataFormatter {
    static func fileSize(_ bytes: Int64) -> String {
        let formatter = ByteCountFormatter()
        formatter.countStyle = .file
        formatter.includesUnit = true
        return formatter.string(fromByteCount: bytes)
    }

    static func dateTime(_ date: Date) -> String {
        let formatter = DateFormatter()
        formatter.dateStyle = .medium
        formatter.timeStyle = .short
        return formatter.string(from: date)
    }

    static func aperture(_ value: Double) -> String { String(format: "ƒ/%.1f", value) }

    static func shutterSpeed(_ seconds: Double) -> String {
        guard seconds > 0, seconds.isFinite else { return "" }
        if seconds < 1 {
            let reciprocal = 1 / seconds
            let nearest = reciprocal.rounded()
            if abs(reciprocal - nearest) < 0.02, nearest <= 1_000_000 {
                return "1/\(Int(nearest)) s"
            }
        }
        return "\(String(format: "%.2g", seconds)) s"
    }

    static func millimeters(_ value: Double) -> String {
        "\(String(format: "%.1f", value)) mm"
    }

    static func ev(_ value: Double) -> String {
        "\(String(format: "%+.1f", value)) EV"
    }
}

/// Reads just ImageIO properties from a Photos original. It never creates a
/// CGImage and never enables PhotoKit's network access for metadata requests.
final class PhotoMetadataService: @unchecked Sendable {
    static let shared = PhotoMetadataService()

    private let lock = NSLock()
    private var cache: [PhotosMedia.AssetCacheKey: PhotoMetadata] = [:]
    private let maximumCachedAssets = 128

    /// Returns the inexpensive PhotoKit fields first. The inspector can show
    /// these while the original's bytes are streamed for exact size and EXIF.
    func basic(for photoURL: URL) async -> PhotoMetadata? {
        guard !Task.isCancelled, let item = PhotosMedia.shared.item(photoURL), item.isStill,
              let cacheKey = item.assetCacheKey else { return nil }
        if let cached = lock.withLock({ cache[cacheKey] }) { return cached }
        return Self.initialMetadata(for: item).metadata
    }

    func load(for photoURL: URL) async throws -> PhotoMetadata {
        try Task.checkCancellation()
        guard let item = PhotosMedia.shared.item(photoURL), item.isStill,
              let cacheKey = item.assetCacheKey else {
            throw PhotosMedia.failure("This photo is no longer available in Photos.")
        }
        if let cached = lock.withLock({ cache[cacheKey] }) { return cached }

        // This method is nonisolated async, so Swift 6 runs its synchronous
        // ImageIO work away from MainActor. Keeping it structured also means a
        // selection change cancels the PhotoKit resource request below.
        let metadata = try await Self.readOriginal(for: item)
        try Task.checkCancellation()
        if metadata.extractionStatus == .complete {
            lock.withLock {
                if cache.count >= maximumCachedAssets { cache.removeAll(keepingCapacity: true) }
                cache[cacheKey] = metadata
            }
        }
        return metadata
    }

    private static func readOriginal(for item: PhotosMedia.Item) async throws -> PhotoMetadata {
        let (base, original) = initialMetadata(for: item)
        guard let original else { return base }

        let options = PHAssetResourceRequestOptions()
        options.isNetworkAccessAllowed = false
        let temporaryURL: URL
        do {
            temporaryURL = try await PhotoResourceDataReader(resource: original, options: options).read()
        } catch is CancellationError {
            throw CancellationError()
        } catch {
            // The filename, PhotoKit dimensions and library dates are still useful
            // when the resource is unavailable locally; never retry with network access enabled.
            var unavailable = base
            unavailable.extractionStatus = .originalUnavailableLocally
            return unavailable
        }
        defer { try? FileManager.default.removeItem(at: temporaryURL) }

        guard let source = CGImageSourceCreateWithURL(temporaryURL as CFURL,
                                                       [kCGImageSourceShouldCache: false] as CFDictionary) else {
            var unreadable = base
            unreadable.extractionStatus = .unreadableOriginal
            return unreadable
        }
        guard CGImageSourceGetCount(source) > 0 else {
            var unreadable = base
            unreadable.extractionStatus = .unreadableOriginal
            return unreadable
        }
        let properties = CGImageSourceCopyPropertiesAtIndex(source, 0,
                                                              [kCGImageSourceShouldCache: false] as CFDictionary)
            as? [CFString: Any] ?? [:]
        let metadata = CGImageSourceCopyMetadataAtIndex(source, 0, nil)
        let size = (try? FileManager.default.attributesOfItem(atPath: temporaryURL.path)[.size] as? NSNumber)?.int64Value
        return PhotoMetadataParser.parse(properties: properties, xmp: metadata,
                                         filename: base.filename,
                                         fallbackWidth: base.width,
                                         fallbackHeight: base.height,
                                         fileSizeBytes: size,
                                         photosCreationDate: base.photosCreationDate,
                                         photosModificationDate: base.photosModificationDate,
                                         sourceType: CGImageSourceGetType(source) as String?)
    }

    private static func initialMetadata(for item: PhotosMedia.Item) -> (metadata: PhotoMetadata, resource: PHAssetResource?) {
        let asset = item.asset
        let resources = PHAssetResource.assetResources(for: asset)
        // `.fullSizePhoto` can be an edited rendering; only inspect the
        // original photo resource or its full-resolution alternate (for
        // example, a paired RAW file).
        let original = resources.first(where: { $0.type == .photo })
            ?? resources.first(where: { $0.type == .alternatePhoto })
        let filename = original?.originalFilename ?? "Photo"
        let metadata = PhotoMetadata(
            filename: filename,
            fileType: PhotoMetadataParser.fileType(filename: filename, identifier: nil),
            width: max(0, original?.pixelWidth ?? asset.pixelWidth),
            height: max(0, original?.pixelHeight ?? asset.pixelHeight), fileSizeBytes: nil,
            captureDate: nil, captureDateSource: nil,
            photosCreationDate: asset.creationDate, photosModificationDate: asset.modificationDate,
            colorProfile: nil, orientation: nil, cameraMake: nil, cameraModel: nil,
            lensMake: nil, lensModel: nil, focalLengthMM: nil, focalLength35MM: nil,
            aperture: nil, exposureTime: nil, iso: nil, exposureCompensationEV: nil,
            exposureProgram: nil, exposureMode: nil, meteringMode: nil, flash: nil, whiteBalance: nil,
            embeddedKeywords: [], title: nil, description: nil,
            extractionStatus: original == nil ? .originalResourceMissing : .loadingOriginal
        )
        return (metadata, original)
    }
}

/// Cancellable, chunked PhotoKit resource access. Chunks are streamed into a
/// temporary file, so even a large RAW original is never retained in memory.
final class PhotoResourceDataReader: @unchecked Sendable {
    typealias StartRequest = (@escaping @Sendable (Data) -> Void,
                              @escaping @Sendable (Error?) -> Void) -> PHAssetResourceDataRequestID

    private let lock = NSLock()
    private let temporaryURL: URL
    private let startRequest: StartRequest
    private let cancelRequest: @Sendable (PHAssetResourceDataRequestID) -> Void
    private var fileHandle: FileHandle?
    private var continuation: CheckedContinuation<URL, Error>?
    private var requestID: PHAssetResourceDataRequestID?
    private var finished = false
    private var started = false

    init(resource: PHAssetResource, options: PHAssetResourceRequestOptions) {
        let manager = PHAssetResourceManager.default()
        temporaryURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("waifu-photo-metadata-\(UUID().uuidString)")
        startRequest = { dataReceived, completion in
            manager.requestData(for: resource, options: options,
                                dataReceivedHandler: dataReceived,
                                completionHandler: completion)
        }
        cancelRequest = { manager.cancelDataRequest($0) }
    }

    /// Injection seam used to pin cancellation without requiring a real Photos library.
    init(temporaryURL: URL,
         startRequest: @escaping StartRequest,
         cancelRequest: @escaping @Sendable (PHAssetResourceDataRequestID) -> Void) {
        self.temporaryURL = temporaryURL
        self.startRequest = startRequest
        self.cancelRequest = cancelRequest
    }

    func read() async throws -> URL {
        try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { continuation in
                do {
                    guard FileManager.default.createFile(atPath: temporaryURL.path, contents: nil) else {
                        throw CocoaError(.fileWriteUnknown)
                    }
                    let handle = try FileHandle(forWritingTo: temporaryURL)
                    let canStart = lock.withLock { () -> Bool in
                        guard !started, !finished else { return false }
                        started = true
                        fileHandle = handle
                        self.continuation = continuation
                        return true
                    }
                    guard canStart else {
                        try? handle.close()
                        try? FileManager.default.removeItem(at: temporaryURL)
                        continuation.resume(throwing: CancellationError())
                        return
                    }
                    if Task.isCancelled {
                        finish(.failure(CancellationError()))
                        return
                    }
                    let id = startRequest(
                        { [weak self] data in self?.append(data) },
                        { [weak self] error in
                            if let error { self?.finish(.failure(error)) }
                            else if let self { self.finish(.success(self.temporaryURL)) }
                        }
                    )
                    let shouldCancel = lock.withLock { () -> Bool in
                        requestID = id
                        return finished
                    }
                    if shouldCancel { cancelRequest(id) }
                } catch {
                    try? FileManager.default.removeItem(at: temporaryURL)
                    continuation.resume(throwing: error)
                }
            }
        } onCancel: {
            self.finish(.failure(CancellationError()))
        }
    }

    private func append(_ data: Data) {
        let error = lock.withLock { () -> Error? in
            guard !finished else { return nil }
            do { try fileHandle?.write(contentsOf: data); return nil }
            catch { return error }
        }
        if let error { finish(.failure(error)) }
    }

    private func finish(_ result: Result<URL, Error>) {
        let state = lock.withLock { () -> (CheckedContinuation<URL, Error>?, FileHandle?, PHAssetResourceDataRequestID?) in
            guard !finished else { return (nil, nil, nil) }
            finished = true
            let continuation = self.continuation
            self.continuation = nil
            let handle = fileHandle
            fileHandle = nil
            let requestID = self.requestID
            self.requestID = nil
            return (continuation, handle, requestID)
        }
        guard let continuation = state.0 else { return }
        try? state.1?.close()
        if case .failure = result { try? FileManager.default.removeItem(at: temporaryURL) }
        continuation.resume(with: result)
        if case .failure = result, let requestID = state.2 { cancelRequest(requestID) }
    }
}

enum PhotoMetadataParser {
    static func parse(properties: [CFString: Any], xmp: CGImageMetadata?, filename: String,
                      fallbackWidth: Int, fallbackHeight: Int, fileSizeBytes: Int64?,
                      photosCreationDate: Date?, photosModificationDate: Date?,
                      sourceType: String?) -> PhotoMetadata {
        let exif = dictionary(properties[kCGImagePropertyExifDictionary])
        let tiff = dictionary(properties[kCGImagePropertyTIFFDictionary])
        let iptc = dictionary(properties[kCGImagePropertyIPTCDictionary])
        let exifDate = date(value(exif, kCGImagePropertyExifDateTimeOriginal),
                            offset: value(exif, kCGImagePropertyExifOffsetTimeOriginal))
        let digitizedDate = date(value(exif, kCGImagePropertyExifDateTimeDigitized),
                                 offset: value(exif, kCGImagePropertyExifOffsetTimeDigitized))
        let xmpValues = xmpFields(xmp)
        let iptcKeywords = strings(value(iptc, kCGImagePropertyIPTCKeywords))
        let keywords = unique(iptcKeywords + xmpValues.keywords)
        let rawOrientation = integer(properties[kCGImagePropertyOrientation])
        let make = cleanString(value(tiff, kCGImagePropertyTIFFMake))
        let exposureProgram = integer(value(exif, kCGImagePropertyExifExposureProgram)).flatMap(programName)
        let exposureMode = integer(value(exif, kCGImagePropertyExifExposureMode)).flatMap(exposureModeName)
        let meteringMode = integer(value(exif, kCGImagePropertyExifMeteringMode)).flatMap(meteringName)
        let whiteBalance = integer(value(exif, kCGImagePropertyExifWhiteBalance)).flatMap { $0 == 0 ? "Auto" : $0 == 1 ? "Manual" : nil }
        let flashValue = integer(value(exif, kCGImagePropertyExifFlash))

        return PhotoMetadata(
            filename: filename,
            fileType: fileType(filename: filename, identifier: sourceType),
            width: positiveInteger(properties[kCGImagePropertyPixelWidth]) ?? max(0, fallbackWidth),
            height: positiveInteger(properties[kCGImagePropertyPixelHeight]) ?? max(0, fallbackHeight),
            fileSizeBytes: fileSizeBytes,
            captureDate: exifDate ?? digitizedDate,
            captureDateSource: exifDate != nil ? .exifOriginal : (digitizedDate != nil ? .exifDigitized : nil),
            photosCreationDate: photosCreationDate, photosModificationDate: photosModificationDate,
            colorProfile: cleanString(properties[kCGImagePropertyProfileName])
                ?? cleanString(properties[kCGImagePropertyColorModel]),
            orientation: rawOrientation.flatMap(orientationName),
            cameraMake: make,
            cameraModel: cleanString(value(tiff, kCGImagePropertyTIFFModel)),
            lensMake: cleanString(value(exif, kCGImagePropertyExifLensMake)),
            lensModel: cleanString(value(exif, kCGImagePropertyExifLensModel))
                ?? cleanString(value(dictionary(properties[kCGImagePropertyExifAuxDictionary]), kCGImagePropertyExifAuxLensModel)),
            focalLengthMM: number(value(exif, kCGImagePropertyExifFocalLength)),
            focalLength35MM: positiveInteger(value(exif, kCGImagePropertyExifFocalLenIn35mmFilm)),
            aperture: number(value(exif, kCGImagePropertyExifFNumber)),
            exposureTime: number(value(exif, kCGImagePropertyExifExposureTime)),
            iso: integer(firstValue(value(exif, kCGImagePropertyExifISOSpeedRatings))
                ?? exif?["PhotographicSensitivity" as CFString]),
            exposureCompensationEV: number(value(exif, kCGImagePropertyExifExposureBiasValue)),
            exposureProgram: exposureProgram,
            exposureMode: exposureMode,
            meteringMode: meteringMode,
            flash: flashValue.map { ($0 & 1) == 1 ? "Fired" : "Did not fire" },
            whiteBalance: whiteBalance,
            embeddedKeywords: keywords,
            title: cleanString(value(iptc, kCGImagePropertyIPTCObjectName)) ?? xmpValues.title,
            description: cleanString(value(iptc, kCGImagePropertyIPTCCaptionAbstract)) ?? xmpValues.description,
            extractionStatus: .complete
        )
    }

    static func fileType(filename: String, identifier: String?) -> String {
        let type = identifier.flatMap(UTType.init) ?? UTType(filenameExtension: URL(fileURLWithPath: filename).pathExtension)
        guard let type else { return URL(fileURLWithPath: filename).pathExtension.uppercased().isEmpty ? "Image" : URL(fileURLWithPath: filename).pathExtension.uppercased() }
        if type.conforms(to: .rawImage) { return "RAW" }
        if type.conforms(to: .jpeg) { return "JPEG" }
        if type.conforms(to: .heic) { return "HEIC" }
        if type.conforms(to: .tiff) { return "TIFF" }
        if type.conforms(to: .png) { return "PNG" }
        if type.conforms(to: .gif) { return "GIF" }
        return type.localizedDescription ?? type.preferredFilenameExtension?.uppercased() ?? "Image"
    }

    private static func xmpFields(_ metadata: CGImageMetadata?) -> (keywords: [String], title: String?, description: String?) {
        guard let metadata else { return ([], nil, nil) }
        var keywords: [String] = []
        var titles: [String] = []
        var descriptions: [String] = []
        let options = [kCGImageMetadataEnumerateRecursively: true] as CFDictionary
        CGImageMetadataEnumerateTagsUsingBlock(metadata, nil, options) { path, tag in
            let path = ((path as String?)?.lowercased() ?? "")
                .replacingOccurrences(of: "\\[[0-9]+\\]", with: "", options: .regularExpression)
            let values = strings(CGImageMetadataTagCopyValue(tag))
            if path.hasSuffix("dc:subject") || path.hasSuffix("photoshop:keywords") || path.hasSuffix("lr:hierarchicalsubject") {
                keywords.append(contentsOf: values)
            } else if path.hasSuffix("dc:title") {
                titles.append(contentsOf: values)
            } else if path.hasSuffix("dc:description") || path.hasSuffix("photoshop:headline") {
                descriptions.append(contentsOf: values)
            }
            return true
        }
        return (unique(keywords), titles.first, descriptions.first)
    }

    private static func dictionary(_ value: Any?) -> [CFString: Any]? {
        if let value = value as? [CFString: Any] { return value }
        if let value = value as? NSDictionary { return value as? [CFString: Any] }
        return nil
    }

    private static func value(_ dictionary: [CFString: Any]?, _ key: CFString) -> Any? {
        dictionary?[key]
    }

    private static func cleanString(_ value: Any?) -> String? {
        guard let text = value as? String else { return nil }
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? nil : trimmed
    }

    private static func strings(_ value: Any?) -> [String] {
        guard let value else { return [] }
        if let text = cleanString(value) { return [text] }
        if let array = value as? [Any] { return array.flatMap(strings) }
        if let array = value as? NSArray { return array.flatMap { strings($0) } }
        if let dictionary = value as? [String: Any] {
            if let inner = dictionary["value"] { return strings(inner) }
            return dictionary.filter { $0.key != "xml:lang" }.values.flatMap(strings)
        }
        if let dictionary = value as? NSDictionary {
            if let inner = dictionary["value"] { return strings(inner) }
            return dictionary.allValues.flatMap(strings)
        }
        return []
    }

    private static func unique(_ values: [String]) -> [String] {
        var seen = Set<String>()
        return values.compactMap { value in
            let cleaned = value.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !cleaned.isEmpty, seen.insert(cleaned.localizedLowercase).inserted else { return nil }
            return cleaned
        }
    }

    private static func firstValue(_ value: Any?) -> Any? {
        if let array = value as? [Any] { return array.first }
        if let array = value as? NSArray { return array.firstObject }
        return value
    }

    private static func number(_ value: Any?) -> Double? {
        guard let number = value as? NSNumber else { return nil }
        let result = number.doubleValue
        return result.isFinite ? result : nil
    }

    private static func integer(_ value: Any?) -> Int? {
        guard let number = value as? NSNumber else { return nil }
        let result = number.doubleValue
        guard result.isFinite, result >= Double(Int.min), result <= Double(Int.max) else { return nil }
        return Int(result)
    }

    private static func positiveInteger(_ value: Any?) -> Int? {
        guard let result = integer(value), result > 0 else { return nil }
        return result
    }

    private static func date(_ value: Any?, offset: Any?) -> Date? {
        guard let raw = cleanString(value) else { return nil }
        let zone = timeZone(offset: cleanString(offset)) ?? .current
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.timeZone = zone
        formatter.dateFormat = "yyyy:MM:dd HH:mm:ss"
        // EXIF subsecond precision is stored in a separate property. Trimming
        // the fractional suffix keeps the capture second parseable everywhere.
        return formatter.date(from: String(raw.prefix(19)))
    }

    private static func timeZone(offset: String?) -> TimeZone? {
        guard let offset, offset.count >= 5, let sign = offset.first,
              sign == "+" || sign == "-" else { return nil }
        let digits = offset.dropFirst().filter(\.isNumber)
        guard digits.count == 4, let hours = Int(digits.prefix(2)), let minutes = Int(digits.suffix(2)) else { return nil }
        let seconds = (hours * 3600 + minutes * 60) * (sign == "-" ? -1 : 1)
        return TimeZone(secondsFromGMT: seconds)
    }

    private static func orientationName(_ value: Int) -> String? {
        switch value {
        case 1: "Up"
        case 2: "Up mirrored"
        case 3: "Down"
        case 4: "Down mirrored"
        case 5: "Left mirrored"
        case 6: "Right (90° clockwise)"
        case 7: "Right mirrored"
        case 8: "Left (90° counterclockwise)"
        default: nil
        }
    }

    private static func programName(_ value: Int) -> String? {
        [0: "Not defined", 1: "Manual", 2: "Normal program", 3: "Aperture priority",
         4: "Shutter priority", 5: "Creative program", 6: "Action program",
         7: "Portrait mode", 8: "Landscape mode"][value]
    }

    private static func exposureModeName(_ value: Int) -> String? {
        [0: "Auto", 1: "Manual", 2: "Auto bracket"][value]
    }

    private static func meteringName(_ value: Int) -> String? {
        [0: "Unknown", 1: "Average", 2: "Center-weighted average", 3: "Spot",
         4: "Multi-spot", 5: "Pattern", 6: "Partial", 255: "Other"][value]
    }
}
