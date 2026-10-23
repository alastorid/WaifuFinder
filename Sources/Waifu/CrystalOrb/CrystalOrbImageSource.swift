import AppKit
import ImageIO
import CoreImage
import UniformTypeIdentifiers

struct CrystalOrbQuery: Sendable {
    let id: UUID
    let image: CGImage
    let metadata: PhotoMetadata
    let fileURL: URL?
    let nativeSize: CGSize
    let isOriginal: Bool
    var fileCreated: Date? = nil
    var fileModified: Date? = nil
}

enum CrystalOrbImageSource {
    static func isImage(_ url: URL) -> Bool {
        guard url.isFileURL else { return false }
        let type = (try? url.resourceValues(forKeys: [.contentTypeKey]))?.contentType
            ?? UTType(filenameExtension: url.pathExtension)
        return type?.conforms(to: .image) == true
    }

    static func read(url: URL, maxPixelSize: Int? = nil, id: UUID = UUID()) throws -> CrystalOrbQuery {
        let access = url.startAccessingSecurityScopedResource()
        defer { if access { url.stopAccessingSecurityScopedResource() } }
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil) else { throw failure }
        let values = try? url.resourceValues(forKeys: [.fileSizeKey, .creationDateKey, .contentModificationDateKey])
        var query = try read(source: source, name: url.lastPathComponent,
                             size: values?.fileSize.map(Int64.init), url: url, maxPixelSize: maxPixelSize, id: id)
        query.fileCreated = values?.creationDate
        query.fileModified = values?.contentModificationDate
        return query
    }

    static func read(data: Data, maxPixelSize: Int? = nil, id: UUID = UUID()) throws -> CrystalOrbQuery {
        guard let source = CGImageSourceCreateWithData(data as CFData, nil) else { throw failure }
        return try read(source: source, name: "Clipboard image", size: nil, url: nil, maxPixelSize: maxPixelSize, id: id)
    }

    private static func read(source: CGImageSource, name: String, size: Int64?, url: URL?, maxPixelSize: Int?, id: UUID) throws -> CrystalOrbQuery {
        let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any] ?? [:]
        let orientation = (properties[kCGImagePropertyOrientation] as? NSNumber)?.int32Value ?? 1
        let width = (properties[kCGImagePropertyPixelWidth] as? NSNumber)?.intValue ?? 0
        let height = (properties[kCGImagePropertyPixelHeight] as? NSNumber)?.intValue ?? 0
        let rotated = (5...8).contains(orientation)
        let native = CGSize(width: rotated ? height : width, height: rotated ? width : height)
        let image: CGImage
        if let maxPixelSize {
            guard let thumbnail = CGImageSourceCreateThumbnailAtIndex(source, 0, [
                kCGImageSourceCreateThumbnailFromImageAlways: true,
                kCGImageSourceCreateThumbnailWithTransform: true,
                kCGImageSourceThumbnailMaxPixelSize: maxPixelSize,
                kCGImageSourceShouldCacheImmediately: true
            ] as CFDictionary) else { throw failure }
            image = thumbnail
        } else {
            guard let raw = CGImageSourceCreateImageAtIndex(source, 0, nil) else { throw failure }
            let oriented = CIImage(cgImage: raw).oriented(forExifOrientation: orientation)
            guard let original = CIContext(options: [.cacheIntermediates: false]).createCGImage(oriented, from: oriented.extent) else { throw failure }
            image = original
        }
        let metadata = PhotoMetadataParser.parse(properties: properties,
            xmp: CGImageSourceCopyMetadataAtIndex(source, 0, nil), filename: name,
            fallbackWidth: width, fallbackHeight: height, fileSizeBytes: size,
            photosCreationDate: nil, photosModificationDate: nil, sourceType: CGImageSourceGetType(source) as String?)
        return CrystalOrbQuery(id: id, image: image, metadata: metadata, fileURL: url,
            nativeSize: native.width > 0 && native.height > 0 ? native : CGSize(width: image.width, height: image.height),
            isOriginal: maxPixelSize == nil || (image.width == Int(native.width) && image.height == Int(native.height)))
    }

    static func candidateSize(source: VideoInfo, frame: AnalyzedFrame) -> CGSize {
        if let photo = PhotosMedia.shared.photoItemURL(for: source.url, seconds: frame.timeSeconds),
           let item = PhotosMedia.shared.item(photo) {
            return CGSize(width: item.asset.pixelWidth, height: item.asset.pixelHeight)
        }
        if let collection = VideoCollection.get(source.url) {
            let member = collection.member(at: frame.timeSeconds)
            return CGSize(width: member.video.width, height: member.video.height)
        }
        return CGSize(width: source.width, height: source.height)
    }

    static var failure: NSError { NSError(domain: "WaifuCrystalOrb", code: 1,
        userInfo: [NSLocalizedDescriptionKey: "This image is unsupported, corrupt, or unreadable."]) }

    @MainActor static func clipboard() -> (URL?, Data?)? {
        let board = NSPasteboard.general
        if let urls = board.readObjects(forClasses: [NSURL.self], options: [.urlReadingFileURLsOnly: true]) as? [URL],
           let url = urls.first(where: isImage) { return (url, nil) }
        for type in [NSPasteboard.PasteboardType.png, .tiff] {
            if let data = board.data(forType: type) { return (nil, data) }
        }
        if let image = NSImage(pasteboard: board), let data = image.tiffRepresentation { return (nil, data) }
        return nil
    }
}
