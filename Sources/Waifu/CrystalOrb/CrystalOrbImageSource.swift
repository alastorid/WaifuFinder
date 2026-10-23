import AppKit
import ImageIO
import CoreImage
import UniformTypeIdentifiers

struct CrystalOrbQuery: Sendable {
    let id: UUID
    let image: CGImage
    let metadata: PhotoMetadata
    let fileURL: URL?
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

    static func read(url: URL) throws -> CrystalOrbQuery {
        let access = url.startAccessingSecurityScopedResource()
        defer { if access { url.stopAccessingSecurityScopedResource() } }
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil) else { throw failure }
        let values = try? url.resourceValues(forKeys: [.fileSizeKey, .creationDateKey, .contentModificationDateKey])
        var query = try read(source: source, name: url.lastPathComponent,
                             size: values?.fileSize.map(Int64.init), url: url)
        query.fileCreated = values?.creationDate
        query.fileModified = values?.contentModificationDate
        return query
    }

    static func read(data: Data) throws -> CrystalOrbQuery {
        guard let source = CGImageSourceCreateWithData(data as CFData, nil) else { throw failure }
        return try read(source: source, name: "Clipboard image", size: nil, url: nil)
    }

    private static func read(source: CGImageSource, name: String, size: Int64?, url: URL?) throws -> CrystalOrbQuery {
        let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any] ?? [:]
        guard let raw = CGImageSourceCreateImageAtIndex(source, 0, nil) else { throw failure }
        let orientation = (properties[kCGImagePropertyOrientation] as? NSNumber)?.int32Value ?? 1
        let oriented = CIImage(cgImage: raw).oriented(forExifOrientation: orientation)
        guard let image = CIContext(options: [.cacheIntermediates: false]).createCGImage(oriented, from: oriented.extent) else { throw failure }
        let metadata = PhotoMetadataParser.parse(properties: properties,
            xmp: CGImageSourceCopyMetadataAtIndex(source, 0, nil), filename: name,
            fallbackWidth: image.width, fallbackHeight: image.height, fileSizeBytes: size,
            photosCreationDate: nil, photosModificationDate: nil, sourceType: CGImageSourceGetType(source) as String?)
        return CrystalOrbQuery(id: UUID(), image: image, metadata: metadata, fileURL: url)
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
