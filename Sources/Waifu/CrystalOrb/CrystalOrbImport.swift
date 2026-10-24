import AppKit
import ImageIO
import UniformTypeIdentifiers

@MainActor
enum CrystalOrbImport {
    /// The concrete image type the provider registered, or `public.image` when it
    /// declares none. Drag sources register what they can actually deliver
    /// (`public.heic` from Photos, `public.png` from browsers); asking for the
    /// abstract supertype instead makes cross-application providers fail to load.
    static func imageType(of provider: NSItemProvider) -> String {
        provider.registeredTypeIdentifiers.first { UTType($0)?.conforms(to: .image) == true }
            ?? UTType.image.identifier
    }

    static func imageData(from provider: NSItemProvider) async throws -> Data {
        let type = imageType(of: provider)
        return try await withCheckedThrowingContinuation { continuation in
            provider.loadDataRepresentation(forTypeIdentifier: type) { data, error in
                if let data { continuation.resume(returning: data) }
                else { continuation.resume(throwing: error ?? CrystalOrbImageSource.failure) }
            }
        }
    }

    /// Whether ImageIO recognises the payload at all; a failed data coercion can
    /// still deliver bytes, and importing them would only report them as corrupt.
    static func isDecodable(_ data: Data) -> Bool {
        guard let source = CGImageSourceCreateWithData(data as CFData, nil) else { return false }
        return CGImageSourceGetCount(source) > 0
    }

    /// Materialises the representation as a file. Some sources (notably Photos
    /// promises) cannot coerce data but can deliver a file, so this is the fallback
    /// when data loading fails or is undecodable. The handed-out URL is valid only
    /// inside the callback, hence the synchronous copy.
    static func stageImageFile(from provider: NSItemProvider) async -> URL? {
        let type = imageType(of: provider)
        return await withCheckedContinuation { continuation in
            provider.loadFileRepresentation(forTypeIdentifier: type) { url, _ in
                let staged = url.flatMap { source -> URL? in
                    let destination = FileManager.default.temporaryDirectory
                        .appendingPathComponent("waifu-orb-drop-\(UUID().uuidString)")
                    return (try? FileManager.default.copyItem(at: source, to: destination)) != nil ? destination : nil
                }
                continuation.resume(returning: staged)
            }
        }
    }

    static func accept(_ providers: [NSItemProvider], model: WaifuCrystalOrbModel, session: WaifuSession) -> Bool {
        guard !providers.isEmpty else { return false }
        let urls = providers.filter { $0.hasItemConformingToTypeIdentifier(UTType.fileURL.identifier) || $0.hasItemConformingToTypeIdentifier(UTType.url.identifier) }
        let imageProvider = providers.first { $0.hasItemConformingToTypeIdentifier(UTType.image.identifier) }
        guard !urls.isEmpty || imageProvider != nil else { return false }
        let ticket = model.beginDrop()
        // Bitmap providers already identify an image: enter Orb before waiting
        // for a cross-application representation to arrive.
        if imageProvider != nil && urls.isEmpty { model.beginImageDrop(session: session) }
        Task {
            let values = await VideoDrop.urls(from: urls)
            guard model.dropTicket == ticket else { return }
            if let image = values.first(where: CrystalOrbImageSource.isImage) {
                model.importImage(url: image, session: session)
            } else if values.contains(where: { $0.isFileURL }) {
                if model.isOpen { model.close() }
                session.loadVideos(at: values)
            } else if let imageProvider {
                // Data first; providers that cannot coerce data still go through
                // the file-promise machinery, which is the reliable cross-process path.
                let payload = try? await imageData(from: imageProvider)
                if let payload, isDecodable(payload) {
                    guard model.dropTicket == ticket else { return }
                    model.importImage(data: payload, session: session)
                } else if let staged = await stageImageFile(from: imageProvider) {
                    guard model.dropTicket == ticket else { return }
                    model.importImage(url: staged, session: session)
                } else {
                    model.failImageDrop(ticket: ticket, message: "The dropped image could not be read.")
                }
            } else if !values.isEmpty {
                if model.isOpen { model.close() }
                session.loadVideos(at: values)
            }
        }
        return true
    }
}
