import AppKit
import ImageIO
import UniformTypeIdentifiers

@MainActor
enum CrystalOrbImport {
    static let webArchiveType = "com.apple.webarchive"
    static var dropTypes: [UTType] {
        [.fileURL, .url, .image, UTType(importedAs: webArchiveType)]
    }

    static func imageTypes(of provider: NSItemProvider) -> [String] {
        let types = provider.registeredTypeIdentifiers.filter { UTType($0)?.conforms(to: .image) == true }
        return types.isEmpty ? [UTType.image.identifier] : types
    }

    /// Safari archives carry the dragged image as the main resource or a subresource.
    /// Read embedded bytes directly, without loading or executing the archived page.
    static func archivedImage(_ data: Data) -> Data? {
        guard let archive = try? PropertyListSerialization.propertyList(from: data, format: nil) as? [String: Any] else { return nil }
        let resources = ([archive["WebMainResource"] as? [String: Any]].compactMap { $0 })
            + (archive["WebSubresources"] as? [[String: Any]] ?? [])
        return resources.compactMap { $0["WebResourceData"] as? Data }.first(where: isDecodable)
    }

    static func archiveData(from provider: NSItemProvider) async -> Data? {
        guard provider.hasItemConformingToTypeIdentifier(webArchiveType) else { return nil }
        return await DropRetry.value {
            let data: Data? = await withCheckedContinuation { continuation in
                provider.loadDataRepresentation(forTypeIdentifier: webArchiveType) { data, _ in
                    continuation.resume(returning: data)
                }
            }
            return data.flatMap(archivedImage)
        }
    }

    static func remoteImage(_ url: URL) async -> Data? {
        guard ["https", "http"].contains(url.scheme?.lowercased() ?? "") else { return nil }
        var request = URLRequest(url: url)
        request.timeoutInterval = 20
        request.setValue("image/*", forHTTPHeaderField: "Accept")
        do {
            let (bytes, response) = try await URLSession.shared.bytes(for: request)
            guard let response = response as? HTTPURLResponse, (200..<300).contains(response.statusCode),
                  response.mimeType?.hasPrefix("image/") == true else { return nil }
            let limit = 64 * 1_024 * 1_024
            guard response.expectedContentLength <= limit else { return nil }
            var data = Data()
            for try await byte in bytes {
                try Task.checkCancellation()
                guard data.count < limit else { return nil }
                data.append(byte)
            }
            return isDecodable(data) ? data : nil
        } catch { return nil }
    }

    /// The concrete image type the provider registered, or `public.image` when it
    /// declares none. Drag sources register what they can actually deliver
    /// (`public.heic` from Photos, `public.png` from browsers); asking for the
    /// abstract supertype instead makes cross-application providers fail to load.
    static func imageType(of provider: NSItemProvider) -> String {
        provider.registeredTypeIdentifiers.first { UTType($0)?.conforms(to: .image) == true }
            ?? UTType.image.identifier
    }

    /// Loads the provider's data representation as decodable image bytes. A load
    /// error and bytes ImageIO rejects are both treated as transient — see
    /// `DropRetry` — so the provider is re-asked before data is given up on.
    static func imageData(from provider: NSItemProvider) async -> Data? {
        return await DropRetry.value {
            for type in imageTypes(of: provider) {
                let data: Data? = await withCheckedContinuation { continuation in
                    provider.loadDataRepresentation(forTypeIdentifier: type) { data, _ in
                        continuation.resume(returning: data)
                    }
                }
                if let data, isDecodable(data) { return data }
            }
            return nil
        }
    }

    /// Whether ImageIO recognises the payload at all; a failed data coercion can
    /// still deliver bytes, and importing them would only report them as corrupt.
    static func isDecodable(_ data: Data) -> Bool {
        guard let source = CGImageSourceCreateWithData(data as CFData, nil) else { return false }
        return CGImageSourceGetCount(source) > 0
    }

    /// Whether a staged file holds a payload ImageIO recognises; reads only the
    /// container header, not the full bitmap.
    static func isDecodableFile(_ url: URL) -> Bool {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil) else { return false }
        return CGImageSourceGetCount(source) > 0
    }

    /// Materialises the representation as a file. Some sources (notably Photos
    /// promises) cannot coerce data but can deliver a file, so this is the fallback
    /// when data loading fails or is undecodable. The handed-out URL is valid only
    /// inside the callback, hence the synchronous copy; a copy ImageIO rejects is
    /// re-asked for like any other transient drop failure (see `DropRetry`).
    static func stageImageFile(from provider: NSItemProvider) async -> URL? {
        let type = imageType(of: provider)
        return await DropRetry.value {
            await withCheckedContinuation { continuation in
                provider.loadFileRepresentation(forTypeIdentifier: type) { url, _ in
                    let staged = url.flatMap { source -> URL? in
                        let destination = FileManager.default.temporaryDirectory
                            .appendingPathComponent("waifu-orb-drop-\(UUID().uuidString)")
                        return (try? FileManager.default.copyItem(at: source, to: destination)) != nil ? destination : nil
                    }
                    continuation.resume(returning: staged)
                }
            }.flatMap { isDecodableFile($0) ? $0 : nil }
        }
    }

    static func accept(_ providers: [NSItemProvider], model: WaifuCrystalOrbModel, session: WaifuSession) -> Bool {
        guard !providers.isEmpty else { return false }
        let urls = providers.filter { $0.hasItemConformingToTypeIdentifier(UTType.fileURL.identifier) || $0.hasItemConformingToTypeIdentifier(UTType.url.identifier) }
        let imageProvider = providers.first { $0.hasItemConformingToTypeIdentifier(UTType.image.identifier) }
        let archiveProvider = providers.first { $0.hasItemConformingToTypeIdentifier(webArchiveType) }
        guard !urls.isEmpty || imageProvider != nil || archiveProvider != nil else { return false }
        let ticket = model.beginDrop()
        // Bitmap providers already identify an image: enter Orb before waiting
        // for a cross-application representation to arrive.
        if imageProvider != nil || archiveProvider != nil { model.beginImageDrop(session: session) }
        Task {
            // Browser and Photos drags often advertise a URL as well as pixels.
            // Resolve the image first: a page URL is not the dragged image, and
            // an unavailable promised URL must not stall an otherwise valid drop.
            if let imageProvider,
               !imageProvider.hasItemConformingToTypeIdentifier(UTType.fileURL.identifier),
               let payload = await imageData(from: imageProvider) {
                guard model.dropTicket == ticket else { return }
                model.importImage(data: payload, session: session)
                return
            }
            if let archiveProvider, let payload = await archiveData(from: archiveProvider) {
                guard model.dropTicket == ticket else { return }
                model.importImage(data: payload, session: session)
                return
            }
            let values = await VideoDrop.urls(from: urls)
            guard model.dropTicket == ticket else { return }
            for url in values where !url.isFileURL {
                let imageURL = UTType(filenameExtension: url.pathExtension)?.conforms(to: .image) == true
                if imageURL || imageProvider != nil || archiveProvider != nil {
                    model.beginImageDrop(session: session)
                    if let payload = await remoteImage(url) {
                        guard model.dropTicket == ticket else { return }
                        model.importImage(data: payload, session: session)
                        return
                    }
                    guard model.dropTicket == ticket else { return }
                    if imageURL && imageProvider == nil && archiveProvider == nil {
                        model.failImageDrop(ticket: ticket, message: "The dropped image could not be downloaded.")
                        return
                    }
                }
            }
            if let image = values.first(where: { $0.isFileURL && isDecodableFile($0) }) {
                model.importImage(url: image, session: session)
            } else if let imageProvider {
                // An image-declaring drag goes to the orb even when no URL
                // resolves as an image: promised files can arrive before their
                // type is derivable from the path, and handing them to the video
                // importer loses the drop. Data first; providers that cannot
                // coerce data still go through the file-promise machinery. Both
                // loads retry transient corruption themselves (see `DropRetry`).
                if let payload = await imageData(from: imageProvider) {
                    guard model.dropTicket == ticket else { return }
                    // Prefer the drag's own file when exactly one is present:
                    // metadata and security scope survive; otherwise use bytes.
                    let files = values.filter { $0.isFileURL && FileManager.default.fileExists(atPath: $0.path) }
                    if files.count == 1, isDecodableFile(files[0]) { model.importImage(url: files[0], session: session) }
                    else { model.importImage(data: payload, session: session) }
                } else if let staged = await stageImageFile(from: imageProvider) {
                    guard model.dropTicket == ticket else { return }
                    model.importImage(url: staged, session: session)
                } else {
                    model.failImageDrop(ticket: ticket, message: "The dropped image could not be read.")
                }
            } else if archiveProvider != nil {
                model.failImageDrop(ticket: ticket, message: "The dropped image could not be read.")
            } else if values.contains(where: { $0.isFileURL }) {
                if model.isOpen { model.close() }
                session.loadVideos(at: values)
            } else if !values.isEmpty {
                if model.isOpen { model.close() }
                session.loadVideos(at: values)
            }
        }
        return true
    }
}
