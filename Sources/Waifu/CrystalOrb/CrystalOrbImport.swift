import AppKit
import UniformTypeIdentifiers

@MainActor
enum CrystalOrbImport {
    static func accept(_ providers: [NSItemProvider], model: WaifuCrystalOrbModel, session: WaifuSession) -> Bool {
        guard !providers.isEmpty else { return false }
        let urls = providers.filter { $0.hasItemConformingToTypeIdentifier(UTType.fileURL.identifier) || $0.hasItemConformingToTypeIdentifier(UTType.url.identifier) }
        let imageProvider = providers.first { $0.hasItemConformingToTypeIdentifier(UTType.image.identifier) }
        guard !urls.isEmpty || imageProvider != nil else { return false }
        let ticket = model.beginDrop()
        Task {
            let values = await VideoDrop.urls(from: urls)
            guard model.dropTicket == ticket else { return }
            if let image = values.first(where: CrystalOrbImageSource.isImage) {
                model.importImage(url: image, session: session)
            } else if let imageProvider {
                do {
                    let data: Data = try await withCheckedThrowingContinuation { continuation in
                        imageProvider.loadDataRepresentation(forTypeIdentifier: UTType.image.identifier) { data, error in
                            if let data { continuation.resume(returning: data) }
                            else { continuation.resume(throwing: error ?? CrystalOrbImageSource.failure) }
                        }
                    }
                    guard model.dropTicket == ticket else { return }
                    model.importImage(data: data, session: session)
                } catch { session.errorMessage = error.localizedDescription }
            } else if !values.isEmpty {
                if model.isOpen { model.close() }
                session.loadVideos(at: values)
            }
        }
        return true
    }
}
