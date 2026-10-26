import AppKit

enum PhotosAppIntegration {
    @MainActor
    static func reveal(_ url: URL) {
        Task {
            do {
                try await PhotosMedia.shared.revealInPhotos(url)
            } catch {
                let alert = NSAlert()
                alert.messageText = "Couldn’t show this photo in Photos"
                alert.informativeText = ErrorAlertText.sanitized(error.localizedDescription)
                alert.addButton(withTitle: "OK")
                alert.runModal()
            }
        }
    }
}
