import SwiftUI

/// Favorite writes stay on the thumbnail and don't change grid selection.
struct PhotosFavoriteButton: View {
    let url: URL
    let isHovered: Bool
    let favoriteTargets: () -> [URL]
    @Environment(\.scenePhase) private var scenePhase
    @State private var isFavorite = false
    @State private var isUpdating = false
    @State private var error: String?

    var body: some View {
        Button {
            guard !isUpdating else { return }
            let next = !isFavorite
            let targets = favoriteTargets()
            let identifiers = targets.compactMap { PhotosMedia.shared.item($0)?.asset.localIdentifier }
            isUpdating = true
            Task {
                do {
                    try await PhotosMedia.shared.setFavorite(targets, isFavorite: next)
                    isFavorite = next
                    NotificationCenter.default.post(
                        name: .photosFavoriteDidUpdate,
                        object: nil,
                        userInfo: ["assetIdentifiers": identifiers, "isFavorite": next]
                    )
                } catch {
                    self.error = ErrorAlertText.sanitized(error.localizedDescription)
                }
                isUpdating = false
            }
        } label: {
            Image(systemName: isFavorite ? "heart.fill" : "heart")
                .font(.system(size: 12, weight: .semibold))
                .foregroundStyle(.white)
                .shadow(color: .black.opacity(0.8), radius: 2)
                .frame(width: 26, height: 26)
                .background(.black.opacity(0.3), in: Circle())
                .contentShape(Circle())
        }
        .buttonStyle(.plain)
        .opacity(isHovered || isFavorite || isUpdating ? 1 : 0)
        .allowsHitTesting(isHovered || isFavorite)
        .disabled(isUpdating)
        .accessibilityLabel(isFavorite ? "Remove from Favorites" : "Add to Favorites")
        .task(id: "\(url.absoluteString):\(scenePhase)") {
            guard scenePhase == .active,
                  let item = PhotosMedia.shared.item(url) else { return }
            // Refetch after returning from Photos or remounting a scrolled cell.
            let favorite = await PhotosMedia.shared.favoriteStatus(url) ?? item.asset.isFavorite
            guard !Task.isCancelled else { return }
            isFavorite = favorite
        }
        .onReceive(NotificationCenter.default.publisher(for: .photosFavoriteDidUpdate)) { notification in
            guard let identifier = PhotosMedia.shared.item(url)?.asset.localIdentifier,
                  let identifiers = notification.userInfo?["assetIdentifiers"] as? [String],
                  identifiers.contains(identifier),
                  let favorite = notification.userInfo?["isFavorite"] as? Bool else { return }
            isFavorite = favorite
        }
        .alert("Couldn't update Favorite", isPresented: Binding(
            get: { error != nil }, set: { if !$0 { error = nil } }
        )) {
            Button("OK", role: .cancel) { error = nil }
        } message: {
            Text(error ?? "")
        }
    }
}
