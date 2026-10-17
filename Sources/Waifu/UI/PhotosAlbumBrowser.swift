import Photos
import SwiftUI

/// Shared order glyph for the toolbar filter and the small synced-photo badge.
struct PhotosOrderIcon: View {
    let filter: PhotosOrderFilter

    var body: some View {
        Text("≡")
            .font(.system(size: 18, weight: filter == .ordered ? .black : .regular))
            .frame(width: 20, height: 20)
            .overlay {
                if filter == .unordered {
                    Path { path in
                        path.move(to: CGPoint(x: 3, y: 18))
                        path.addLine(to: CGPoint(x: 17, y: 2))
                    }
                    .stroke(style: StrokeStyle(lineWidth: 1.5, lineCap: .round))
                }
            }
    }
}

/// Favorite writes stay on the thumbnail and don't change grid selection.
struct PhotosFavoriteButton: View {
    let url: URL
    let isHovered: Bool
    @Environment(\.scenePhase) private var scenePhase
    @State private var isFavorite = false
    @State private var isUpdating = false
    @State private var error: String?

    var body: some View {
        Button {
            guard !isUpdating else { return }
            let next = !isFavorite
            isUpdating = true
            Task {
                do {
                    try await PhotosMedia.shared.setFavorite(url, isFavorite: next)
                    isFavorite = next
                } catch {
                    self.error = error.localizedDescription
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
        .help(isFavorite ? "Remove from Favorites" : "Add to Favorites")
        .accessibilityLabel(isFavorite ? "Remove from Favorites" : "Add to Favorites")
        .task(id: "\(url.absoluteString):\(scenePhase)") {
            guard scenePhase == .active,
                  let item = PhotosMedia.shared.item(url) else { return }
            // Refetch after returning from Photos or remounting a scrolled cell.
            isFavorite = PHAsset.fetchAssets(withLocalIdentifiers: [item.asset.localIdentifier], options: nil)
                .firstObject?.isFavorite ?? item.asset.isFavorite
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
