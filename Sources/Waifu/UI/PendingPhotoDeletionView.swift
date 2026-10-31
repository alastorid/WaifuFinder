import SwiftUI

struct PendingPhotoDeletionView: View {
    @ObservedObject var session: WaifuSession

    var body: some View {
        if !session.pendingPhotoDeletions.isEmpty {
            HStack(spacing: 10) {
                Image(systemName: "trash")
                VStack(alignment: .leading, spacing: 2) {
                    Text("\(session.pendingPhotoDeletions.count) photos queued")
                        .font(.caption.weight(.semibold))
                    Text(sizeText).font(.caption2).foregroundStyle(.secondary)
                }
                Button("Clear", action: session.undoPendingPhotoDeletions)
                Button("Delete All", role: .destructive, action: session.commitPendingPhotoDeletions)
                    .disabled(session.progress.isRunning || session.isSyncingPhotosOrder || session.isWritingPhotosTags)
                if session.isDeletingPhotos { ProgressView().controlSize(.small) }
            }
            .controlSize(.small)
            .padding(10)
            .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 12))
            .shadow(radius: 4, y: 2)
            .disabled(session.isDeletingPhotos)
            .transition(.move(edge: .bottom).combined(with: .opacity))
            .animation(.easeInOut(duration: 0.33), value: session.pendingPhotoDeletions.count)
        }
    }

    private var sizeText: String {
        let size = PhotoMetadataFormatter.fileSize(session.pendingDeletionBytes)
        if session.pendingDeletionUnknownSizes == 0 { return size }
        if session.pendingDeletionBytes == 0 { return "Size unavailable for \(session.pendingDeletionUnknownSizes) photos" }
        return "\(size) + \(session.pendingDeletionUnknownSizes) sizes unavailable"
    }
}
