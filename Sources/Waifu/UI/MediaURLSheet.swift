import SwiftUI

struct MediaURLSheet: View {
    @ObservedObject var session: WaifuSession
    @Environment(\.dismiss) private var dismiss
    @State private var address = ""
    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Load from URL").font(.headline)
            TextField("https://example.com/media", text: $address)
                .textFieldStyle(.roundedBorder)
                .onSubmit { load() }
            Text("Paste a direct media URL or a completed MPEG-TS HLS playlist. The content determines the format, including links without an extension. Media is downloaded before analysis. Encrypted playlists, live streams and protected media are not supported.")
                .font(.callout).foregroundStyle(.secondary)
            HStack {
                Spacer()
                Button("Cancel", role: .cancel) { dismiss() }
                Button("Load") { load() }.keyboardShortcut(.defaultAction)
                    .disabled(MediaImport.webURL(address) == nil)
            }
        }.padding(24).frame(width: 500)
    }
    private func load() {
        guard let url = MediaImport.webURL(address) else { return }
        session.loadVideo(at: url)
        dismiss()
    }
}
