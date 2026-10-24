import SwiftUI

struct PhotosKeywordJobView: View {
    @ObservedObject var status: PhotosKeywordJobStatus
    let cancel: () -> Void

    var body: some View {
        if let job = status.progress {
            HStack(spacing: 7) {
                Text(job.title)
                Text("\(job.completed)/\(job.total)")
                    .monospacedDigit()
                    .foregroundStyle(.secondary)
                Button(action: cancel) {
                    Image(systemName: "xmark.circle.fill")
                        .font(.system(size: 14))
                }
                .buttonStyle(.borderless)
                .disabled(job.phase == .stopping)
                .accessibilityLabel("Cancel setting keywords from tags")
            }
            .font(.system(size: 12))
            .padding(.horizontal, 9)
            .padding(.vertical, 5)
            .background(.quaternary, in: Capsule())
            .accessibilityElement(children: .contain)
            .accessibilityLabel("Keyword job")
        }
    }
}
