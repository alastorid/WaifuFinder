import AppKit
import SwiftUI

/// Animated Photos preview with a thumbnail filmstrip that also acts as a scrubber.
/// The supplied content view can be a plain fitted image or the zoomable ranked
/// preview, so both Photos entry points use the same playback controls.
struct AnimatedPhotoPlayer<Content: View>: View {
    let animation: PhotosAnimatedImage
    let isActive: Bool
    private let maximumPixelSize: Int
    private let content: (NSImage, Int) -> Content

    @State private var index = 0
    @State private var isPlaying = true
    @State private var currentImage: NSImage?
    @State private var loadedIndex: Int?
    @State private var filmstrip: [PhotosFilmstripFrame] = []

    init(animation: PhotosAnimatedImage, isActive: Bool = true, maximumPixelSize: Int = 2400,
         @ViewBuilder content: @escaping (NSImage, Int) -> Content) {
        self.animation = animation
        self.isActive = isActive
        self.maximumPixelSize = maximumPixelSize
        self.content = content
    }

    var body: some View {
        VStack(spacing: 10) {
            Group {
                if let currentImage {
                    content(currentImage, index)
                } else {
                    ProgressView("Loading animation…")
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)

            if animation.frameCount > 1 {
                filmstripView
                controls
            }
        }
        .task(id: "\(animation.identity):\(index)") {
            guard let pixels = await animation.frameAsync(at: index, maximumPixelSize: maximumPixelSize),
                  !Task.isCancelled else { return }
            currentImage = NSImage(cgImage: pixels, size: NSSize(width: pixels.width, height: pixels.height))
            loadedIndex = index
        }
        .task(id: animation.identity) {
            filmstrip = []
            try? await Task.sleep(for: .milliseconds(120))
            guard !Task.isCancelled else { return }
            filmstrip = await animation.filmstripFrames()
        }
        .task(id: "\(animation.identity):\(index):\(isPlaying):\(isActive):\(loadedIndex == index)") {
            guard isPlaying, isActive, animation.frameCount > 1, loadedIndex == index else { return }
            try? await Task.sleep(for: .seconds(animation.duration(at: index)))
            guard !Task.isCancelled else { return }
            index = (index + 1) % animation.frameCount
        }
    }

    private var filmstripView: some View {
        GeometryReader { geometry in
            let width = max(1, geometry.size.width)
            let frameWidth = width / CGFloat(max(1, filmstrip.count))
            ZStack(alignment: .leading) {
                HStack(spacing: 0) {
                    ForEach(filmstrip) { frame in
                        Image(nsImage: NSImage(cgImage: frame.image,
                                              size: NSSize(width: frame.image.width, height: frame.image.height)))
                            .resizable()
                            .aspectRatio(contentMode: .fill)
                            .frame(width: frameWidth, height: 56)
                            .clipped()
                    }
                }
                RoundedRectangle(cornerRadius: 3)
                    .strokeBorder(Color.white, lineWidth: 2)
                    .frame(width: max(frameWidth, 12), height: 56)
                    .shadow(color: .black.opacity(0.75), radius: 2)
                    .offset(x: playheadX(width: width, markerWidth: max(frameWidth, 12)))
                    .allowsHitTesting(false)
            }
            .frame(height: 56)
            .clipShape(RoundedRectangle(cornerRadius: 5))
            .contentShape(Rectangle())
            .gesture(DragGesture(minimumDistance: 0).onChanged { value in
                seek(to: value.location.x, width: width)
            })
            .accessibilityLabel("Animation frame strip")
            .accessibilityHint("Drag or click to seek through the animation")
        }
        .frame(height: 56)
    }

    private var controls: some View {
        HStack(spacing: 10) {
            Button {
                isPlaying = false
                index = max(0, index - 1)
            } label: {
                Image(systemName: "backward.frame.fill")
            }

            Button { isPlaying.toggle() } label: {
                Image(systemName: isPlaying ? "pause.fill" : "play.fill")
                    .frame(width: 16)
            }

            Button {
                isPlaying = false
                index = min(animation.frameCount - 1, index + 1)
            } label: {
                Image(systemName: "forward.frame.fill")
            }

            Text(String(format: "%.2f / %.2f sec", animation.elapsed(before: index), animation.duration))
                .font(.system(size: 11, design: .monospaced))
                .foregroundStyle(.secondary)
                .monospacedDigit()
            Spacer(minLength: 8)
            Text("Frame \(index + 1) of \(animation.frameCount)")
                .font(.system(size: 11, design: .monospaced))
                .foregroundStyle(.tertiary)
                .monospacedDigit()
        }
        .buttonStyle(.borderless)
    }

    private func playheadX(width: CGFloat, markerWidth: CGFloat) -> CGFloat {
        guard animation.frameCount > 1 else { return 0 }
        return min(max(0, width - markerWidth),
                   CGFloat(index) / CGFloat(animation.frameCount - 1) * (width - markerWidth))
    }

    private func seek(to location: CGFloat, width: CGFloat) {
        guard animation.frameCount > 1 else { return }
        isPlaying = false
        let fraction = min(1, max(0, location / width))
        index = Int((fraction * CGFloat(animation.frameCount - 1)).rounded())
    }
}
