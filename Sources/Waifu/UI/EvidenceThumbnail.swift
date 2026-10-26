import AppKit
import SwiftUI

/// A real, bounded raster thumbnail; never retains a second full-size output.
enum EvidenceThumbnailRaster {
    static let maximumSide = 310

    static func make(_ image: CGImage) -> CGImage? {
        let scale = min(1, Double(maximumSide) / Double(max(image.width, image.height)))
        let width = max(1, Int((Double(image.width) * scale).rounded()))
        let height = max(1, Int((Double(image.height) * scale).rounded()))
        guard let context = CGContext(data: nil, width: width, height: height, bitsPerComponent: 8,
            bytesPerRow: width * 4, space: CGColorSpace(name: CGColorSpace.sRGB)!,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return nil }
        context.interpolationQuality = .high
        context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
        return context.makeImage()
    }
}

struct EvidenceThumbnail: View {
    let image: CGImage?
    let label: String
    let detail: String
    let selected: Bool
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            VStack(spacing: 4) {
                ZStack {
                    Color.secondary.opacity(0.08)
                    if let image {
                        Image(decorative: image, scale: 2).resizable().aspectRatio(contentMode: .fit)
                    } else {
                        Image(systemName: "photo").foregroundStyle(.tertiary)
                    }
                }
                .frame(width: 155, height: 88)
                .clipped()
                Text(label).font(.caption).lineLimit(1).truncationMode(.tail)
                    .frame(width: 145, height: 18)
            }
            .frame(width: 155, height: 116)
            .background(selected ? Color.accentColor.opacity(0.12) : Color.clear)
            .overlay {
                RoundedRectangle(cornerRadius: 5)
                    .strokeBorder(selected ? Color.accentColor : Color.secondary.opacity(0.25), lineWidth: selected ? 2 : 1)
            }
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .help(detail)
        .accessibilityLabel(label)
        .accessibilityAddTraits(selected ? .isSelected : [])
    }
}
