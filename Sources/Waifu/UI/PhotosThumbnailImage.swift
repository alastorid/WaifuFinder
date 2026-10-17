import SwiftUI

/// The container sets the layout before the image is fitted or cropped, so an
/// image's intrinsic size cannot override the chosen thumbnail shape.
struct PhotosThumbnailImage: View {
    let image: NSImage?
    let sourceAspect: CGFloat
    let square: Bool

    var body: some View {
        let aspect = square ? 1 : (sourceAspect.isFinite && sourceAspect > 0 ? sourceAspect : 1)
        Rectangle()
            .fill(Color(nsColor: .controlBackgroundColor))
            .aspectRatio(aspect, contentMode: .fit)
            .overlay {
                if let image {
                    GeometryReader { geometry in
                        Image(nsImage: image).resizable()
                            .aspectRatio(contentMode: square ? .fill : .fit)
                            .frame(width: geometry.size.width, height: geometry.size.height)
                    }
                }
            }
            .clipped()
    }
}
