import AppKit
import QuartzCore
import SwiftUI

enum GridThumbnailLayout {
    static func size(width: CGFloat, sourceAspect: CGFloat, square: Bool) -> CGSize {
        let aspect = sourceAspect.isFinite && sourceAspect > 0 ? sourceAspect : 16.0 / 9
        return CGSize(width: width, height: square ? width : width / aspect)
    }
}

/// A decoded image with an explicit compositor animation. A prefetched image
/// can be present in makeNSView; it still fades on its first viewport entry.
struct GridThumbnailImage: NSViewRepresentable {
    let image: CGImage?
    let identity: String
    let cropToFill: Bool
    let isVisible: Bool

    func makeNSView(context: Context) -> GridThumbnailImageView {
        GridThumbnailImageView(frame: .zero)
    }

    func updateNSView(_ view: GridThumbnailImageView, context: Context) {
        view.configure(image: image, identity: identity, cropToFill: cropToFill, isVisible: isVisible)
    }

    static func dismantleNSView(_ view: GridThumbnailImageView, coordinator: ()) {
        view.configure(image: nil, identity: "", cropToFill: false, isVisible: false)
    }
}

@MainActor
final class GridThumbnailImageView: NSView {
    private let picture = CALayer()
    private let loader: GridThumbnailLoader
    private let mediaTime: () -> CFTimeInterval
    private var identity: String?
    private var image: CGImage?
    private var revealed = false
    private var wasVisible = false

    init(frame: NSRect, loader: GridThumbnailLoader = .shared,
         mediaTime: @escaping () -> CFTimeInterval = CACurrentMediaTime) {
        self.loader = loader
        self.mediaTime = mediaTime
        super.init(frame: frame)
        wantsLayer = true
        picture.masksToBounds = true
        picture.cornerRadius = 3
        picture.opacity = 0
        layer?.addSublayer(picture)
    }

    required init?(coder: NSCoder) { fatalError("init(coder:) has not been implemented") }

    // SwiftUI owns selection, double-click, hover, and favorite controls.
    override func hitTest(_ point: NSPoint) -> NSView? { nil }

    override func layout() {
        super.layout()
        CATransaction.begin()
        CATransaction.setDisableActions(true)
        picture.frame = bounds
        CATransaction.commit()
    }

    func configure(image: CGImage?, identity: String, cropToFill: Bool, isVisible: Bool) {
        CATransaction.begin()
        CATransaction.setDisableActions(true)
        picture.frame = bounds
        picture.contentsGravity = cropToFill ? .resizeAspectFill : .resizeAspect
        effectiveAppearance.performAsCurrentDrawingAppearance {
            picture.backgroundColor = NSColor.windowBackgroundColor.cgColor
        }
        if self.identity != identity || self.image !== image {
            picture.removeAnimation(forKey: "thumbnailReveal")
            self.identity = identity
            self.image = image
            revealed = false
            picture.contents = image
            picture.opacity = 0
        }
        if !isVisible { picture.removeAnimation(forKey: "thumbnailReveal") }
        let firstVisiblePixels = isVisible && image != nil && (!revealed || !wasVisible)
        wasVisible = isVisible
        if firstVisiblePixels {
            revealed = true
            picture.opacity = 1
        }
        CATransaction.commit()

        guard firstVisiblePixels else { return }
        let now = mediaTime()
        let start = loader.revealStart(identity: identity, at: now)
        guard now - start < 0.33 else { return }
        let fade = CABasicAnimation(keyPath: "opacity")
        fade.fromValue = Float(0)
        fade.toValue = Float(1)
        fade.duration = 0.33
        fade.timingFunction = CAMediaTimingFunction(name: .easeIn)
        fade.beginTime = picture.convertTime(start, from: nil)
        picture.add(fade, forKey: "thumbnailReveal")
    }
}
