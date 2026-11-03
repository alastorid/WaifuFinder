import Foundation

/// Pure wrapping geometry. No tag data, score reads or scroll-origin input.
struct TagFlowGeometry {
    let placements: [CGRect]
    let size: CGSize

    init(sizes: [CGSize], width: CGFloat, spacing: CGFloat) {
        var placements: [CGRect] = []
        placements.reserveCapacity(sizes.count)
        var x: CGFloat = 0, y: CGFloat = 0, lineHeight: CGFloat = 0, usedWidth: CGFloat = 0
        for size in sizes {
            if x > 0 && x + size.width > width {
                x = 0
                y += lineHeight + spacing
                lineHeight = 0
            }
            placements.append(CGRect(origin: CGPoint(x: x, y: y), size: size))
            x += size.width + spacing
            lineHeight = max(lineHeight, size.height)
            usedWidth = max(usedWidth, x - spacing)
        }
        self.placements = placements
        self.size = CGSize(width: width.isFinite ? min(width, usedWidth) : usedWidth,
                           height: sizes.isEmpty ? 0 : y + lineHeight)
    }
}
