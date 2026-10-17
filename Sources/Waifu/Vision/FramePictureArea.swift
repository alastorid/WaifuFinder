import CoreGraphics

/// Size of the picture after contiguous near-black outer padding is excluded.
/// A small raster is sufficient for border bounds; native pixels remain the source
/// for detection and definition. A completely black frame keeps its full area.
enum FramePictureArea {
    static func fraction(_ image: CGImage) -> Float {
        let picture = bounds(image)
        return Float(picture.width * picture.height) / Float(image.width * image.height)
    }

    static func bounds(_ image: CGImage) -> CGRect {
        let full = CGRect(x: 0, y: 0, width: image.width, height: image.height)
        let scale = min(1, 256.0 / Double(max(image.width, image.height)))
        let width = max(1, Int((Double(image.width) * scale).rounded()))
        let height = max(1, Int((Double(image.height) * scale).rounded()))
        var pixels = [UInt8](repeating: 0, count: width * height)
        let rendered = pixels.withUnsafeMutableBytes { bytes -> Bool in
            guard let context = CGContext(data: bytes.baseAddress, width: width, height: height,
                bitsPerComponent: 8, bytesPerRow: width, space: CGColorSpaceCreateDeviceGray(),
                bitmapInfo: CGImageAlphaInfo.none.rawValue) else { return false }
            context.interpolationQuality = .low
            context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
            return true
        }
        guard rendered else { return full }
        var left = width, right = -1, top = height, bottom = -1
        for y in 0..<height {
            for x in 0..<width where pixels[y * width + x] > 8 {
                left = min(left, x); right = max(right, x)
                top = min(top, y); bottom = max(bottom, y)
            }
        }
        guard right >= left, bottom >= top else { return full }
        // Require substantial black padding on opposing edges. Small dark edges
        // and ordinary dark pictures should not change the analysis geometry.
        let horizontalPadding = left >= max(2, width / 25) && width - right - 1 >= max(2, width / 25)
        let verticalPadding = top >= max(2, height / 25) && height - bottom - 1 >= max(2, height / 25)
        guard horizontalPadding || verticalPadding else { return full }
        // Ignore single-pixel rounding changes, and never normalize by zero area.
        left = max(0, left - 1); right = min(width - 1, right + 1)
        top = max(0, top - 1); bottom = min(height - 1, bottom + 1)
        let sx = Double(image.width) / Double(width), sy = Double(image.height) / Double(height)
        return CGRect(x: Double(left) * sx, y: Double(top) * sy,
                      width: Double(right - left + 1) * sx, height: Double(bottom - top + 1) * sy)
            .integral.intersection(full)
    }
}
