import AppKit
import Darwin
import Foundation
import ImageIO
import CoreImage
import MLX
import UniformTypeIdentifiers

public struct Tile {
    public let x:Int,y:Int,width:Int,height:Int
    public init(x:Int,y:Int,width:Int,height:Int) { self.x=x; self.y=y; self.width=width; self.height=height }
}

public func fail(_ message: String) -> Never { fputs(message + "\n", stderr); exit(1) }
public func option(_ name: String) -> String? {
    guard let i = CommandLine.arguments.firstIndex(of: name), i + 1 < CommandLine.arguments.count else { return nil }
    return CommandLine.arguments[i + 1]
}
public func readRGB(_ url: URL, straightAlpha: Bool = false) throws -> (bytes: [UInt8], width: Int, height: Int) {
    guard let source = CGImageSourceCreateWithURL(url as CFURL, nil), let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
        throw NSError(domain: "WaifuRuntime", code: 1, userInfo: [NSLocalizedDescriptionKey: "Cannot decode degraded frame."])
    }
    // Pillow convert("RGB") reads straight PNG channels, ignoring alpha. Drawing to
    // a premultiplied CGContext darkens translucent Lanczos crop-edge pixels.
    // Keep the established SeedVR2 reader default; FlashVSR explicitly requests
    // straight channels to match its official input preparation exactly.
    if straightAlpha, image.bitsPerComponent == 8, image.bitsPerPixel == 32,
       image.alphaInfo == .last, image.bitmapInfo.intersection(.byteOrderMask).rawValue == 0,
       let stored = image.dataProvider?.data {
        let source = stored as Data
        var bytes = [UInt8](repeating: 0, count: image.width * image.height * 4)
        for y in 0..<image.height {
            bytes.replaceSubrange((y*image.width*4)..<((y+1)*image.width*4),
                                  with: source[(y*image.bytesPerRow)..<(y*image.bytesPerRow+image.width*4)])
        }
        return (bytes,image.width,image.height)
    }
    var bytes = [UInt8](repeating: 0, count: image.width * image.height * 4)
    let drawn = bytes.withUnsafeMutableBytes { buffer -> Bool in
        guard let context = CGContext(data: buffer.baseAddress, width: image.width, height: image.height,
            bitsPerComponent: 8, bytesPerRow: image.width * 4, space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return false }
        context.draw(image, in: CGRect(x: 0, y: 0, width: image.width, height: image.height))
        return true
    }
    guard drawn else { throw NSError(domain: "WaifuRuntime", code: 2) }
    if straightAlpha {
        for offset in stride(from: 0, to: bytes.count, by: 4) where bytes[offset+3] > 0 && bytes[offset+3] < 255 {
            let alpha = Double(bytes[offset+3])
            for c in 0..<3 { bytes[offset+c] = UInt8(min(255,(Double(bytes[offset+c])*255/alpha).rounded())) }
        }
    }
    return (bytes, image.width, image.height)
}
// Pillow's a=-0.5 bicubic kernel, separable and rounded to 8-bit per pass. The
// conditioning upscale is performed on degraded pixels only.
public func cubic(_ value: Double) -> Double {
    let x = abs(value)
    if x < 1 { return ((1.5 * x - 2.5) * x) * x + 1 }
    if x < 2 { return ((-0.5 * x + 2.5) * x - 4) * x + 2 }
    return 0
}
public func resize(_ input: (bytes: [UInt8], width: Int, height: Int), factor: Int, rect: Tile) -> [Float] {
    precondition(factor > 0)
    let width = rect.width, height = rect.height
    var horizontal = [UInt8](repeating: 0, count: width * input.height * 3)
    func coefficients(_ i: Int, length: Int) -> [(Int, Double)] {
        let center = (Double(i) + 0.5) / Double(factor) - 0.5
        let lo = max(0, Int(floor(center)) - 1), hi = min(length - 1, Int(floor(center)) + 2)
        let weights = (lo...hi).map { ($0, cubic(Double($0) - center)) }
        let total = weights.reduce(0) { $0 + $1.1 }
        return weights.map { ($0.0, $0.1 / total) }
    }
    let xs = (0..<width).map { coefficients(min(input.width * factor - 1, $0 + rect.x), length: input.width) }
    let ys = (0..<height).map { coefficients(min(input.height * factor - 1, $0 + rect.y), length: input.height) }
    for y in 0..<input.height { for x in 0..<width { for c in 0..<3 {
        let value = xs[x].reduce(0.0) { $0 + Double(input.bytes[(y * input.width + $1.0) * 4 + c]) * $1.1 }
        horizontal[(y * width + x) * 3 + c] = UInt8(max(0, min(255, value.rounded())))
    } } }
    var output = [Float](repeating: 0, count: width * height * 3)
    for y in 0..<height { for x in 0..<width { for c in 0..<3 {
        let value = ys[y].reduce(0.0) { $0 + Double(horizontal[($1.0 * width + x) * 3 + c]) * $1.1 }
        output[(y * width + x) * 3 + c] = Float(max(0, min(255, value.rounded()))) / 127.5 - 1
    } } }
    return output
}
public func resize(_ input: (bytes: [UInt8], width: Int, height: Int),
                   toWidth targetWidth: Int, toHeight targetHeight: Int, rect: Tile) -> [Float] {
    precondition(input.width > 0 && input.height > 0 && targetWidth > 0 && targetHeight > 0)
    let width = rect.width, height = rect.height
    var horizontal = [UInt8](repeating: 0, count: width * input.height * 3)
    func coefficients(_ i: Int, offset: Int, sourceLength: Int, destinationLength: Int) -> [(Int, Double)] {
        let destination = min(destinationLength - 1, max(0, i + offset))
        let center = (Double(destination) + 0.5) * Double(sourceLength) / Double(destinationLength) - 0.5
        let lo = max(0, Int(floor(center)) - 1), hi = min(sourceLength - 1, Int(floor(center)) + 2)
        let weights = (lo...hi).map { ($0, cubic(Double($0) - center)) }
        let total = weights.reduce(0) { $0 + $1.1 }
        return weights.map { ($0.0, $0.1 / total) }
    }
    let xs = (0..<width).map {
        coefficients($0, offset: rect.x, sourceLength: input.width, destinationLength: targetWidth)
    }
    let ys = (0..<height).map {
        coefficients($0, offset: rect.y, sourceLength: input.height, destinationLength: targetHeight)
    }
    for y in 0..<input.height { for x in 0..<width { for c in 0..<3 {
        let value = xs[x].reduce(0.0) { $0 + Double(input.bytes[(y * input.width + $1.0) * 4 + c]) * $1.1 }
        horizontal[(y * width + x) * 3 + c] = UInt8(max(0, min(255, value.rounded())))
    } } }
    var output = [Float](repeating: 0, count: width * height * 3)
    for y in 0..<height { for x in 0..<width { for c in 0..<3 {
        let value = ys[y].reduce(0.0) { $0 + Double(horizontal[($1.0 * width + x) * 3 + c]) * $1.1 }
        output[(y * width + x) * 3 + c] = Float(max(0, min(255, value.rounded()))) / 127.5 - 1
    } } }
    return output
}
public func save(_ array: MLXArray, width: Int, height: Int, targetWidth: Int, targetHeight: Int, to url: URL) throws {
    let pixels = clip((array + 1) * 127.5, min: 0, max: 255).asType(.uint8).asArray(UInt8.self)
    let data = Data(pixels)
    guard let provider = CGDataProvider(data: data as CFData),
          let image = CGImage(width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 24,
            bytesPerRow: width * 3, space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGBitmapInfo(rawValue: 0),
            provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent),
          let destination = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil) else {
        throw NSError(domain: "WaifuRuntime", code: 3)
    }
    let sx = Double(targetWidth) / Double(width), sy = Double(targetHeight) / Double(height)
    let resized = CIImage(cgImage: image).applyingFilter("CILanczosScaleTransform", parameters: [
        kCIInputScaleKey: sy, kCIInputAspectRatioKey: sx / sy])
    let context = CIContext(options: [.useSoftwareRenderer: true, .cacheIntermediates: false])
    guard let final = (width == targetWidth && height == targetHeight ? image
        : context.createCGImage(resized, from: CGRect(x: 0, y: 0, width: targetWidth, height: targetHeight))) else {
        throw NSError(domain: "WaifuRuntime", code: 5)
    }
    CGImageDestinationAddImage(destination, final, nil)
    guard CGImageDestinationFinalize(destination) else { throw NSError(domain: "WaifuRuntime", code: 4) }
}
public func footprint() -> UInt64 {
    var info = task_vm_info_data_t()
    var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
    let status = withUnsafeMutablePointer(to: &info) { pointer in
        pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
            task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
        }
    }
    return status == KERN_SUCCESS ? info.phys_footprint : 0
}
public final class MemoryMonitor: @unchecked Sendable {
    let lock = NSLock()
    var peak: UInt64 = 0
    let timer = DispatchSource.makeTimerSource(queue: DispatchQueue(label: "native-memory"))
    public init(budget: Int) {
        timer.schedule(deadline: .now(), repeating: .milliseconds(20))
        timer.setEventHandler { [weak self] in
            let bytes = footprint()
            guard let self else { return }
            self.lock.lock(); self.peak = max(self.peak, bytes); self.lock.unlock()
            if bytes > UInt64(budget) {
                fputs(String(format:"Native inference stopped at its physical-memory budget (%.3f GB / %.3f GB). Reduce the selection or interval.\n",Double(bytes)/1e9,Double(budget)/1e9), stderr)
                fflush(stderr)
                // GPU work may still own MLX synchronization objects. A hard
                // resource stop must not run their teardown on this monitor
                // thread while inference continues on another thread.
                _exit(75)
            }
        }
        timer.resume()
    }
    deinit { timer.cancel() }
    public var peakBytes: UInt64 { lock.lock(); defer { lock.unlock() }; return max(peak, footprint()) }
}
