import AppKit
import CoreML
import Foundation
import Darwin

/// Core ML x2 image restoration used only by the dedicated RealESRGAN algorithm.
final class NativeSpatialUpscaler {
    private struct RGBImage {
        let width: Int
        let height: Int
        var pixels: [UInt8] // interleaved RGB
    }

    private let model: MLModel
    private let inputName = "input"
    private let outputName: String
    private let modelSize = 522
    private let supportedSizes: [Int]
    private let contextHalo: Int
    private let inputType: MLMultiArrayDataType
    private var inputBuffers: [Int:MLMultiArray] = [:]
    private(set) var predictionSeconds = 0.0
    private(set) var predictionSizes = [Int]()
    let selectedComputeUnits: String
    private let tileSize = 512
    private let overlap = 32
    private var rgbaBitmapInfo: UInt32 {
        CGBitmapInfo.byteOrder32Big.rawValue | CGImageAlphaInfo.premultipliedLast.rawValue
    }

    private static var hardwareModel: String {
        var size = 0
        guard sysctlbyname("hw.model",nil,&size,nil,0) == 0 else { return "unknown" }
        var bytes = [UInt8](repeating:0,count:size)
        guard sysctlbyname("hw.model",&bytes,&size,nil,0) == 0 else { return "unknown" }
        return String(decoding:bytes.prefix { $0 != 0 },as:UTF8.self)
    }

    init(modelURL: URL) throws {
        let configuration = MLModelConfiguration(); configuration.computeUnits = .all
        configuration.allowLowPrecisionAccumulationOnGPU = false
        // Validated packages record the measured device choice. Legacy imports
        // retain Core ML's original automatic selection.
        let probe = try MLModel(contentsOf:modelURL,configuration:configuration)
        let metadata = probe.modelDescription.metadata[.creatorDefinedKey] as? [String:String] ?? [:]
        let choice = metadata["waifu.profileHardware"] == Self.hardwareModel ? (metadata["waifu.computeUnits"] ?? "all") : "all"
        switch choice {
        case "gpu": configuration.computeUnits = .cpuAndGPU
        case "ane": configuration.computeUnits = .cpuAndNeuralEngine
        default: configuration.computeUnits = .all
        }
        configuration.allowLowPrecisionAccumulationOnGPU = false
        selectedComputeUnits = choice
        model = choice == "all" ? probe : try MLModel(contentsOf:modelURL,configuration:configuration)
        guard model.modelDescription.inputDescriptionsByName[inputName]?.multiArrayConstraint != nil,
              let outputName = model.modelDescription.outputDescriptionsByName.keys.sorted().first else {
            throw UncoverError.backend("The installed spatial upscaler has an unsupported model interface.")
        }
        let verified = metadata["waifu.optimizationVersion"] == "rrdb-fused-v1" && metadata["waifu.validated"] == "true"
            && metadata["waifu.weightsSHA256"] == "49fafd45f8fd7aa8d31ab2a22d14d91b536c34494a5cfe31eb5d89c2fa266abb"
        guard let input = model.modelDescription.inputDescriptionsByName[inputName]?.multiArrayConstraint,
              input.shape.map(\.intValue) == [1, 3, modelSize, modelSize],
              let output = model.modelDescription.outputDescriptionsByName[outputName]?.multiArrayConstraint,
              (output.shape.map(\.intValue) == [1,3,modelSize*2,modelSize*2] || (verified && output.shape.isEmpty)) else {
            throw UncoverError.backend("The installed spatial upscaler has an unexpected tensor size.")
        }
        self.outputName = outputName
        inputType = input.dataType
        guard inputType == .float16 || inputType == .float32 else { throw UncoverError.backend("Unsupported spatial input precision.") }
        let shapes = input.shapeConstraint.enumeratedShapes.map { $0.map(\.intValue) }
        supportedSizes = verified ? shapes.compactMap { shape in
            shape.count == 4 && shape[0] == 1 && shape[1] == 3 && shape[2] == shape[3] && [256,384,522].contains(shape[2]) ? shape[2] : nil
        }.sorted() : [522]
        contextHalo = verified ? 160 : 10
    }

    func upscale(_ source: CGImage) throws -> CGImage {
        var image = try rgbImage(source)
        image = try upscaleOnce(image)
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        var rgba = [UInt8](repeating: 255, count: image.width * image.height * 4)
        for pixel in 0..<(image.width * image.height) {
            let sourceOffset = pixel * 3, destination = pixel * 4
            rgba[destination] = image.pixels[sourceOffset]
            rgba[destination + 1] = image.pixels[sourceOffset + 1]
            rgba[destination + 2] = image.pixels[sourceOffset + 2]
        }
        guard let provider = CGDataProvider(data: Data(rgba) as CFData),
              let result = CGImage(width: image.width, height: image.height, bitsPerComponent: 8,
                                   bitsPerPixel: 32, bytesPerRow: image.width * 4, space: colorSpace,
                                   bitmapInfo: CGBitmapInfo(rawValue: rgbaBitmapInfo),
                                   provider: provider, decode: nil, shouldInterpolate: false,
                                   intent: .defaultIntent) else {
            throw UncoverError.backend("Could not create the upscaled frame.")
        }
        return result
    }

    private func rgbImage(_ image: CGImage) throws -> RGBImage {
        RGBImage(width:image.width,height:image.height,pixels:try Self.straightRGBPixels(image))
    }

    static func straightRGBPixels(_ image: CGImage) throws -> [UInt8] {
        var pixels = [UInt8](repeating:0,count:image.width*image.height*3)
        // PNG stores straight channels. Ignoring alpha matches the original
        // model's RGB conversion and avoids darkening translucent Lanczos edges.
        if image.bitsPerComponent == 8, image.bitsPerPixel == 32, image.alphaInfo == .last,
           image.bitmapInfo.intersection(.byteOrderMask).rawValue == 0, let data = image.dataProvider?.data {
            let bytes = data as Data
            for y in 0..<image.height { for x in 0..<image.width { for c in 0..<3 {
                pixels[(y*image.width+x)*3+c] = bytes[y*image.bytesPerRow+x*4+c]
            } } }
            return pixels
        }
        var rgba = [UInt8](repeating:0,count:image.width*image.height*4)
        let drawn = rgba.withUnsafeMutableBytes { buffer -> Bool in
            guard let context = CGContext(data:buffer.baseAddress,width:image.width,height:image.height,
                bitsPerComponent:8,bytesPerRow:image.width*4,space:CGColorSpaceCreateDeviceRGB(),
                bitmapInfo:CGBitmapInfo.byteOrder32Big.rawValue | CGImageAlphaInfo.premultipliedLast.rawValue) else { return false }
            context.draw(image,in:CGRect(x:0,y:0,width:image.width,height:image.height))
            return true
        }
        guard drawn else { throw UncoverError.backend("Could not read the spatial restoration input.") }
        for pixel in 0..<image.width*image.height {
            let alpha = Int(rgba[pixel*4+3])
            for c in 0..<3 {
                let value = Int(rgba[pixel*4+c])
                pixels[pixel*3+c] = UInt8(alpha > 0 && alpha < 255 ? min(255,(value*255+alpha/2)/alpha) : value)
            }
        }
        return pixels
    }

    private func starts(_ size: Int) -> [Int] {
        guard size > tileSize else { return [0] }
        var values = [0]
        while let previous = values.last, previous + tileSize < size {
            var next = previous + tileSize - overlap
            let remaining = size - next
            if remaining > tileSize && remaining - tileSize <= overlap { next = size - tileSize }
            guard next > previous else { break }
            values.append(next)
        }
        return values
    }

    static func reflected(_ index: Int, count: Int) -> Int {
        guard count > 1 else { return 0 }
        let period = 2 * (count - 1)
        let value = ((index % period) + period) % period
        return value < count ? value : period - value
    }

    private func upscaleOnce(_ input: RGBImage) throws -> RGBImage {
        let outputWidth = input.width * 2, outputHeight = input.height * 2
        if input.width <= tileSize && input.height <= tileSize {
            // A single tile needs no accumulation, weight image or blend pass.
            let floats = try predict(input,x:0,y:0,width:input.width,height:input.height)
            let bytes = floats.map { UInt8(min(255,max(0,($0*255).rounded()))) }
            return RGBImage(width:outputWidth,height:outputHeight,pixels:bytes)
        }
        var sums = [Float](repeating: 0, count: outputWidth * outputHeight * 3)
        var weights = [Float](repeating: 0, count: outputWidth * outputHeight)
        let ys = starts(input.height), xs = starts(input.width)

        for (row, y0) in ys.enumerated() {
            for (column, x0) in xs.enumerated() {
                let tileWidth = min(tileSize, input.width - x0)
                let tileHeight = min(tileSize, input.height - y0)
                try autoreleasepool {
                let pixels = try predict(input, x: x0, y: y0, width: tileWidth, height: tileHeight)
                let blendInX = column > 0 ? (xs[column - 1] + tileSize - x0) * 2 : 0
                let blendOutX = column + 1 < xs.count ? (x0 + tileWidth - xs[column + 1]) * 2 : 0
                let blendInY = row > 0 ? (ys[row - 1] + tileSize - y0) * 2 : 0
                let blendOutY = row + 1 < ys.count ? (y0 + tileHeight - ys[row + 1]) * 2 : 0
                for localY in 0..<(tileHeight * 2) {
                    let wy = ramp(localY, length: tileHeight * 2, blendIn: blendInY, blendOut: blendOutY)
                    for localX in 0..<(tileWidth * 2) {
                        let wx = ramp(localX, length: tileWidth * 2, blendIn: blendInX, blendOut: blendOutX)
                        let weight = wx * wy
                        let destinationPixel = (y0 * 2 + localY) * outputWidth + x0 * 2 + localX
                        let sourcePixel = localY * tileWidth * 2 + localX
                        weights[destinationPixel] += weight
                        for channel in 0..<3 {
                            sums[destinationPixel * 3 + channel] += pixels[sourcePixel * 3 + channel] * weight
                        }
                    }
                }
                }
            }
        }
        var pixels = [UInt8](repeating: 0, count: sums.count)
        for pixel in weights.indices {
            let weight = max(weights[pixel], 1e-6)
            for channel in 0..<3 {
                pixels[pixel * 3 + channel] = UInt8(min(255, max(0, (sums[pixel * 3 + channel] / weight * 255).rounded())))
            }
        }
        return RGBImage(width: outputWidth, height: outputHeight, pixels: pixels)
    }

    private func ramp(_ position: Int, length: Int, blendIn: Int, blendOut: Int) -> Float {
        var value: Float = 1
        if blendIn > 0 && position < blendIn { value *= Float(position + 1) / Float(blendIn + 1) }
        if blendOut > 0 && position >= length - blendOut { value *= Float(length - position) / Float(blendOut + 1) }
        return value
    }

    private func predict(_ image: RGBImage, x: Int, y: Int, width: Int, height: Int) throws -> [Float] {
        guard width > 0, height > 0, x >= 0, y >= 0,
              x + width <= image.width, y + height <= image.height,
              image.pixels.count == image.width * image.height * 3 else {
            throw UncoverError.backend("The spatial upscaler received an invalid frame tile.")
        }
        let size = supportedSizes.first { $0 >= max(width,height)+contextHalo } ?? 522
        let input: MLMultiArray
        if let reused = inputBuffers[size] { input = reused }
        else {
            input = try MLMultiArray(shape:[1,3,NSNumber(value:size),NSNumber(value:size)],dataType:inputType)
            if inputBuffers.count >= 2 { inputBuffers.removeAll(keepingCapacity:true) }
            inputBuffers[size] = input
        }
        let inputStrides = input.strides.map(\.intValue)
        let floatValues = input.dataPointer.assumingMemoryBound(to:Float.self)
        let halfValues = input.dataPointer.assumingMemoryBound(to:Float16.self)
        let reflectedX = (0..<size).map { Self.reflected($0,count:width) }
        let reflectedY = (0..<size).map { Self.reflected($0,count:height) }
        for row in 0..<size {
            // Reflect every padded sample inside the actual tile. Reflecting against
            // `height + 10`/`width + 10` lets partial edge tiles address pixels past
            // the end of the source image (for example, a 33px final tile).
            let localY = reflectedY[row]
            let sourceY = y + localY
            for column in 0..<size {
                let localX = reflectedX[column]
                let sourceX = x + localX
                let sourcePixel = (sourceY * image.width + sourceX) * 3
                for channel in 0..<3 {
                    let target = channel * inputStrides[1] + row * inputStrides[2] + column * inputStrides[3]
                    let value = Float(image.pixels[sourcePixel + channel])/255
                    if inputType == .float16 { halfValues[target] = Float16(value) }
                    else { floatValues[target] = value }
                }
            }
        }
        let provider = try MLDictionaryFeatureProvider(dictionary: [inputName: MLFeatureValue(multiArray: input)])
        let tick = Date()
        let prediction = try model.prediction(from: provider)
        predictionSeconds += Date().timeIntervalSince(tick)
        predictionSizes.append(size)
        guard let output = prediction.featureValue(for: outputName)?.multiArrayValue,
              output.shape.count == 4, output.shape[0].intValue == 1,
              output.shape[1].intValue == 3, output.shape[2].intValue == size * 2,
              output.shape[3].intValue == size * 2 else {
            throw UncoverError.backend("The spatial upscaler returned an unexpected frame size.")
        }
        let strides = output.strides.map(\.intValue)
        var result = [Float](repeating: 0, count: width * height * 4 * 3)
        for row in 0..<(height * 2) {
            for column in 0..<(width * 2) {
                let pixel = row * width * 2 + column
                for channel in 0..<3 {
                    let index = channel * strides[1] + row * strides[2] + column * strides[3]
                    let value: Float
                    switch output.dataType {
                    case .float32: value = output.dataPointer.assumingMemoryBound(to: Float.self)[index]
                    case .float16: value = Float(output.dataPointer.assumingMemoryBound(to: Float16.self)[index])
                    default: throw UncoverError.backend("The spatial upscaler returned an unsupported pixel type.")
                    }
                    guard value.isFinite else { throw UncoverError.backend("The spatial upscaler returned a nonfinite pixel.") }
                    result[pixel * 3 + channel] = min(1, max(0, value))
                }
            }
        }
        return result
    }
}
