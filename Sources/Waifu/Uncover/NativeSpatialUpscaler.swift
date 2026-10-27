import AppKit
import CoreML
import Foundation
import Darwin

enum RealESRGANVariant: String, CaseIterable, Codable, Sendable, Identifiable {
    case x2plus, x4plus
    var id: String { rawValue }
    var scale: Int { self == .x2plus ? 2 : 4 }
    var label: String { self == .x2plus ? "x2" : "x4" }
    var modelName: String { "RealESRGAN \(rawValue) · Core ML FP16" }
    var weightsSHA256: String {
        self == .x2plus ? "49fafd45f8fd7aa8d31ab2a22d14d91b536c34494a5cfe31eb5d89c2fa266abb" : "4fa0d38905f75ac06eb49a7951b426670021be3018265fd191d2125df9d682f1"
    }
    static let maximumOutputPixels = 256_000_000
    func accepts(width: Int, height: Int) -> Bool {
        width > 0 && height > 0 && width <= 16_384 / scale && height <= 16_384 / scale
            && width * height <= Self.maximumOutputPixels / (scale * scale)
    }
    static let runtimeRevision = "realesrgan-native-images-v5-ane"
    var configurationName: String { self == .x2plus ? "realesrgan-model.txt" : "realesrgan-x4plus-model.txt" }
    var configurationURL: URL {
        FileManager.default.urls(for:.applicationSupportDirectory,in:.userDomainMask)[0]
            .appendingPathComponent("Waifu/Uncover/" + configurationName)
    }
    var modelURL: URL? {
        if let data = try? Data(contentsOf:configurationURL),
           let text = String(data:data,encoding:.utf8)?.trimmingCharacters(in:.whitespacesAndNewlines), !text.isEmpty {
            return URL(fileURLWithPath:text)
        }
        guard self == .x2plus,
              let data = try? Data(contentsOf:configurationURL.deletingLastPathComponent().appendingPathComponent("backend.json")),
              let fields = try? JSONSerialization.jsonObject(with:data) as? [String:Any],
              let path = fields["spatialUpscaler"] as? String else { return nil }
        return URL(fileURLWithPath:path)
    }
    var cacheIdentity: String {
        let url = modelURL
        let modified = url.flatMap { try? $0.resourceValues(forKeys:[.contentModificationDateKey]).contentModificationDate }?.timeIntervalSince1970 ?? 0
        return "\(Self.runtimeRevision):\(rawValue):\(weightsSHA256):\(url?.path ?? "missing"):\(modified)"
    }
}

/// Shared native Core ML inference; x2 preprocessing and installed paths stay compatible.
final class NativeSpatialUpscaler {
    private struct RGBImage {
        let width: Int
        let height: Int
        var pixels: [UInt8] // interleaved RGB
    }

    private let model: MLModel
    private let inputName = "input"
    private let outputName: String
    private let modelSize: Int
    private let scale: Int
    private var x4Prediction: (x:Int, y:Int, tensor:MLMultiArray)?
    private let supportedSizes: [Int]
    private let x4ModelSize: Int
    private let contextHalo: Int
    private let inputType: MLMultiArrayDataType
    private var inputBuffers: [Int:MLMultiArray] = [:]
    #if !arch(arm64)
    private var floatInputBuffers: [Int:MLMultiArray] = [:]
    #endif
    private(set) var predictionSeconds = 0.0
    private(set) var predictionSizes = [Int]()
    let selectedComputeUnits: String
    private let tileSize = 512
    private let overlap = 32
    private var rgbaBitmapInfo: UInt32 {
        CGBitmapInfo.byteOrder32Big.rawValue | CGImageAlphaInfo.premultipliedLast.rawValue
    }

    private static var hardwareModel: String {
        var count = 0
        guard sysctlbyname("hw.model",nil,&count,nil,0) == 0 else { return "unknown" }
        var bytes = [UInt8](repeating:0,count:count)
        guard sysctlbyname("hw.model",&bytes,&count,nil,0) == 0 else { return "unknown" }
        return String(decoding:bytes.prefix { $0 != 0 },as:UTF8.self)
    }

    // One policy for legacy packages and current exports; package metadata cannot
    // authorize GPU execution. CPU is reserved for correctness diagnostics.
    static func configuration(cpuOnly: Bool = false) -> MLModelConfiguration {
        let configuration = MLModelConfiguration()
        configuration.computeUnits = cpuOnly ? .cpuOnly : .cpuAndNeuralEngine
        return configuration
    }

    init(modelURL: URL, variant: RealESRGANVariant = .x2plus) throws {
        scale = variant.scale
        modelSize = variant == .x2plus ? 522 : 384
        let configuration = Self.configuration()
        selectedComputeUnits = "ane"
        model = try MLModel(contentsOf:modelURL,configuration:configuration)
        let metadata = model.modelDescription.metadata[.creatorDefinedKey] as? [String:String] ?? [:]
        guard model.modelDescription.inputDescriptionsByName[inputName]?.multiArrayConstraint != nil,
              let outputName = model.modelDescription.outputDescriptionsByName.keys.sorted().first else {
            throw UncoverError.backend("The installed spatial upscaler has an unsupported model interface.")
        }
        let version = metadata["waifu.optimizationVersion"] ?? ""
        let verified = [variant == .x2plus ? "rrdb-fused-v1" : "rrdb-x4-v1", "rrdb-ane-identity-v2", "rrdb-ane-scale-v2", "rrdb-ane-none-v2"].contains(version)
            && metadata["waifu.validated"] == "true" && metadata["waifu.weightsSHA256"] == variant.weightsSHA256
        guard variant == .x2plus || (verified && metadata["waifu.variant"] == "x4plus" && metadata["waifu.scale"] == "4") else {
            throw UncoverError.backend("Choose the validated official RealESRGAN x4plus package.")
        }
        guard let input = model.modelDescription.inputDescriptionsByName[inputName]?.multiArrayConstraint,
              input.shape.map(\.intValue) == [1, 3, modelSize, modelSize],
              let output = model.modelDescription.outputDescriptionsByName[outputName]?.multiArrayConstraint,
              (output.shape.map(\.intValue) == [1,3,modelSize*scale,modelSize*scale] || (verified && output.shape.isEmpty)) else {
            throw UncoverError.backend("The installed spatial upscaler has an unexpected tensor size.")
        }
        self.outputName = outputName
        inputType = input.dataType
        if variant == .x4plus {
            guard inputType == .float16,
                  model.modelDescription.outputDescriptionsByName[outputName]?.multiArrayConstraint?.dataType == .float16 else {
                throw UncoverError.backend("RealESRGAN x4plus requires the validated FP16 model interface.")
            }
        }
        guard inputType == .float16 || inputType == .float32 else { throw UncoverError.backend("Unsupported spatial input precision.") }
        let shapes = input.shapeConstraint.enumeratedShapes.map { $0.map(\.intValue) }
        supportedSizes = verified ? shapes.compactMap { shape in
            shape.count == 4 && shape[0] == 1 && shape[1] == 3 && shape[2] == shape[3] && (variant == .x2plus ? [256,384,522] : [128,256,384,512]).contains(shape[2]) ? shape[2] : nil
        }.sorted() : [522]
        x4ModelSize = verified && metadata["waifu.tileTuningValidated"] == "true" && metadata["waifu.profileHardware"] == Self.hardwareModel && metadata["waifu.x4TileSize"] == "512" && supportedSizes.contains(512) ? 512 : 384
        contextHalo = verified ? 160 : 10
    }

    func resetDiagnostics() { predictionSeconds = 0; predictionSizes.removeAll(keepingCapacity:true) }

    func upscale(_ source: CGImage) throws -> CGImage {
        try Task.checkCancellation()
        let variant: RealESRGANVariant = scale == 4 ? .x4plus : .x2plus
        guard variant.accepts(width:source.width,height:source.height) else {
            throw UncoverError.backend("RealESRGAN output is limited to 16,384 pixels per side and 256 million pixels. Choose a smaller image or region.")
        }
        defer { x4Prediction = nil }
        let input = try rgbImage(source)
        if scale == 4 { return try upscaleFour(input) }
        let image = try upscaleOnce(input)
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
        RGBImage(width:image.width,height:image.height,pixels:try Self.straightRGBPixels(image, convertToSRGB: scale == 4))
    }

    static func straightRGBPixels(_ image: CGImage, convertToSRGB: Bool = false) throws -> [UInt8] {
        var pixels = [UInt8](repeating:0,count:image.width*image.height*3)
        // PNG stores straight channels. Ignoring alpha matches the original
        // model's RGB conversion and avoids darkening translucent Lanczos edges.
        if (!convertToSRGB || image.colorSpace?.name == CGColorSpace.sRGB), image.bitsPerComponent == 8, image.bitsPerPixel == 32, image.alphaInfo == .last,
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
                bitsPerComponent:8,bytesPerRow:image.width*4,space:convertToSRGB ? CGColorSpace(name:CGColorSpace.sRGB)! : CGColorSpaceCreateDeviceRGB(),
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

    /// Coalesce leading cores sharing the same reflected context. Each interior
    /// span retains 160 pixels on both sides within its fixed prediction size.
    static func x4Spans(_ count: Int, modelSize: Int = 384) -> [(origin: Int, count: Int)] {
        guard count > 0 else { return [] }
        precondition([384,512].contains(modelSize))
        var spans = [(origin: 0, count: min(modelSize-160,count))]
        var next = spans[0].count
        while next < count {
            let length = min(modelSize-320,count-next)
            spans.append((origin:next,count:length))
            next += length
        }
        return spans
    }

    private func upscaleFour(_ input: RGBImage) throws -> CGImage {
        let width = input.width * 4, height = input.height * 4
        let count = width * height * 3
        let bytes = UnsafeMutablePointer<UInt8>.allocate(capacity:count)
        bytes.initialize(repeating:0,count:count)
        var transferred = false
        defer { if !transferred { bytes.deallocate() } }
        let destination = UnsafeMutableBufferPointer(start:bytes,count:count)
        // Crop directly into the final RGB allocation. No Float tile, RGBA image,
        // or Data copy is retained alongside this output, including at 512px.
        for vertical in Self.x4Spans(input.height,modelSize:x4ModelSize) {
            for horizontal in Self.x4Spans(input.width,modelSize:x4ModelSize) {
                try Task.checkCancellation()
                try autoreleasepool {
                    _ = try predict(input,x:horizontal.origin,y:vertical.origin,
                        width:horizontal.count,height:vertical.count,
                        destination:destination,destinationWidth:width)
                }
            }
        }
        guard let provider = CGDataProvider(dataInfo:nil,data:bytes,size:count,releaseData:{ _,data,_ in
            UnsafeMutableRawPointer(mutating:data).deallocate()
        }) else { throw UncoverError.backend("Could not create the x4 image provider.") }
        transferred = true // Provider owns the allocation, including on CGImage failure.
        guard let result = CGImage(width:width,height:height,bitsPerComponent:8,
            bitsPerPixel:24,bytesPerRow:width*3,space:CGColorSpace(name:CGColorSpace.sRGB)!,
            bitmapInfo:CGBitmapInfo(rawValue:CGImageAlphaInfo.none.rawValue),provider:provider,
            decode:nil,shouldInterpolate:false,intent:.defaultIntent) else {
            throw UncoverError.backend("Could not create the x4 image.")
        }
        return result
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

    private func predict(_ image: RGBImage, x: Int, y: Int, width: Int, height: Int,
                         destination: UnsafeMutableBufferPointer<UInt8>? = nil, destinationWidth: Int = 0) throws -> [Float] {
        guard width > 0, height > 0, x >= 0, y >= 0,
              x + width <= image.width, y + height <= image.height,
              image.pixels.count == image.width * image.height * 3 else {
            throw UncoverError.backend("The spatial upscaler received an invalid frame tile.")
        }
        try Task.checkCancellation()
        // A whole small x4 image needs trailing reflection context, just like x2.
        // Interior x4 cores keep the validated 160px halo for either fixed size.
        let wholeSmallX4 = scale == 4 && image.width <= x4ModelSize-160 && image.height <= x4ModelSize-160
        let size = scale == 4 && !wholeSmallX4 ? x4ModelSize : (supportedSizes.first { $0 >= max(width,height)+contextHalo } ?? modelSize)
        let originX = scale == 4 ? max(0,x-160) : x
        let originY = scale == 4 ? max(0,y-160) : y
        let tick = Date()
        let tensor: MLMultiArray?
        if scale == 4, let cached = x4Prediction, cached.x == originX, cached.y == originY {
            tensor = cached.tensor
        } else {
            // A different origin cannot reuse the prior prediction. Release it before
            // Core ML allocates the next output, keeping only one large tile alive.
            x4Prediction = nil
            let input: MLMultiArray
            if let reused = inputBuffers[size] { input = reused }
            else {
                input = try MLMultiArray(shape:[1,3,NSNumber(value:size),NSNumber(value:size)],dataType:inputType)
                if inputBuffers.count >= 2 { inputBuffers.removeAll(keepingCapacity:true) }
                inputBuffers[size] = input
            }
        #if arch(arm64)
            let packingInput = input
        #else
            // Swift Float16 is unavailable on Intel macOS. Core ML transfers convert
            // whole tensors, including their strides, without boxing every pixel.
            let packingInput: MLMultiArray
            if inputType == .float16 {
                if let reused = floatInputBuffers[size] { packingInput = reused }
                else {
                    packingInput = try MLMultiArray(shape: input.shape, dataType: .float32)
                    if floatInputBuffers.count >= 2 { floatInputBuffers.removeAll(keepingCapacity: true) }
                    floatInputBuffers[size] = packingInput
                }
            } else { packingInput = input }
        #endif
            let inputStrides = packingInput.strides.map(\.intValue)
            let floatValues = packingInput.dataPointer.assumingMemoryBound(to: Float.self)
        #if arch(arm64)
            let halfValues = input.dataPointer.assumingMemoryBound(to: Float16.self)
        #endif
            let reflectedX = (0..<size).map { Self.reflected($0,count:width) }
            let reflectedY = (0..<size).map { Self.reflected($0,count:height) }
            for row in 0..<size {
                if row % 32 == 0 { try Task.checkCancellation() }
                // Reflect every padded sample inside the actual tile. Reflecting against
                // `height + 10`/`width + 10` lets partial edge tiles address pixels past
                // the end of the source image (for example, a 33px final tile).
                let localY = reflectedY[row]
                let sourceY = scale == 4 ? Self.reflected(originY+row,count:image.height) : y + localY
                for column in 0..<size {
                    let localX = reflectedX[column]
                    let sourceX = scale == 4 ? Self.reflected(originX+column,count:image.width) : x + localX
                    let sourcePixel = (sourceY * image.width + sourceX) * 3
                    for channel in 0..<3 {
                        let target = channel * inputStrides[1] + row * inputStrides[2] + column * inputStrides[3]
                        let value = Float(image.pixels[sourcePixel + channel])/255
                    #if arch(arm64)
                        if inputType == .float16 { halfValues[target] = Float16(value) }
                        else { floatValues[target] = value }
                    #else
                        floatValues[target] = value
                    #endif
                    }
                }
            }
        #if !arch(arm64)
            if packingInput !== input { packingInput.transfer(to: input) }
        #endif
            let provider = try MLDictionaryFeatureProvider(dictionary: [inputName: MLFeatureValue(multiArray: input)])
            tensor = try model.prediction(from: provider).featureValue(for:outputName)?.multiArrayValue
            if scale == 4, let tensor { x4Prediction = (originX,originY,tensor) }
        }
        try Task.checkCancellation()
        predictionSeconds += Date().timeIntervalSince(tick)
        predictionSizes.append(size)
        guard let output = tensor,
              output.shape.count == 4, output.shape[0].intValue == 1,
              output.shape[1].intValue == 3, output.shape[2].intValue == size * scale,
              output.shape[3].intValue == size * scale else {
            throw UncoverError.backend("The spatial upscaler returned an unexpected frame size.")
        }
        #if arch(arm64)
        let readableOutput = output
        #else
        let readableOutput: MLMultiArray
        if output.dataType == .float16 {
            readableOutput = try MLMultiArray(shape: output.shape, dataType: .float32)
            output.transfer(to: readableOutput)
        } else { readableOutput = output }
        #endif
        let strides = readableOutput.strides.map(\.intValue)
        var result = [Float](repeating: 0, count: destination == nil ? width * height * scale * scale * 3 : 0)
        for row in 0..<(height * scale) {
            for column in 0..<(width * scale) {
                let pixel = row * width * scale + column
                for channel in 0..<3 {
                    let offsetY = scale == 4 ? (y-originY)*4 : 0
                    let offsetX = scale == 4 ? (x-originX)*4 : 0
                    let index = channel * strides[1] + (row+offsetY) * strides[2] + (column+offsetX) * strides[3]
                    let value: Float
                    switch readableOutput.dataType {
                    case .float32: value = readableOutput.dataPointer.assumingMemoryBound(to: Float.self)[index]
                    #if arch(arm64)
                    case .float16: value = Float(readableOutput.dataPointer.assumingMemoryBound(to: Float16.self)[index])
                    #endif
                    default: throw UncoverError.backend("The spatial upscaler returned an unsupported pixel type.")
                    }
                    guard value.isFinite else { throw UncoverError.backend("The spatial upscaler returned a nonfinite pixel.") }
                    if let destination {
                        destination[((y*scale+row)*destinationWidth+x*scale+column)*3+channel] = UInt8(min(255,max(0,(value*255).rounded())))
                    } else {
                        result[pixel * 3 + channel] = min(1, max(0, value))
                    }
                }
            }
        }
        return result
    }
}
