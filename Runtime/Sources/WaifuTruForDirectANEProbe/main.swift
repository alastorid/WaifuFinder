import CoreML
import CoreGraphics
import CryptoKit
import Darwin
import Foundation
import ImageIO
import MLX
import TruForDirectANE
import TruForMLX

private let inputSide = 546
private let halo = 17
private let outputSide = 512
private let noiseLayers = [0, 2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35, 38, 41, 44, 47]
private let maxAbsoluteErrorGate = 0.001
private let relativeRMSGate = 0.001

private enum ProbeError: Error, CustomStringConvertible {
    case invalid(String)
    var description: String { if case .invalid(let text) = self { text } else { "Invalid direct-ANE probe input" } }
}

private struct TensorInfo {
    let dtype: String
    let shape: [Int]
    let offsets: [Int]
}

private struct WeightBlob {
    let path: String
    let bytes: [UInt8]
}

private struct FusedWeights {
    let bytes: [UInt8]
    let offsets: [(weight: UInt64, bias: UInt64)]
}

private struct ImageTile {
    let rgba: [UInt8]
    let width: Int
    let height: Int
    let rgb: [Float]
    let valid: [Float]
    let fp16: [UInt16]
    let originX: Int
    let originY: Int
}

private struct Reference {
    let data: Data
    let width: Int
    let height: Int
    let score: Float
    let noiseOffset: Int

    init(url: URL) throws {
        let loaded = try Data(contentsOf: url, options: .mappedIfSafe)
        func u32(_ offset: Int) -> UInt32 {
            loaded.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: offset, as: UInt32.self).littleEndian }
        }
        guard loaded.count >= 20, String(data: loaded[0..<4], encoding: .ascii) == "TFM1" else {
            throw ProbeError.invalid("Reference must be a TFM1 TruFor evidence file")
        }
        let parsedWidth = Int(u32(4)), parsedHeight = Int(u32(8))
        let maps = Int(u32(12))
        guard maps == 3, parsedWidth >= outputSide, parsedHeight >= outputSide,
              loaded.count == 20 + parsedWidth * parsedHeight * maps * MemoryLayout<Float>.size else {
            throw ProbeError.invalid("TFM1 reference dimensions or map layout are invalid")
        }
        data = loaded
        width = parsedWidth
        height = parsedHeight
        score = Float(bitPattern: u32(16))
        noiseOffset = 20 + 2 * parsedWidth * parsedHeight * MemoryLayout<Float>.size
    }

    func noise(_ x: Int, _ y: Int) -> Float {
        let offset = noiseOffset + (y * width + x) * MemoryLayout<Float>.size
        let bits = data.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: offset, as: UInt32.self).littleEndian }
        return Float(bitPattern: bits)
    }
}

private final class PeakFootprint: @unchecked Sendable {
    private let lock = NSLock()
    private var peak: UInt64 = 0
    private let timer = DispatchSource.makeTimerSource(queue: DispatchQueue(label: "trufor-direct-ane-memory"))

    init() {
        timer.schedule(deadline: .now(), repeating: .milliseconds(5))
        timer.setEventHandler { [weak self] in self?.sample() }
        timer.resume()
    }

    private func sample() {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
        let status = withUnsafeMutablePointer(to: &info) { pointer in
            pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        guard status == KERN_SUCCESS else { return }
        lock.lock(); peak = max(peak, UInt64(info.phys_footprint)); lock.unlock()
    }

    func bytes() -> UInt64 { sample(); lock.lock(); defer { lock.unlock() }; return peak }
    deinit { timer.cancel() }
}

private func hex(_ digest: SHA256.Digest) -> String { digest.map { String(format: "%02x", $0) }.joined() }

private func fileSHA256(_ url: URL) throws -> String {
    let handle = try FileHandle(forReadingFrom: url)
    defer { try? handle.close() }
    var digest = SHA256()
    while let part = try handle.read(upToCount: 8 * 1024 * 1024), !part.isEmpty { digest.update(data: part) }
    return hex(digest.finalize())
}

private func readExactly(_ handle: FileHandle, count: Int) throws -> Data {
    guard let data = try handle.read(upToCount: count), data.count == count else {
        throw ProbeError.invalid("Safetensors file ended before a requested tensor")
    }
    return data
}

private final class SafetensorsReader {
    private let handle: FileHandle
    private let dataStart: UInt64
    private let tensors: [String: TensorInfo]

    init(file: URL) throws {
        handle = try FileHandle(forReadingFrom: file)
        let prefix = try readExactly(handle, count: 8)
        let headerSize = prefix.withUnsafeBytes { Int($0.loadUnaligned(as: UInt64.self).littleEndian) }
        guard headerSize > 0, headerSize < 64 * 1024 * 1024 else {
            throw ProbeError.invalid("Safetensors header size is invalid")
        }
        let header = try readExactly(handle, count: headerSize)
        guard let root = try JSONSerialization.jsonObject(with: header) as? [String: Any] else {
            throw ProbeError.invalid("Safetensors header is not an object")
        }
        var parsed: [String: TensorInfo] = [:]
        for (name, raw) in root where name != "__metadata__" {
            guard let item = raw as? [String: Any], let dtype = item["dtype"] as? String,
                  let shape = item["shape"] as? [NSNumber], let offsets = item["data_offsets"] as? [NSNumber], offsets.count == 2 else {
                throw ProbeError.invalid("Safetensors tensor metadata is invalid for \(name)")
            }
            parsed[name] = TensorInfo(dtype: dtype, shape: shape.map(\.intValue), offsets: offsets.map(\.intValue))
        }
        tensors = parsed
        dataStart = UInt64(8 + headerSize)
    }

    deinit { try? handle.close() }

    func floats(_ name: String, shape expectedShape: [Int]) throws -> [Float] {
        guard let tensor = tensors[name], tensor.dtype == "F32", tensor.shape == expectedShape,
              tensor.offsets.count == 2, tensor.offsets[0] >= 0, tensor.offsets[1] >= tensor.offsets[0] else {
            throw ProbeError.invalid("Missing or incompatible FP32 TruFor tensor \(name)")
        }
        let count = expectedShape.reduce(1, *)
        guard tensor.offsets[1] - tensor.offsets[0] == count * MemoryLayout<Float>.size else {
            throw ProbeError.invalid("TruFor tensor byte size is inconsistent for \(name)")
        }
        try handle.seek(toOffset: dataStart + UInt64(tensor.offsets[0]))
        let bytes = try readExactly(handle, count: count * MemoryLayout<Float>.size)
        return bytes.withUnsafeBytes { raw in
            (0..<count).map { Float(bitPattern: raw.loadUnaligned(fromByteOffset: $0 * 4, as: UInt32.self).littleEndian) }
        }
    }
}

private func appendLE32(_ value: UInt32, to bytes: inout [UInt8]) {
    var little = value.littleEndian
    withUnsafeBytes(of: &little) { bytes.append(contentsOf: $0) }
}

private func makeFP16Blob(_ values: [Float]) -> [UInt8] {
    let dataCount = values.count * MemoryLayout<UInt16>.size
    var blob = [UInt8](repeating: 0, count: 128 + dataCount)
    blob[0] = 1; blob[4] = 2
    blob[64] = 0xEF; blob[65] = 0xBE; blob[66] = 0xAD; blob[67] = 0xDE; blob[68] = 1
    var size = UInt32(dataCount).littleEndian
    withUnsafeBytes(of: &size) { blob.replaceSubrange(72..<76, with: $0) }
    var offset = UInt32(128).littleEndian
    withUnsafeBytes(of: &offset) { blob.replaceSubrange(80..<84, with: $0) }
    for (index, value) in values.enumerated() {
        let bits = Float16(value).bitPattern.littleEndian
        let destination = 128 + index * 2
        blob[destination] = UInt8(truncatingIfNeeded: bits)
        blob[destination + 1] = UInt8(truncatingIfNeeded: bits >> 8)
    }
    return blob
}

private func buildWeights(_ reader: SafetensorsReader) throws -> FusedWeights {
    var packed = [UInt8](repeating: 0, count: 64)
    packed[0] = 1; packed[4] = 2
    var offsets: [(weight: UInt64, bias: UInt64)] = []
    func appendChunk(_ values: [Float]) -> UInt64 {
        let tensorBlob = makeFP16Blob(values)
        let chunkStart = UInt64(packed.count)
        var chunk = Array(tensorBlob.dropFirst(64))
        var absoluteDataOffset = UInt32(packed.count + 64).littleEndian
        withUnsafeBytes(of: &absoluteDataOffset) { chunk.replaceSubrange(16..<20, with: $0) }
        packed.append(contentsOf: chunk)
        return chunkStart
    }
    for layer in noiseLayers {
        let sourceInputChannels = layer == 0 ? 3 : 64
        let sourceOutputChannels = layer == 47 ? 1 : 64
        let weightName = "dncnn.\(layer).weight"
        var weights = try reader.floats(weightName, shape: [sourceOutputChannels, sourceInputChannels, 3, 3])
        var bias: [Float]
        if layer == 0 {
            bias = try reader.floats("dncnn.0.bias", shape: [64])
        } else if layer == 47 {
            bias = try reader.floats("dncnn.47.bias", shape: [1])
        } else {
            let batchNorm = layer + 1
            let gamma = try reader.floats("dncnn.\(batchNorm).weight", shape: [64])
            let beta = try reader.floats("dncnn.\(batchNorm).bias", shape: [64])
            let mean = try reader.floats("dncnn.\(batchNorm).running_mean", shape: [64])
            let variance = try reader.floats("dncnn.\(batchNorm).running_var", shape: [64])
            bias = Array(repeating: 0, count: 64)
            for channel in 0..<64 {
                let scale = gamma[channel] / sqrt(variance[channel] + 1e-5)
                bias[channel] = beta[channel] - mean[channel] * scale
                let channelOffset = channel * sourceInputChannels * 9
                for index in channelOffset..<(channelOffset + sourceInputChannels * 9) { weights[index] *= scale }
            }
        }
        if layer == 0 {
            var padded = [Float](repeating: 0, count: 64 * 8 * 9)
            for output in 0..<64 {
                padded.replaceSubrange((output * 8 * 9)..<(output * 8 * 9 + 3 * 9),
                                       with: weights[(output * 3 * 9)..<(output * 3 * 9 + 3 * 9)])
            }
            weights = padded
        } else if layer == 47 {
            var padded = [Float](repeating: 0, count: 8 * 64 * 9)
            padded.replaceSubrange(0..<(64 * 9), with: weights)
            weights = padded
            bias += Array(repeating: 0, count: 7)
        }
        offsets.append((appendChunk(weights), appendChunk(bias)))
    }
    return FusedWeights(bytes: packed, offsets: offsets)
}

private func milProgram(inputTileSide: Int, offsets: [(weight: UInt64, bias: UInt64)]) -> String {
    var lines = [
        "program(1.3)",
        "[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, {\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, {\"coremltools-version\", \"9.0\"}})]",
        "{",
        "    func main<ios18>(tensor<fp16, [1, 8, \(inputTileSide), \(inputTileSide)]> x) {",
        "        string c_pad_type = const()[name = string(\"c_pad_type\"), val = string(\"same\")];",
        "        tensor<int32, [2]> c_strides = const()[name = string(\"c_strides\"), val = tensor<int32, [2]>([1, 1])];",
        "        tensor<int32, [4]> c_pad = const()[name = string(\"c_pad\"), val = tensor<int32, [4]>([0, 0, 0, 0])];",
        "        tensor<int32, [2]> c_dilations = const()[name = string(\"c_dilations\"), val = tensor<int32, [2]>([1, 1])];",
        "        int32 c_groups = const()[name = string(\"c_groups\"), val = int32(1)];"
    ]
    var height = inputTileSide
    var inputChannels = 8
    var inputName = "x"
    for (index, layer) in noiseLayers.enumerated() {
        let outputChannels = layer == 47 ? 8 : 64
        let outputHeight = height
        let weight = "W\(index)"
        let bias = "B\(index)"
        let convolution = "C\(index)"
        lines.append("        tensor<fp16, [\(outputChannels), \(inputChannels), 3, 3]> \(weight) = const()[name = string(\"\(weight)\"), val = tensor<fp16, [\(outputChannels), \(inputChannels), 3, 3]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64(\(offsets[index].weight))))];")
        lines.append("        tensor<fp16, [\(outputChannels)]> \(bias) = const()[name = string(\"\(bias)\"), val = tensor<fp16, [\(outputChannels)]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64(\(offsets[index].bias))))];")
        lines.append("        tensor<fp16, [1, \(outputChannels), \(outputHeight), \(outputHeight)]> \(convolution) = conv(dilations = c_dilations, groups = c_groups, pad = c_pad, pad_type = c_pad_type, strides = c_strides, bias = \(bias), weight = \(weight), x = \(inputName))[name = string(\"\(convolution)\")];")
        if layer != 47 {
            let activation = "R\(index)"
            lines.append("        tensor<fp16, [1, 64, \(outputHeight), \(outputHeight)]> \(activation) = relu(x = \(convolution))[name = string(\"\(activation)\")];")
            inputName = activation
        } else {
            inputName = convolution
        }
        inputChannels = outputChannels
        height = outputHeight
    }
    lines.append("    } -> (\(inputName));")
    lines.append("}")
    return lines.joined(separator: "\n")
}

private func decodeTile(_ url: URL, originX: Int = 512, originY: Int = 512) throws -> ImageTile {
    guard let source = CGImageSourceCreateWithURL(url as CFURL, [kCGImageSourceShouldCache: false] as CFDictionary),
          let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
        throw ProbeError.invalid("Could not decode the supplied input image")
    }
    let width = image.width, height = image.height
    guard originX >= halo, originY >= halo,
          originX + outputSide + halo < width, originY + outputSide + halo < height else {
        throw ProbeError.invalid("Input image is too small for the fixed interior comparison tile")
    }
    var rgba = [UInt8](repeating: 0, count: width * height * 4)
    let rendered = rgba.withUnsafeMutableBytes { storage -> Bool in
        guard let context = CGContext(data: storage.baseAddress, width: width, height: height, bitsPerComponent: 8,
                                      bytesPerRow: width * 4, space: CGColorSpace(name: CGColorSpace.sRGB)!,
                                      bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return false }
        context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
        return true
    }
    guard rendered else { throw ProbeError.invalid("Could not render input image as sRGB") }
    let pixels = inputSide * inputSide
    var rgb = [Float](repeating: 0, count: 3 * pixels)
    var valid = [Float](repeating: 0, count: pixels)
    var fp16 = [UInt16](repeating: 0, count: 3 * pixels)
    for y in 0..<inputSide {
        let sourceY = originY + y - halo
        for x in 0..<inputSide {
            let sourceX = originX + x - halo
            let point = y * inputSide + x
            if sourceX >= 0 && sourceY >= 0 && sourceX < width && sourceY < height {
                let sourceOffset = (sourceY * width + sourceX) * 4
                for channel in 0..<3 {
                    let value = Float(rgba[sourceOffset + channel]) * (1.0 / 256.0)
                    rgb[channel * pixels + point] = value
                    fp16[channel * pixels + point] = Float16(value).bitPattern
                }
                valid[point] = 1
            }
        }
    }
    return ImageTile(rgba: rgba, width: width, height: height, rgb: rgb, valid: valid, fp16: fp16,
                     originX: originX, originY: originY)
}

private struct ErrorMetrics {
    let maximumAbsolute: Double
    let relativeRMS: Double
    let finite: Bool
}

private func metrics(_ values: [Float], reference: Reference, side: Int, offsetX: Int, offsetY: Int) -> ErrorMetrics {
    var maximum = 0.0, squaredError = 0.0, squaredReference = 0.0
    var finite = true
    for index in 0..<(side * side) {
        let actual = Double(values[index])
        let expected = Double(reference.noise(offsetX + index % side, offsetY + index / side))
        if !actual.isFinite { finite = false; continue }
        let error = actual - expected
        maximum = max(maximum, abs(error)); squaredError += error * error; squaredReference += expected * expected
    }
    return ErrorMetrics(maximumAbsolute: finite ? maximum : .infinity,
                        relativeRMS: finite ? sqrt(squaredError / max(squaredReference, 1e-30)) : .infinity,
                        finite: finite)
}

private func readDirectOutput(_ kernel: OpaquePointer, inputSide: Int, outputSide: Int) throws -> [Float] {
    guard let pointer = trufor_direct_ane_lock_output(kernel) else { throw ProbeError.invalid("Could not lock ANE output IOSurface") }
    defer { trufor_direct_ane_unlock_output(kernel) }
    let values = pointer.assumingMemoryBound(to: UInt16.self)
    var cropped = [Float](repeating: 0, count: outputSide * outputSide)
    for y in 0..<outputSide {
        for x in 0..<outputSide {
            let source = (halo + y) * inputSide + halo + x
            cropped[y * outputSide + x] = Float(Float16(bitPattern: values[source]))
        }
    }
    return cropped
}

private func buildDirectKernel(_ modelDirectory: URL, inputTileSide: Int) throws -> (OpaquePointer, String, Int, Int) {
    let modelURL = modelDirectory.appendingPathComponent("model.safetensors")
    let manifestURL = modelDirectory.appendingPathComponent("manifest.json")
    let architectureURL = modelDirectory.appendingPathComponent("architecture.json")
    let manifest = try JSONSerialization.jsonObject(with: Data(contentsOf: manifestURL)) as? [String: Any]
    let hashes = manifest?["fileSHA256"] as? [String: String]
    guard hashes?["model.safetensors"] == "3b2c7146303e984c43b51f26e38d093c8b2ebba9a13e41bb029a98aa5b29783c",
          try fileSHA256(modelURL) == hashes?["model.safetensors"] else {
        throw ProbeError.invalid("TruFor model hash does not match the published checkpoint manifest")
    }
    let architecture = try JSONSerialization.jsonObject(with: Data(contentsOf: architectureURL)) as? [String: Any]
    guard architecture?["sourceSHA256"] as? String == "ac1d90e329a72e0d66e8665e123a19e94bfae3209c3ef8a4f9ca3b91578c7844",
          architecture?["preprocessingVersion"] as? String == "trufor-native-rgb-div256-v1" else {
        throw ProbeError.invalid("Expected the verified official TruFor checkpoint and preprocessing")
    }
    let reader = try SafetensorsReader(file: modelURL)
    let weights = try buildWeights(reader)
    let blobs = [WeightBlob(path: "@model_path/weights/weight.bin", bytes: weights.bytes)]
    let mil = milProgram(inputTileSide: inputTileSide, offsets: weights.offsets)
    var error = [CChar](repeating: 0, count: 2048)
    let nameBuffers = blobs.map { strdup($0.path) }
    guard nameBuffers.allSatisfy({ $0 != nil }) else { throw ProbeError.invalid("Could not allocate MIL weight names") }
    defer { nameBuffers.forEach { if let value = $0 { free(value) } } }
    let dataBuffers = blobs.map { blob -> UnsafeMutablePointer<UInt8> in
        let value = UnsafeMutablePointer<UInt8>.allocate(capacity: blob.bytes.count)
        value.initialize(from: blob.bytes, count: blob.bytes.count)
        return value
    }
    defer { for (index, value) in dataBuffers.enumerated() { value.deinitialize(count: blobs[index].bytes.count); value.deallocate() } }
    let names: [UnsafePointer<CChar>?] = nameBuffers.map { UnsafePointer<CChar>($0!) }
    let bytes: [UnsafePointer<UInt8>?] = dataBuffers.map { UnsafePointer<UInt8>($0) }
    let lengths = blobs.map { $0.bytes.count }
    var kernel: OpaquePointer?
    let compileStart = DispatchTime.now().uptimeNanoseconds
    mil.withCString { milPointer in
        names.withUnsafeBufferPointer { namePointer in
            bytes.withUnsafeBufferPointer { bytePointer in
                lengths.withUnsafeBufferPointer { lengthPointer in
                    kernel = trufor_direct_ane_compile(milPointer, mil.utf8.count, Int32(blobs.count),
                        namePointer.baseAddress!, bytePointer.baseAddress!, lengthPointer.baseAddress!,
                        8 * inputTileSide * inputTileSide * 2,
                        8 * inputTileSide * inputTileSide * 2,
                        &error, error.count)
                }
            }
        }
    }
    let compileMs = Int((DispatchTime.now().uptimeNanoseconds - compileStart) / 1_000_000)
    guard let kernel else { throw ProbeError.invalid("Private ANE compilation/loading failed: \(String(cString: error))") }
    return (kernel, mil, compileMs, blobs.reduce(0) { $0 + $1.bytes.count })
}

private func runDirect(model: URL, image: ImageTile, reference: Reference, samples: Int, outputTileSide: Int) throws -> [String: Any] {
    let sampler = PeakFootprint()
    var availabilityError = [CChar](repeating: 0, count: 1024)
    guard trufor_direct_ane_available(&availabilityError, availabilityError.count) else {
        throw ProbeError.invalid("Private ANE runtime unavailable: \(String(cString: availabilityError))")
    }
    let directInputSide = outputTileSide + 2 * halo
    let (kernel, _, compileMs, weightBytes) = try buildDirectKernel(model, inputTileSide: directInputSide)
    defer { trufor_direct_ane_free(kernel) }
    guard let inputPointer = trufor_direct_ane_lock_input(kernel) else { throw ProbeError.invalid("Could not lock ANE input IOSurface") }
    let inputPixels = directInputSide * directInputSide
    var tileFP16 = [UInt16](repeating: 0, count: 8 * inputPixels)
    for channel in 0..<3 {
        for row in 0..<directInputSide {
            let sourceOffset = channel * inputSide * inputSide + row * inputSide
            let destinationOffset = channel * inputPixels + row * directInputSide
            tileFP16[destinationOffset..<(destinationOffset + directInputSide)] = image.fp16[sourceOffset..<(sourceOffset + directInputSide)]
        }
    }
    tileFP16.withUnsafeBytes { inputPointer.copyMemory(from: $0.baseAddress!, byteCount: $0.count) }
    trufor_direct_ane_unlock_input(kernel)

    var latencies: [Double] = []
    var output: [Float] = []
    for sample in 0..<samples {
        var error = [CChar](repeating: 0, count: 2048)
        let start = DispatchTime.now().uptimeNanoseconds
        guard trufor_direct_ane_evaluate(kernel, &error, error.count) else {
            throw ProbeError.invalid("Private ANE evaluation failed: \(String(cString: error))")
        }
        latencies.append(Double(DispatchTime.now().uptimeNanoseconds - start) / 1_000_000)
        if sample == samples - 1 { output = try readDirectOutput(kernel, inputSide: directInputSide, outputSide: outputTileSide) }
    }
    let measured = metrics(output, reference: reference, side: outputTileSide,
                           offsetX: image.originX, offsetY: image.originY)
    let warm = Array(latencies.dropFirst())
    let sorted = warm.sorted()
    return [
        "backend": "experimental direct ANE",
        "precision": "FP16 input, weights, and activations; BN folded into convolution; RGB and single-channel output padded to 8 channels",
        "compileCount": trufor_direct_ane_compile_count(),
        "compileAndLoadMilliseconds": compileMs,
        "firstEvaluationMilliseconds": latencies[0],
        "warmEvaluationMilliseconds": warm,
        "warmP50Milliseconds": sorted[sorted.count / 2],
        "warmP95Milliseconds": sorted[min(sorted.count - 1, max(0, Int(ceil(Double(sorted.count) * 0.95)) - 1))],
        "sampleCount": samples,
        "inputShape": [1, 8, directInputSide, directInputSide],
        "outputShape": [1, 8, directInputSide, directInputSide],
        "fp16WeightBlobBytes": weightBytes,
        "inputIOSurfaceBytes": 8 * directInputSide * directInputSide * 2,
        "outputIOSurfaceBytes": 8 * directInputSide * directInputSide * 2,
        "logicalOutputBytes": 8 * directInputSide * directInputSide * 2,
        "physicalPeakBytes": sampler.bytes(),
        "referenceScore": reference.score,
        "comparedNoiseprintPixels": outputTileSide * outputTileSide,
        "noiseprintMaxAbsoluteError": measured.maximumAbsolute,
        "noiseprintRelativeRMS": measured.relativeRMS,
        "finiteOutput": measured.finite,
        "accuracyGate": ["maxAbsolute": maxAbsoluteErrorGate, "relativeRMS": relativeRMSGate],
        "accepted": measured.finite && measured.maximumAbsolute <= maxAbsoluteErrorGate && measured.relativeRMS <= relativeRMSGate,
        "note": "Private API and IOSurface execution are experimental; this result is not an app backend selection."
    ]
}

private func makeCoreMLProvider(_ image: ImageTile, rgbName: String, validName: String) throws -> MLDictionaryFeatureProvider {
    let shapeRGB = [1, 3, inputSide, inputSide].map { NSNumber(value: $0) }
    let shapeMask = [1, 1, inputSide, inputSide].map { NSNumber(value: $0) }
    let rgb = try MLMultiArray(shape: shapeRGB, dataType: .float32)
    let mask = try MLMultiArray(shape: shapeMask, dataType: .float32)
    image.rgb.withUnsafeBufferPointer { rgb.dataPointer.copyMemory(from: $0.baseAddress!, byteCount: $0.count * 4) }
    image.valid.withUnsafeBufferPointer { mask.dataPointer.copyMemory(from: $0.baseAddress!, byteCount: $0.count * 4) }
    return try MLDictionaryFeatureProvider(dictionary: [
        rgbName: MLFeatureValue(multiArray: rgb), validName: MLFeatureValue(multiArray: mask)
    ])
}

private func runCoreML(modelURL: URL, image: ImageTile, reference: Reference, samples: Int) throws -> [String: Any] {
    let sampler = PeakFootprint()
    let configuration = MLModelConfiguration(); configuration.computeUnits = .all
    var compileMs: Double = 0
    let compiledURL: URL
    if modelURL.pathExtension == "mlpackage" {
        let compileStart = DispatchTime.now().uptimeNanoseconds
        compiledURL = try MLModel.compileModel(at: modelURL)
        compileMs = Double(DispatchTime.now().uptimeNanoseconds - compileStart) / 1_000_000
    } else {
        compiledURL = modelURL
    }
    let loadStart = DispatchTime.now().uptimeNanoseconds
    let model = try MLModel(contentsOf: compiledURL, configuration: configuration)
    let loadMs = Double(DispatchTime.now().uptimeNanoseconds - loadStart) / 1_000_000
    let descriptions = model.modelDescription.inputDescriptionsByName
    guard let rgbName = descriptions.keys.first(where: { descriptions[$0]?.multiArrayConstraint?.shape.count == 4 && descriptions[$0]?.multiArrayConstraint?.shape[1].intValue == 3 }),
          let validName = descriptions.keys.first(where: { $0 != rgbName && descriptions[$0]?.multiArrayConstraint?.shape.count == 4 }),
          let outputName = model.modelDescription.outputDescriptionsByName.keys.sorted().first else {
        throw ProbeError.invalid("Core ML fallback model has unexpected inputs or outputs")
    }
    let provider = try makeCoreMLProvider(image, rgbName: rgbName, validName: validName)
    var latencies: [Double] = []
    var outputValues: [Float] = []
    for sample in 0..<samples {
        let start = DispatchTime.now().uptimeNanoseconds
        let result = try model.prediction(from: provider)
        latencies.append(Double(DispatchTime.now().uptimeNanoseconds - start) / 1_000_000)
        guard let output = result.featureValue(for: outputName)?.multiArrayValue, output.dataType == .float32 else {
            throw ProbeError.invalid("Core ML fallback returned a non-FP32 output")
        }
        if sample == samples - 1 {
            let strides = output.strides.map(\.intValue)
            let base = output.dataPointer.assumingMemoryBound(to: Float.self)
            outputValues.reserveCapacity(outputSide * outputSide)
            for y in 0..<outputSide {
                for x in 0..<outputSide {
                    outputValues.append(base[(halo + y) * strides[2] + (halo + x) * strides[3]])
                }
            }
        }
    }
    let measured = metrics(outputValues, reference: reference, side: outputSide,
                           offsetX: image.originX, offsetY: image.originY)
    guard measured.finite && measured.maximumAbsolute <= maxAbsoluteErrorGate && measured.relativeRMS <= relativeRMSGate else {
        throw ProbeError.invalid(String(format: "Public Core ML FP32 fallback failed its accuracy gate (max abs %.6g, relative RMS %.6g)",
                                        measured.maximumAbsolute, measured.relativeRMS))
    }
    let warm = Array(latencies.dropFirst())
    let sorted = warm.sorted()
    return [
        "backend": "public Core ML fallback",
        "precision": "FP32 validated model",
        "computeUnits": "all",
        "modelCompileMilliseconds": compileMs,
        "modelLoadMilliseconds": loadMs,
        "firstEvaluationMilliseconds": latencies[0],
        "warmEvaluationMilliseconds": warm,
        "warmP50Milliseconds": sorted[sorted.count / 2],
        "warmP95Milliseconds": sorted[min(sorted.count - 1, max(0, Int(ceil(Double(sorted.count) * 0.95)) - 1))],
        "sampleCount": samples,
        "inputShape": [1, 3, inputSide, inputSide],
        "outputComparedPixels": outputSide * outputSide,
        "physicalPeakBytes": sampler.bytes(),
        "referenceScore": reference.score,
        "noiseprintMaxAbsoluteError": measured.maximumAbsolute,
        "noiseprintRelativeRMS": measured.relativeRMS,
        "finiteOutput": measured.finite,
        "accuracyGate": ["maxAbsolute": maxAbsoluteErrorGate, "relativeRMS": relativeRMSGate],
        "accepted": true,
        "placementEvidence": "Public Core ML .all policy; no claim about actual per-operation ANE placement."
    ]
}

private func runMLXTile(modelDirectory: URL, image: ImageTile, reference: Reference, samples: Int) throws -> [String: Any] {
    let sampler = PeakFootprint()
    let loadStart = DispatchTime.now().uptimeNanoseconds
    let model = try TruForModel(directory: modelDirectory)
    let loadMs = Double(DispatchTime.now().uptimeNanoseconds - loadStart) / 1_000_000
    let pixels = inputSide * inputSide
    var nhwc = [Float](repeating: 0, count: 3 * pixels)
    for pixel in 0..<pixels {
        nhwc[pixel * 3] = image.rgb[pixel]
        nhwc[pixel * 3 + 1] = image.rgb[pixels + pixel]
        nhwc[pixel * 3 + 2] = image.rgb[2 * pixels + pixel]
    }
    let input = MLXArray(nhwc, [1, inputSide, inputSide, 3])
    eval(input)
    var latencies: [Double] = []
    var outputValues: [Float] = []
    for sample in 0..<samples {
        let start = DispatchTime.now().uptimeNanoseconds
        let output = model.probeNoiseprintTile(input)
        eval(output)
        latencies.append(Double(DispatchTime.now().uptimeNanoseconds - start) / 1_000_000)
        if sample == samples - 1 { outputValues = output.asType(.float32).asArray(Float.self) }
    }
    let strides = inputSide
    var cropped = [Float](repeating: 0, count: outputSide * outputSide)
    for y in 0..<outputSide {
        for x in 0..<outputSide {
            cropped[y * outputSide + x] = outputValues[(halo + y) * strides + halo + x]
        }
    }
    let measured = metrics(cropped, reference: reference, side: outputSide,
                           offsetX: image.originX, offsetY: image.originY)
    let warm = Array(latencies.dropFirst())
    let sorted = warm.sorted()
    return [
        "backend": "MLX / Metal",
        "precision": model.precision.rawValue,
        "modelLoadMilliseconds": loadMs,
        "firstEvaluationMilliseconds": latencies[0],
        "warmEvaluationMilliseconds": warm,
        "warmP50Milliseconds": sorted[sorted.count / 2],
        "warmP95Milliseconds": sorted[min(sorted.count - 1, max(0, Int(ceil(Double(sorted.count) * 0.95)) - 1))],
        "sampleCount": samples,
        "inputShape": [1, inputSide, inputSide, 3],
        "outputComparedPixels": outputSide * outputSide,
        "physicalPeakBytes": sampler.bytes(),
        "modelParameterBytes": model.parameterBytes,
        "referenceScore": reference.score,
        "noiseprintMaxAbsoluteError": measured.maximumAbsolute,
        "noiseprintRelativeRMS": measured.relativeRMS,
        "finiteOutput": measured.finite
    ]
}

private func emit(_ report: [String: Any]) throws {
    let data = try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
    print(String(decoding: data, as: UTF8.self))
}

private func runBridgeSmoke(depth: Int = 1, side: Int = 32) throws -> [String: Any] {
    let channels = 64
    guard (1...17).contains(depth), [32, 36, 66, 162].contains(side) else {
        throw ProbeError.invalid("Smoke chain depth must be 1 through 17 and side must be 32, 36, 66, or 162")
    }
    var identity = [Float](repeating: 0, count: channels * channels * 9)
    for output in 0..<channels { identity[(output * channels + output) * 9 + 4] = 1 }
    let blobs = [
        WeightBlob(path: "@model_path/weights/w.bin", bytes: makeFP16Blob(identity)),
        WeightBlob(path: "@model_path/weights/b.bin", bytes: makeFP16Blob(Array(repeating: 0.01, count: channels)))
    ]
    var milLines = [
        "program(1.3)",
        "[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, {\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, {\"coremltools-version\", \"9.0\"}})]",
        "{",
        "    func main<ios18>(tensor<fp16, [1, 64, \(side), \(side)]> x) {",
        "        string c_pad_type = const()[name = string(\"c_pad_type\"), val = string(\"custom\")];",
        "        tensor<int32, [2]> c_strides = const()[name = string(\"c_strides\"), val = tensor<int32, [2]>([1, 1])];",
        "        tensor<int32, [4]> c_pad = const()[name = string(\"c_pad\"), val = tensor<int32, [4]>([1, 1, 1, 1])];",
        "        tensor<int32, [2]> c_dilations = const()[name = string(\"c_dilations\"), val = tensor<int32, [2]>([1, 1])];",
        "        int32 c_groups = const()[name = string(\"c_groups\"), val = int32(1)];"
    ]
    for index in 0..<depth {
        let weight = "W\(index)", bias = "B\(index)", convolution = "C\(index)"
        milLines.append("        tensor<fp16, [64, 64, 3, 3]> \(weight) = const()[name = string(\"\(weight)\"), val = tensor<fp16, [64, 64, 3, 3]>(BLOBFILE(path = string(\"@model_path/weights/w.bin\"), offset = uint64(64)))];")
        milLines.append("        tensor<fp16, [64]> \(bias) = const()[name = string(\"\(bias)\"), val = tensor<fp16, [64]>(BLOBFILE(path = string(\"@model_path/weights/b.bin\"), offset = uint64(64)))];")
        milLines.append("        tensor<fp16, [1, 64, \(side), \(side)]> \(convolution) = conv(dilations = c_dilations, groups = c_groups, pad = c_pad, pad_type = c_pad_type, strides = c_strides, bias = \(bias), weight = \(weight), x = \(index == 0 ? "x" : "R\(index - 1)"))[name = string(\"\(convolution)\")];")
        if index + 1 < depth {
            let activation = "R\(index)"
            milLines.append("        tensor<fp16, [1, 64, \(side), \(side)]> \(activation) = relu(x = \(convolution))[name = string(\"\(activation)\")];")
        }
    }
    milLines.append("    } -> (C\(depth - 1));")
    milLines.append("}")
    let mil = milLines.joined(separator: "\n")
    let nameBuffers = blobs.map { strdup($0.path) }
    defer { nameBuffers.forEach { if let value = $0 { free(value) } } }
    let dataBuffers = blobs.map { blob -> UnsafeMutablePointer<UInt8> in
        let value = UnsafeMutablePointer<UInt8>.allocate(capacity: blob.bytes.count)
        value.initialize(from: blob.bytes, count: blob.bytes.count)
        return value
    }
    defer { for (index, value) in dataBuffers.enumerated() { value.deinitialize(count: blobs[index].bytes.count); value.deallocate() } }
    let names: [UnsafePointer<CChar>?] = nameBuffers.map { UnsafePointer<CChar>($0!) }
    let bytes: [UnsafePointer<UInt8>?] = dataBuffers.map { UnsafePointer<UInt8>($0) }
    let lengths = blobs.map { $0.bytes.count }
    var error = [CChar](repeating: 0, count: 2048)
    let started = DispatchTime.now().uptimeNanoseconds
    var kernel: OpaquePointer?
    mil.withCString { milText in
        names.withUnsafeBufferPointer { namePointer in
            bytes.withUnsafeBufferPointer { bytePointer in
                lengths.withUnsafeBufferPointer { lengthPointer in
                    kernel = trufor_direct_ane_compile(milText, mil.utf8.count, Int32(blobs.count), namePointer.baseAddress!, bytePointer.baseAddress!, lengthPointer.baseAddress!,
                                                       channels * side * side * 2, channels * side * side * 2,
                                                       &error, error.count)
                }
            }
        }
    }
    let compileMs = Double(DispatchTime.now().uptimeNanoseconds - started) / 1_000_000
    guard let kernel else { throw ProbeError.invalid("Minimal direct-ANE compile failed: \(String(cString: error))") }
    defer { trufor_direct_ane_free(kernel) }
    guard let input = trufor_direct_ane_lock_input(kernel) else { throw ProbeError.invalid("Smoke input surface could not be locked") }
    let inputValues = [UInt16](repeating: Float16(0.25).bitPattern, count: channels * side * side)
    inputValues.withUnsafeBytes { input.copyMemory(from: $0.baseAddress!, byteCount: $0.count) }
    trufor_direct_ane_unlock_input(kernel)
    var evalError = [CChar](repeating: 0, count: 1024)
    let evalStart = DispatchTime.now().uptimeNanoseconds
    guard trufor_direct_ane_evaluate(kernel, &evalError, evalError.count) else {
        throw ProbeError.invalid("Minimal direct-ANE evaluation failed: \(String(cString: evalError))")
    }
    let evalMs = Double(DispatchTime.now().uptimeNanoseconds - evalStart) / 1_000_000
    guard let output = trufor_direct_ane_lock_output(kernel) else { throw ProbeError.invalid("Smoke output surface could not be locked") }
    let values = output.assumingMemoryBound(to: UInt16.self)
    let finite = (0..<(channels * side * side)).allSatisfy { Float16(bitPattern: values[$0]).isFinite }
    trufor_direct_ane_unlock_output(kernel)
    return ["backend": "direct ANE bridge smoke", "chainDepth": depth, "side": side, "compiled": true, "evaluated": true,
            "compileAndLoadMilliseconds": compileMs, "evaluationMilliseconds": evalMs,
            "compileCount": trufor_direct_ane_compile_count(), "finiteOutput": finite]
}

@main
private struct DirectANEProbe {
    static func main() {
        do {
            let args = CommandLine.arguments
            if args.count == 2 && args[1] == "--smoke" {
                try emit(runBridgeSmoke(depth: 1))
                return
            }
            if (args.count == 3 || args.count == 4) && args[1] == "--chain-smoke", let depth = Int(args[2]) {
                let side = args.count == 4 ? (Int(args[3]) ?? 32) : 32
                try emit(runBridgeSmoke(depth: depth, side: side))
                return
            }
            let fallbackOnly = args.count >= 2 && args[1] == "--coreml-only"
            let mlxOnly = args.count >= 2 && args[1] == "--mlx-only"
            let isolatedOnly = fallbackOnly || mlxOnly
            let base = isolatedOnly ? 2 : 1
            guard args.count == base + 4 || args.count == base + 5 || args.count == base + 6 else {
                throw ProbeError.invalid("Usage: waifu-trufor-direct-ane-probe [--coreml-only|--mlx-only] MODEL_DIR IMAGE.png REFERENCE.tfm1 COREML_FP32.mlpackage|mlmodelc [samples] [directOutputSide]")
            }
            let modelDirectory = URL(fileURLWithPath: args[base], isDirectory: true)
            let imageURL = URL(fileURLWithPath: args[base + 1])
            let reference = try Reference(url: URL(fileURLWithPath: args[base + 2]))
            let coreMLURL = URL(fileURLWithPath: args[base + 3], isDirectory: true)
            let samples = args.count == base + 5 || args.count == base + 6 ? (Int(args[base + 4]) ?? 7) : 7
            let outputTileSide = args.count == base + 6 ? (Int(args[base + 5]) ?? 128) : outputSide
            guard samples >= 2, samples <= 100, [32, 64, 128, 256, 512].contains(outputTileSide) else {
                throw ProbeError.invalid("Choose 2 to 100 samples and a direct output tile side of 32, 64, 128, 256, or 512")
            }
            let image = try decodeTile(imageURL)
            guard image.width == reference.width && image.height == reference.height else {
                throw ProbeError.invalid("Input image and evidence reference dimensions differ")
            }

            if fallbackOnly {
                try emit(["usedFallback": true, "fallbackReason": "Direct-ANE worker exited before returning a result.",
                          "fallback": try runCoreML(modelURL: coreMLURL, image: image, reference: reference, samples: samples)])
                return
            }
            if mlxOnly {
                try emit(runMLXTile(modelDirectory: modelDirectory, image: image, reference: reference, samples: samples))
                return
            }

            var directReport: [String: Any]?
            var failureReason: String?
            do { directReport = try runDirect(model: modelDirectory, image: image, reference: reference,
                                              samples: samples, outputTileSide: outputTileSide) }
            catch { failureReason = String(describing: error) }
            let passed = directReport?["accepted"] as? Bool == true
            if passed {
                try emit(["usedFallback": false, "directANE": directReport!])
            } else {
                let fallback = try runCoreML(modelURL: coreMLURL, image: image, reference: reference, samples: samples)
                try emit(["usedFallback": true,
                          "fallbackReason": failureReason ?? "Direct ANE did not pass the strict FP16 output-accuracy gate.",
                          "directANE": directReport ?? NSNull(), "fallback": fallback])
            }
        } catch {
            fputs("TruFor direct ANE probe: \(error)\n", stderr)
            exit(1)
        }
    }
}
