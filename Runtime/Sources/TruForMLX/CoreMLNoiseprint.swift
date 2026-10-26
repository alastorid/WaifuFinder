import CoreML
import Foundation
import MLX

/// Core ML replacement for TruFor's DnCNN noiseprint body. The packaged graph
/// uses FP16 internal compute and fixed 546x546 inputs; its input and output
/// interface remains FP32. A separate native MLX path remains available as a
/// runtime fallback.
final class CoreMLNoiseprint {
    private let model: MLModel
    private let rgbName: String
    private let validName: String
    private let outputName: String
    private let side: Int
    private let halo = 17

    init(modelURL: URL) throws {
        let configuration = MLModelConfiguration()
        configuration.computeUnits = .all
        model = try MLModel(contentsOf: modelURL, configuration: configuration)
        let descriptions = model.modelDescription.inputDescriptionsByName
        guard descriptions.count == 2,
              let rgb = descriptions.keys.first(where: {
                  let shape = descriptions[$0]?.multiArrayConstraint?.shape.map(\.intValue) ?? []
                  return shape.count == 4 && shape[0] == 1 && shape[1] == 3 && shape[2] == shape[3]
              }),
              let valid = descriptions.keys.first(where: {
                  let shape = descriptions[$0]?.multiArrayConstraint?.shape.map(\.intValue) ?? []
                  return shape.count == 4 && shape[0] == 1 && shape[1] == 1 && shape[2] == shape[3]
              }),
              let output = model.modelDescription.outputDescriptionsByName.keys.sorted().first,
              let inputSide = descriptions[rgb]?.multiArrayConstraint?.shape.last?.intValue,
              inputSide == 546 else {
            throw TruForError.invalid("Core ML noiseprint model has an unexpected interface")
        }
        rgbName = rgb
        validName = valid
        outputName = output
        side = inputSide
    }

    func predict(_ input: TruForInput, tileSide: Int) throws -> MLXArray {
        guard tileSide > 0, side == tileSide + halo * 2 else {
            throw TruForError.invalid("Core ML noiseprint tile does not match its receptive-field halo")
        }
        let plane = side * side
        let rgb = try MLMultiArray(shape: [1, 3, side, side].map(NSNumber.init(value:)), dataType: .float32)
        let valid = try MLMultiArray(shape: [1, 1, side, side].map(NSNumber.init(value:)), dataType: .float32)
        let rgbBase = rgb.dataPointer.assumingMemoryBound(to: Float.self)
        let validBase = valid.dataPointer.assumingMemoryBound(to: Float.self)
        var result = [Float](repeating: 0, count: input.height * input.width)

        for y in stride(from: 0, to: input.height, by: tileSide) {
            let tileHeight = min(tileSide, input.height - y)
            let imageY0 = max(0, y - halo)
            let imageY1 = min(input.height, y + tileSide + halo)
            let originY = y - halo
            for x in stride(from: 0, to: input.width, by: tileSide) {
                let tileWidth = min(tileSide, input.width - x)
                let imageX0 = max(0, x - halo)
                let imageX1 = min(input.width, x + tileSide + halo)
                let originX = x - halo

                try autoreleasepool {
                    let sourceTile = try input.read(imageY0..<imageY1, columns: imageX0..<imageX1)
                    let sourceValues = sourceTile.asType(.float32).contiguous().asArray(Float.self)
                    rgbBase.update(repeating: 0, count: plane * 3)
                    validBase.update(repeating: 0, count: plane)

                    for sourceY in imageY0..<imageY1 {
                        let targetY = sourceY - originY
                        for sourceX in imageX0..<imageX1 {
                            let targetX = sourceX - originX
                            let source = ((sourceY - imageY0) * (imageX1 - imageX0) + sourceX - imageX0) * 3
                            let target = targetY * side + targetX
                            rgbBase[target] = sourceValues[source]
                            rgbBase[plane + target] = sourceValues[source + 1]
                            rgbBase[2 * plane + target] = sourceValues[source + 2]
                            validBase[target] = 1
                        }
                    }

                    let provider = try MLDictionaryFeatureProvider(dictionary: [
                        rgbName: MLFeatureValue(multiArray: rgb),
                        validName: MLFeatureValue(multiArray: valid)
                    ])
                    let prediction = try model.prediction(from: provider)
                    guard let output = prediction.featureValue(for: outputName)?.multiArrayValue,
                          output.dataType == .float32, output.shape.map(\.intValue) == [1, 1, side, side] else {
                        throw TruForError.invalid("Core ML noiseprint returned an unexpected tensor")
                    }
                    let values = output.dataPointer.assumingMemoryBound(to: Float.self)
                    let strides = output.strides.map(\.intValue)
                    for localY in 0..<tileHeight {
                        let outputY = (halo + localY) * strides[2]
                        let destinationY = (y + localY) * input.width + x
                        for localX in 0..<tileWidth {
                            result[destinationY + localX] = values[outputY + (halo + localX) * strides[3]]
                        }
                    }
                }
            }
        }
        let output = MLXArray(result, [1, input.height, input.width, 1])
        eval(output)
        return output
    }
}
