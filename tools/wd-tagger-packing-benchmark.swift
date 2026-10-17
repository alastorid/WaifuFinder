// Compare the production RGB packer with Accelerate, checking every value.
import Accelerate
import CoreML
import Foundation

@inline(never) private func accelerated(_ rgba: [UInt8], input: MLMultiArray) {
    let plane = 448 * 448
    let destination = input.dataPointer.assumingMemoryBound(to: Float.self)
    rgba.withUnsafeBufferPointer { source in
        for channel in 0..<3 {
            vDSP_vfltu8(source.baseAddress!.advanced(by: channel), 4,
                       destination.advanced(by: channel * plane), 1, vDSP_Length(plane))
        }
    }
}

@main struct PackingBenchmark {
    static func main() throws {
        let plane = 448 * 448, iterations = 1000
        let rgba = (0..<(plane * 4)).map { UInt8(truncatingIfNeeded: $0 &* 71 &+ $0 / 4) }
        let input = try MLMultiArray(shape: [8, 3, 448, 448], dataType: .float32)
        var records = [[String: Any]]()
        for round in 0..<6 {
            for method in (round.isMultiple(of: 2) ? ["scalar", "accelerate"] : ["accelerate", "scalar"]) {
                let start = Date()
                for _ in 0..<iterations {
                    if method == "scalar" { WDTaggerBackend.writeRGB(rgba, to: input, batchIndex: 0) }
                    else { accelerated(rgba, input: input) }
                }
                let seconds = Date().timeIntervalSince(start)
                let output = input.dataPointer.assumingMemoryBound(to: Float.self)
                for pixel in 0..<plane {
                    for channel in 0..<3 {
                        precondition(output[channel * plane + pixel] == Float(rgba[pixel * 4 + channel]))
                    }
                }
                records.append(["method": method, "seconds": seconds, "iterations": iterations])
            }
        }
        let data = try JSONSerialization.data(withJSONObject: records, options: [.prettyPrinted, .sortedKeys])
        if CommandLine.arguments.count == 2 { try data.write(to: URL(fileURLWithPath: CommandLine.arguments[1])) }
        else { print(String(data: data, encoding: .utf8)!) }
    }
}
