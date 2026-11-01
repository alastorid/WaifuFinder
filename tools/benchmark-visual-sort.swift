import Foundation
import Metal

@main
struct VisualSortBenchmark {
    static func distanceAccess() throws {
        print("photos,baseline_ms,optimized_ms,speedup,extra_matrix_bytes")
        for count in [128, 256, 511] {
            let dimensions = 768
            let data = (0..<(count * dimensions)).map { sin(Float($0 % 1009) * 0.01) }
            let vectors = VisualFeatureVectors(count: count, dimensions: dimensions, values: data)!
            func baseline() throws -> TSPDistanceMatrix {
                var matrix = [Float](repeating: 0, count: count * count)
                for left in 0..<count {
                    for right in (left + 1)..<count {
                        let value = vectors.distance(left, right)
                        matrix[left * count + right] = value
                        matrix[right * count + left] = value
                    }
                }
                return try TSPDistanceMatrix(count: count, values: matrix)
            }
            var before: [Double] = [], after: [Double] = []
            for _ in 0..<9 {
                let start = ProcessInfo.processInfo.systemUptime
                let expected = try baseline()
                let middle = ProcessInfo.processInfo.systemUptime
                let actual = try vectors.matrix()
                let end = ProcessInfo.processInfo.systemUptime
                guard expected.values == actual.values else { throw TSPError.invalidMatrix }
                before.append((middle - start) * 1000)
                after.append((end - middle) * 1000)
            }
            let a = before.sorted()[4], b = after.sorted()[4]
            print(String(format: "%d,%.3f,%.3f,%.3f,0", count, a, b, a / b))
        }
    }

    static func main() throws {
        if CommandLine.arguments.contains("--distance-access") { try distanceAccess(); return }
        let metal = try CuOptMetalTSP()
        print("device=\(MTLCreateSystemDefaultDevice()?.name ?? "unknown"), optimized Swift, 16 search iterations, four initial routes")
        print("photos,cpu_ms,metal_ms,cpu_cost,metal_cost,max_distance_error")
        for count in [32, 64, 128, 256, 512, 1_024] {
            let dimensions = 768
            var data: [Float] = []
            for row in 0..<count {
                let position = Float((row * 7_919) % count) / Float(count)
                for column in 0..<dimensions {
                    data.append(sin(position * Float(column % 7 + 1) * 3) + Float((row + column * 17) % 256) * 0.0001)
                }
            }
            let vectors = VisualFeatureVectors(count: count, dimensions: dimensions, values: data)!
            let seeds = CuOptTSP.seeds(from: Array(0..<count))
            let cpuStart = ProcessInfo.processInfo.systemUptime
            let matrix = try vectors.matrix()
            let cpu = try CuOptTSP.solveCPU(matrix, seeds: seeds, iterations: 16)
            let cpuTime = (ProcessInfo.processInfo.systemUptime - cpuStart) * 1_000
            let gpuStart = ProcessInfo.processInfo.systemUptime
            let gpu = try metal.solve(vectors, seeds: seeds, iterations: 16)
            let gpuTime = (ProcessInfo.processInfo.systemUptime - gpuStart) * 1_000
            let gpuMatrix = try metal.distances(vectors)
            let error = zip(matrix.values, gpuMatrix.values).map { abs($0 - $1) }.max() ?? 0
            guard Set(gpu) == Set(0..<count), matrix.cost(gpu) <= matrix.cost(seeds[0]) + 1e-4, error < 2e-5 else {
                throw TSPError.invalidTour
            }
            print(String(format: "%d,%.3f,%.3f,%.6f,%.6f,%.9f", count, cpuTime, gpuTime, matrix.cost(cpu), matrix.cost(gpu), error))
        }
    }
}
