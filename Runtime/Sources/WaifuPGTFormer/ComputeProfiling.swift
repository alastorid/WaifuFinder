import Foundation
import CoreML

// GPU execution preserves the model's stored precision. ANE experiments are
// exposed only through offline verification.
func computeConfiguration(_ units: String) throws -> MLModelConfiguration {
    let config = MLModelConfiguration()
    switch units {
    case "cpu": config.computeUnits = .cpuOnly
    case "gpu": config.computeUnits = .cpuAndGPU
    case "ane": config.computeUnits = .cpuAndNeuralEngine
    case "all": config.computeUnits = .all
    default: throw NSError(domain:"PGTFormer",code:8,userInfo:[NSLocalizedDescriptionKey:"Unknown compute device \(units)"])
    }
    config.allowLowPrecisionAccumulationOnGPU = false
    return config
}

func devicePlan(_ url: URL, _ config: MLModelConfiguration) throws -> [String:Any] {
    let semaphore = DispatchSemaphore(value:0)
    var result: Result<[String:Any],Error>!
    Task.detached {
        do {
            let plan = try await MLComputePlan.load(contentsOf:url,configuration:config)
            var counts = [String:Int]()
            if case .program(let program) = plan.modelStructure {
                func visit(_ block: MLModelStructure.Program.Block) {
                    for operation in block.operations {
                        let device: String
                        switch plan.deviceUsage(for:operation)?.preferred {
                        case .cpu: device = "cpu"
                        case .gpu: device = "gpu"
                        case .neuralEngine: device = "ane"
                        default: device = "unknown"
                        }
                        counts[device,default:0] += 1
                        for nested in operation.blocks { visit(nested) }
                    }
                }
                for function in program.functions.values { visit(function.block) }
            }
            result = .success(["preferredOperationCounts":counts,"evidence":"Core ML anticipated placement; not hardware counters"])
        } catch { result = .failure(error) }
        semaphore.signal()
    }
    let deadline = Date().addingTimeInterval(120)
    while semaphore.wait(timeout:.now()) != .success {
        guard Date() < deadline else { throw NSError(domain:"PGTFormer",code:9,userInfo:[NSLocalizedDescriptionKey:"Compute plan timed out"]) }
        RunLoop.current.run(until:Date().addingTimeInterval(0.02))
    }
    return try result.get()
}
