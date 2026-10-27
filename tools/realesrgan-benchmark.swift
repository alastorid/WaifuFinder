import AppKit
import CoreML
import Foundation
import Darwin

enum UncoverError: Error, LocalizedError {
    case backend(String)
    var errorDescription: String? { if case .backend(let message) = self { return message }; return nil }
}
func argument(_ name: String) -> String? {
    guard let i = CommandLine.arguments.firstIndex(of:name),i+1 < CommandLine.arguments.count else { return nil }; return CommandLine.arguments[i+1]
}
func footprint() -> UInt64 {
    var info = task_vm_info_data_t(); var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
    let status = withUnsafeMutablePointer(to:&info) { $0.withMemoryRebound(to:integer_t.self,capacity:Int(count)) { task_info(mach_task_self_,task_flavor_t(TASK_VM_INFO),$0,&count) } }
    return status == KERN_SUCCESS ? info.phys_footprint : 0
}
final class PeakMemory: @unchecked Sendable {
    private let lock = NSLock(); private var peak:UInt64 = 0
    private let timer = DispatchSource.makeTimerSource(queue:DispatchQueue(label:"spatial-memory"))
    init() { timer.schedule(deadline:.now(),repeating:.milliseconds(20));timer.setEventHandler { [weak self] in
        guard let self else { return };let bytes=footprint();self.lock.lock();self.peak=max(self.peak,bytes);self.lock.unlock()
        if bytes > 6_000_000_000 { fputs("RealESRGAN exceeded 6 GB physical memory\n",stderr);_exit(75) }
    };timer.resume() }
    deinit { timer.cancel() }
    var bytes:UInt64 { lock.lock();defer {lock.unlock()};return max(peak,footprint()) }
}
@main struct Benchmark {
 static func main() {
    do { try run() }
    catch { fputs("RealESRGAN validation: \(error.localizedDescription)\n",stderr);exit(1) }
 }
 private static func run() throws {
    guard let path = argument("--model"),let reportPath=argument("--report") else { throw UncoverError.backend("Supply --model and --report") }
    let monitor=PeakMemory(),start=Date(),url=URL(fileURLWithPath:path).standardizedFileURL
    let scale=Int(argument("--scale") ?? "2") ?? 2
    guard [2,4].contains(scale) else { throw UncoverError.backend("Scale must be 2 or 4") }
    let units=argument("--units") ?? "ane"
    guard units == "ane" || units == "cpu" else { throw UncoverError.backend("RealESRGAN permits only --units ane or cpu") }
    let configuration=NativeSpatialUpscaler.configuration(cpuOnly:units == "cpu")
    if CommandLine.arguments.contains("--fast-prediction") { configuration.optimizationHints.specializationStrategy = .fastPrediction }
    let warmups=Int(argument("--warmups") ?? "1") ?? 1
    guard warmups >= 0 && warmups <= 3 else { throw UncoverError.backend("Use 0...3 warmups") }
    var report:[String:Any]=["units":units,"scale":scale,"warmups":warmups,"memoryScope":"whole benchmark process; includes image IO; excludes unowned Core ML/ANE services","specialization":CommandLine.arguments.contains("--fast-prediction") ? "fastPrediction" : "default","thermalStateAtStart":ProcessInfo.processInfo.thermalState.rawValue]
    if let fixturePath=argument("--fixtures") {
        let model=try MLModel(contentsOf:url,configuration:configuration)
        report["loadSeconds"]=Date().timeIntervalSince(start)
        let directory=URL(fileURLWithPath:fixturePath),cases=try JSONSerialization.jsonObject(with:Data(contentsOf:directory.appendingPathComponent("cases.json"))) as! [[String:Any]]
        var results=[[String:Any]]()
        let repeats=Int(argument("--repeats") ?? "1") ?? 1
        for item in cases {
            let name=item["name"] as! String,size=item["size"] as! Int,w=item["width"] as! Int,h=item["height"] as! Int
            if let requested=argument("--size"),size != Int(requested) { continue }
            for repeatIndex in 0..<repeats {
                let result:[String:Any]=try autoreleasepool {
                    let data=try Data(contentsOf:directory.appendingPathComponent(name+"-input.bin")),gold=try Data(contentsOf:directory.appendingPathComponent(name+"-output.bin"))
                    guard w > 0, h > 0, w <= size, h <= size, data.count == 3*size*size*4, gold.count == 3*w*h*scale*scale*4 else { throw UncoverError.backend("Invalid fixture dimensions or byte counts") }
                    let dtype=model.modelDescription.inputDescriptionsByName["input"]!.multiArrayConstraint!.dataType
                    let input=try MLMultiArray(shape:[1,3,NSNumber(value:size),NSNumber(value:size)],dataType:dtype)
                    data.withUnsafeBytes { raw in
                        let values=raw.bindMemory(to:Float.self)
                        if dtype == .float16 { let p=input.dataPointer.assumingMemoryBound(to:Float16.self);for i in values.indices { p[i]=Float16(values[i]) } }
                        else { input.dataPointer.assumingMemoryBound(to:Float.self).update(from:values.baseAddress!,count:values.count) }
                    }
                    let provider=try MLDictionaryFeatureProvider(dictionary:["input":input])
                    if repeatIndex == 0 { for _ in 0..<warmups { _ = try model.prediction(from:provider) } }
                    let tick=Date(),output=try model.prediction(from:provider)
                    let seconds=Date().timeIntervalSince(tick)
                    let tensor=output.featureValue(for:output.featureNames.sorted()[0])!.multiArrayValue!,strides=tensor.strides.map(\.intValue)
                    guard tensor.shape.map(\.intValue) == [1,3,size*scale,size*scale] else { throw UncoverError.backend("Bad output shape") }
                    var maximum:Float=0,sum:Double=0,pixels=[Float]();pixels.reserveCapacity(3*w*h*scale*scale)
                    try gold.withUnsafeBytes { raw in
                        let values=raw.bindMemory(to:Float.self)
                        for c in 0..<3 { for y in 0..<h*scale { for x in 0..<w*scale {
                            let index=c*strides[1]+y*strides[2]+x*strides[3]
                            let value=tensor.dataType == .float16 ? Float(tensor.dataPointer.assumingMemoryBound(to:Float16.self)[index]) : tensor.dataPointer.assumingMemoryBound(to:Float.self)[index]
                            guard value.isFinite else { throw UncoverError.backend("Nonfinite restored pixel") }
                            let clipped=min(1,max(0,value)),error=abs(clipped-values[(c*h*scale+y)*w*scale+x]);maximum=max(maximum,error);sum+=Double(error);pixels.append(clipped)
                        } } }
                    }
                    if let dump=argument("--dump") { let destination=URL(fileURLWithPath:dump);try FileManager.default.createDirectory(at:destination,withIntermediateDirectories:true);try pixels.withUnsafeBytes { try Data($0).write(to:destination.appendingPathComponent(name+".bin")) } }
                    let mean=sum/Double(pixels.count)
                    return ["name":name,"size":size,"scale":scale,"iteration":repeatIndex,"seconds":seconds,"maximumAbsoluteError":maximum,"meanAbsoluteError":mean,"passed":maximum <= 0.01 && mean <= 0.001]
                }
                results.append(result);print("Verified \(name) \(repeatIndex+1)");fflush(stdout)
            }
        }
        guard !results.isEmpty else { throw UncoverError.backend("No matching fixtures") }
        report["measurements"]=results;report["passed"]=results.allSatisfy { $0["passed"] as! Bool }
        if CommandLine.arguments.contains("--compute-plan") { do { report["plan"]=try devicePlan(url,configuration) } catch { report["planError"]=error.localizedDescription } }
    } else if let imagePath=argument("--image") {
        guard units == "ane" else { throw UncoverError.backend("Image benchmarks use the production ANE policy; use fixtures for CPU diagnostics") }
        let upscaler=try NativeSpatialUpscaler(modelURL:url,variant:scale == 4 ? .x4plus : .x2plus)
        report["loadSeconds"]=Date().timeIntervalSince(start)
        report["actualComputeUnits"]=upscaler.selectedComputeUnits
        guard let source=CGImageSourceCreateWithURL(URL(fileURLWithPath:imagePath) as CFURL,nil),let image=CGImageSourceCreateImageAtIndex(source,0,nil) else { throw UncoverError.backend("Bad image") }
        if let small=image.cropping(to:CGRect(x:0,y:0,width:min(48,image.width),height:min(33,image.height))) {
            for _ in 0..<warmups { _ = try upscaler.upscale(small) }
        }
        upscaler.resetDiagnostics()
        var times=[Double](),hashes=[String]()
        for i in 0..<(Int(argument("--repeats") ?? "3") ?? 3) {
            let tick=Date()
            try autoreleasepool {
                let result=try upscaler.upscale(image)
                guard result.width == image.width*scale && result.height == image.height*scale else { throw UncoverError.backend("Incorrect upscale geometry") }
                if let dump=argument("--dump") { let directory=URL(fileURLWithPath:dump);try FileManager.default.createDirectory(at:directory,withIntermediateDirectories:true);let dest=CGImageDestinationCreateWithURL(directory.appendingPathComponent("\(i).png") as CFURL,"public.png" as CFString,1,nil)!;CGImageDestinationAddImage(dest,result,nil);guard CGImageDestinationFinalize(dest) else { throw UncoverError.backend("Write failed") } }
                hashes.append(String(result.width)+"x"+String(result.height))
            }
            times.append(Date().timeIntervalSince(tick));print("Restored image \(i+1)");fflush(stdout)
        }
        report["predictionSizes"]=upscaler.predictionSizes;report["predictionSeconds"]=upscaler.predictionSeconds;report["secondsPerImage"]=times;report["geometry"]=hashes;report["passed"]=true
    } else { throw UncoverError.backend("Supply --fixtures or --image") }
    report["peakPhysicalFootprintBytes"]=monitor.bytes;report["thermalStateAtEnd"]=ProcessInfo.processInfo.thermalState.rawValue
    try JSONSerialization.data(withJSONObject:report,options:[.sortedKeys,.prettyPrinted]).write(to:URL(fileURLWithPath:reportPath),options:.atomic)
    if report["passed"] as? Bool == false { exit(1) }
 }
}
