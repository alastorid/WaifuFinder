import Foundation
import Darwin
import CoreML
import CoreImage
import ImageIO
import UniformTypeIdentifiers
import CryptoKit
import NativeVideoIO
import MLX

struct Frame: Decodable { let index: Int }
struct Manifest: Decodable { let scale: Int; let regionWidth: Int; let regionHeight: Int; let frames: [Frame] }
let revision = "e78a26cd463182f4a5aa96b8681110da8615d44d"
func progress(_ message: String) { print("Progress: " + message); fflush(stdout) }
func sha256(_ url: URL) throws -> String {
    let handle = try FileHandle(forReadingFrom: url); defer { try? handle.close() }
    var hash = SHA256()
    while try autoreleasepool(invoking: {
        guard let data = try handle.read(upToCount:1024*1024), !data.isEmpty else { return false }
        hash.update(data:data)
        return true
    }) {}
    return hash.finalize().map { String(format: "%02x", $0) }.joined()
}
func checkedModel(_ root: URL) throws -> URL {
    let metadata = try JSONSerialization.jsonObject(with: Data(contentsOf: root.appendingPathComponent("manifest.json"))) as? [String: Any]
    guard metadata?["algorithm"] as? String == "pgtformer", metadata?["upstreamRevision"] as? String == revision,
          metadata?["validated"] as? Bool == true,
          metadata?["weightsSHA256"] as? String == "37f9a42b9e35e0ccb89016ca6cea6728a7bf783e70b99e886ac52bb37c6855f3",
          metadata?["runtime"] as? String == "cpu-codebook-gpu-decoder-v1",
          let hashes = metadata?["splitPackageSHA256"] as? [String: String], !hashes.isEmpty else {
        throw NSError(domain: "PGTFormer", code: 1, userInfo: [NSLocalizedDescriptionKey:"Install a validated original PGTFormer package first."])
    }
    let package = root
    for (file, expected) in hashes {
        guard !file.hasPrefix("/"), !file.split(separator: "/").contains(".."),
              try sha256(package.appendingPathComponent(file)) == expected else {
            throw NSError(domain:"PGTFormer",code:2,userInfo:[NSLocalizedDescriptionKey:"PGTFormer package integrity check failed."])
        }
    }
    return package
}
func plane(_ source: (bytes:[UInt8],width:Int,height:Int)) -> [Float] {
    var result = [Float](repeating:0,count:3*512*512)
    // Only the selected ROI is transformed into the model's required 512 square.
    // Half-pixel bilinear coordinates, edge clamping, RGB uint8 then /255.
    for y in 0..<512 {
        let sy = (Double(y)+0.5)*Double(source.height)/512-0.5, iy = Int(floor(sy)), fy = sy-floor(sy)
        let y0 = max(0,min(source.height-1,iy)), y1 = max(0,min(source.height-1,iy+1))
        for x in 0..<512 {
            let sx = (Double(x)+0.5)*Double(source.width)/512-0.5, ix = Int(floor(sx)), fx = sx-floor(sx)
            let x0 = max(0,min(source.width-1,ix)), x1 = max(0,min(source.width-1,ix+1))
            for c in 0..<3 {
                func at(_ yy:Int,_ xx:Int)->Double { Double(source.bytes[(yy*source.width+xx)*4+c]) }
                let value = (at(y0,x0)*(1-fx)+at(y0,x1)*fx)*(1-fy)+(at(y1,x0)*(1-fx)+at(y1,x1)*fx)*fy
                result[c*512*512+y*512+x] = Float(value.rounded())/255
            }
        }
    }
    return result
}
func writeOutput(_ tensor: MLMultiArray, width: Int, height: Int, url: URL, context: CIContext) throws {
    guard tensor.dataType == .float32, tensor.shape.map(\.intValue) == [1,3,512,512] else { throw NSError(domain:"PGTFormer",code:3) }
    let pointer = tensor.dataPointer.assumingMemoryBound(to:Float.self), strides = tensor.strides.map(\.intValue)
    var pixels = [UInt8](repeating:0,count:512*512*3)
    for y in 0..<512 { for x in 0..<512 { for c in 0..<3 {
        let value = pointer[c*strides[1]+y*strides[2]+x*strides[3]]
        guard value.isFinite else { throw NSError(domain:"PGTFormer",code:4) }
        pixels[(y*512+x)*3+c] = UInt8(max(0,min(255,value*255)))
    } } }
    guard let provider = CGDataProvider(data:Data(pixels) as CFData),
          let image = CGImage(width:512,height:512,bitsPerComponent:8,bitsPerPixel:24,bytesPerRow:512*3,
              space:CGColorSpaceCreateDeviceRGB(),bitmapInfo:CGBitmapInfo(rawValue:0),provider:provider,decode:nil,shouldInterpolate:false,intent:.defaultIntent) else { throw NSError(domain:"PGTFormer",code:5) }
    let resized = CIImage(cgImage:image).applyingFilter("CILanczosScaleTransform",parameters:[kCIInputScaleKey:Double(height)/512,kCIInputAspectRatioKey:Double(width)/Double(height)])
    guard let output = width == 512 && height == 512 ? image : context.createCGImage(resized,from:CGRect(x:0,y:0,width:width,height:height)),
          let destination = CGImageDestinationCreateWithURL(url as CFURL,UTType.png.identifier as CFString,1,nil) else { throw NSError(domain:"PGTFormer",code:6) }
    CGImageDestinationAddImage(destination,output,nil)
    guard CGImageDestinationFinalize(destination) else { throw NSError(domain:"PGTFormer",code:7) }
}
setpgid(0,0)
do {
    guard let path = option("--model") else { fail("Supply --model with the validated PGTFormer directory") }
    if let referencePath = option("--verify-split") {
        // Offline installation verification only; this never configures the app.
        let root = URL(fileURLWithPath:path).standardizedFileURL
        let monitor = MemoryMonitor(budget: 6_000_000_000)
        let start = Date()
        let encURL = try MLModel.compileModel(at:root.appendingPathComponent("PGTEncoder.mlpackage"))
        defer { try? FileManager.default.removeItem(at:encURL) }
        let decURL = try MLModel.compileModel(at:root.appendingPathComponent("PGTDecoder.mlpackage"))
        defer { try? FileManager.default.removeItem(at:decURL) }
        let ec = try computeConfiguration(option("--encoder-units") ?? "gpu")
        let dc = try computeConfiguration(option("--decoder-units") ?? "gpu")
        let encoder = try MLModel(contentsOf:encURL,configuration:ec)
        let decoder = try MLModel(contentsOf:decURL,configuration:dc)
        let loadSeconds = Date().timeIntervalSince(start)
        let referenceURL = URL(fileURLWithPath:referencePath)
        var isDirectory: ObjCBool = false
        FileManager.default.fileExists(atPath:referencePath,isDirectory:&isDirectory)
        let references = isDirectory.boolValue ? try FileManager.default.contentsOfDirectory(at:referenceURL,includingPropertiesForKeys:nil).filter { $0.pathExtension == "safetensors" }.sorted { $0.path < $1.path } : [referenceURL]
        guard !references.isEmpty else { fail("No original reference tensors") }
        let iterations = Int(option("--iterations") ?? "1") ?? 0
        guard (1...10).contains(iterations) else { fail("Invalid verification iteration count") }
        var measurements = [[String:Any]](), passed = true
        for referenceURL in references {
            for iteration in 0..<iterations {
                let measurement: [String:Any] = try autoreleasepool {
                    let reference = try MLX.loadArrays(url:referenceURL)
                    guard let frames = reference["frames"], let expected = reference["restored"],
                          frames.shape == [3,3,512,512], expected.shape == [1,3,512,512] else { fail("Invalid original reference tensors") }
                    let input = try MLMultiArray(shape:[3,3,512,512],dataType:.float32)
                    let values = frames.asArray(Float.self)
                    values.withUnsafeBufferPointer { input.dataPointer.assumingMemoryBound(to:Float.self).update(from:$0.baseAddress!,count:$0.count) }
                    let tick = Date()
                    let features = try encoder.prediction(from:MLDictionaryFeatureProvider(dictionary:["frames":input]))
                    let encoderSeconds = Date().timeIntervalSince(tick), decoderTick = Date()
                    let output = try decoder.prediction(from:features)
                    let decoderSeconds = Date().timeIntervalSince(decoderTick)
                    guard let actual = output.featureValue(for:"restored")?.multiArrayValue,
                          actual.shape.map(\.intValue) == [1,3,512,512], actual.dataType == .float32 else { fail("Invalid split model output") }
                    let gold = expected.asArray(Float.self), strides = actual.strides.map(\.intValue)
                    let pointer = actual.dataPointer.assumingMemoryBound(to:Float.self)
                    var maximum:Float = 0, sum:Double = 0
                    for c in 0..<3 { for y in 0..<512 { for x in 0..<512 {
                        let error = abs(pointer[c*strides[1]+y*strides[2]+x*strides[3]]-gold[c*512*512+y*512+x])
                        guard error.isFinite else { fail("Nonfinite split output") }; maximum = max(maximum,error); sum += Double(error)
                    } } }
                    let mean = sum/Double(gold.count)
                    return ["reference":referenceURL.lastPathComponent,"iteration":iteration,"passed":maximum <= 0.02 && mean <= 0.002,"maximumAbsoluteError":maximum,"meanAbsoluteError":mean,"encoderSeconds":encoderSeconds,"decoderSeconds":decoderSeconds]
                }
                passed = passed && measurement["passed"] as! Bool
                measurements.append(measurement)
                progress("Verified \(referenceURL.lastPathComponent) iteration \(iteration+1)")
            }
        }
        var report:[String:Any] = ["passed":passed,"backend":"native Swift Core ML", "encoderUnits":option("--encoder-units") ?? "gpu","decoderUnits":option("--decoder-units") ?? "gpu","measurements":measurements,"loadSeconds":loadSeconds,"peakPhysicalFootprintBytes":monitor.peakBytes]
        report["maximumAbsoluteError"] = measurements.map { $0["maximumAbsoluteError"] as! Float }.max()!
        report["meanAbsoluteError"] = measurements.map { $0["meanAbsoluteError"] as! Double }.max()!
        if CommandLine.arguments.contains("--compute-plan") {
            do { report["encoderPlan"] = try devicePlan(encURL,ec) }
            catch { report["encoderPlanError"] = error.localizedDescription }
            do { report["decoderPlan"] = try devicePlan(decURL,dc) }
            catch { report["decoderPlanError"] = error.localizedDescription }
        }
        let reportURL = option("--report").map { URL(fileURLWithPath:$0) } ?? root.appendingPathComponent("split-validation.json")
        try JSONSerialization.data(withJSONObject:report,options:[.prettyPrinted,.sortedKeys]).write(to:reportURL,options:.atomic)
        print(report); exit(passed ? 0 : 1)
    }
    let root = URL(fileURLWithPath:path).standardizedFileURL, package = try checkedModel(root)
    if CommandLine.arguments.contains("--configure") {
        let directory = FileManager.default.urls(for:.applicationSupportDirectory,in:.userDomainMask)[0].appendingPathComponent("Waifu/Uncover")
        try FileManager.default.createDirectory(at:directory,withIntermediateDirectories:true)
        let config: [String:Any] = ["executable":URL(fileURLWithPath:CommandLine.arguments[0]).standardizedFileURL.path,"model":root.standardizedFileURL.path,"memoryBudgetGB":min(6,Double(ProcessInfo.processInfo.physicalMemory)/2e9)]
        try JSONSerialization.data(withJSONObject:config,options:[.prettyPrinted,.sortedKeys]).write(to:directory.appendingPathComponent("pgtformer.json"),options:.atomic)
        print("PGTFormer configured for Uncover Waifu"); exit(0)
    }
    guard let jobPath = option("--job") else { fail("Supply --job") }
    let budget = Double(option("--memory-budget-gb") ?? "6") ?? 0
    guard budget.isFinite, budget > 0, budget*1e9 <= Double(ProcessInfo.processInfo.physicalMemory)/2 else { fail("Invalid memory budget") }
    let monitor = MemoryMonitor(budget:Int(budget*1e9))
    let lockRoot = FileManager.default.urls(for:.applicationSupportDirectory,in:.userDomainMask)[0].appendingPathComponent("Waifu/Uncover")
    try FileManager.default.createDirectory(at:lockRoot,withIntermediateDirectories:true)
    let descriptor = open(lockRoot.appendingPathComponent("native.lock").path,O_CREAT|O_RDWR|O_NOFOLLOW,0o600)
    guard descriptor >= 0, flock(descriptor,LOCK_EX|LOCK_NB) == 0 else { fail("Another native restoration job is running") }
    defer { flock(descriptor,LOCK_UN); close(descriptor) }
    let job = URL(fileURLWithPath:jobPath), manifest = try JSONDecoder().decode(Manifest.self,from:Data(contentsOf:job.appendingPathComponent("manifest.json")))
    guard [1,2,4,8,16,32,64].contains(manifest.scale), !manifest.frames.isEmpty, manifest.frames.count <= 300,
          manifest.regionWidth > 0, manifest.regionHeight > 0, manifest.frames.enumerated().allSatisfy({$0.offset == $0.element.index}) else { fail("Invalid PGTFormer video manifest") }
    let start = Date(), initialThermalState = ProcessInfo.processInfo.thermalState.rawValue; progress("Loading PGTFormer")
    // Package integrity was checked above. Content/OS-addressed compilation
    // persists across worker processes without accepting stale model weights.
    let encoderCompiled = try compiledModel(package,"PGTEncoder")
    let decoderCompiled = try compiledModel(package,"PGTDecoder")
    // Keep the validated FP32 encoder graph, but allow its convolutions and
    // transformer to run on Metal too. CPU-only here left the GPU idle for
    // roughly half of every frame. The override supports paired benchmarks.
    let encoderUnits = option("--encoder-units") ?? "gpu"
    guard ["cpu", "gpu"].contains(encoderUnits) else { fail("Use cpu or gpu for the encoder") }
    let encoderConfig = try computeConfiguration(encoderUnits)
    let decoderConfig = try computeConfiguration("gpu")
    let encoder = try MLModel(contentsOf:encoderCompiled,configuration:encoderConfig)
    let decoder = try MLModel(contentsOf:decoderCompiled,configuration:decoderConfig)
    guard encoder.modelDescription.inputDescriptionsByName["frames"]?.multiArrayConstraint?.shape.map(\.intValue) == [3,3,512,512],
          decoder.modelDescription.outputDescriptionsByName["restored"]?.multiArrayConstraint?.shape.map(\.intValue) == [1,3,512,512] else { fail("Incompatible PGTFormer interface") }
    let loadSeconds = Date().timeIntervalSince(start), context = CIContext(options:[.cacheIntermediates:false])
    let input = try MLMultiArray(shape:[3,3,512,512],dataType:.float32)
    try FileManager.default.createDirectory(at:job.appendingPathComponent("reconstructed"),withIntermediateDirectories:true)
    var cached: [Int:[Float]] = [:], windows: [[Int]] = [], inferenceSeconds = 0.0, encoderSeconds = 0.0, decoderSeconds = 0.0
    for frame in manifest.frames {
        try autoreleasepool {
            let indices = [max(0,frame.index-1),frame.index,min(manifest.frames.count-1,frame.index+1)]
            cached = cached.filter { indices.contains($0.key) }
            for index in Set(indices) where cached[index] == nil {
                let url = job.appendingPathComponent(String(format:"degraded/%06d.png",index))
                let image = try readRGB(url,straightAlpha:true)
                guard image.width == (manifest.regionWidth+manifest.scale-1)/manifest.scale, image.height == (manifest.regionHeight+manifest.scale-1)/manifest.scale else { fail("PGTFormer crop dimensions do not match job") }
                cached[index] = plane(image)
            }
            for (slot,index) in indices.enumerated() {
                cached[index]!.withUnsafeBufferPointer { p in input.dataPointer.assumingMemoryBound(to:Float.self).advanced(by:slot*3*512*512).update(from:p.baseAddress!,count:p.count) }
            }
            progress("PGTFormer · frame \(frame.index+1)/\(manifest.frames.count)")
            let tick = Date()
            let encoded = try encoder.prediction(from:MLDictionaryFeatureProvider(dictionary:["frames":input]))
            let decoderTick = Date()
            encoderSeconds += decoderTick.timeIntervalSince(tick)
            let output = try decoder.prediction(from:encoded)
            decoderSeconds += Date().timeIntervalSince(decoderTick)
            inferenceSeconds += Date().timeIntervalSince(tick)
            guard let tensor = output.featureValue(for:"restored")?.multiArrayValue else { fail("PGTFormer returned no restored center frame") }
            if let validationPath = option("--validation-dir") {
                let directory = URL(fileURLWithPath:validationPath)
                try FileManager.default.createDirectory(at:directory,withIntermediateDirectories:true)
                var restored = [Float](); restored.reserveCapacity(3*512*512)
                let strides = tensor.strides.map(\.intValue), p = tensor.dataPointer.assumingMemoryBound(to:Float.self)
                for c in 0..<3 { for y in 0..<512 { for x in 0..<512 { restored.append(p[c*strides[1]+y*strides[2]+x*strides[3]]) } } }
                try MLX.save(arrays:["frames":MLXArray(indices.flatMap { cached[$0]! }).reshaped([3,3,512,512]),
                    "restored":MLXArray(restored).reshaped([1,3,512,512])],url:directory.appendingPathComponent(String(format:"%06d.safetensors",frame.index)))
            }
            try writeOutput(tensor,width:manifest.regionWidth,height:manifest.regionHeight,url:job.appendingPathComponent(String(format:"reconstructed/%06d.png",frame.index)),context:context)
            windows.append(indices)
        }
    }
    let modelMetadata = try JSONSerialization.jsonObject(with:Data(contentsOf:root.appendingPathComponent("manifest.json"))) as! [String:Any]
    let report:[String:Any] = ["thermalStateAtStart":initialThermalState,"thermalStateAtEnd":ProcessInfo.processInfo.thermalState.rawValue,"optimizations":modelMetadata["optimizations"] ?? [:],"model":"pgtformer","backend":"native Swift Core ML / Apple GPU","pythonRuntime":false,"frames":manifest.frames.count,"scale":manifest.scale,"temporalWindows":windows,"modelInputShape":[3,3,512,512],"centerFrame":1,"sourceDownsampling":manifest.scale,"loadSeconds":loadSeconds,"inferenceSeconds":inferenceSeconds,"encoderSeconds":encoderSeconds,"decoderSeconds":decoderSeconds,"compiledModelCache":true,"executionPolicy":"gpu-encoder-decoder-v2","encoderUnits":encoderUnits,"decoderUnits":"gpu","seconds":Date().timeIntervalSince(start),"peakPhysicalFootprintBytes":monitor.peakBytes,"memoryBudgetBytes":Int(budget*1e9)]
    try JSONSerialization.data(withJSONObject:report,options:[.prettyPrinted,.sortedKeys]).write(to:job.appendingPathComponent("report.json"),options:.atomic)
    progress("PGTFormer restoration complete")
} catch { fail("PGTFormer: \(error.localizedDescription)") }
