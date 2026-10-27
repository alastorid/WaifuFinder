import AppKit
import Foundation
import CoreML

func realESRGANFixture(width: Int = 33, height: Int = 19) throws -> CGImage {
    var bytes = [UInt8](repeating:255,count:width*height*4)
    for y in 0..<height { for x in 0..<width {
        let i=(y*width+x)*4
        bytes[i]=UInt8((x*7+y*3)%256);bytes[i+1]=UInt8((y*11+x*3)%256);bytes[i+2]=UInt8(((x/4+y/3)%2)*255)
    } }
    return CGImage(width:width,height:height,bitsPerComponent:8,bitsPerPixel:32,bytesPerRow:width*4,
        space:CGColorSpace(name:CGColorSpace.sRGB)!,bitmapInfo:CGBitmapInfo(rawValue:CGImageAlphaInfo.last.rawValue),
        provider:CGDataProvider(data:Data(bytes) as CFData)!,decode:nil,shouldInterpolate:false,intent:.defaultIntent)!
}

private actor RealESRGANPreviewGate {
    var pending: [RealESRGANVariant: CheckedContinuation<CGImage, Error>] = [:]
    func run(_ variant: RealESRGANVariant) async throws -> CGImage {
        try await withCheckedThrowingContinuation { pending[variant] = $0 }
    }
    func ready(_ variant: RealESRGANVariant) -> Bool { pending[variant] != nil }
    func finish(_ variant: RealESRGANVariant, image: CGImage) { pending.removeValue(forKey:variant)?.resume(returning:image) }
}

func registerRealESRGANTests() {
    let suite="RealESRGAN integration"
    Registry.shared.add(suite:suite,TestCase(name:"production compute policy excludes GPU for both legacy and current packages",knownBug:nil) {
        checkEqual(NativeSpatialUpscaler.configuration().computeUnits,.cpuAndNeuralEngine,"production always uses CPU and ANE")
        checkEqual(NativeSpatialUpscaler.configuration(cpuOnly:true).computeUnits,.cpuOnly,"diagnostics remain CPU only")
    })
    Registry.shared.add(suite:suite,TestCase(name:"x4 leading cores coalesce without losing context or coverage",knownBug:nil) {
        for modelSize in [384,512] {
            for count in [1,33,96,224,225,321,352,353,896,1134] {
                let spans=NativeSpatialUpscaler.x4Spans(count,modelSize:modelSize)
                checkEqual(spans.map(\.count).reduce(0,+),count,"every source pixel covered")
                var next=0
                for span in spans {
                    checkEqual(span.origin,next,"no gap or overlap")
                    let origin=max(0,span.origin-160)
                    check(span.origin+span.count <= origin+modelSize-160,"160 trailing context pixels retained")
                    check(span.origin == 0 || span.origin-origin >= 160,"leading context retained away from image boundary")
                    next += span.count
                }
            }
        }
    })
    Registry.shared.add(suite:suite,TestCase(name:"variants retain independent paths, identity and model registry entries",knownBug:nil) {
        checkEqual(RealESRGANVariant.x2plus.configurationName,"realesrgan-model.txt","legacy x2 path unchanged")
        check(RealESRGANVariant.x4plus.configurationName != RealESRGANVariant.x2plus.configurationName,"independent configuration")
        check(RealESRGANVariant.x4plus.accepts(width:4032,height:3024),"a standard 12MP photo fits the native x4 output guard")
        check(!RealESRGANVariant.x4plus.accepts(width:6000,height:4000),"excess output dimensions are bounded")
        checkEqual(ManagedModel.real4.algorithm,.realESRGANFour,"x4 owns its backend")
        checkEqual(UncoverAlgorithm.realESRGANFour.makeBackend().cacheIdentity,RealESRGANBackend(variant:.x4plus).cacheIdentity,"Uncover selects x4")
        _ = checkThrows("variant mismatches fail") { try RealESRGANImageRequest(.x2plus).validate(.x4plus) }
        let old = UncoverManifest(source:URL(fileURLWithPath:"/source"),sourceBytes:0,sourceModified:0,roi:.init(x:0,y:0,width:1,height:1),width:1,height:1,regionX:0,regionY:0,regionWidth:1,regionHeight:1,scale:.one,frames:[])
        checkNil(try JSONDecoder().decode(UncoverManifest.self,from:JSONEncoder().encode(old)).realESRGANImage,"legacy jobs decode without native-photo extension")
    })
    Registry.shared.add(suite:suite,TestCase(name:"late results cannot replace Off or newer model selection; navigation hides prior output",knownBug:nil) {
        try await Task { @MainActor in
            let gate=RealESRGANPreviewGate()
            let model=RealESRGANPreviewModel { _,variant in try await gate.run(variant) }
            let image=try realESRGANFixture(),input=PreviewEvidenceInput(photoURL:URL(fileURLWithPath:"/stale-\(UUID()).png"),image:image,isOriginal:true)
            model.select(.x2plus)
            let first=Task { await model.load(input:input) }
            while !(await gate.ready(.x2plus)) { await Task.yield() }
            model.select(.x4plus)
            let newer=Task { await model.load(input:input) }
            while !(await gate.ready(.x4plus)) { await Task.yield() }
            await gate.finish(.x4plus,image:image);await newer.value
            let selected=model.rendered
            await gate.finish(.x2plus,image:try realESRGANFixture(width:5,height:5));await first.value
            check(model.rendered === selected,"late x2 cannot replace completed x4")
            let next=PreviewEvidenceInput(photoURL:URL(fileURLWithPath:"/next.png"),image:image,isOriginal:true)
            checkNil(model.image(for:next),"old photo output never appears on next photo")
            model.select(.x2plus)
            let pending=Task { await model.load(input:input) }
            while !(await gate.ready(.x2plus)) { await Task.yield() }
            checkNil(model.image(for:input),"switching models shows original until the replacement completes")
            model.select(nil)
            checkNil(model.image(for:input),"Off immediately shows original")
            await gate.finish(.x2plus,image:image);await pending.value
            checkNil(model.image(for:input),"late completion cannot replace Off")
            model.select(.x4plus);await model.load(input:input)
            check(model.image(for:input) != nil,"completed result is cached")
        }.value
    })
    Registry.shared.add(suite:suite,TestCase(name:"actual native x2/x4 output, tiled x4 parity and cancellation",knownBug:nil) {
        let env=ProcessInfo.processInfo.environment
        guard let x2=env["WAIFU_TEST_REALESRGAN_X2"],let x4=env["WAIFU_TEST_REALESRGAN_X4"],let reference=env["WAIFU_TEST_REALESRGAN_REFERENCE"] else { throw SkipTest(reason:"set native x2/x4 model paths and offline reference directory") }
        let small=try realESRGANFixture()
        for (variant,path) in [(RealESRGANVariant.x2plus,x2),(.x4plus,x4)] {
            let model=try NativeSpatialUpscaler(modelURL:URL(fileURLWithPath:path),variant:variant)
            checkEqual(model.selectedComputeUnits,"ane","both actual model loads enforce ANE policy")
            let output=try model.upscale(small)
            checkEqual(output.width,small.width*variant.scale,"actual native width")
            checkEqual(output.height,small.height*variant.scale,"actual native height")
        }
        // Exercise the ordinary Uncover contract as well as native photo output.
        let input=PreviewEvidenceInput(photoURL:URL(fileURLWithPath:"/uncover-x4.png"),image:small,isOriginal:true)
        let photo=try await RealESRGANImageRuntime.shared.prepare(input,variant:.x4plus)
        defer { try? FileManager.default.removeItem(at:photo.directory) }
        var manifest=photo.manifest;manifest.realESRGANImage=nil
        let restored=try await RealESRGANBackend(variant:.x4plus,modelURL:URL(fileURLWithPath:x4)).reconstruct(
            UncoverPrepared(directory:photo.directory,manifest:manifest),allowUnsafeMemory:false,progress:{ _ in })
        let ordinary=try RegionImages.read(RegionImages.frameURL(restored.directory,"reconstructed",0))
        checkEqual(ordinary.width,small.width,"x4 Uncover restoration preserves ROI width")
        checkEqual(ordinary.height,small.height,"x4 Uncover restoration preserves ROI height")
        checkEqual(restored.model,RealESRGANVariant.x4plus.modelName,"Uncover uses the actual x4 network")
        let model=try NativeSpatialUpscaler(modelURL:URL(fileURLWithPath:x4),variant:.x4plus)
        let output=try model.upscale(realESRGANFixture(width:321,height:65))
        let actual=try NativeSpatialUpscaler.straightRGBPixels(output)
        let gold=try Data(contentsOf:URL(fileURLWithPath:reference).appendingPathComponent("reference.bin"))
        guard gold.count == actual.count * 4 else { throw ModelInstaller.fail("Reference byte count differs") }
        var maximum:Float=0,sum:Double=0
        gold.withUnsafeBytes { raw in
            let values=raw.bindMemory(to:Float.self),w=output.width,h=output.height
            for y in 0..<h { for x in 0..<w { for c in 0..<3 {
                let error=abs(Float(actual[(y*w+x)*3+c])/255-values[(c*h+y)*w+x]);maximum=max(maximum,error);sum+=Double(error)
            } } }
        }
        check(maximum <= 0.01 && sum/Double(actual.count) <= 0.002,"native tiled output matches rectangular original FP32 reference including byte quantization")
        let report:[String:Any]=["passed":maximum<=0.01 && sum/Double(actual.count)<=0.002,"width":output.width,"height":output.height,"maximumAbsoluteError":maximum,"meanAbsoluteError":sum/Double(actual.count),"kind":"correctness only; no profiling"]
        if let reportPath=env["WAIFU_TEST_REALESRGAN_REPORT"] { try JSONSerialization.data(withJSONObject:report,options:[.prettyPrinted,.sortedKeys]).write(to:URL(fileURLWithPath:reportPath)) }
        let cancelled=Task.detached {
            try Task.checkCancellation()
            let model=try NativeSpatialUpscaler(modelURL:URL(fileURLWithPath:x4),variant:.x4plus)
            return try model.upscale(small)
        }
        cancelled.cancel()
        _ = await checkThrowsAsync("native request honors cancellation") { _ = try await cancelled.value }
    })
    Registry.shared.add(suite:suite,TestCase(name:"published x2/x4 install independently and preserve x2 activation",knownBug:nil) {
        guard ProcessInfo.processInfo.environment["WAIFU_TEST_REALESRGAN_INSTALL"] == "1" else { throw SkipTest(reason:"opt in to real published-model installer check") }
        let root=try Temporary.directory("realesrgan-install"),config=root.appendingPathComponent("config")
        let installer=ModelInstaller(configurationRoot:config)
        let x2=try await installer.install(.real,root:root,progress:{ _,_ in })
        let original=try Data(contentsOf:config.appendingPathComponent("realesrgan-model.txt"))
        let x4=try await installer.install(.real4,root:root,progress:{ _,_ in })
        check(x2 != x4,"distinct installed packages")
        checkEqual(try Data(contentsOf:config.appendingPathComponent("realesrgan-model.txt")),original,"x4 preserves x2 activation")
        try await installer.validate(.real,directory:x2,progress:{ _,_ in })
        try await installer.validate(.real4,directory:x4,progress:{ _,_ in })
        for (variant,directory) in [(RealESRGANVariant.x2plus,x2),(.x4plus,x4)] {
            let model=try NativeSpatialUpscaler(modelURL:directory.appendingPathComponent("model.mlmodelc"),variant:variant)
            let image=try model.upscale(realESRGANFixture())
            checkEqual(image.width,33*variant.scale,"freshly downloaded installed model executes")
            checkEqual(image.height,19*variant.scale,"installed model keeps native geometry")
        }
        checkEqual(String(decoding:try Data(contentsOf:config.appendingPathComponent("realesrgan-x4plus-model.txt")),as:UTF8.self),x4.appendingPathComponent("model.mlmodelc").path,"independent x4 activation")
    })
}
