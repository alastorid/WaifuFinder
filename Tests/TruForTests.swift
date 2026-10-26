import AppKit
import Metal

func registerTruForTests() {
    let suite="TruFor regression"
    Registry.shared.add(suite:suite,TestCase(name:"production pins original FP32 and removes Core ML companion overrides",knownBug:nil) {
        let environment=TruForConfiguration.workerEnvironment([
            "WAIFU_TRUFOR_PRECISION":"fp16", "WAIFU_TRUFOR_COREML_FP16_MODEL":"/companion.mlmodelc", "PATH":"preserved"])
        checkEqual(environment["WAIFU_TRUFOR_PRECISION"],"fp32","production never inherits FP16")
        checkNil(environment["WAIFU_TRUFOR_COREML_FP16_MODEL"],"companion cannot change inference")
        checkEqual(environment["PATH"],"preserved","unrelated worker environment remains intact")
        checkEqual(TruForConfiguration.execution,"trufor-fp32-v2","FP32 capability and score cache revision are fixed")
        checkEqual(TruForConfiguration(directory:URL(fileURLWithPath:"/unused")).fingerprint,TruForConfiguration.weightsSHA,"wire fingerprint names the weights, not the execution namespace")
    })
    Registry.shared.add(suite:suite,TestCase(name:"outline pixels match clamped morphology for portrait and landscape maps",knownBug:nil) {
        guard MTLCreateSystemDefaultDevice() != nil else { throw SkipTest(reason:"Metal unavailable") }
        for (width,height) in [(65,32),(32,65)] {
            let original=try ImageFixture.cgImage(width:width,height:height)
            // Include changes below y=width and on every image edge. A width-only
            // clamp either reads past the last landscape row or loses portrait rows.
            let anomaly=(0..<(width*height)).map { i in Float((i/width)%7<3 ? 1:0) }
            let map=TruForSpatial(width:width,height:height,score:0.3,anomaly:anomaly,reliability:Array(repeating:1,count:width*height),noiseprint:Array(repeating:0,count:width*height))
            let output=try await TruForEvidenceRenderer.shared.render(original:original,evidence:map,mode:.outlines)
            func pixels(_ image:CGImage)->[UInt8] {
                var bytes=[UInt8](repeating:0,count:width*height*4)
                bytes.withUnsafeMutableBytes { data in
                    let context=CGContext(data:data.baseAddress,width:width,height:height,bitsPerComponent:8,bytesPerRow:width*4,space:CGColorSpace(name:CGColorSpace.sRGB)!,bitmapInfo:CGImageAlphaInfo.premultipliedLast.rawValue)!
                    context.draw(image,in:CGRect(x:0,y:0,width:width,height:height))
                }
                return bytes
            }
            let actual=pixels(output),source=pixels(original)
            var mismatches=0
            for y in 0..<height { for x in 0..<width {
                var low:Float=1,high:Float=0
                for dy in -1...1 { for dx in -1...1 {
                    let value=anomaly[min(height-1,max(0,y+dy))*width+min(width-1,max(0,x+dx))]
                    low=min(low,value);high=max(high,value)
                } }
                let index=(y*width+x)*4
                let expected: [Int]=high>low ? [255,38,12]:[Int(source[index]),Int(source[index+1]),Int(source[index+2])]
                if (0..<3).contains(where:{abs(Int(actual[index+$0])-expected[$0])>1}) { mismatches+=1 }
            } }
            checkEqual(mismatches,0,"\(width)x\(height) outline geometry, border handling and photo alignment")
        }
    })
    Registry.shared.add(suite:suite,TestCase(name:"unscored photos remain visible and sort last in either direction",knownBug:nil) {
        var low=FrameFixture.make(id:"f000001",time:0),high=FrameFixture.make(id:"f000002",time:1)
        low.truforScore=0;high.truforScore=0.9
        let missing=FrameFixture.make(id:"f000003",time:2)
        var invalid=FrameFixture.make(id:"f000004",time:3);invalid.truforScore = .nan
        let frames=[missing,high,low,invalid]
        var order=FrameOrder.only([.trufor])
        let grouping=GroupingSettings(similarityThreshold:0.35,maximumGroupSize:60,faceWeight:0.5,minimumFaceAreaFraction:0.01)
        let ascending=FrameArrangement.arrange(frames,order:order,perGroupTopCount:99,grouping:grouping).visible
        checkEqual(Array(ascending.prefix(2)).map(\.id),[low.id,high.id],"zero is scored, missing and invalid are last")
        order.setAscending(false,for:.trufor)
        let descending=FrameArrangement.arrange(frames,order:order,perGroupTopCount:99,grouping:grouping).visible
        checkEqual(Array(descending.prefix(2)).map(\.id),[high.id,low.id],"missing stays last when direction reverses")
        var limits=FrameLimits.none;limits.setLimit(.init(low:0.2,high:0.7),for:.trufor)
        checkEqual(Set(FrameArrangement.arrange(frames,order:order,limits:limits,perGroupTopCount:99,grouping:grouping).visible.map(\.id)),Set([missing.id,invalid.id]),"incomplete coverage does not silently disappear")
    })
    Registry.shared.add(suite:suite,TestCase(name:"legacy preset decoding defaults TruFor off and video strips its filter",knownBug:nil) {
        let old=Data("{}".utf8)
        checkEqual(try JSONDecoder().decode(FrameOrder.self,from:old).trufor.enabled,false,"old order has no forensic term")
        checkEqual(try JSONDecoder().decode(FrameLimits.self,from:old).trufor,.unlimited,"old ranges have no forensic limit")
        var order=FrameOrder.only([.trufor]);order.setAscending(false,for:.trufor)
        var limits=FrameLimits.none;limits.trufor = .init(low:0.1,high:0.5)
        checkEqual(try JSONDecoder().decode(FrameOrder.self,from:JSONEncoder().encode(order)),order,"TruFor direction and enablement round trip")
        checkEqual(try JSONDecoder().decode(FrameLimits.self,from:JSONEncoder().encode(limits)),limits,"TruFor range round trips")
        check(!order.withoutForensic.trufor.enabled && order.withoutForensic.enabledCount>0,"video keeps a valid nonforensic order")
        check(limits.withoutForensic.isEmpty,"video has no hidden forensic filter")
    })
    Registry.shared.add(suite:suite,TestCase(name:"scalar cache preserves zero and rejects edited assets, other models and old precision",knownBug:nil) {
        let root=try Temporary.directory("trufor-cache-review"),file=root.appendingPathComponent("scores.bin")
        let photo=TruForPhoto(url:URL(string:"waifu-photos://cache-test")!,assetKey:.init(identifier:"A",modifiedMilliseconds:1,pixelWidth:65,pixelHeight:32))
        let cache=TruForScoreMemory(fileURL:file)
        try await cache.remember(photo,score:0,fingerprint:TruForConfiguration.scoreFingerprint)
        let reopened=TruForScoreMemory(fileURL:file)
        checkEqual(try await reopened.read([photo],fingerprint:TruForConfiguration.scoreFingerprint)[photo.contentIdentity],0,"zero survives an independent cache reader")
        let edited=TruForPhoto(url:photo.url,assetKey:.init(identifier:"A",modifiedMilliseconds:2,pixelWidth:65,pixelHeight:32))
        check(try await reopened.read([edited],fingerprint:TruForConfiguration.scoreFingerprint).isEmpty,"edits do not reuse stale scores")
        check(try await reopened.read([photo],fingerprint:"different model").isEmpty,"weights change invalidates score")
        let oldIdentity=TagResultCacheIdentity(sourcePath:"trufor-score-v1/\(TruForConfiguration.weightsSHA)/\(TruForConfiguration.preprocessing)/A",sourceSize:(UInt64(65)<<32)|32,sourceModifiedMilliseconds:1,analysisWidth:0,vocabularyFingerprint:0)
        check(TruForScoreMemory.identity(photo,fingerprint:TruForConfiguration.scoreFingerprint) != oldIdentity,"old FP16 namespace cannot satisfy FP32 work")
        let raw=try Data(contentsOf:file)
        check(raw.count<8192,"only scalar records are persisted")
        _=await checkThrowsAsync("nonfinite scores are refused") { try await cache.remember(photo,score:.infinity,fingerprint:"bad") }
    })
    Registry.shared.add(suite:suite,TestCase(name:"invalid evidence geometry is rejected before size arithmetic",knownBug:nil) {
        let map=TruForSpatial(width:Int.max,height:32,score:0,anomaly:[],reliability:[],noiseprint:[])
        _=checkThrows("oversized evidence must not overflow") { try map.validate() }
    })
}
