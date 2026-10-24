import Foundation
import CoreGraphics
import CoreImage
import ImageIO

private func lineagePhoto(_ id:String,revision:String="1") -> LineagePhoto {
    .init(id:id,contentIdentity:LineageIdentity.hash(id+revision),name:id+".png",width:320,height:240,format:"PNG",url:URL(string:"waifu-photos://fixture/"+id)!,offset:Double(id.utf8.first ?? 0))
}
private func lineageRelation(_ a:String,_ b:String) -> LineageRelation {
    .init(a:a,b:b,correlation:0.999,meanAbsoluteDifference:0.002,overlap:1,registration:.init())
}
private func lineageMask(_ values:[Float],width:Int=4,height:Int=3,photo:LineagePhoto=lineagePhoto("a")) -> SafireSpatialResult {
    .init(assetID:photo.id,contentIdentity:photo.contentIdentity,modelFingerprint:String(repeating:"a",count:64),
          runtimeRevision:SafireSpatialResult.runtimeRevision,preprocessingVersion:SafireSpatialResult.preprocessingVersion,
          width:width,height:height,probabilities:values,partitions:nil)
}
func registerLineageTests() {
    let suite="Image Lineage"
    Registry.shared.add(suite:suite,TestCase(name:"mask components preserve top-left pixel edges and separate regions",knownBug:nil) {
        let map=lineageMask([1,1,0,0, 1,0,0,1, 0,0,0,1])
        let regions=try map.regions(minimumPixels:1)
        checkEqual(regions.count,2,"two disconnected components")
        checkEqual(regions[0].bounds,CGRect(x:0,y:0,width:2,height:2),"exact first extent")
        checkEqual(regions[1].bounds,CGRect(x:3,y:1,width:1,height:2),"exact second extent")
        checkEqual(regions[0].pixelCount,3,"mask area is exact")
        checkEqual(regions[0].contour.count,8,"L shape has eight exposed edges")
        checkEqual(map.nativeBounds(regions[1],size:CGSize(width:400,height:300)),CGRect(x:300,y:100,width:100,height:200),"non-square native registration")
    })
    Registry.shared.add(suite:suite,TestCase(name:"mask holes keep their contours and noise is filtered",knownBug:nil) {
        let regions=try lineageMask([1,1,1,1,0,1,1,1,1],width:3,height:3).regions()
        checkEqual(regions.count,1,"ring is connected")
        checkEqual(regions[0].contour.count,16,"outer perimeter and hole retained")
        let filtered=try lineageMask([0,0,0,1],width:2,height:2).regions()
        check(filtered.isEmpty,"single pixel excluded at default minimum")
    })
    Registry.shared.add(suite:suite,TestCase(name:"invalid remote spatial maps fail closed",knownBug:nil) {
        for map in [lineageMask([.nan],width:1,height:1),lineageMask([1.01],width:1,height:1),lineageMask([0],width:2,height:1)] {
            do { try map.validate();check(false,"invalid map accepted") } catch { }
        }
    })
    Registry.shared.add(suite:suite,TestCase(name:"cache identity includes model runtime preprocessing and content",knownBug:nil) {
        let a=lineageMask([0],width:1,height:1)
        let b=lineageMask([0],width:1,height:1,photo:lineagePhoto("a",revision:"2"))
        check(a.cacheKey != b.cacheKey,"content edit changes cache identity")
        checkEqual(a.cacheKey,SafireEvidenceCache.key(photo:lineagePhoto("a"),fingerprint:a.modelFingerprint),"disk cache contract identical")
    })
    Registry.shared.add(suite:suite,TestCase(name:"unresolved pair never asserts chronology or larger original",knownBug:nil) {
        var value=LineageInvestigation()
        value.merge(photos:[lineagePhoto("a"),lineagePhoto("b")],relations:[lineageRelation("a","b")])
        checkEqual(value.families.count,1,"verified correspondence creates family")
        check(value.families[0].sourceCandidate==nil,"no invented source")
        check(value.families[0].relations[0].confirmedParent==nil,"no invented direction")
    })
    Registry.shared.add(suite:suite,TestCase(name:"DAG permits multiple parents but rejects a cycle",knownBug:nil) {
        var f=LineageFamily(id:"f",photoIDs:["a","b","c"],relations:[lineageRelation("a","c"),lineageRelation("b","c"),lineageRelation("a","b")])
        check(f.confirm(f.relations[0].id,parent:"a"),"first parent")
        check(f.confirm(f.relations[1].id,parent:"b"),"second parent")
        checkEqual(f.reachable(from:"c",descendants:false),Set(["a","b","c"]),"all ancestors traced")
        check(!f.confirm(f.relations[0].id,parent:"c") || !f.confirm(f.relations[2].id,parent:"a"),"directed cycle refused")
    })
    Registry.shared.add(suite:suite,TestCase(name:"successful deletion prunes edges evidence keepers and selection atomically",knownBug:nil) {
        var value=LineageInvestigation()
        value.merge(photos:[lineagePhoto("a"),lineagePhoto("b"),lineagePhoto("c")],relations:[lineageRelation("a","b"),lineageRelation("b","c")])
        value.keepers=["b"];value.selectedPhotoID="b"
        try value.accept(lineageMask([1],width:1,height:1,photo:lineagePhoto("b")))
        value.remove(["b"])
        checkEqual(Set(value.photos.map(\.id)),Set(["a","c"]),"deleted photo removed")
        check(value.families.flatMap(\.relations).isEmpty,"dangling relations removed")
        checkEqual(value.families.count,2,"deleting a bridge splits unsupported family membership")
        check(value.evidence.isEmpty && value.keepers.isEmpty,"deleted state removed")
        check(value.selectedPhotoID != "b","selection moved")
        do { try value.accept(lineageMask([1],width:1,height:1,photo:lineagePhoto("b")));check(false,"late result resurrected deleted asset") } catch { }
        value.remove(["a","c"]);check(value.families.isEmpty && value.selectedPhotoID==nil,"empty family supported")
    })
    Registry.shared.add(suite:suite,TestCase(name:"surviving selection and stable family order survive progressive results",knownBug:nil) {
        var value=LineageInvestigation()
        let photos=[lineagePhoto("a"),lineagePhoto("b"),lineagePhoto("c")]
        value.merge(photos:photos,relations:[lineageRelation("a","b")])
        let first=value.families[0].id;value.selectedPhotoID="a"
        value.merge(photos:photos,relations:[lineageRelation("a","b"),lineageRelation("b","c")])
        checkEqual(value.families[0].id,first,"family identity retained")
        checkEqual(value.families[0].photoIDs,["a","b","c"],"stable layout order")
        value.remove(["b"]);checkEqual(value.selectedPhotoID,"a","surviving selection retained")
    })
    Registry.shared.add(suite:suite,TestCase(name:"an edited asset rejects old evidence and clears confirmed history",knownBug:nil) {
        var value=LineageInvestigation()
        let a=lineagePhoto("a"),b=lineagePhoto("b")
        value.merge(photos:[a,b],relations:[lineageRelation("a","b")])
        _=value.families[0].confirm(value.families[0].relations[0].id,parent:"a")
        try value.accept(lineageMask([1],width:1,height:1))
        value.merge(photos:[lineagePhoto("a",revision:"2"),b],relations:[lineageRelation("a","b")])
        check(value.evidence.isEmpty,"old evidence invalidated")
        check(value.families[0].relations[0].confirmedParent==nil,"edited content invalidates confirmations")
        do { try value.accept(lineageMask([1],width:1,height:1));check(false,"stale edit accepted") } catch { }
    })
    Registry.shared.add(suite:suite,TestCase(name:"Photos-only routing rejects files and video URLs",knownBug:nil) {
        await MainActor.run {
            check(!ImageLineageModel.allowed(source:nil),"no stale source")
            check(!ImageLineageModel.allowed(source:URL(fileURLWithPath:"/tmp/video.mov")),"no video entry")
            check(!ImageLineageModel.allowed(source:URL(string:"waifu-photos://missing")),"stale Photos handle rejected")
        }
    })
    Registry.shared.add(suite:suite,TestCase(name:"investigation persistence retains review without embedding mask rasters",knownBug:nil) {
        let root=try Temporary.directory("lineage")
        let store=LineageStore(root:root);var value=LineageInvestigation()
        value.merge(photos:[lineagePhoto("a")],relations:[]);value.keepers=["a"]
        try value.accept(lineageMask([1],width:1,height:1))
        try await store.save(value,album:"album")
        let restored=await store.read(album:"album")
        checkEqual(restored?.keepers,Set(["a"]),"keepers persisted")
        check(restored?.evidence.isEmpty==true,"spatial cache is separate")
    })
    Registry.shared.add(suite:suite,TestCase(name:"remote forensic input requires lossless validated image data",knownBug:nil) {
        let request=RemoteSafireRequest(photo:lineagePhoto("a"),png:Data([0,1,2]),modelFingerprint:String(repeating:"a",count:64))
        do { try request.validate();check(false,"invalid transport accepted") } catch { }
    })
}

func registerLineageCoordinateTests() {
    Registry.shared.add(suite:"Image Lineage",TestCase(name:"mask coordinates remain registered through aspect scale rotation and viewport zoom",knownBug:nil) {
        let alignment=CrystalOrbAlignment(target:CGSize(width:800,height:600),candidate:CGSize(width:320,height:240),registration:.init(x:23,y:-17,scale:1.2,degrees:11))
        let viewport=CrystalOrbViewport(size:CGSize(width:900,height:700),center:CGPoint(x:310,y:290),scale:2.3)
        let mapping=LineagePixelMapping.transform(width:1024,height:1024,alignment:alignment,viewport:viewport)
        for pixel in [CGPoint.zero,CGPoint(x:1024,y:1024),CGPoint(x:512,y:256)] {
            let native=CGPoint(x:pixel.x/1024*320,y:240-pixel.y/1024*240)
            let expected=viewport.screen(alignment.comparison(native,candidate:true))
            let actual=pixel.applying(mapping)
            checkClose(Double(actual.x),Double(expected.x),tolerance:1e-7,"registered x")
            checkClose(Double(actual.y),Double(expected.y),tolerance:1e-7,"registered y")
            let recovered=actual.applying(mapping.inverted())
            checkClose(Double(recovered.x),Double(pixel.x),tolerance:1e-7,"inverse x")
            checkClose(Double(recovered.y),Double(pixel.y),tolerance:1e-7,"inverse y")
        }
    })
}

func registerLineageInspectionTests() {
    Registry.shared.add(suite:"Image Lineage",TestCase(name:"preview upgrades preserve native coordinates navigation and another workspace's ownership",knownBug:nil) {
        try await Task { @MainActor in
            let session=WaifuSession(settingsURL:try Temporary.directory("lineage-inspection").appendingPathComponent("settings.json"))
            let viewer=WaifuCrystalOrbModel()
            let preview=try ImageFixture.cgImage(width:104,height:78)
            let original=try ImageFixture.cgImage(width:416,height:312)
            let metadata=PhotoMetadataParser.parse(properties:[:],xmp:nil,filename:"fixture.png",fallbackWidth:416,fallbackHeight:312,fileSizeBytes:nil,photosCreationDate:nil,photosModificationDate:nil,sourceType:nil)
            let frame=LineageDiscovery.indexFrame(lineagePhoto("a"),existing:[])
            viewer.inspectLineage(target:preview,candidate:preview,metadata:metadata,frame:frame,session:session,original:false,targetNativeSize:CGSize(width:416,height:312),candidateNativeSize:CGSize(width:416,height:312),pairIdentity:"a|b")
            checkEqual(viewer.pair?.alignment.size,CGSize(width:416,height:312),"preview declares native geometry")
            viewer.zoom=3;viewer.navigationCenter=CGPoint(x:190,y:120)
            viewer.registration = .init(x:7,y:-3,scale:0.98,degrees:2)
            viewer.inspectLineage(target:original,candidate:original,metadata:metadata,frame:frame,session:session,original:true,targetNativeSize:CGSize(width:416,height:312),candidateNativeSize:CGSize(width:416,height:312),pairIdentity:"a|b")
            checkEqual(viewer.zoom,3,"native upgrade retains zoom")
            checkEqual(viewer.navigationCenter,CGPoint(x:190,y:120),"native upgrade retains focal point")
            checkEqual(viewer.registration.x,7,"native upgrade retains registration")
            session.crystalOrbIsOpen=true
            viewer.close()
            check(session.crystalOrbIsOpen,"embedded viewer cannot close another CrystalOrb workspace")
        }.value
    })
    Registry.shared.add(suite:"Image Lineage",TestCase(name:"a family merge follows the surviving selected photo rather than an unrelated first family",knownBug:nil) {
        var value=LineageInvestigation()
        let photos=[lineagePhoto("x"),lineagePhoto("a"),lineagePhoto("b")]
        value.merge(photos:photos,relations:[])
        value.selectedPhotoID="b";value.selectedFamilyID=value.families.first{$0.photoIDs.contains("b")}?.id
        value.merge(photos:photos,relations:[lineageRelation("a","b")])
        let selected=value.families.first{$0.id==value.selectedFamilyID}
        checkEqual(Set(selected?.photoIDs ?? []),Set(["a","b"]),"selection tracks merged family")
    })
}

private func lineageTexture(_ variant:Int=0,width:Int=256,height:Int=256) throws -> CGImage {
    var bytes=[UInt8](repeating:255,count:width*height*4)
    for y in 0..<height { for x in 0..<width {
        let u=Double(x),v=Double(y)
        let value=variant==0 ? 125+42*sin(u*0.071+v*0.027)+35*cos(v*0.091-u*0.013)+22*sin((u+v)*0.193) : 125+50*cos(u*0.119-v*0.053)+33*sin(v*0.163)
        for c in 0..<3 { bytes[(y*width+x)*4+c]=UInt8(clamping:Int(value)) }
    } }
    guard let provider=CGDataProvider(data:Data(bytes) as CFData),let result=CGImage(width:width,height:height,bitsPerComponent:8,bitsPerPixel:32,bytesPerRow:width*4,space:CGColorSpace(name:CGColorSpace.sRGB)!,bitmapInfo:CGBitmapInfo(rawValue:CGImageAlphaInfo.premultipliedLast.rawValue),provider:provider,decode:nil,shouldInterpolate:true,intent:.defaultIntent) else { throw PhotosMedia.failure("Unable to generate texture") }
    return result
}
func registerLineageSourceTests() {
    Registry.shared.add(suite:"Image Lineage",TestCase(name:"native resampling supports a smaller parent without treating dates alone as proof",knownBug:nil) {
        let source=try lineageTexture(),unrelated=try lineageTexture(1)
        let ci=CIImage(cgImage:source).transformed(by:CGAffineTransform(scaleX:2,y:2))
        let child=CrystalOrbDifferenceRenderer.context.createCGImage(ci,from:ci.extent)!
        let assessor=LineageSourceAssessor(),early=Date(timeIntervalSince1970:100),late=Date(timeIntervalSince1970:200)
        let proof=try await assessor.assessPixels(parent:source,child:child,parentID:"small",childID:"large",parentDate:early,childDate:late)
        checkEqual(proof?.parentID,"small","lower resolution parent permitted")
        check(proof?.explanation.contains("not proof") == true,"tentative chronology made explicit")
        let different=try await assessor.assessPixels(parent:source,child:unrelated,parentID:"a",childID:"b",parentDate:early,childDate:late)
        check(different==nil,"dates and similar texture cannot establish direction")
        let tied=try await assessor.assessPixels(parent:source,child:source,parentID:"a",childID:"b",parentDate:early,childDate:early)
        check(tied==nil,"identical dates leave order unresolved")
    })
    Registry.shared.add(suite:"Image Lineage",TestCase(name:"crop correspondence registers contained pixels and supports tentative direction",knownBug:nil) {
        let source=try lineageTexture(),child=source.cropping(to:CGRect(x:32,y:32,width:192,height:192))!
        let geometry=CrystalOrbAlignment(target:CGSize(width:256,height:256),candidate:CGSize(width:192,height:192))
        let registration=try await LineageCropRegistrar.shared.align(target:source,candidate:child,geometry:geometry)
        check(registration != nil,"shared crop detail is registered")
        checkClose(Double(registration?.registration.scale ?? 0),0.75,tolerance:0.005,"correct relative field of view")
        checkClose(Double(registration?.registration.x ?? 99),0,tolerance:1,"centered crop has no translation")
        let evidence=try await LineageSourceAssessor().assessPixels(parent:source,child:child,parentID:"full",childID:"crop",parentDate:Date(timeIntervalSince1970:100),childDate:Date(timeIntervalSince1970:200))
        checkEqual(evidence?.method,"contained crop correspondence","native crop proof retained")
        check(evidence?.explanation.contains("contained crop") == true,"explanation identifies deterministic crop evidence")
        let relation=try await LineageCorrespondence.verify(source,child,photo:lineagePhoto("full"),other:lineagePhoto("crop"),generation:UUID())
        check(relation != nil,"verified crop is admitted into a family")
    })
    Registry.shared.add(suite:"Image Lineage",TestCase(name:"inferred ancestry participates in cycle protection and source ranking",knownBug:nil) {
        var f=LineageFamily(id:"f",photoIDs:["a","b","c"],relations:[lineageRelation("a","b"),lineageRelation("b","c"),lineageRelation("a","c")])
        f.relations[0].inferredParent="a";f.relations[1].inferredParent="b"
        checkEqual(f.sourceCandidate,"a","unique supported root")
        check(!f.confirm(f.relations[2].id,parent:"c"),"manual confirmation cannot create cycle through hypotheses")
        checkEqual(f.reachable(from:"a",descendants:true),Set(["a","b","c"]),"tracing follows supported hypotheses")
    })
}

func registerLineagePhotographicFixtureTests() {
    Registry.shared.add(suite:"Image Lineage photographic fixtures",TestCase(name:"actual known-history photographs expose measured correspondence without forced chronology",knownBug:nil) {
        guard let path=ProcessInfo.processInfo.environment["WAIFU_TEST_LINEAGE_FIXTURES"] else { throw SkipTest(reason: "set WAIFU_TEST_LINEAGE_FIXTURES to Apple-generated history fixtures") }
        let root=URL(fileURLWithPath:path,isDirectory:true)
        func read(_ name:String) throws -> CGImage {
            guard let source=CGImageSourceCreateWithURL(root.appendingPathComponent(name) as CFURL,nil),let image=CGImageSourceCreateImageAtIndex(source,0,nil) else { throw PhotosMedia.failure("Missing photographic fixture " + name) }
            return image
        }
        let baseline=try read("baseline.png")
        let names=["recompressed.jpg","downsampled.png","upscaled.png","watermarked.png","watermark-removed.png","locally-inpainted.png","generation-3.jpg","sibling-tone.png","sibling-crop.png","ai-restored.png","distinct-capture.jpg"]
        var results:[[String:Any]]=[]
        for name in names {
            let image=try read(name)
            let value=try await LineageCorrespondence.verify(baseline,image,photo:lineagePhoto("baseline"),other:lineagePhoto(name),generation:UUID())
            results.append(["name":name,"verifiedRelated":value != nil,"correlation":value.map{$0.correlation as Any} ?? NSNull(),"meanDifference":value.map{$0.meanAbsoluteDifference as Any} ?? NSNull()])
            if name=="watermark-removed.png" { check(value != nil,"exact recovered baseline is related") }
            if name=="sibling-crop.png" { check(value != nil,"real photographic crop is verified") }
            if name=="distinct-capture.jpg" { check(value==nil,"adjacent distinct Apollo photograph is not grouped by subject alone") }
        }
        try JSONSerialization.data(withJSONObject:["scope":"Deterministic pair verification of known-history fixture against baseline; no claim of full lineage recovery", "results":results],options:[.prettyPrinted,.sortedKeys]).write(to:root.appendingPathComponent("correspondence.json"),options:.atomic)
    })
}

func registerLineageScopeAndBinaryTests() {
    Registry.shared.add(suite:"Image Lineage",TestCase(name:"context action targets a complete Photos group in rows or only the clicked photo otherwise",knownBug:nil) {
        let first=FrameGroupID(raw:"g1"),other=FrameGroupID(raw:"g2")
        var frames=(0..<5).map { i in AnalyzedFrame(sampleIndex:i,timeSeconds:Double(i),artScore:0,faceScore:nil,faceCount:0,faceAreaFraction:nil,featurePrint:nil) }
        for i in 0..<3 { frames[i].groupID=first };frames[3].groupID=other
        let groups=[FrameGroup(id:first,memberIDs:Array(frames.prefix(3)).map{$0.id.raw}),FrameGroup(id:other,memberIDs:[frames[3].id.raw])]
        let selected=LineageGroupTarget.frames(anchor:frames[1],frames:frames,groups:groups,photosSource:true,usesGroupRows:true)
        checkEqual(selected?.map(\.id),Array(frames.prefix(3)).map(\.id),"complete group, including capped or filtered members; excludes unrelated selection")
        check(LineageGroupTarget.frames(anchor:frames[1],frames:frames,groups:groups,photosSource:false,usesGroupRows:true)==nil,"video source denied")
        checkEqual(LineageGroupTarget.frames(anchor:frames[1],frames:frames,groups:groups,photosSource:true,usesGroupRows:false)?.map(\.id),[frames[1].id],"flow or grouping off targets only the clicked photo")
        checkEqual(LineageGroupTarget.frames(anchor:frames[4],frames:frames,groups:groups,photosSource:true,usesGroupRows:true)?.map(\.id),[frames[4].id],"ungrouped thumbnail targets one photo")
        checkEqual(LineageGroupTarget.frames(anchor:frames[1],frames:frames,groups:[],photosSource:true,usesGroupRows:true)?.map(\.id),[frames[1].id],"stale group cannot expand scope")
    })
    Registry.shared.add(suite:"Image Lineage",TestCase(name:"binary masks preserve exact floats partitions identity and reject truncated data",knownBug:nil) {
        let photo=lineagePhoto("a")
        let result=SafireSpatialResult(assetID:photo.id,contentIdentity:photo.contentIdentity,modelFingerprint:String(repeating:"a",count:64),runtimeRevision:SafireSpatialResult.runtimeRevision,preprocessingVersion:SafireSpatialResult.preprocessingVersion,width:3,height:2,probabilities:[0,0.125,0.49999997,0.5,0.75,1],partitions:[0,1,2,2,1,0])
        let binary=try result.encodedSpatialData(),decoded=try SafireSpatialResult.decodeSpatialData(binary)
        checkEqual(decoded.probabilities.map(\.bitPattern),result.probabilities.map(\.bitPattern),"lossless Float32 transport")
        checkEqual(decoded.partitions,result.partitions,"lossless labels")
        checkEqual(decoded.assetID,result.assetID,"asset identity preserved")
        do { _=try SafireSpatialResult.decodeSpatialData(binary.dropLast());check(false,"truncated mask accepted") } catch { }
        let legacy=try SafireSpatialResult.decodeSpatialData(JSONEncoder().encode(result))
        checkEqual(legacy.probabilities,result.probabilities,"old JSON cache remains readable")
        let summary=LineageSafireSummary(result)
        checkEqual(summary.flaggedPixels,3,"story facts derived from real thresholded mask")
        checkEqual(summary.maskPixels,6,"summary describes exact spatial coverage without authenticity claim")
    })
}

func registerSafireRankingTests() {
    Registry.shared.add(suite:"Image Lineage",TestCase(name:"SAFIRE orders real mask coverage ascending with stable ties and pending photos last",knownBug:nil) {
        let photos=[lineagePhoto("a"),lineagePhoto("b"),lineagePhoto("c"),lineagePhoto("d")]
        let findings=Dictionary(uniqueKeysWithValues:photos.prefix(3).enumerated().map { index,photo in
            (photo.id,LineageSafireSummary(lineageMask(index==0 ? [1,1,1,0] : [0,0,0,0],width:2,height:2,photo:photo)))
        })
        checkEqual(LineageMaskRanking.ordered(photos,findings:findings).map(\.id),["b","c","a","d"],"ascending ratio, stable zero-area ties, pending last")
        checkEqual(findings["a"]?.ratio,0.75,"ratio is actual thresholded area divided by full map area")
        let changed=[lineagePhoto("b",revision:"changed"),photos[0]]
        checkEqual(LineageMaskRanking.ordered(changed,findings:findings).map(\.id),["a","b"],"old content cannot win source-candidate ranking")
    })
    Registry.shared.add(suite:"Image Lineage",TestCase(name:"visible overlay and separate binary mask preserve oriented corners",knownBug:nil) {
        let image=try lineageTexture(width:8,height:8)
        var values=[Float](repeating:0,count:64);values[0]=1
        let result=lineageMask(values,width:8,height:8)
        let material=try await SafireCardRenderer.shared.make(image:image,result:result)
        let data=material.mask.dataProvider!.data! as Data
        check(data[0]>250,"top-left flagged corner remains white")
        check(data[7*material.mask.bytesPerRow]<5,"bottom-left unflagged corner remains black")
        checkEqual(material.highlighted.width,image.width,"single-photo overlay geometry preserved")
    })
}

func registerSafireJobWaitTests() {
    Registry.shared.add(suite:"Image Lineage",TestCase(name:"busy SAFIRE requests resume automatically without repeated waiting updates",knownBug:nil) {
        try await Task { @MainActor in
            var attempts=0,waits=0,announcements=0
            let result:Int=try await SafireJobWait.run(onWaiting:{announcements+=1},wait:{waits+=1}) {
                attempts+=1
                if attempts<3 { throw RemoteError.busy }
                return 42
            }
            checkEqual(result,42,"busy jobs eventually return their actual result")
            checkEqual(attempts,3,"only temporary busy failures retried")
            checkEqual(waits,2,"retries yield rather than spin")
            checkEqual(announcements,1,"waiting does not repeatedly redraw the interface")
        }.value
    })
    Registry.shared.add(suite:"Image Lineage",TestCase(name:"SAFIRE waiting respects cancellation and never retries a permanent failure",knownBug:nil) {
        try await Task { @MainActor in
            var attempts=0
            do {
                let _:Int=try await SafireJobWait.run(onWaiting:{},wait:{throw CancellationError()}) {
                    attempts+=1;throw RemoteError.busy
                }
                check(false,"cancelled waiting returned a result")
            } catch is CancellationError { }
            checkEqual(attempts,1,"cancelled preview or closed workspace cannot launch another attempt")
            attempts=0
            do {
                let _:Int=try await SafireJobWait.run(onWaiting:{check(false,"permanent failure announced as waiting")},wait:{check(false,"permanent failure retried")}) {
                    attempts+=1;throw RemoteError.modelUnavailable("Model missing")
                }
                check(false,"missing model returned a result")
            } catch RemoteError.modelUnavailable { }
            checkEqual(attempts,1,"installation and model errors surface immediately")
        }.value
    })
}
