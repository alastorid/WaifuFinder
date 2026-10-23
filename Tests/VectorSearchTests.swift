import Foundation
import Metal
import Vision

private func vectorFixture(_ count: Int, dimensions: Int, metric: WaifuVectorIndex.Metric = .squaredEuclidean) -> WaifuVectorIndex {
    var index = WaifuVectorIndex(source: UUID(),dimensions: dimensions,metric: metric)!
    for row in 0..<count {
        let values = (0..<dimensions).map { Float((row*17+$0*3)%137)/137 }
        index.append(id: String(format:"id-%05d",count-row),vector: values)
    }
    return index
}
private func vectorGPU() throws -> WaifuMetalVectorSearch {
    guard MTLCreateSystemDefaultDevice()?.hasUnifiedMemory == true else { throw SkipTest(reason:"Unified-memory Metal unavailable") }
    return try WaifuMetalVectorSearch()
}
func registerVectorSearchTests() {
    let suite = "Vector search"
    Registry.shared.add(suite:suite,TestCase(name:"packed index rejects invalid rows and keeps stable source identities",knownBug:nil) {
        check(WaifuVectorIndex(source:UUID(),dimensions:0,metric:.euclidean)==nil,"zero dimension rejected")
        var index=vectorFixture(2,dimensions:3)
        let old=index.revision
        check(!index.append(id:"nan",vector:[.nan,0,0]),"NaN rejected")
        check(!index.append(id:"inf",vector:[0,.infinity,0]),"infinity rejected")
        check(!index.append(id:"bad",vector:[0]),"dimension mismatch rejected")
        check(!index.append(id:index.ids[0],vector:[0,0,0]),"duplicate identity rejected")
        checkEqual(index.revision,old,"invalid appends do not invalidate resident buffers")
        check(index.append(id:"new",vector:[0,0,0]),"incremental append")
        checkEqual(index.count,3,"rows append without replacing old identities")
        checkEqual(index.bytes,192,"64-byte row padding stays linear")
    })
    Registry.shared.add(suite:suite,TestCase(name:"CPU exact Top-K handles empty sets K limits ties and zero vectors",knownBug:nil) {
        var index=WaifuVectorIndex(source:UUID(),dimensions:3,metric:.squaredEuclidean)!
        check(try WaifuVectorCPU.search(index,query:[0,0,0],k:19).isEmpty,"empty safe")
        for id in ["z","a","m"] { index.append(id:id,vector:[0,0,0]) }
        checkEqual(try WaifuVectorCPU.search(index,query:[0,0,0],k:2).map(\.id),["a","m"],"ties use identity instead of row order")
        checkEqual(try WaifuVectorCPU.search(index,query:[0,0,0],k:99).map(\.id),["a","m","z"],"K clamps to collection")
        check(try WaifuVectorCPU.search(index,query:[0,0,0],k:0).isEmpty,"zero K safe")
        do { _=try WaifuVectorCPU.search(index,query:[.nan,0,0],k:1); check(false,"invalid query must throw") } catch WaifuVectorError.invalidQuery {}
    })
    Registry.shared.add(suite:suite,TestCase(name:"Metal SIMD and scalar kernels match CPU through tails and hierarchical Top-K",knownBug:nil) {
        let gpu=try vectorGPU()
        for dimensions in [1,37,768,1025] {
            let index=vectorFixture(1067,dimensions:dimensions)
            let query=(0..<dimensions).map { Float($0%17)/17 }
            let cpu=try WaifuVectorCPU.search(index,query:query,k:19)
            for scalar in [false,true] {
                let result=try await gpu.search(index,query:query,k:19,scalar:scalar).0
                checkEqual(result.map(\.id),cpu.map(\.id),"GPU identities including SIMD tails D=\(dimensions)")
                for (a,b) in zip(result,cpu) { checkClose(Double(a.distance),Double(b.distance),tolerance:2e-4,"GPU distance") }
            }
        }
    })
    Registry.shared.add(suite:suite,TestCase(name:"GPU duplicate ties zero rows and all K sizes are deterministic",knownBug:nil) {
        let gpu=try vectorGPU()
        var index=WaifuVectorIndex(source:UUID(),dimensions:37,metric:.euclidean)!
        for row in 0..<577 { index.append(id:String(format:"%04d",577-row),vector:[Float](repeating:0,count:37)) }
        for k in [1,19,32,64] {
            let result=try await gpu.search(index,query:[Float](repeating:0,count:37),k:k).0
            checkEqual(result.map(\.id),Array(index.ids.sorted().prefix(k)),"ties are stable across tiles for K=\(k)")
            check(result.allSatisfy { $0.distance==0 },"exact identical zeros")
        }
        let small=vectorFixture(3,dimensions:37)
        checkEqual(try await gpu.search(small,query:[Float](repeating:0,count:37),k:19).0.count,3,"partial threadgroup and K=N")
    })
    Registry.shared.add(suite:suite,TestCase(name:"real Vision query vectors preserve native distance semantics and ranking",knownBug:nil) {
        var observations:[FeaturePrintObservation]=[]
        for seed in 0..<12 { observations.append(try await makeFeaturePrintObservation(seed)) }
        var index=WaifuVisionVectorIndex(source:UUID())
        for (row,observation) in observations.enumerated() { index.append(id:String(format:"%03d",row),observation:observation) }
        let gpu=try vectorGPU()
        guard let packed=checkNotNil(index.packed,"native representation packed") else { return }
        checkEqual(packed.dimensions,observations[0].elementCount,"actual dimensions read from Vision")
        for query in observations {
            guard let q=checkNotNil(index.validatedQuery(query),"metric validates against native query") else { return }
            let result=try await gpu.search(packed,query:q,k:8).0
            var native:[WaifuVectorHit]=[]
            for row in observations.indices {
                let distance=try query.distance(to:observations[row])
                native.append(.init(row:row,id:packed.ids[row],distance:Float(distance)))
            }
            native.sort(by:WaifuVectorHit.precedes)
            checkEqual(result.map(\.id),Array(native.prefix(8)).map(\.id),"Recall@8 1.0 against Vision")
            for hit in result { check(VisualFeatureVectors.matches(try query.distance(to:observations[hit.row]),Double(hit.distance)),"native distance agreement") }
        }
    })
    Registry.shared.add(suite:suite,TestCase(name:"resident generations and appends cannot reuse stale GPU rows",knownBug:nil) {
        let gpu=try vectorGPU()
        var first=vectorFixture(800,dimensions:37)
        let q=[Float](repeating:0,count:37)
        _=try await gpu.search(first,query:q,k:19)
        first.append(id:"winner",vector:q)
        checkEqual(try await gpu.search(first,query:q,k:19).0.first?.id,"winner","append refreshes revision")
        var replacement=WaifuVectorIndex(source:UUID(),dimensions:37,metric:.squaredEuclidean)!
        replacement.append(id:"other-source",vector:q)
        checkEqual(try await gpu.search(replacement,query:q,k:19).0.map(\.id),["other-source"],"different source has its own resident identity")
    })
    Registry.shared.add(suite:suite,TestCase(name:"concurrent GPU queries own scratch and cancelled commands do not poison replacements",knownBug:nil) {
        let gpu=try vectorGPU(), index=vectorFixture(10000,dimensions:37)
        let q=[Float](repeating:0.1,count:37), other=[Float](repeating:0.9,count:37)
        let expected=try WaifuVectorCPU.search(index,query:q,k:19)
        async let a=gpu.search(index,query:q,k:19)
        async let b=gpu.search(index,query:other,k:19)
        let results=try await (a,b)
        checkEqual(results.0.0.map(\.id),expected.map(\.id),"first query unaffected by second staging")
        checkEqual(results.1.0.map(\.id),try WaifuVectorCPU.search(index,query:other,k:19).map(\.id),"second query has its own output")
        let task=Task { try await gpu.search(index,query:other,k:19) }
        task.cancel()
        do { _=try await task.value; check(false,"cancelled query must throw") } catch is CancellationError {}
        checkEqual(try await gpu.search(index,query:q,k:19).0.map(\.id),expected.map(\.id),"replacement retains valid resources")
    })
    Registry.shared.add(suite:suite,TestCase(name:"GPU failures and oversized K fall back to complete CPU search",knownBug:nil) {
        let index=vectorFixture(800,dimensions:37), q=[Float](repeating:0,count:37)
        let service=WaifuVectorSearch(metalFactory:{ try WaifuMetalVectorSearch(allocationLimit:16) })
        let result=try await service.search(index,query:q,k:19,backend:.metal)
        checkEqual(result.backend,.accelerate,"allocation failure selects CPU")
        checkEqual(result.hits,try WaifuVectorCPU.search(index,query:q,k:19),"fallback preserves exact results")
        let unavailable=WaifuVectorSearch(metalFactory:{ throw WaifuVectorError.unavailable })
        checkEqual(try await unavailable.search(index,query:q,k:19,backend:.metal).hits,result.hits,"no Metal device remains usable")
        let normal=WaifuVectorSearch()
        checkEqual(try await normal.search(index,query:q,k:800,backend:.metal).hits.count,800,"large K CPU fallback is complete")
    })
    Registry.shared.add(suite:suite,TestCase(name:"batch queries retain ordering and bounded results",knownBug:nil) {
        let index=vectorFixture(1000,dimensions:37)
        let queries=[[Float](repeating:0,count:37),[Float](repeating:0.5,count:37)]
        let results=try await WaifuVectorSearch().searchBatch(index,queries:queries,k:19)
        for (q,result) in zip(queries,results) { checkEqual(result.hits,try WaifuVectorCPU.search(index,query:q,k:19),"batch preserves input order") }
    })
    Registry.shared.add(suite:suite,TestCase(name:"planner requires measured warm GPU advantage and avoids unprofiled dispatch",knownBug:nil) {
        var planner=WaifuVectorPlanner()
        check(!planner.usesMetal(count:10000,dimensions:768,resident:true),"unknown workload uses CPU")
        for _ in 0..<3 {
            planner.record(count:10000,dimensions:768,gpu:false,seconds:0.001)
            planner.record(count:10000,dimensions:768,gpu:true,seconds:0.0005)
        }
        check(planner.usesMetal(count:10000,dimensions:768,resident:true),"measured faster GPU is eligible")
        check(!planner.usesMetal(count:10000,dimensions:768,resident:false),"cold upload is not free")
        check(!planner.usesMetal(count:1000,dimensions:768,resident:true),"a different shape must be measured")
    })

    Registry.shared.add(suite:suite,TestCase(name:"Orb integrates packed search with payload invalidation progressive results and source replacement",knownBug:nil) {
        let a=try await makeFeaturePrintObservation(0), b=try await makeFeaturePrintObservation(4)
        let image=try ImageFixture.cgImage(width:128,height:96,gray:0.5)
        let bytesA=FeaturePrintCodec.encode(a)!, bytesB=FeaturePrintCodec.encode(b)!
        func frame(_ row: Int,_ bytes:Data) -> AnalyzedFrame {
            AnalyzedFrame(sampleIndex:row,timeSeconds:Double(row),artScore:0,faceScore:nil,faceCount:0,faceAreaFraction:nil,featurePrint:bytes)
        }
        let generation=UUID(), query=UUID(), engine=CrystalOrbSearchService()
        let frames=(0..<256).map { frame($0,$0%2==0 ? bytesA:bytesB) }
        let first=try await engine.search(query:image,queryID:query,frames:frames,sourceURL:URL(fileURLWithPath:"/unused"),generation:generation,resultLimit:19) { _,_ in }
        checkEqual(first.count,19,"production returns only nineteen")
        checkEqual(await engine.readableCount,256,"summary keeps full readable count")
        let again=try await engine.search(query:image,queryID:query,frames:frames,sourceURL:URL(fileURLWithPath:"/unused"),generation:generation,resultLimit:19) { _,_ in }
        checkEqual(first,again,"unchanged query reuses exact ranked identities")
        checkEqual(await engine.queryComputations,1,"query encoder is reused")
        var updated=frames
        updated[0]=frame(0,bytesB)
        let actual=try await engine.search(query:image,queryID:query,frames:updated,sourceURL:URL(fileURLWithPath:"/unused"),generation:generation,resultLimit:19) { _,_ in }
        let queryFeature=try await GenerateImageFeaturePrintRequest(.revision2).perform(on:image)
        var expected:[CrystalOrbMatch]=[]
        for f in updated { expected.append(.init(frame:f,distance:try queryFeature.distance(to:FeaturePrintCodec.decode(f.featurePrint!)!))) }
        checkEqual(actual,Array(CrystalOrbMatch.ranked(expected).prefix(19)),"changed payload invalidates the same-generation vector")
        let replaced=try await engine.search(query:image,queryID:query,frames:[frame(999,bytesA)],sourceURL:URL(fileURLWithPath:"/unused"),generation:UUID(),resultLimit:19) { _,_ in }
        checkEqual(replaced.map(\.id.sampleIndex),[999],"source generation removes stale rows")
    })

    Registry.shared.add(suite:suite,TestCase(name:"public Metal backend canonically reranks near-ties without changing CPU scores",knownBug:nil) {
        var index=WaifuVectorIndex(source:UUID(),dimensions:37,metric:.squaredEuclidean)!
        for row in 0..<700 {
            var vector=[Float](repeating:0.5,count:37)
            vector[row%37] += Float(row%31)*0.0000001
            index.append(id:String(format:"near-%04d",700-row),vector:vector)
        }
        let query=[Float](repeating:0.49,count:37)
        let actual=try await WaifuVectorSearch().search(index,query:query,k:19,backend:.metal)
        checkEqual(actual.hits,try WaifuVectorCPU.search(index,query:query,k:19),"near-tie identities and distances match canonical CPU exactly")
    })

}
