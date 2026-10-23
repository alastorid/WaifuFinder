import Foundation
import CoreGraphics
import Vision
import Metal

@main struct VectorBenchmark {
    static func clock() -> Double { ProcessInfo.processInfo.systemUptime }
    static func summary(_ label: String, _ values: [Double]) {
        let sorted = values.sorted()
        print(String(format: "%@: median %.3f ms, p95 %.3f ms", label, sorted[sorted.count/2]*1000, sorted[min(sorted.count-1,Int(Double(sorted.count)*0.95))]*1000))
    }
    static func image(_ seed: Int) -> CGImage {
        let c = CGContext(data: nil,width: 320,height: 240,bitsPerComponent: 8,bytesPerRow: 0,space: CGColorSpace(name: CGColorSpace.sRGB)!,bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
        c.setFillColor(CGColor(red: CGFloat(seed%7)/7,green: CGFloat(seed%11)/11,blue: CGFloat(seed%13)/13,alpha: 1)); c.fill(CGRect(x: 0,y: 0,width: 320,height: 240))
        for i in 0..<18 {
            c.setFillColor(CGColor(red: CGFloat((seed*7+i*11)%31)/31,green: CGFloat((seed+i*3)%19)/19,blue: CGFloat((i+seed*3)%17)/17,alpha: 1))
            c.fillEllipse(in: CGRect(x: (seed*13+i*31)%290,y: (seed*29+i*13)%210,width: 20+i*2,height: 19+i*3))
        }
        return c.makeImage()!
    }
    static func main() async throws {
        setvbuf(stdout,nil,_IOLBF,0)
        print("Device: \(MTLCreateSystemDefaultDevice()?.name ?? "CPU only"); maximum benchmark size 10,000")
        let start = clock()
        var observations: [FeaturePrintObservation] = [], extraction: [Double] = []
        for seed in 0..<48 {
            let t = clock(); observations.append(try await GenerateImageFeaturePrintRequest(.revision2).perform(on: image(seed))); extraction.append(clock()-t)
        }
        print("Vision real feature representation: \(observations[0].elementType), \(observations[0].elementCount) elements, \(observations[0].data.count) bytes")
        summary("Vision encoding (48 native still fixtures)",extraction)
        guard let validated = VisualFeatureVectors(observations: observations) else { fatalError("Vision metric unsupported") }
        print("Validated metric: \(validated.metric); extraction total \(clock()-start)s")
        let query = WaifuVisionVectorIndex.vector(observations[0])!
        let encodingStart = clock()
        let encoded = observations.compactMap(FeaturePrintCodec.encode)
        let decodeStart = clock()
        let decoded = encoded.compactMap(FeaturePrintCodec.decode)
        print(String(format: "48 cache encode %.3f ms; decode %.3f ms (%d retained)",(decodeStart-encodingStart)*1000,(clock()-decodeStart)*1000,decoded.count))
        // Native feature parity includes all pairs, duplicates and neighboring query ranks.
        var worst = 0.0
        for a in observations.indices {
            for b in observations.indices {
                worst = max(worst,abs(try observations[a].distance(to: observations[b])-Double(validated.distance(a,b))))
            }
        }
        print("Vision/Accelerate worst absolute distance error across 2304 pairs: \(worst)")
        let pipelineStart=clock()
        let gpu = try WaifuMetalVectorSearch()
        print(String(format:"Metal device/queue/pipeline initialization: %.3f ms",(clock()-pipelineStart)*1000))
        let service = WaifuVectorSearch()
        for count in [1000,10000] {
            let buildStart = clock()
            var index = WaifuVectorIndex(source: UUID(),dimensions: query.count,metric: validated.metric == .squaredEuclidean ? .squaredEuclidean : .euclidean)!
            // Scaling uses actual Vision vectors with deterministic small perturbations;
            // the unmodified 48-observation parity check above is the native reference.
            for row in 0..<count {
                var vector = WaifuVisionVectorIndex.vector(observations[row%observations.count])!
                let offset = Float(row/observations.count) * 0.00001
                vector[(row*17)%vector.count] += offset
                precondition(index.append(id: String(format: "%08d",row),vector: vector))
            }
            let build = clock()-buildStart
            print("\nN=\(count), D=\(query.count), packed=\(index.bytes) bytes, build \(build*1000) ms")
            let reference = try WaifuVectorCPU.search(index,query: query,k: 19)
            let coldStart = clock(); let cold = try await gpu.search(index,query: query,k: 19)
            print(String(format: "GPU cold e2e %.3f ms; upload/staging %.3f ms; allocated %d bytes",(clock()-coldStart)*1000,cold.1.upload*1000,cold.1.allocatedBytes))
            var cpu: [Double] = [], simd: [Double] = [], scalar: [Double] = [], readback: [Double] = [], native: [Double] = []
            var kernel: [Double] = [], selection: [Double] = [], uploads: [Double] = [], read: [Double] = [], wall: [Double] = []
            for _ in 0..<21 {
                var t=clock(); _ = try WaifuVectorCPU.search(index,query: query,k: 19); cpu.append(clock()-t)
                t=clock(); let result=try await gpu.search(index,query: query,k: 19); simd.append(clock()-t)
                precondition(result.0.map(\.id)==reference.map(\.id),"SIMD Top-K ranking mismatch")
                kernel.append(result.1.distanceGPU); selection.append(result.1.selectionGPU); uploads.append(result.1.upload); read.append(result.1.readback); wall.append(result.1.commandWall)
                t=clock(); let simple=try await gpu.search(index,query: query,k: 19,scalar: true); scalar.append(clock()-t)
                precondition(simple.0.map(\.id)==reference.map(\.id),"Scalar Top-K ranking mismatch")
                t=clock(); let host=try await gpu.search(index,query: query,k: 19,readback: true); readback.append(clock()-t)
                precondition(host.0.map(\.id)==reference.map(\.id),"Readback ranking mismatch")
                t=clock(); var top=WaifuVectorTopK(19)
                for row in 0..<count { let distance=try observations[0].distance(to: observations[row%observations.count]); top.insert(.init(row:row,id:index.ids[row],distance:Float(distance))) }
                _=top.sorted; native.append(clock()-t)
            }
            summary("Vision loop baseline (repeated native corpus)",native)
            summary("Accelerate exact + bounded Top-K",cpu)
            summary("Metal SIMD + GPU Top-K warm e2e",simd)
            summary("Metal scalar float4 + GPU Top-K warm e2e",scalar)
            summary("Metal distances + CPU Top-K warm e2e",readback)
            summary("GPU distance execution",kernel); summary("GPU Top-K execution",selection)
            summary("Warm upload/staging",uploads); summary("Command wall including scheduling",wall); summary("Compact readback",read)
            let batchStart=clock(); let batch=try await service.searchBatch(index,queries: Array(repeating:query,count:4),k:19,backend:.metal)
            precondition(batch.allSatisfy { $0.hits.map(\.id)==reference.map(\.id) })
            print(String(format:"Four-query batch, including first upload: %.3f ms; Recall@19 1.0 on these exact reference queries",(clock()-batchStart)*1000))
        }
        var corpus=WaifuVectorIndex(source: UUID(),dimensions:query.count,metric:.squaredEuclidean)!
        for row in observations.indices { corpus.append(id:String(format:"%03d",row),vector:WaifuVisionVectorIndex.vector(observations[row])!) }
        for obs in observations {
            let q=WaifuVisionVectorIndex.vector(obs)!
            let result=try await gpu.search(corpus,query:q,k:19)
            var native: [(id: String,distance: Double)] = []
            for row in observations.indices { native.append((corpus.ids[row],try obs.distance(to: observations[row]))) }
            native.sort { a,b in a.distance == b.distance ? a.id < b.id : a.distance < b.distance }
            precondition(result.0.map(\.id)==Array(native.prefix(19)).map(\.id),"Native Vision ranking disagreement")
        }
        var near=WaifuVectorIndex(source:UUID(),dimensions:37,metric:.squaredEuclidean)!
        for row in 0..<700 {
            var vector=[Float](repeating:0.5,count:37)
            vector[row%37] += Float(row%31)*0.0000001
            near.append(id:String(format:"near-%04d",700-row),vector:vector)
        }
        let nearQuery=[Float](repeating:0.49,count:37)
        let nearExpected=try WaifuVectorCPU.search(near,query:nearQuery,k:19)
        let nearActual=try await service.search(near,query:nearQuery,k:19,backend:.metal)
        precondition(nearActual.hits==nearExpected,"Canonical near-tie ranking mismatch")
        print("Public Metal backend near-tie guard matches exact CPU IDs and distances")
        print("48 native queries: Metal Recall@19 = 1.0; no approximate index or reduced-precision storage")
        var usage=rusage()
        if getrusage(RUSAGE_SELF,&usage)==0 { print("Peak process RSS (includes Vision/frameworks/CPU+GPU data): \(usage.ru_maxrss) bytes") }
        #if WAIFU_ORB_BENCHMARK
        try await benchmarkOrb(observations)
        #endif
    }
    #if WAIFU_ORB_BENCHMARK
    static func benchmarkOrb(_ observations:[FeaturePrintObservation]) async throws {
        let payloads=observations.compactMap(FeaturePrintCodec.encode)
        let queryImage=image(0)
        let query=try await GenerateImageFeaturePrintRequest(.revision2).perform(on:queryImage)
        for count in [1000,10000] {
            let frames=(0..<count).map { row in
                AnalyzedFrame(sampleIndex:row,timeSeconds:Double(row),artScore:0,faceScore:nil,faceCount:0,faceAreaFraction:nil,featurePrint:payloads[row%payloads.count])
            }
            let generation=UUID(), id=UUID(), engine=CrystalOrbSearchService()
            let before=clock()
            var reference:[CrystalOrbMatch]=[]
            var decoded:[FeaturePrintObservation]=[]
            for frame in frames {
                let feature=FeaturePrintCodec.decode(frame.featurePrint!)!
                decoded.append(feature)
                reference.append(.init(frame:frame,distance:try query.distance(to:feature)))
            }
            let expected=Array(CrystalOrbMatch.ranked(reference).prefix(19))
            let coldBaseline=clock()-before
            let t=clock()
            let cold=try await engine.search(query:queryImage,queryID:id,frames:frames,sourceURL:URL(fileURLWithPath:"/unused"),generation:generation,resultLimit:19) { _,_ in }
            precondition(cold==expected,"Cold Orb must match original native loop")
            print(String(format:"Orb N=%d original cold decode+Vision+sort %.3f ms; new cold with query encode %.3f ms",count,coldBaseline*1000,(clock()-t)*1000))
            var native:[Double]=[], warm:[Double]=[]
            for _ in 0..<11 {
                var start=clock(); var values:[CrystalOrbMatch]=[]
                for row in frames.indices { values.append(.init(frame:frames[row],distance:try query.distance(to:decoded[row]))) }
                _=CrystalOrbMatch.ranked(values); native.append(clock()-start)
                start=clock()
                let result=try await engine.search(query:queryImage,queryID:id,frames:frames,sourceURL:URL(fileURLWithPath:"/unused"),generation:generation,resultLimit:19) { _,_ in }
                warm.append(clock()-start); precondition(result==expected,"Warm Orb ranking mismatch")
            }
            summary("Orb original warm Vision + full sort N=\(count)",native)
            summary("Orb integrated warm N=\(count)",warm)
        }
    }
    #endif

}
