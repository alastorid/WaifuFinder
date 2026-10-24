import Foundation
import MLX

public struct SAFIREPrediction {
    public let probabilities: [Float]
    public let partitions: [UInt16]
    public let selectedPrompts: [Int]
}

extension SAFIREModel {
    /// PyTorch align_corners=False interpolation for BHWC tensors, with clamped
    /// boundary samples. Also used to sample the exact nearest-downsampled binary
    /// mask for the encoder-feature pooling step.
    public static func resize(_ x: MLXArray,height:Int,width:Int) -> MLXArray {
        let h=x.dim(1), w=x.dim(2)
        let ys=(0..<height).map { max(0,min(Float(h-1),(Float($0)+0.5)*Float(h)/Float(height)-0.5)) }
        let xs=(0..<width).map { max(0,min(Float(w-1),(Float($0)+0.5)*Float(w)/Float(width)-0.5)) }
        let y0=MLXArray(ys.map { Int32($0) }), y1=minimum(y0+1,MLXArray(Int32(h-1)))
        let x0=MLXArray(xs.map { Int32($0) }), x1=minimum(x0+1,MLXArray(Int32(w-1)))
        let fy=MLXArray(ys.map { $0-floor($0) },[1,height,1,1]), fx=MLXArray(xs.map { $0-floor($0) },[1,1,width,1])
        let a=take(x,y0,axis:1), b=take(x,y1,axis:1)
        let rows=a*(1-fy)+b*fy
        return take(rows,x0,axis:2)*(1-fx)+take(rows,x1,axis:2)*fx
    }

    /// Official automatic mode: 16x16 positive prompts; cosine clustering of
    /// mean image embeddings inside each prompt mask; highest IoU representative
    /// per source cluster. The larger binary representative is background.
    /// Clustering initialization is deterministic (upstream uses random seeds).
    public func predict(_ normalized:MLXArray,clusters:Int=3) throws -> SAFIREPrediction {
        guard (2...8).contains(clusters) else { throw SAFIREError.invalid("Expected 2…8 source partitions") }
        let embedding=try encode(normalized)
        var masks:[MLXArray]=[], features:[[Float]]=[], qualities:[Float]=[], areas:[Int]=[]
        for start in stride(from:0,to:256,by:diagnostics.promptBatch) {
            let end=min(256,start+diagnostics.promptBatch)
            var coords: [Float] = []
            for i in start..<end { coords.append(Float(i%16)*64+32); coords.append(Float(i/16)*64+32) }
            let result=try decode(embedding:embedding,points:MLXArray(coords,[end-start,2]))
            // Keep full-mask thresholding, areas and feature pooling resident
            // on the GPU for the decoder batch. One completion fence per batch
            // replaces the per-prompt scalar reads and feature readback fences.
            let batch=end-start
            let full=Self.resize(result.masks.reshaped(batch,256,256,1),height:1024,width:1024)
            let mask:MLXArray=full .> Float(0)
            let areaTensor=sum(mask.asType(.int32),axes:[1,2,3])
            let samples=mask[0...,MLXSlice.stride(by:16),MLXSlice.stride(by:16),0].asType(.float32).reshaped(batch,64,64,1)
            let counts=sum(samples,axes:[1,2,3])
            let pooled=sum(embedding*samples,axes:[1,2])/maximum(counts.reshaped(batch,1),MLXArray(Float(1)))
            eval(result.quality,areaTensor,counts,pooled)
            let quality=result.quality.asArray(Float.self)
            let batchAreas=areaTensor.asArray(Int32.self),batchCounts=counts.asArray(Float.self)
            let batchFeatures=pooled.asArray(Float.self)
            for j in 0..<batch where batchAreas[j]>0 && batchCounts[j]>0 {
                masks.append(result.masks[j])
                features.append(Array(batchFeatures[(j*256)..<((j+1)*256)]))
                qualities.append(quality[j]);areas.append(Int(batchAreas[j]))
            }
        }
        guard features.count >= 2 else { throw SAFIREError.invalid("SAFIRE produced fewer than two usable region masks") }
        func representatives(_ count:Int) -> [Int] {
            let assignments=SAFIREClustering.cosine(features,k:count)
            return Set(assignments).sorted().compactMap { label in
                assignments.indices.filter({assignments[$0]==label}).max(by:{qualities[$0] < qualities[$1]})
            }
        }
        let binary=representatives(2), multi=representatives(clusters).sorted()
        guard !binary.isEmpty,!multi.isEmpty else { throw SAFIREError.invalid("No source partition survived clustering") }
        func probability(_ i:Int) -> MLXArray {
            sigmoid(Self.resize(masks[i].reshaped(1,256,256,1),height:1024,width:1024)).reshaped(1024,1024)
        }
        let partitions=argMax(stacked(multi.map(probability),axis:0),axis:0).asType(.uint16)
        let probabilities:MLXArray
        if binary.count == 2 {
            let foreground=areas[binary[0]] > areas[binary[1]] ? binary[1] : binary[0]
            let background=foreground == binary[0] ? binary[1] : binary[0]
            probabilities=(probability(foreground)+1-probability(background))/2
        } else { probabilities=probability(binary[0]) }
        eval(probabilities,partitions)
        return .init(probabilities:probabilities.asArray(Float.self),partitions:partitions.asArray(UInt16.self),selectedPrompts:binary)
    }
}

public enum SAFIREClustering {
    public static func cosine(_ input:[[Float]],k:Int) -> [Int] {
        guard !input.isEmpty else { return [] }
        let values=input
        func normalized(_ row:[Float]) -> [Float] {
            let norm=sqrt(row.reduce(Float(0)) { $0+$1*$1 })
            return row.map { $0/max(1e-12,norm) }
        }
        let unit=values.map(normalized)
        func distance(_ a:[Float],_ b:[Float]) -> Float { 1-zip(a,b).reduce(Float(0)) { $0+$1.0*$1.1 } }
        var centers=[values[0]]
        for _ in 1..<min(k,values.count) {
            let units=centers.map(normalized)
            let next=values.indices.max { a,b in
                units.map { distance(unit[a],$0) }.min()! < units.map { distance(unit[b],$0) }.min()!
            }!
            centers.append(values[next])
        }
        var labels=[Int](repeating:-1,count:values.count)
        for _ in 0..<1000 {
            let units=centers.map(normalized)
            let next=unit.map { row in centers.indices.min { distance(row,units[$0]) < distance(row,units[$1]) }! }
            if next==labels { break }; labels=next
            var shift:Float=0
            for c in centers.indices {
                let previous=centers[c]
                let members=values.indices.filter { labels[$0]==c }
                guard !members.isEmpty else { continue }
                var total=[Float](repeating:0,count:values[0].count)
                for i in members { for d in total.indices { total[d]+=values[i][d] } }
                centers[c]=total.map { $0/Float(members.count) }
                shift += sqrt(zip(previous,centers[c]).reduce(Float(0)) { $0+($1.0-$1.1)*($1.0-$1.1) })
            }
            if shift*shift < 1e-4 { break }
        }
        return labels
    }
}
