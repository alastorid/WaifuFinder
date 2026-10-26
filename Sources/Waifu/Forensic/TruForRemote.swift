import Foundation
import ImageIO

struct RemoteTruForRequest:Codable,Sendable {
    let images:[Data]
    let evidence:Bool
    let fingerprint:String
    func validate()throws {
        guard fingerprint==TruForConfiguration.weightsSHA,(1...4).contains(images.count),!evidence || images.count==1,
              images.reduce(0,{$0+$1.count})<=32*1024*1024 else { throw RemoteError.invalidRequest("Invalid TruFor batch") }
        for image in images {
            guard image.prefix(8)==Data([137,80,78,71,13,10,26,10]),let source=CGImageSourceCreateWithData(image as CFData,[kCGImageSourceShouldCache:false] as CFDictionary),let p=CGImageSourceCopyPropertiesAtIndex(source,0,nil) as? [CFString:Any],let w=p[kCGImagePropertyPixelWidth] as? Int,let h=p[kCGImagePropertyPixelHeight] as? Int,w>=32,h>=32,w<=32768,h<=32768,w*h<=24_000_000 else { throw RemoteError.invalidRequest("TruFor requires supported native PNG geometry") }
        }
    }
}
private struct TruForReply:Codable {
    let fingerprint:String
    let scores:[Float]?
    let bytes:Int?
}
extension TruForSpatial {
    func encoded()throws->Data {
        try validate();var data=Data("TFM1".utf8)
        for value in [UInt32(width),UInt32(height),UInt32(3)] { var v=value.littleEndian;withUnsafeBytes(of:&v){data.append(contentsOf:$0)} }
        var value=score;withUnsafeBytes(of:&value){data.append(contentsOf:$0)}
        for map in [anomaly,reliability,noiseprint] { map.withUnsafeBytes{data.append(contentsOf:$0)} };return data
    }
}
extension RemoteServerSession {
    func handleTruFor(_ envelope:RemoteEnvelope)async {
        do {
            try Task.checkCancellation()
            let request=try JSONDecoder().decode(RemoteTruForRequest.self,from:envelope.data);try request.validate()
            let result=try await catalog.trufor(request)
            try Task.checkCancellation()
            switch result {
            case .scores(let values):
                try await connection.send(.truforResult,payload:[UInt8](JSONEncoder().encode(TruForReply(fingerprint:request.fingerprint,scores:values,bytes:nil))),streamID:envelope.streamID)
            case .evidence(let map):
                let bytes=try map.encoded()
                try await connection.send(.truforResult,payload:[UInt8](JSONEncoder().encode(TruForReply(fingerprint:request.fingerprint,scores:nil,bytes:bytes.count))),streamID:envelope.streamID)
                for start in stride(from:0,to:bytes.count,by:8*1024*1024) {
                    try Task.checkCancellation()
                    try await connection.send(.truforData,payload:[UInt8](bytes[start..<min(bytes.count,start+8*1024*1024)]),streamID:envelope.streamID)
                }
                try await connection.send(.truforEnd,payload:[],streamID:envelope.streamID)
            }
        } catch { try? await connection.sendError(envelope,Task.isCancelled ? .cancelled:Self.describe(error)) }
        await resources.finish(streamID:envelope.streamID)
    }
}
extension RemoteClient {
    var offersTruFor:Bool { status.isConnected && capabilities?.offersTruFor==true }
    func truforScores(_ images:[Data])async throws->[Float] {
        let result=try await trufor(images,evidence:false)
        guard case .scores(let scores)=result else { throw RemoteError.framing("Unexpected TruFor evidence") };return scores
    }
    func truforEvidence(_ image:Data)async throws->TruForSpatial {
        let result=try await trufor([image],evidence:true)
        guard case .evidence(let map)=result else { throw RemoteError.framing("Missing TruFor evidence") };return map
    }
    private func trufor(_ images:[Data],evidence:Bool)async throws->TruForJobResult {
        guard offersTruFor,let connection else { throw RemoteError.modelUnavailable("No matching TruFor server connected") }
        let request=RemoteTruForRequest(images:images,evidence:evidence,fingerprint:TruForConfiguration.weightsSHA);try request.validate()
        let stream=try await connection.beginRequest()
        do {
            let result=try await withTaskCancellationHandler {
                try Task.checkCancellation()
                try await connection.send(.truforRequest,payload:[UInt8](JSONEncoder().encode(request)),streamID:stream)
                guard let response=await connection.next(streamID:stream) else { try Task.checkCancellation();throw RemoteError.framing("TruFor stream ended") }
                if response.isError { throw RemoteConnection.remoteError(from:response) }
                guard response.opcode == .truforResult else { throw RemoteError.framing("Unexpected TruFor response") }
                let header=try JSONDecoder().decode(TruForReply.self,from:response.data)
                guard header.fingerprint==request.fingerprint else { throw RemoteError.framing("TruFor model fingerprint mismatch") }
                if !evidence {
                    guard let values=header.scores,values.count==images.count,values.allSatisfy({$0.isFinite && (0...1).contains($0)}) else { throw RemoteError.framing("Invalid TruFor scores") }
                    return TruForJobResult.scores(values)
                }
                guard let expected=header.bytes,expected>=20,expected<=24_000_000*12+20 else { throw RemoteError.framing("Invalid TruFor evidence size") }
                var data=Data();data.reserveCapacity(expected)
                while let chunk=await connection.next(streamID:stream) {
                    try Task.checkCancellation();if chunk.isError { throw RemoteConnection.remoteError(from:chunk) }
                    if chunk.opcode == .truforEnd { guard data.count==expected else { throw RemoteError.framing("Truncated TruFor evidence") };return .evidence(try TruForSpatial.decode(data)) }
                    guard chunk.opcode == .truforData,data.count+chunk.payload.count<=expected else { throw RemoteError.framing("Invalid TruFor evidence chunk") };data.append(chunk.data)
                }
                try Task.checkCancellation();throw RemoteError.framing("TruFor evidence stream ended")
            } onCancel:{ Task { await connection.cancelRequest(stream) } }
            await connection.endRequest();return result
        } catch { await connection.endRequest();throw error }
    }
}
