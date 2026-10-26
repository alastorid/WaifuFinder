import Foundation

/// One warm native worker. This protocol carries only private file paths; PNGs
/// and transient evidence are deleted after use. Scoring exports scalars only.
private final class TruForWorker:@unchecked Sendable {
    let root:URL
    let process=Process()
    private let input:FileHandle
    private let output:FileHandle
    init(configuration:TruForConfiguration,executable:URL)throws {
        root=FileManager.default.temporaryDirectory.appendingPathComponent("trufor-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true,attributes:[.posixPermissions:0o700])
        let incoming=Pipe(),outgoing=Pipe();input=incoming.fileHandleForWriting;output=outgoing.fileHandleForReading
        process.executableURL=executable;process.arguments=["--model",configuration.directory.path,"--serve"];process.qualityOfService = .utility
        // A diagnostic shell override must not silently alter persisted app scores.
        var environment=ProcessInfo.processInfo.environment
        environment["WAIFU_TRUFOR_PRECISION"]="fp32"
        let coreMLNoiseprint=configuration.directory.appendingPathComponent("CoreMLNoiseprintFP16.mlmodelc")
        if FileManager.default.fileExists(atPath:coreMLNoiseprint.path) {
            environment["WAIFU_TRUFOR_COREML_FP16_MODEL"]=coreMLNoiseprint.path
        } else {
            environment.removeValue(forKey:"WAIFU_TRUFOR_COREML_FP16_MODEL")
        }
        process.environment=environment
        process.standardInput=incoming;process.standardOutput=outgoing
        FileManager.default.createFile(atPath:root.appendingPathComponent("worker.log").path,contents:nil)
        process.standardError=try FileHandle(forWritingTo:root.appendingPathComponent("worker.log"))
        try process.run();try incoming.fileHandleForReading.close();try outgoing.fileHandleForWriting.close()
        guard try line()=="TRF1" else { cancel();throw ModelInstaller.fail("Incompatible TruFor worker") }
    }
    deinit { cancel();try? FileManager.default.removeItem(at:root) }
    func cancel() { if process.isRunning { process.terminate() } }
    private func line()throws->String {
        var bytes=Data()
        while bytes.count<8192 {
            guard let byte=try output.read(upToCount:1),!byte.isEmpty else {
                let log=(try? String(contentsOf:root.appendingPathComponent("worker.log"),encoding:.utf8)) ?? ""
                let detail=log.split(separator:"\n").last.map(String.init) ?? ""
                throw ModelInstaller.fail(detail.hasPrefix("TruFor stopped") ? detail:"TruFor worker stopped")
            }
            if byte[0]==10 { return String(decoding:bytes,as:UTF8.self) };bytes.append(byte)
        }
        throw ModelInstaller.fail("Invalid TruFor worker reply")
    }
    func run(_ images:[Data],evidence:Bool)throws->TruForJobResult {
        let files=images.indices.map{root.appendingPathComponent("input-\($0).png")},result=root.appendingPathComponent("result.bin")
        defer { for url in files+[result] { try? FileManager.default.removeItem(at:url) } }
        for (index,data) in images.enumerated() { try data.write(to:files[index],options:.atomic) }
        var request=try JSONSerialization.data(withJSONObject:["inputs":files.map(\.path),"output":result.path,"mode":evidence ? "evidence":"scores"]);request.append(10)
        try input.write(contentsOf:request)
        let response=try line();guard response=="OK" else { throw ModelInstaller.fail(response.replacingOccurrences(of:"ERROR ",with:"")) }
        if evidence { return .evidence(try TruForSpatial.decode(Data(contentsOf:result,options:.mappedIfSafe))) }
        struct Scores:Decodable { let scores:[Float] }
        let values=try JSONDecoder().decode(Scores.self,from:Data(contentsOf:result)).scores
        guard values.count==images.count,values.allSatisfy({$0.isFinite && (0...1).contains($0)}) else { throw ModelInstaller.fail("Invalid TruFor detection scores") }
        return .scores(values)
    }
}

/// Interactive evidence is chosen before queued background batches. A running
/// batch finishes normally; cancellation removes queued work or stops its worker.
actor TruForBackend {
    static let shared=TruForBackend()
    private struct Request {
        let id:UUID,images:[Data],evidence:Bool,cacheKey:String?
        let continuation:CheckedContinuation<TruForJobResult,Error>
    }
    private let override:TruForConfiguration?
    private let executableOverride:URL?
    private var worker:TruForWorker?
    private var fingerprint:String?
    private var pending:[Request]=[]
    private var current:UUID?
    private var cancelled=Set<UUID>()
    private var draining=false
    private var idle:Task<Void,Never>?
    private var cached:[String:TruForSpatial]=[:]
    private var order:[String]=[]
    private var bytes=0
    private let cacheLimit=256*1024*1024
    init(configuration:TruForConfiguration?=nil,executable:URL?=nil) { override=configuration;executableOverride=executable }
    func submit(images:[Data],evidence:Bool,cacheKey:String?=nil)async throws->TruForJobResult {
        try Task.checkCancellation()
        if evidence,let cacheKey,let value=cached[cacheKey] { order.removeAll{$0==cacheKey};order.append(cacheKey);return .evidence(value) }
        let id=UUID()
        return try await withTaskCancellationHandler {
            try Task.checkCancellation()
            return try await withCheckedThrowingContinuation { continuation in
                pending.append(.init(id:id,images:images,evidence:evidence,cacheKey:cacheKey,continuation:continuation))
                idle?.cancel()
                if !draining { draining=true;Task { await drain() } }
            }
        } onCancel:{ Task { await self.cancel(id) } }
    }
    private func cancel(_ id:UUID) {
        if let i=pending.firstIndex(where:{$0.id==id}) { pending.remove(at:i).continuation.resume(throwing:CancellationError()) }
        if current==id { cancelled.insert(id);worker?.cancel() }
    }
    private func drain()async {
        while !pending.isEmpty {
            let index=pending.firstIndex(where:{$0.evidence}) ?? 0
            let request=pending.remove(at:index);current=request.id
            do {
                guard let configuration=override ?? TruForConfiguration.load(),let executable=executableOverride ?? TruForConfiguration.executable else { throw ModelInstaller.fail("Set up a locally validated TruFor folder in Settings → Models.") }
                if fingerprint != configuration.fingerprint { await release() }
                if worker==nil {
                    guard await ModelLibrary.shared.active==nil else { throw RemoteError.busy }
                    await ModelLibrary.shared.acquireTruFor()
                    do {
                        worker=try await Task.detached(priority:.utility){try TruForWorker(configuration:configuration,executable:executable)}.value
                        fingerprint=configuration.fingerprint
                    } catch { await ModelLibrary.shared.releaseTruFor();throw error }
                }
                if cancelled.contains(request.id) { throw CancellationError() }
                guard let worker else { throw RemoteError.busy }
                let value=try await Task.detached(priority:request.evidence ? .userInitiated:.utility){try worker.run(request.images,evidence:request.evidence)}.value
                if cancelled.contains(request.id) { throw CancellationError() }
                if case .evidence(let map)=value,let key=request.cacheKey,map.byteCount<=cacheLimit {
                    if let old=cached.removeValue(forKey:key) { bytes-=old.byteCount;order.removeAll{$0==key} }
                    while bytes+map.byteCount>cacheLimit,let oldest=order.first { order.removeFirst();bytes-=cached.removeValue(forKey:oldest)?.byteCount ?? 0 }
                    cached[key]=map;order.append(key);bytes+=map.byteCount
                }
                request.continuation.resume(returning:value)
            } catch {
                worker?.cancel();await release();request.continuation.resume(throwing:error)
            }
            cancelled.remove(request.id);current=nil
        }
        draining=false
        idle=Task { do { try await Task.sleep(for:.seconds(60));await release() } catch {} }
    }
    private func release()async {
        worker?.cancel();worker=nil
        if fingerprint != nil { fingerprint=nil;await ModelLibrary.shared.releaseTruFor() }
    }
}
