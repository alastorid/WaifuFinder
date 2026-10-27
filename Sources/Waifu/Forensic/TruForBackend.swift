import Foundation
import Darwin

/// One warm FP32 Swift/MLX worker. PNGs and evidence travel through pipes;
/// neither transport nor activation storage uses scratch files.
private final class TruForWorker:@unchecked Sendable {
    let process=Process()
    private let input:FileHandle
    private let output:FileHandle
    init(configuration:TruForConfiguration,executable:URL)throws {
        let incoming=Pipe(),outgoing=Pipe();input=incoming.fileHandleForWriting;output=outgoing.fileHandleForReading
        process.executableURL=executable;process.arguments=["--model",configuration.directory.path,"--serve-pipe"];process.qualityOfService = .utility
        process.environment=TruForConfiguration.workerEnvironment(ProcessInfo.processInfo.environment)
        process.standardInput=incoming;process.standardOutput=outgoing
        process.standardError=FileHandle.standardError
        // A cancelled/failed worker must produce a write error, never SIGPIPE
        // in the app while a large PNG is still being sent.
        guard fcntl(input.fileDescriptor,F_SETNOSIGPIPE,1) != -1 else { throw ModelInstaller.fail("Unable to configure TruFor pipe") }
        try process.run()
        do {
            try incoming.fileHandleForReading.close();try outgoing.fileHandleForWriting.close()
            guard try TruForPipe.line(output)=="TRF2 FP32 MLX RAM" else { throw ModelInstaller.fail("Incompatible TruFor worker; rebuild the FP32 runtime") }
        } catch { cancel();throw error }
    }
    deinit { cancel();try? input.close();try? output.close() }
    func cancel() { if process.isRunning { process.terminate() } }
    func run(_ images:[Data],evidence:Bool)throws->TruForJobResult {
        guard !images.isEmpty,images.count<=4,(!evidence || images.count==1),
              images.allSatisfy({!$0.isEmpty && $0.count<=24_000_000*4+1_048_576}) else { throw ModelInstaller.fail("Invalid TruFor request") }
        var request=try JSONSerialization.data(withJSONObject:["sizes":images.map(\.count),"evidence":evidence]);request.append(10)
        try input.write(contentsOf:request)
        for image in images { try input.write(contentsOf:image) }
        let response=try TruForPipe.line(output)
        guard response.hasPrefix("OK "),let count=Int(response.dropFirst(3)),count>0 else { throw ModelInstaller.fail(response.replacingOccurrences(of:"ERROR ",with:"")) }
        if evidence { return .evidence(try TruForPipe.spatial(output,byteCount:count)) }
        guard count<=4096 else { throw ModelInstaller.fail("Oversized TruFor score response") }
        struct Scores:Decodable { let scores:[Float] }
        let values=try JSONDecoder().decode(Scores.self,from:TruForPipe.read(output,count:count)).scores
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
