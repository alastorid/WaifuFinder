import AppKit
import Foundation
import ImageIO
import UniformTypeIdentifiers

struct SafireConfiguration: Sendable {
    let directory: URL
    let fingerprint: String
    static let sourceSHA256 = "65bdf6880380162d57f77d71b3fd367c3f16aebca316f942a821da16e9225b12"
    static var pointerURL: URL { SeedVR2Configuration.configurationURL.deletingLastPathComponent().appendingPathComponent("safire-model.txt") }
    static var executable: URL? { Bundle.main.url(forResource:"waifu-safire",withExtension:nil,subdirectory:"Runtime") }
    static var availableDirectory: URL? {
        guard let data=try? Data(contentsOf:pointerURL), let text=String(data:data,encoding:.utf8)?.trimmingCharacters(in:.whitespacesAndNewlines),!text.isEmpty else { return nil }
        return URL(fileURLWithPath:text)
    }
    static func load() -> Self? {
        guard let directory=availableDirectory,let executable,FileManager.default.isExecutableFile(atPath:executable.path) else { return nil }
        return try? validate(directory)
    }
    static func validate(_ directory:URL) throws -> Self {
        let architecture=try JSONSerialization.jsonObject(with:Data(contentsOf:directory.appendingPathComponent("architecture.json"))) as? [String:Any]
        let validation=try JSONSerialization.jsonObject(with:Data(contentsOf:directory.appendingPathComponent("validation.json"))) as? [String:Any]
        let packageRevision=architecture?["runtimeRevision"] as? String
        let validationRevision=validation?["runtimeRevision"] as? String
        let supportedCheckpointRevisions:Set<String>=["safire-native-v1",SafireSpatialResult.runtimeRevision]
        guard architecture?["architecture"] as? String == "safire-vit-b-adaptor",
              architecture?["sourceSHA256"] as? String == sourceSHA256,
              packageRevision == validationRevision,
              packageRevision.map(supportedCheckpointRevisions.contains) == true,
              architecture?["preprocessingVersion"] as? String == SafireSpatialResult.preprocessingVersion,
              validation?["status"] as? String == "passed",
              validation?["sourceSHA256"] as? String == sourceSHA256,
              validation?["spatialParityPassed"] as? Bool == true else {
            throw ModelInstaller.fail("This SAFIRE package has not passed native reference and spatial validation. Install a validated release from Models.")
        }
        let hashes=try ModelInstaller.hashes(directory)
        guard let fingerprint=hashes["model.safetensors"],ModelInstaller.safeFile("model.safetensors",hash:fingerprint) else {
            throw ModelInstaller.fail("Missing SAFIRE checkpoint fingerprint.")
        }
        return .init(directory:directory,fingerprint:fingerprint)
    }
}

/// Warm native worker shared by local and remote callers. A dedicated actor
/// serializes requests; cancellation terminates the process and discards its lane.
final class SafireWorker: @unchecked Sendable {
    private let model:URL
    private let executable:URL
    private let root:URL
    private let lock=NSLock()
    private var process:Process?
    private var input:FileHandle?
    private var output:FileHandle?
    private var cancelled=false
    init(model:URL,executable:URL) throws {
        self.model=model;self.executable=executable
        root=FileManager.default.temporaryDirectory.appendingPathComponent("safire-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true,attributes:[.posixPermissions:0o700])
    }
    deinit { cancel();try? FileManager.default.removeItem(at:root) }
    func cancel() { lock.withLock { cancelled=true;if let process,process.isRunning { process.terminate() } } }
    private func line() throws -> String {
        guard let output else { throw ModelInstaller.fail("SAFIRE worker has no response stream.") }
        var data=Data()
        while data.count < 8192 {
            guard let byte=try output.read(upToCount:1),!byte.isEmpty else { throw ModelInstaller.fail("SAFIRE worker stopped. Retry the analysis.") }
            if byte[0]==10 { return String(decoding:data,as:UTF8.self) };data.append(byte)
        }
        throw ModelInstaller.fail("Invalid SAFIRE worker response.")
    }
    private func start() throws {
        if process != nil { return }
        let process=Process(),incoming=Pipe(),outgoing=Pipe()
        process.executableURL=executable;process.arguments=["--model",model.path,"--serve"]
        process.standardInput=incoming;process.standardOutput=outgoing
        let logURL=root.appendingPathComponent("worker.log")
        FileManager.default.createFile(atPath:logURL.path,contents:nil)
        process.standardError=try FileHandle(forWritingTo:logURL)
        try lock.withLock {
            guard !cancelled else { throw CancellationError() }
            self.process=process;try process.run()
        }
        input=incoming.fileHandleForWriting;output=outgoing.fileHandleForReading
        try incoming.fileHandleForReading.close();try outgoing.fileHandleForWriting.close()
        guard try line()=="SAF1" else { throw ModelInstaller.fail("Incompatible SAFIRE worker.") }
    }
    func infer(_ png:Data,photo:LineagePhoto,configuration:SafireConfiguration) throws -> SafireSpatialResult {
        try start();try Task.checkCancellation()
        let inputURL=root.appendingPathComponent("input.png"),outputURL=root.appendingPathComponent("result.json")
        defer { try? FileManager.default.removeItem(at:inputURL);try? FileManager.default.removeItem(at:outputURL) }
        try png.write(to:inputURL,options:.atomic)
        var request=try JSONSerialization.data(withJSONObject:["input":inputURL.path,"output":outputURL.path,"format":"sfm1"]);request.append(10)
        try input?.write(contentsOf:request)
        guard try line()=="OK" else { throw ModelInstaller.fail("Native SAFIRE analysis failed. Retry or verify the model in Settings → Models.") }
        try Task.checkCancellation()
        let raw=try SafireRasterCodec.decode(Data(contentsOf:outputURL))
        let result=SafireSpatialResult(assetID:photo.id,contentIdentity:photo.contentIdentity,modelFingerprint:configuration.fingerprint,
            runtimeRevision:SafireSpatialResult.runtimeRevision,preprocessingVersion:SafireSpatialResult.preprocessingVersion,
            width:raw.width,height:raw.height,probabilities:raw.probabilities,partitions:raw.partitions)
        try result.validate();return result
    }
}

actor SafireBackend {
    static let shared=SafireBackend()
    private var worker:SafireWorker?
    private var current:SafireConfiguration?
    private var busy=false
    private var idle:Task<Void,Never>?
    func release() async {
        guard !busy else { return }
        worker?.cancel();worker=nil
        if current != nil { current=nil;await ModelLibrary.shared.releaseSafire() }
    }
    func infer(png:Data,photo:LineagePhoto) async throws -> SafireSpatialResult {
        guard !busy else { throw RemoteError.busy }
        busy=true;idle?.cancel()
        defer {
            busy=false
            idle=Task { [weak self] in do { try await Task.sleep(for:.seconds(60));await self?.release() } catch {} }
        }
        guard let config=SafireConfiguration.load(),let executable=SafireConfiguration.executable else {
            throw ModelInstaller.fail("Install a validated SAFIRE native checkpoint in Settings → Models.")
        }
        if current?.fingerprint != config.fingerprint { worker?.cancel();worker=nil;if current != nil { await ModelLibrary.shared.releaseSafire() };current=nil }
        if worker==nil {
            guard await ModelLibrary.shared.active==nil else { throw RemoteError.busy }
            await ModelLibrary.shared.acquireSafire()
            do { worker=try SafireWorker(model:config.directory,executable:executable);current=config }
            catch { await ModelLibrary.shared.releaseSafire();throw error }
        }
        guard let worker else { throw RemoteError.busy }
        do {
            return try await withTaskCancellationHandler {
                try await Task.detached(priority:.utility) { try worker.infer(png,photo:photo,configuration:config) }.value
            } onCancel: { worker.cancel() }
        } catch {
            worker.cancel();self.worker=nil;current=nil;await ModelLibrary.shared.releaseSafire();throw error
        }
    }
    static func png(_ image:CGImage) throws -> Data {
        let data=NSMutableData()
        guard let destination=CGImageDestinationCreateWithData(data,UTType.png.identifier as CFString,1,nil) else { throw CrystalOrbImageSource.failure }
        CGImageDestinationAddImage(destination,image,nil)
        guard CGImageDestinationFinalize(destination) else { throw CrystalOrbImageSource.failure }
        return data as Data
    }
}

/// Lossless PNG plus explicit asset/content identity. No JPEG re-encoding and no
/// Photos identifiers reconstructed from a local filesystem path on the server.
struct RemoteSafireRequest: Codable, Sendable {
    let photo:LineagePhoto
    let png:Data
    let modelFingerprint:String
    func validate() throws {
        guard png.count >= 8,png.count <= 32*1024*1024,
              png.prefix(8) == Data([137,80,78,71,13,10,26,10]),
              photo.width>0,photo.height>0,photo.width<=32768,photo.height<=32768,
              photo.url.scheme == "waifu-photos",!photo.id.isEmpty,photo.contentIdentity.count==64,photo.contentIdentity.allSatisfy(\.isHexDigit),
              modelFingerprint.count==64 else { throw RemoteError.invalidRequest("Invalid lossless SAFIRE request.") }
        guard let source=CGImageSourceCreateWithData(png as CFData,[kCGImageSourceShouldCache:false] as CFDictionary),
              let props=CGImageSourceCopyPropertiesAtIndex(source,0,nil) as? [CFString:Any],
              let w=props[kCGImagePropertyPixelWidth] as? Int,let h=props[kCGImagePropertyPixelHeight] as? Int,
              w>0,h>0,w<=32768,h<=32768,w*h<=160_000_000 else { throw RemoteError.invalidRequest("Image exceeds the forensic decode limit.") }
    }
}

extension RemoteServerSession {
    func handleSafire(_ envelope:RemoteEnvelope) async {
        do {
            let request=try JSONDecoder().decode(RemoteSafireRequest.self,from:envelope.data)
            try request.validate()
            let result=try await catalog.safire(request)
            try Task.checkCancellation()
            try await connection.send(.safireResult,payload:[UInt8](result.encodedSpatialData()),streamID:envelope.streamID)
        } catch {
            if !Task.isCancelled { try? await connection.sendError(envelope,Self.describe(error)) }
        }
        await resources.finish(streamID:envelope.streamID)
    }
}

extension RemoteClient {
    var offersSafire:Bool { status.isConnected && capabilities?.safire?.available==true }
    func safire(png:Data,photo:LineagePhoto) async throws -> SafireSpatialResult {
        guard offersSafire,let capability=capabilities?.safire,let connection else { throw RemoteError.modelUnavailable("No SAFIRE worker is connected.") }
        let request=RemoteSafireRequest(photo:photo,png:png,modelFingerprint:capability.modelFingerprint)
        try request.validate()
        let response=try await connection.request(.safireRequest,payload:[UInt8](JSONEncoder().encode(request)))
        guard response.opcode == .safireResult else { throw RemoteError.framing("Unexpected SAFIRE response") }
        let result=try SafireSpatialResult.decodeSpatialData(response.data)
        try result.validate()
        guard result.assetID==photo.id,result.contentIdentity==photo.contentIdentity,result.modelFingerprint==capability.modelFingerprint else { throw RemoteError.framing("SAFIRE response identity mismatch") }
        return result
    }
}
