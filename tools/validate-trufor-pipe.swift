// Native, noninteractive production-protocol check. No Python inference or
// temporary input/output files. Usage: swift SCRIPT WORKER MODEL PNG BASELINE_BIN
import Foundation
import CryptoKit
import Darwin

func require(_ value:Bool,_ message:String)throws {
    if !value { throw NSError(domain:"TruFor pipe validation",code:1,userInfo:[NSLocalizedDescriptionKey:message]) }
}
func read(_ handle:FileHandle,_ count:Int)throws->Data {
    var result=Data();result.reserveCapacity(count)
    while result.count<count {
        guard let part=try handle.read(upToCount:min(65_536,count-result.count)),!part.isEmpty else {
            throw NSError(domain:"TruFor pipe validation",code:2,userInfo:[NSLocalizedDescriptionKey:"Unexpected worker EOF"])
        }
        result.append(part)
    }
    return result
}
func line(_ handle:FileHandle)throws->String {
    var result=Data()
    while result.count<8192 {
        let byte=try read(handle,1)
        if byte[0]==10 { return String(decoding:result,as:UTF8.self) }
        result.append(byte)
    }
    throw NSError(domain:"TruFor pipe validation",code:3)
}
func word(_ data:Data,_ offset:Int)->UInt32 { data.withUnsafeBytes { UInt32(littleEndian:$0.loadUnaligned(fromByteOffset:offset,as:UInt32.self)) } }
func value(_ data:Data,_ offset:Int)->Float { data.withUnsafeBytes { $0.loadUnaligned(fromByteOffset:offset,as:Float.self) } }
let args=CommandLine.arguments
try require(args.count==5,"Expected WORKER MODEL PNG BASELINE_BIN")
let png=try Data(contentsOf:URL(fileURLWithPath:args[3]))
let baseline=try Data(contentsOf:URL(fileURLWithPath:args[4]))
var results:[[String:Any]]=[]
for streamed in [false,true] {
    let worker=Process(),incoming=Pipe(),outgoing=Pipe(),errors=Pipe()
    worker.executableURL=URL(fileURLWithPath:"/usr/bin/time")
    worker.arguments=["-l",args[1],"--model",args[2],"--serve-pipe"]
    worker.standardInput=incoming;worker.standardOutput=outgoing;worker.standardError=errors
    var env=ProcessInfo.processInfo.environment
    env["WAIFU_TRUFOR_PRECISION"]="fp32";env["WAIFU_TRUFOR_MEMORY_REPORT"]="1"
    env["WAIFU_TRUFOR_STREAM_ROWS"]=streamed ? "1":"0"
    worker.environment=env
    try worker.run()
    try incoming.fileHandleForReading.close();try outgoing.fileHandleForWriting.close();try errors.fileHandleForWriting.close()
    let input=incoming.fileHandleForWriting,output=outgoing.fileHandleForReading
    try require(try line(output)=="TRF2 FP32 MLX RAM","Worker must attest FP32 MLX RAM protocol")
    func request(_ count:Int,_ evidence:Bool)throws->Data {
        var header=try JSONSerialization.data(withJSONObject:["sizes":Array(repeating:png.count,count:count),"evidence":evidence]);header.append(10)
        try input.write(contentsOf:header)
        for _ in 0..<count { try input.write(contentsOf:png) }
        let response=try line(output)
        try require(response.hasPrefix("OK "),response)
        guard let count=Int(response.dropFirst(3)),count>0,count<=288_000_020 else { throw NSError(domain:"TruFor pipe validation",code:4) }
        return try read(output,count)
    }
    let evidence=try request(1,true)
    try require(evidence.count==baseline.count && evidence.prefix(16)==baseline.prefix(16),"Evidence geometry differs")
    let pixels=Int(word(evidence,4))*Int(word(evidence,8))
    let score=value(evidence,16),referenceScore=value(baseline,16)
    try require(abs(score-referenceScore)<0.001,"Reference score mismatch")
    var planes:[[String:Any]]=[]
    for (plane,name) in ["anomaly","reliability","noiseprint"].enumerated() {
        var maximum:Float=0,squared:Double=0,referenceSquared:Double=0
        for index in 0..<pixels {
            let offset=20+(plane*pixels+index)*4,a=value(evidence,offset),b=value(baseline,offset)
            try require(a.isFinite,"Nonfinite evidence")
            maximum=max(maximum,abs(a-b));squared+=Double(a-b)*Double(a-b);referenceSquared+=Double(b)*Double(b)
        }
        let relative=sqrt(squared/max(referenceSquared,1e-20))
        try require(maximum<0.001 && relative<0.001,"\(name) parity failed: max \(maximum), relative \(relative)")
        planes.append(["name":name,"maximumAbsoluteError":maximum,"relativeRMS":relative])
    }
    let batch=try request(4,false)
    let scores=(try JSONSerialization.jsonObject(with:batch) as! [String:Any])["scores"] as! [Double]
    try require(scores.count==4 && scores.allSatisfy { abs($0-Double(score))<0.00001 },"Repeated score/evidence mismatch")
    // Exercise one more frame on the same warm worker after its batch response.
    let repeated=try request(1,true)
    try require(repeated==evidence,"Repeated evidence differs")
    try input.close();worker.waitUntilExit()
    let log=String(decoding:errors.fileHandleForReading.readDataToEndOfFile(),as:UTF8.self)
    try require(worker.terminationStatus==0,"Worker failed: \(log)")
    let memory=log.split(separator:"\n").compactMap { row -> [String:Any]? in
        guard let data=String(row).data(using:.utf8) else { return nil }
        return (try? JSONSerialization.jsonObject(with:data)) as? [String:Any]
    }
    try require(memory.count>=4,"Missing whole-process memory reports")
    try require(memory.allSatisfy { ($0["peakPhysicalBytes"] as? Int ?? Int.max)<1_700_000_000 },"Physical memory guard failed")
    try require(memory.allSatisfy { ($0["swapinsDuringRequest"] as? Int ?? -1)==0 },"Worker incurred swap-ins")
    results.append(["streamed":streamed,"width":word(evidence,4),"height":word(evidence,8),"score":score,"referenceScore":referenceScore,"planes":planes,"memory":memory,"externalTimeReport":log.split(separator:"\n").filter{!$0.hasPrefix("{")}.joined(separator:"\n")])
}
let manifest=try JSONSerialization.jsonObject(with:Data(contentsOf:URL(fileURLWithPath:args[2]).appendingPathComponent("manifest.json"))) as! [String:Any]
let report:[String:Any]=["status":"passed","date":"2026-10-26",
    "inputSHA256":SHA256.hash(data:png).map{String(format:"%02x",$0)}.joined(),
    "baselineSHA256":SHA256.hash(data:baseline).map{String(format:"%02x",$0)}.joined(),
    "weightsSHA256":(manifest["fileSHA256"] as! [String:String])["model.safetensors"]!,
    "os":ProcessInfo.processInfo.operatingSystemVersionString,"precision":"fp32","backend":"Swift/MLX","transport":"pipes; no temporary image or map files","results":results]
print(String(decoding:try JSONSerialization.data(withJSONObject:report,options:[.prettyPrinted,.sortedKeys]),as:UTF8.self))
