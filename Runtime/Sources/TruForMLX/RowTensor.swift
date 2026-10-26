import Foundation
import Darwin
import MLX

/// Owned FP32 chunks in RAM. Consumed chunks are released as each operator
/// advances. There is no backing file, memory mapping, compression or spill.
final class TruForRows {
    let height:Int,width:Int,channels:Int
    private struct Chunk { let first:Int;let end:Int;var values:[Float]? }
    private var chunks:[Chunk]=[]
    private var written=0
    init(height:Int,width:Int,channels:Int)throws {
        self.height=height;self.width=width;self.channels=channels
    }
    var stripRows:Int { max(1,min(TruForMemoryPolicy.rowPositions/width,TruForMemoryPolicy.rowElements/(width*channels))) }
    func discard(before row:Int) {
        for i in chunks.indices where chunks[i].end<=row { chunks[i].values=nil }
    }
    func read(_ rows:Range<Int>)throws->MLXArray { try read(rows,columns:0..<width) }
    func read(_ rows:Range<Int>,columns:Range<Int>)throws->MLXArray {
        guard rows.lowerBound>=0,rows.upperBound<=written,columns.lowerBound>=0,columns.upperBound<=width else { throw TruForError.invalid("Invalid TruFor RAM region") }
        var result=[Float](repeating:0,count:rows.count*columns.count*channels)
        var copied=0
        try result.withUnsafeMutableBufferPointer { output in
            for chunk in chunks where chunk.first<rows.upperBound && chunk.end>rows.lowerBound {
                guard let values=chunk.values else { throw TruForError.invalid("TruFor RAM rows were already consumed") }
                values.withUnsafeBufferPointer { source in
                    for y in max(rows.lowerBound,chunk.first)..<min(rows.upperBound,chunk.end) {
                        let count=columns.count*channels
                        output.baseAddress!.advanced(by:(y-rows.lowerBound)*count).update(from:source.baseAddress!.advanced(by:((y-chunk.first)*width+columns.lowerBound)*channels),count:count)
                        copied+=count
                    }
                }
            }
        }
        guard copied==result.count else { throw TruForError.invalid("Incomplete TruFor RAM region") }
        return MLXArray(result,[1,rows.count,columns.count,channels])
    }
    func append(_ x:MLXArray)throws {
        guard x.ndim==4,x.dim(0)==1,x.dim(2)==width,x.dim(3)==channels,written+x.dim(1)<=height else { throw TruForError.invalid("Invalid TruFor RAM tensor shape") }
        let values=x.asType(.float32).asArray(Float.self)
        chunks.append(Chunk(first:written,end:written+x.dim(1),values:values));written+=x.dim(1)
        TruForMemoryPolicy.releaseUnusedCPU()
    }
}

/// Takes ownership of its own storage choice. Callers can release the decoded
/// MLX input before inference starts, rather than pinning a full RGB buffer.
public final class TruForInput {
    public let width:Int,height:Int
    let array:MLXArray?
    let rows:TruForRows?
    private var rgba:[UInt8]?
    public init(rgba:[UInt8],width:Int,height:Int)throws {
        try TruForMemoryPolicy.validate(width:width,height:height)
        guard rgba.count==width*height*4 else { throw TruForError.invalid("Incomplete native RGBA image") }
        self.width=width;self.height=height
        let streamed=Memory.activeMemory+width*height*396>1_000_000_000 || ProcessInfo.processInfo.environment["WAIFU_TRUFOR_STREAM_ROWS"]=="1"
        if streamed {
            array=nil;rows=nil;self.rgba=rgba
        } else {
            let input=try TruForPreprocessing.rgba(rgba,width:width,height:height);eval(input)
            array=input;rows=nil;self.rgba=nil
        }
    }
    public init(_ rgb:MLXArray)throws {
        guard rgb.ndim==4,rgb.dim(0)==1,rgb.dim(3)==3 else { throw TruForError.invalid("Expected one native RGB image") }
        width=rgb.dim(2);height=rgb.dim(1);rgba=nil
        try TruForMemoryPolicy.validate(width:width,height:height)
        let streamed=Memory.activeMemory+width*height*384>1_000_000_000 || ProcessInfo.processInfo.environment["WAIFU_TRUFOR_STREAM_ROWS"]=="1"
        if streamed {
            let storage=try TruForRows(height:height,width:width,channels:3)
            for y in stride(from:0,to:height,by:storage.stripRows) {
                try storage.append(rgb[0...,y..<min(height,y+storage.stripRows),0...,0...])
            }
            array=nil;rows=storage
        } else { array=rgb;rows=nil }
    }
    var usesRows:Bool { array==nil }
    func releasePixels() { rgba=nil;rows?.discard(before:height) }
    public func materialize()throws->MLXArray { try read(0..<height) }
    func read(_ range:Range<Int>)throws->MLXArray { try read(range,columns:0..<width) }
    func read(_ y:Range<Int>,columns:Range<Int>)throws->MLXArray {
        if let array { return array[0...,y,columns,0...] }
        if let rows { return try rows.read(y,columns:columns) }
        guard let rgba else { throw TruForError.invalid("TruFor input pixels were already consumed") }
        var bytes=[UInt8]();bytes.reserveCapacity(y.count*columns.count*4)
        for row in y {
            bytes.append(contentsOf:rgba[((row*width+columns.lowerBound)*4)..<((row*width+columns.upperBound)*4)])
        }
        return TruForPreprocessing.strip(bytes,width:columns.count,height:y.count)
    }
}
