import Foundation

/// Bounded binary transport. Decode final planes directly into their owned RAM
/// arrays instead of retaining a second full evidence blob or mapping a file.
enum TruForPipe {
    static func read(_ handle:FileHandle,count:Int)throws->Data {
        guard count>=0 else { throw ModelInstaller.fail("Invalid TruFor pipe size") }
        var data=Data();data.reserveCapacity(count)
        while data.count<count {
            guard let part=try handle.read(upToCount:min(65_536,count-data.count)),!part.isEmpty else { throw ModelInstaller.fail("TruFor worker stopped or sent an incomplete response") }
            data.append(part)
        }
        return data
    }
    static func line(_ handle:FileHandle)throws->String {
        var data=Data()
        while data.count<8192 {
            let byte=try read(handle,count:1)
            if byte[0]==10 { return String(decoding:data,as:UTF8.self) }
            data.append(byte)
        }
        throw ModelInstaller.fail("Invalid TruFor worker header")
    }
    static func spatial(_ handle:FileHandle,byteCount:Int)throws->TruForSpatial {
        guard byteCount>=20,byteCount<=20+24_000_000*12 else { throw ModelInstaller.fail("Invalid TruFor evidence size") }
        let header=try read(handle,count:20)
        func word(_ offset:Int)->UInt32 { header.withUnsafeBytes { UInt32(littleEndian:$0.loadUnaligned(fromByteOffset:offset,as:UInt32.self)) } }
        let width=Int(word(4)),height=Int(word(8))
        guard header.prefix(4)==Data("TFM1".utf8),width>=32,height>=32,width<=32768,height<=32768,
              width*height<=24_000_000,word(12)==3,byteCount==20+width*height*12 else { throw ModelInstaller.fail("Invalid TruFor evidence geometry") }
        let score=header.withUnsafeBytes { $0.loadUnaligned(fromByteOffset:16,as:Float.self) }
        func plane()throws->[Float] {
            var values=[Float](repeating:0,count:width*height)
            try values.withUnsafeMutableBytes { destination in
                for offset in stride(from:0,to:destination.count,by:65_536) {
                    let count=min(65_536,destination.count-offset)
                    let chunk=try read(handle,count:count)
                    chunk.withUnsafeBytes { source in
                        UnsafeMutableRawBufferPointer(rebasing:destination[offset..<offset+count]).copyMemory(from:source)
                    }
                }
            }
            return values
        }
        let result=TruForSpatial(width:width,height:height,score:score,anomaly:try plane(),reliability:try plane(),noiseprint:try plane())
        try result.validate();return result
    }
}
