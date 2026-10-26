// Native-resolution TruFor working-set policy. No additional dependencies.
import Darwin
import Foundation
public enum TruForMemoryPolicy {
    public static let budgetBytes=2_000_000_000
    // Reserve space for the current Metal allocation and framework bookkeeping.
    public static let stopBytes=1_700_000_000
    public static let maximumPixels=24_000_000
    /// Bounds the largest row-wise activation chunk. Larger values reduce MLX
    /// dispatch and host-copy overhead; keep experimental choices explicit.
    public static let rowPositions:Int={
        let value=Int(ProcessInfo.processInfo.environment["WAIFU_TRUFOR_ROW_POSITIONS"] ?? "16384") ?? 16384
        return [8_192,16_384,32_768,65_536].contains(value) ? value:16_384
    }()
    public static let rowElements=rowPositions*64
    /// CPU staging arrays can otherwise remain in libmalloc's free-page cache,
    /// outside MLX's active/cache counters, across many photo geometries.
    public static func releaseUnusedCPU() { _ = malloc_zone_pressure_relief(nil,0) }
    public static func validate(width:Int,height:Int) throws {
        guard width>=32,height>=32,width<=32768,height<=32768 else {
            throw TruForError.invalid("Unsupported native TruFor image dimensions")
        }
        guard width*height<=maximumPixels else {
            throw TruForError.invalid("Unsupported native TruFor image geometry (original 24 megapixel envelope); images are never silently resized.")
        }
    }
}
