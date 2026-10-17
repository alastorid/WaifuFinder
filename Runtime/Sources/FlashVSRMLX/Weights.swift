import CryptoKit
import Foundation
import MLX
import MLXNN

public enum FlashError: Error, CustomStringConvertible {
    case invalid(String)
    public var description: String {
        switch self {
        case .invalid(let s): return s
        }
    }
}

final class FlashWeights {
    let arrays: [String: MLXArray]
    init(directory: URL, component: String) throws {
        let manifest =
            try JSONSerialization.jsonObject(
                with: Data(contentsOf: directory.appendingPathComponent("manifest.json"))) as? [String: Any]
        guard manifest?["format"] as? Int == 1,
            manifest?["revision"] as? String == "27561b186ded3402d7c975f4fd722e2885b6135f",
            manifest?["precision"] as? String == "bfloat16",
            let map = manifest?["weightMap"] as? [String: [String: Any]]
        else { throw FlashError.invalid("Incompatible FlashVSR weight manifest") }
        guard let hashes = manifest?["fileSHA256"] as? [String: String] else {
            throw FlashError.invalid("Missing checkpoint checksums")
        }
        let expectedKeys = Set(
            ["dit", "projection", "decoder"].flatMap { part in
                Self.shapes(part).keys.map { part + ":" + $0 }
            })
        guard Set(map.keys) == expectedKeys,
            Set(map.values.compactMap { $0["file"] as? String }).union(["prompt.safetensors"])
                == Set(hashes.keys),
            hashes.count == 34
        else { throw FlashError.invalid("Unexpected or incomplete FlashVSR manifest coverage") }
        let entries = map.filter { $0.key.hasPrefix(component + ":") }
        let files = Set(entries.values.compactMap { $0["file"] as? String })
        var loaded: [String: MLXArray] = [:]
        for file in files.sorted() {
            guard !file.contains("/") && !file.contains("..") else {
                throw FlashError.invalid("Invalid checkpoint path")
            }
            guard let expected = hashes[file],
                try Self.digest(directory.appendingPathComponent(file)) == expected
            else { throw FlashError.invalid("Checkpoint checksum mismatch: \(file)") }
            for (key, array) in try loadArrays(url: directory.appendingPathComponent(file)) {
                guard loaded[key] == nil else { throw FlashError.invalid("Duplicate weight: \(key)") }
                loaded[key] = array
                eval(array)
            }
            Memory.clearCache()
        }
        guard Set(entries.keys.map { String($0.dropFirst(component.count + 1)) }) == Set(loaded.keys) else {
            throw FlashError.invalid("Incomplete checkpoint parameter coverage")
        }
        let shapes = Self.shapes(component)
        guard Set(shapes.keys) == Set(loaded.keys) else {
            throw FlashError.invalid("Missing or unexpected \(component) parameters")
        }
        for (key, shape) in shapes {
            guard loaded[key]?.shape == shape && loaded[key]?.dtype == .bfloat16 else {
                throw FlashError.invalid("Incompatible shape or precision: \(key)")
            }
        }
        arrays = loaded
    }
    subscript(_ key: String) -> MLXArray { arrays[key]! }
    static func digest(_ url: URL) throws -> String {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        var hash = SHA256()
        while try autoreleasepool(invoking: { () throws -> Bool in
            guard let data = try handle.read(upToCount: 8 * 1024 * 1024), !data.isEmpty else { return false }
            hash.update(data: data)
            return true
        }) {}
        return hash.finalize().map { String(format: "%02x", $0) }.joined()
    }
    func linear(_ x: MLXArray, _ name: String) -> MLXArray {
        if let bias = arrays[name + ".bias"] { return addMM(bias, x, self[name + ".weight"].T) }
        return matmul(x, self[name + ".weight"].T)
    }
    func conv2(_ x: MLXArray, _ name: String, padding: Int = 1) -> MLXArray {
        var y = conv2d(x, self[name + ".weight"], padding: IntOrPair(padding))
        if let bias = arrays[name + ".bias"] { y = y + bias }
        return y
    }
    static func shapes(_ component: String) -> [String: [Int]] {
        var d: [String: [Int]] = [:]
        func linear(_ name: String, _ inputs: Int, _ outputs: Int, _ bias: Bool = true) {
            d[name + ".weight"] = [outputs, inputs]
            if bias { d[name + ".bias"] = [outputs] }
        }
        func conv(_ name: String, _ inputs: Int, _ outputs: Int, _ kernel: Int = 3, _ bias: Bool = true) {
            d[name + ".weight"] = [outputs, kernel, kernel, inputs]
            if bias { d[name + ".bias"] = [outputs] }
        }
        if component == "dit" {
            for i in 0..<30 {
                let p = "blocks.\(i)."
                for a in ["self_attn", "cross_attn"] {
                    for op in ["q", "k", "v", "o"] { linear(p + a + "." + op, 1536, 1536) }
                    for n in ["norm_q", "norm_k"] { d[p + a + "." + n + ".weight"] = [1536] }
                }
                linear(p + "ffn.0", 1536, 8960)
                linear(p + "ffn.2", 8960, 1536)
                d[p + "modulation"] = [1, 6, 1536]
                d[p + "norm3.weight"] = [1536]
                d[p + "norm3.bias"] = [1536]
            }
            d["patch_embedding.weight"] = [1536, 1, 2, 2, 16]
            d["patch_embedding.bias"] = [1536]
            linear("text_embedding.0", 4096, 1536)
            linear("text_embedding.2", 1536, 1536)
            linear("time_embedding.0", 256, 1536)
            linear("time_embedding.2", 1536, 1536)
            linear("time_projection.1", 1536, 9216)
            linear("head.head", 1536, 64)
            d["head.modulation"] = [1, 2, 1536]
        } else if component == "projection" {
            d["conv1.weight"] = [2048, 4, 3, 3, 768]
            d["conv1.bias"] = [2048]
            d["conv2.weight"] = [3072, 4, 3, 3, 2048]
            d["conv2.bias"] = [3072]
            d["norm1.gamma"] = [2048, 1, 1, 1]
            d["norm2.gamma"] = [3072, 1, 1, 1]
            linear("linear_layers.0", 3072, 1536)
        } else {
            conv("decoder.1", 784, 512)
            conv("decoder.3", 512, 512, 3, false)
            for (i, c) in [
                (5, 512), (6, 512), (7, 512), (11, 256), (12, 256), (13, 256), (17, 128), (18, 128),
                (19, 128),
            ] {
                for (n, a) in [(0, c * 2), (2, c), (4, c)] { conv("decoder.\(i).conv.\(n)", a, c) }
            }
            for (i, a, b) in [(9, 512, 512), (15, 256, 512), (21, 128, 256)] {
                conv("decoder.\(i).conv", a, b, 1, false)
            }
            for (i, a, b) in [(10, 512, 256), (16, 256, 128), (22, 128, 128), (24, 128, 128)] {
                conv("decoder.\(i)", a, b, 3, false)
            }
            conv("decoder.26", 128, 3)
        }
        return d
    }
}

func flashEval(_ arrays: MLXArray...) {
    eval(arrays)
    Memory.clearCache()
}
func flashGELU(_ x: MLXArray) -> MLXArray { MLXNN.geluApproximate(x.asType(.float32)).asType(x.dtype) }
func flashSiLU(_ x: MLXArray) -> MLXArray { MLXNN.silu(x.asType(.float32)).asType(x.dtype) }
func rms(_ x: MLXArray, _ weight: MLXArray) -> MLXArray {
    let value = x.asType(.float32)
    return (value * rsqrt(mean(value * value, axis: -1, keepDims: true) + Float(1e-6))).asType(x.dtype)
        * weight
}
func norm(_ x: MLXArray, weight: MLXArray? = nil, bias: MLXArray? = nil) -> MLXArray {
    MLXFast.layerNorm(x, weight: weight, bias: bias, eps: 1e-6)
}
