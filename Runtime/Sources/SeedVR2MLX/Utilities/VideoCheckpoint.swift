import Foundation
import MLX
import MLXNN

/// Select lazy safetensors keys before evaluation: only the requested block is read.
public final class SeedVR2VideoCheckpoint {
    public let directory: URL
    public let config: SeedVR2Config
    public let variant: String
    public let fusedVAE: Bool
    private let maps: [String: [String: String]]
    private let transformerAliases: [String: String]
    public init(directory: URL) throws {
        self.directory = directory
        let url = directory.appendingPathComponent("manifest.json")
        let manifest = FileManager.default.fileExists(atPath: url.path)
            ? try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any] ?? [:] : [:]
        variant = manifest["algorithm"] as? String ?? "seedvr2-3b"
        guard ["seedvr2-3b", "seedvr2-7b"].contains(variant) else { throw CheckpointError.invalid("Unsupported architecture") }
        var cfg = variant == "seedvr2-7b" ? SeedVR2Config.r7B : .r3B
        if let fusion = manifest["fusion"] as? String {
            guard ["swiglu-packed-v1", "packed-projections-v1"].contains(fusion) else { throw CheckpointError.invalid("Unsupported fusion layout") }
            cfg.fusedMLP = variant == "seedvr2-3b"
            // 7B's GELU/residual range can overflow F16. Keep its transformer
            // at F32 while both architectures use the matched F16 video VAE.
            cfg.halfPrecision = variant == "seedvr2-3b" && fusion == "packed-projections-v1"
        }
        let vaeFusion = manifest["vaeFusion"] as? String
        guard vaeFusion == nil || vaeFusion == "qkv-packed-v1" else { throw CheckpointError.invalid("Unsupported VAE fusion layout") }
        fusedVAE = vaeFusion != nil
        guard (manifest["groupSize"] as? Int ?? 64) == 64,
              (manifest["precision"] as? String ?? "mlx-4bit") == "mlx-4bit" else { throw CheckpointError.invalid("Expected q4/group64 weights") }
        var result: [String: [String: String]] = [:]
        for name in ["transformer", "vae"] {
            let raw = try JSONSerialization.jsonObject(with: Data(contentsOf: directory.appendingPathComponent("\(name)/model.safetensors.index.json"))) as? [String: Any]
            guard let map = raw?["weight_map"] as? [String: String], !map.isEmpty,
                  map.values.allSatisfy({ !$0.contains("/") && !$0.contains("\\") && $0.hasSuffix(".safetensors") }) else { throw CheckpointError.invalid("Invalid weight index") }
            result[name] = map
        }
        let aliases = manifest["transformerAliases"] as? [String: String] ?? [:]
        for (alias, canonical) in aliases {
            let parts = alias.split(separator: ".")
            guard parts.count >= 4, parts[0] == "blocks", let index = Int(parts[1]),
                  index >= cfg.mmLayers, index < cfg.numLayers, parts[2] == "attn",
                  alias.contains("_txt."), canonical == alias.replacingOccurrences(of: "_txt.", with: "_vid."),
                  result["transformer"]?[alias] == nil, result["transformer"]?[canonical] != nil,
                  aliases[canonical] == nil else { throw CheckpointError.invalid("Invalid shared-weight alias") }
        }
        transformerAliases = aliases
        let sharedKeys = ["proj_qkv_txt.weight", "proj_qkv_txt.scales", "proj_qkv_txt.biases",
            "proj_out_txt.weight", "proj_out_txt.scales", "proj_out_txt.biases", "proj_out_txt.bias"]
        cfg.sharedAttentionFusion = variant == "seedvr2-3b" && (cfg.mmLayers..<cfg.numLayers).allSatisfy { index in
            sharedKeys.allSatisfy { aliases["blocks.\(index).attn." + $0] != nil }
        }
        config = cfg
        maps = result
    }
    public func weights(_ name: String, matching predicate: (String) -> Bool = { _ in true }) throws -> [String: MLXArray] {
        guard let map = maps[name] else { throw CheckpointError.invalid("Unknown component") }
        var selected = map.filter { predicate($0.key) }
        let aliases = name == "transformer" ? transformerAliases.filter { predicate($0.key) } : [:]
        for canonical in aliases.values { selected[canonical] = map[canonical] }
        var arrays: [String: MLXArray] = [:]
        for file in Set(selected.values).sorted() {
            let lazy = try loadArrays(url: directory.appendingPathComponent("\(name)/\(file)"))
            for key in selected.keys where selected[key] == file {
                guard let value = lazy[key] else { throw CheckpointError.invalid("Missing indexed parameter \(key)") }
                arrays[key] = value
            }
        }
        for (alias, canonical) in aliases { arrays[alias] = arrays[canonical] }
        return arrays.filter { predicate($0.key) }
    }
    public func apply(_ arrays: [String: MLXArray], to module: Module, verifyAll: Bool = true) throws {
        let replacements: [(String, Module)] = module.leafModules().flattened().compactMap { path, layer in
            guard let scales = arrays[path + ".scales"], let weight = arrays[path + ".weight"], layer is Linear else { return nil }
            return (path, QuantizedLinear(weight: weight, bias: arrays[path + ".bias"], scales: scales,
                biases: arrays[path + ".biases"], groupSize: 64, bits: 4))
        }
        module.update(modules: ModuleChildren.unflattened(replacements))
        try module.update(parameters: ModuleParameters.unflattened(arrays), verify: verifyAll ? [.all] : [.noUnusedKeys, .shapeMismatch])
        // Submit a whole bounded component/block together instead of waiting on
        // every parameter separately. All these weights remain live either way.
        eval(Array(arrays.values))
        Memory.clearCache()
    }
    public func loadBlock(_ index: Int, into block: TransformerBlock) throws {
        let prefix = "blocks.\(index)."
        let arrays = try weights("transformer") { $0.hasPrefix(prefix) }
        let local = Dictionary(uniqueKeysWithValues: arrays.map { (String($0.key.dropFirst(prefix.count)), $0.value) })
        try apply(local, to: block)
    }
    public func estimatedWorkingSet(streamed: Bool) throws -> Int {
        let vaeBytes = try weights("vae").values.reduce(0) { $0 + $1.nbytes }
        let transformer = try weights("transformer").filter { transformerAliases[$0.key] == nil }
        let rootBytes = transformer.filter { !$0.key.hasPrefix("blocks.") }.values.reduce(0) { $0 + $1.nbytes }
        var blocks: [Int: Int] = [:]
        for (key, value) in transformer {
            let parts = key.split(separator: ".")
            if parts.first == "blocks", parts.count > 1, let index = Int(parts[1]) { blocks[index, default: 0] += value.nbytes }
        }
        let weightBytes = streamed ? max(vaeBytes, rootBytes + (blocks.values.max() ?? 0))
            : vaeBytes + transformer.values.reduce(0) { $0 + $1.nbytes }
        return weightBytes + 2_700_000_000
    }
    enum CheckpointError: Error { case invalid(String) }
}
