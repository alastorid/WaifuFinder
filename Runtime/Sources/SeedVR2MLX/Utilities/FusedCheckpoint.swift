import CryptoKit
import Foundation
import MLX

public enum SeedVR2FusedCheckpoint {
    /// Native, bounded conversion. Each shard contains one block for low-memory
    /// loading. Existing packed q4 values/scales/offsets are reordered, never rounded.
    public static func export(source: URL, destination: URL) throws {
        guard !FileManager.default.fileExists(atPath: destination.path) else { throw SeedVR2VideoCheckpoint.CheckpointError.invalid("Export destination already exists") }
        let checkpoint = try SeedVR2VideoCheckpoint(directory: source)
        guard !checkpoint.config.fusedMLP && !checkpoint.fusedVAE else { throw SeedVR2VideoCheckpoint.CheckpointError.invalid("Checkpoint already fused") }
        try FileManager.default.createDirectory(at: destination.appendingPathComponent("transformer"), withIntermediateDirectories: true)
        var map: [String: String] = [:], aliases: [String: String] = [:], total = 0
        for index in -1..<checkpoint.config.numLayers {
            let prefix = "blocks.\(index)."
            var arrays = try checkpoint.weights("transformer") { index < 0 ? !$0.hasPrefix("blocks.") : $0.hasPrefix(prefix) }
            if index >= checkpoint.config.mmLayers {
                for key in arrays.keys.sorted() where key.contains(".attn.") && key.contains("_txt.") {
                    let canonical = key.replacingOccurrences(of: "_txt.", with: "_vid.")
                    if let value = arrays[key], let shared = arrays[canonical],
                       value.shape == shared.shape, value.dtype == shared.dtype,
                       value.asData(access: .noCopyIfContiguous).data == shared.asData(access: .noCopyIfContiguous).data {
                        aliases[key] = canonical
                        arrays.removeValue(forKey: key)
                    }
                }
            }
            let gates = arrays.keys.filter { $0.hasSuffix(".proj_in_gate.weight") }
            for key in gates {
                for suffix in ["weight", "scales", "biases"] {
                    let gateKey = String(key.dropLast("weight".count)) + suffix
                    let valueKey = gateKey.replacingOccurrences(of: ".proj_in_gate.", with: ".proj_in.")
                    guard let gate = arrays[gateKey], let value = arrays[valueKey],
                          gate.dtype == value.dtype, gate.shape == value.shape else { throw SeedVR2VideoCheckpoint.CheckpointError.invalid("Incompatible packed projection") }
                    arrays[valueKey] = concatenated([gate, value], axis: 0)
                    arrays.removeValue(forKey: gateKey)
                }
            }
            let file = index < 0 ? "root.safetensors" : String(format: "block-%03d.safetensors", index)
            for array in arrays.values { eval(array) }
            try save(arrays: arrays, url: destination.appendingPathComponent("transformer/\(file)"))
            for (key, value) in arrays { map[key] = file; total += value.nbytes }
            Memory.clearCache()
            print("Fused checkpoint shard \(index + 1)/\(checkpoint.config.numLayers)")
        }
        let index: [String: Any] = ["metadata": ["total_size": total], "weight_map": map]
        try JSONSerialization.data(withJSONObject: index, options: [.prettyPrinted, .sortedKeys])
            .write(to: destination.appendingPathComponent("transformer/model.safetensors.index.json"))
        try FileManager.default.createDirectory(at: destination.appendingPathComponent("vae"), withIntermediateDirectories: true)
        var vae = try checkpoint.weights("vae")
        for stream in ["encoder", "decoder"] {
            let prefix = "\(stream).mid_block.attentions.0."
            for suffix in ["weight", "scales", "biases", "bias"] {
                let keys = ["to_q", "to_k", "to_v"].map { prefix + $0 + "." + suffix }
                let values = keys.compactMap { vae[$0] }
                if values.isEmpty && suffix == "bias" { continue }
                guard values.count == 3, values.allSatisfy({ $0.shape == values[0].shape && $0.dtype == values[0].dtype })
                else { throw SeedVR2VideoCheckpoint.CheckpointError.invalid("Incompatible VAE QKV projection") }
                vae[keys[0]] = concatenated(values, axis: 0)
                keys.dropFirst().forEach { vae.removeValue(forKey: $0) }
            }
        }
        for array in vae.values { eval(array) }
        try save(arrays: vae, url: destination.appendingPathComponent("vae/0.safetensors"))
        let vaeIndex: [String: Any] = ["metadata": ["total_size": vae.values.reduce(0) { $0 + $1.nbytes }],
            "weight_map": Dictionary(uniqueKeysWithValues: vae.keys.map { ($0, "0.safetensors") })]
        try JSONSerialization.data(withJSONObject: vaeIndex, options: [.prettyPrinted, .sortedKeys])
            .write(to: destination.appendingPathComponent("vae/model.safetensors.index.json"))
        try FileManager.default.copyItem(at: source.appendingPathComponent("pos_emb.safetensors"), to: destination.appendingPathComponent("pos_emb.safetensors"))
        var manifest = try JSONSerialization.jsonObject(with: Data(contentsOf: source.appendingPathComponent("manifest.json"))) as! [String: Any]
        manifest["fusion"] = "packed-projections-v1"
        manifest["vaeFusion"] = "qkv-packed-v1"
        manifest["transformerAliases"] = aliases
        var hashes: [String: String] = [:]
        let files = FileManager.default.enumerator(at: destination, includingPropertiesForKeys: [.isRegularFileKey])!
        for case let file as URL in files where (try file.resourceValues(forKeys: [.isRegularFileKey])).isRegularFile == true {
            let handle = try FileHandle(forReadingFrom: file)
            var hash = SHA256()
            while let data = try handle.read(upToCount: 8 * 1024 * 1024), !data.isEmpty { hash.update(data: data) }
            try handle.close()
            hashes[String(file.path.dropFirst(destination.path.count + 1))] = hash.finalize().map { String(format: "%02x", $0) }.joined()
        }
        manifest["fileSHA256"] = hashes
        try JSONSerialization.data(withJSONObject: manifest, options: [.prettyPrinted, .sortedKeys])
            .write(to: destination.appendingPathComponent("manifest.json"))
    }
}
