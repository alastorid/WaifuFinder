import MLX
import MLXNN

public enum SeedVR2ProjectionFusion {
    /// Join compatible output rows without changing the checkpoint's numeric format.
    public static func join(_ layers: [Linear]) -> Linear? {
        guard let first = layers.first, layers.count > 1,
              layers.allSatisfy({ $0.shape.1 == first.shape.1 && $0.weight.dtype == first.weight.dtype && $0.bias == nil }) else { return nil }
        if let q = first as? QuantizedLinear {
            let quantized = layers.compactMap { $0 as? QuantizedLinear }
            guard quantized.count == layers.count,
                  quantized.allSatisfy({ $0.bits == q.bits && $0.groupSize == q.groupSize && $0.mode == q.mode && ($0.biases == nil) == (q.biases == nil) }) else { return nil }
            return QuantizedLinear(weight: concatenated(quantized.map(\.weight), axis: 0),
                scales: concatenated(quantized.map(\.scales), axis: 0),
                biases: q.biases == nil ? nil : concatenated(quantized.compactMap(\.biases), axis: 0),
                groupSize: q.groupSize, bits: q.bits, mode: q.mode)
        }
        guard layers.allSatisfy({ !($0 is QuantizedLinear) }) else { return nil }
        return Linear(weight: concatenated(layers.map(\.weight), axis: 0), bias: nil)
    }
}
