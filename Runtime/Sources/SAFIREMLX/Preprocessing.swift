// OpenCV-compatible 11-bit linear interpolation and SAM normalization.
// Algorithm reference: OpenCV resize.cpp, Intel/Willow Garage/Itseez (BSD-3).
// See SAFIRE-NOTICE.md and SAFIRE-OPENCV-LICENSE.
import Foundation
import MLX

public enum SAFIREPreprocessing {
    private static let kernel=MLXFast.metalKernel(name:"safire_rgba_linear_normalize",inputNames:["rgba","xs","ys","alpha","beta"],outputNames:["out"],source:"""
        uint i = thread_position_in_grid.x;
        if (i >= 1024u * 1024u * 3u) return;
        uint c = i % 3u, x = (i / 3u) % 1024u, y = i / (3u * 1024u);
        int a0 = alpha[x*2], a1 = alpha[x*2+1];
        int b0 = beta[y*2], b1 = beta[y*2+1];
        int row0 = ys[y*2] * W, row1 = ys[y*2+1] * W;
        int s0 = int(rgba[(row0+xs[x*2])*4+c])*a0 + int(rgba[(row0+xs[x*2+1])*4+c])*a1;
        int s1 = int(rgba[(row1+xs[x*2])*4+c])*a0 + int(rgba[(row1+xs[x*2+1])*4+c])*a1;
        int pixel = ((((b0*(s0>>4))>>16) + ((b1*(s1>>4))>>16) + 2)>>2);
        const float mean[3] = {123.675f,116.28f,103.53f};
        const float stddev[3] = {58.395f,57.12f,57.375f};
        out[i] = (float(pixel)-mean[c])/stddev[c];
        """)
    private static func coordinates(_ size:Int) -> ([Int32],[Int32]) {
        var indices:[Int32]=[],coefficients:[Int32]=[]
        for i in 0..<1024 {
            let position=Float((Double(i)+0.5)*Double(size)/1024-0.5)
            let base=Int(floor(position)),fraction=position-Float(base)
            indices.append(Int32(max(0,min(size-1,base))))
            indices.append(Int32(max(0,min(size-1,base+1))))
            coefficients.append(Int32(((1-fraction)*2048).rounded(.toNearestOrEven)))
            coefficients.append(Int32((fraction*2048).rounded(.toNearestOrEven)))
        }
        return (indices,coefficients)
    }
    public static func rgba(_ bytes:[UInt8],width:Int,height:Int) throws -> MLXArray {
        guard width>0,height>0,width<=32768,height<=32768,bytes.count==width*height*4 else { throw SAFIREError.invalid("Invalid RGBA input") }
        let (xs,a)=coordinates(width),(ys,b)=coordinates(height)
        return kernel([MLXArray(bytes),MLXArray(xs),MLXArray(ys),MLXArray(a),MLXArray(b)],template:[("W",width)],
            grid:(1024*1024*3,1,1),threadGroup:(256,1,1),outputShapes:[[1,1024,1024,3]],outputDTypes:[.float32])[0]
    }
    /// Scalar oracle for the fused Metal conversion, independent of GPU dispatch.
    public static func reference(_ bytes:[UInt8],width:Int,height:Int) -> [Float] {
        let (xs,a)=coordinates(width),(ys,b)=coordinates(height)
        let mean:[Float]=[123.675,116.28,103.53],std:[Float]=[58.395,57.12,57.375]
        var result=[Float](repeating:0,count:1024*1024*3)
        for y in 0..<1024 { for x in 0..<1024 { for c in 0..<3 {
            let s0=Int(bytes[(Int(ys[2*y])*width+Int(xs[2*x]))*4+c])*Int(a[2*x])+Int(bytes[(Int(ys[2*y])*width+Int(xs[2*x+1]))*4+c])*Int(a[2*x+1])
            let s1=Int(bytes[(Int(ys[2*y+1])*width+Int(xs[2*x]))*4+c])*Int(a[2*x])+Int(bytes[(Int(ys[2*y+1])*width+Int(xs[2*x+1]))*4+c])*Int(a[2*x+1])
            let pixel=((((Int(b[2*y])*(s0>>4))>>16)+((Int(b[2*y+1])*(s1>>4))>>16)+2)>>2)
            result[(y*1024+x)*3+c]=(Float(pixel)-mean[c])/std[c]
        } } }
        return result
    }
}
