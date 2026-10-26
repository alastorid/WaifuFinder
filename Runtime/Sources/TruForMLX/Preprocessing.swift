// GRIP-UNINA TruFor input convention; informational/nonprofit-only adaptation.
import MLX
public enum TruForPreprocessing {
    private static let kernel=MLXFast.metalKernel(name:"trufor_native_rgba_rgb",inputNames:["rgba"],outputNames:["out"],source:"""
        uint i=thread_position_in_grid.x;
        if (i>=PIXELS*3u) return;
        out[i]=float(rgba[(i/3u)*4u+i%3u])*(1.0f/256.0f);
        """)
    public static func rgba(_ pixels:[UInt8],width:Int,height:Int)throws->MLXArray {
        try TruForMemoryPolicy.validate(width:width,height:height)
        guard pixels.count==width*height*4 else { throw TruForError.invalid("Unsupported native image dimensions") }
        return strip(pixels,width:width,height:height)
    }
    static func strip(_ pixels:[UInt8],width:Int,height:Int)->MLXArray {
        kernel([MLXArray(pixels)],template:[("PIXELS",width*height)],grid:(width*height*3,1,1),threadGroup:(256,1,1),outputShapes:[[1,height,width,3]],outputDTypes:[.float32])[0]
    }
}
