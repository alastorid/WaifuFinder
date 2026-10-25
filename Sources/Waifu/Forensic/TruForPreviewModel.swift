import AppKit
import Combine
import Metal

enum TruForDisplay:String,CaseIterable,Identifiable,Sendable {
    case anomaly,reliability,outlines,combined,noiseprint
    var id:String { rawValue }
    var label:String { switch self { case .anomaly:"Anomaly map";case .reliability:"Reliability map";case .outlines:"Suspicious outlines";case .combined:"Combined overlay";case .noiseprint:"Noiseprint++" } }
}

/// Only the selected layer is rendered. Evidence stays transient and switching
/// layers never requests another inference pass. One Metal pass composites the
/// chosen layer over the photo; the outline mode takes its edges from a 3×3
/// morphological gradient of the real anomaly threshold, never a synthetic mask.
actor TruForEvidenceRenderer {
    static let shared=TruForEvidenceRenderer()
    private let pipeline:MTLComputePipelineState?
    private let source=MTLCreateSystemDefaultDevice()
    private init() {
        guard let device=source,let library=try? device.makeLibrary(source:Self.metal,options:nil),
              let function=library.makeFunction(name:"truforComposite") else { pipeline=nil;return }
        pipeline=try? device.makeComputePipelineState(function:function)
    }
    private static let metal="""
        #include <metal_stdlib>
        using namespace metal;
        kernel void truforComposite(
            texture2d<float, access::read> photo [[texture(0)]],
            device const float* anomaly [[buffer(0)]],
            device const float* reliability [[buffer(1)]],
            device const float* noise [[buffer(2)]],
            device unsigned char* output [[buffer(3)]],
            constant unsigned int& mode [[buffer(4)]],
            constant float& contrast [[buffer(5)]],
            constant uint2& dimensions [[buffer(6)]],
            uint2 position [[thread_position_in_grid]]) {
            if (position.x>=dimensions.x || position.y>=dimensions.y) return;
            uint width=dimensions.x,index=position.y*width+position.x;
            float a=clamp(anomaly[index],0.0f,1.0f),r=clamp(reliability[index],0.0f,1.0f);
            float3 color;
            switch (mode) {
                case 0u: color=float3(a,a*a*0.8f,0.0f);break;
                case 1u: color=float3(r);break;
                case 2u: {
                    float low=1.0f,high=0.0f;
                    for (int y=-1;y<=1;y++) for (int x=-1;x<=1;x++) {
                        int2 around=int2(position)+int2(x,y);
                        around=clamp(around,int2(0),int2(dimensions)-1);
                        float mask=step(0.5f,anomaly[around.y*width+around.x]);
                        low=min(low,mask);high=max(high,mask);
                    }
                    color=mix(photo.read(position).rgb,float3(1.0f,0.15f,0.05f),clamp((high-low)*2.0f,0.0f,1.0f));
                    break;
                }
                case 3u: color=mix(photo.read(position).rgb,mix(float3(0.6f,0.25f,1.0f),float3(1.0f,0.2f,0.0f),r),a*0.65f);break;
                default: { float n=clamp(0.5f+noise[index]*contrast,0.0f,1.0f);color=float3(n); }
            }
            output[index*4u+0u]=uchar(clamp(color.r,0.0f,1.0f)*255.0f);
            output[index*4u+1u]=uchar(clamp(color.g,0.0f,1.0f)*255.0f);
            output[index*4u+2u]=uchar(clamp(color.b,0.0f,1.0f)*255.0f);
            output[index*4u+3u]=255u;
        }
        """
    func render(original:CGImage,evidence:TruForSpatial,mode:TruForDisplay)throws->CGImage {
        try evidence.validate()
        guard let device=source,let pipeline else { throw ModelInstaller.fail("TruFor evidence renderer unavailable") }
        let width=evidence.width,height=evidence.height,pixels=width*height
        // The photo is resampled to the evidence geometry once, here; maps are
        // exported in top-to-bottom rows and the bitmap rows match, so no pixel
        // needs a second transform.
        var rgba=[UInt8](repeating:0,count:pixels*4)
        let drew=rgba.withUnsafeMutableBytes { data in
            guard let context=CGContext(data:data.baseAddress,width:width,height:height,bitsPerComponent:8,bytesPerRow:width*4,space:CGColorSpace(name:CGColorSpace.sRGB)!,bitmapInfo:CGImageAlphaInfo.premultipliedLast.rawValue) else { return false }
            context.interpolationQuality = .medium
            context.draw(original,in:CGRect(x:0,y:0,width:width,height:height));return true
        }
        guard drew else { throw ModelInstaller.fail("Could not scale the photo for TruFor evidence") }
        func buffer(_ values:[Float])throws->MTLBuffer {
            guard let buffer=device.makeBuffer(bytes:values,length:pixels*4,options:.storageModeShared) else {
                throw ModelInstaller.fail("Not enough GPU memory to render TruFor evidence")
            }
            return buffer
        }
        let anomaly=try buffer(evidence.anomaly),reliability=try buffer(evidence.reliability),noise=try buffer(evidence.noiseprint)
        guard let output=device.makeBuffer(length:pixels*4,options:.storageModeShared),
              let photo=device.makeTexture(descriptor:MTLTextureDescriptor.texture2DDescriptor(pixelFormat:.rgba8Unorm,width:width,height:height,mipmapped:false)) else { throw ModelInstaller.fail("TruFor evidence renderer unavailable") }
        rgba.withUnsafeBytes { data in photo.replace(region:MTLRegion(origin:MTLOrigin(x:0,y:0,z:0),size:MTLSize(width:width,height:height,depth:1)),mipmapLevel:0,withBytes:data.baseAddress!,bytesPerRow:width*4) }
        guard let queue=device.makeCommandQueue(),let commands=queue.makeCommandBuffer(),let encoder=commands.makeComputeCommandEncoder() else { throw ModelInstaller.fail("TruFor evidence renderer unavailable") }
        encoder.setComputePipelineState(pipeline)
        encoder.setTexture(photo,index:0)
        encoder.setBuffer(anomaly,offset:0,index:0)
        encoder.setBuffer(reliability,offset:0,index:1)
        encoder.setBuffer(noise,offset:0,index:2)
        encoder.setBuffer(output,offset:0,index:3)
        var selection=UInt32(TruForDisplay.allCases.firstIndex(of:mode)!);encoder.setBytes(&selection,length:MemoryLayout<UInt32>.size,index:4)
        let rms=sqrt(evidence.noiseprint.reduce(0.0){$0+Double($1)*Double($1)}/Double(evidence.noiseprint.count))
        var contrast=Float(1/max(1e-6,6*rms));encoder.setBytes(&contrast,length:MemoryLayout<Float>.size,index:5)
        var dimensions=SIMD2<UInt32>(UInt32(width),UInt32(height));encoder.setBytes(&dimensions,length:MemoryLayout<SIMD2<UInt32>>.size,index:6)
        encoder.dispatchThreadgroups(MTLSize(width:(width+15)/16,height:(height+15)/16,depth:1),threadsPerThreadgroup:MTLSize(width:16,height:16,depth:1))
        encoder.endEncoding();commands.commit();commands.waitUntilCompleted()
        guard commands.error==nil else { throw ModelInstaller.fail("Could not render TruFor evidence: \(commands.error!.localizedDescription)") }
        var bytes=[UInt8](repeating:0,count:pixels*4)
        bytes.withUnsafeMutableBytes { data in _ = memcpy(data.baseAddress!,output.contents(),pixels*4) }
        guard let rendered=CGImage(width:width,height:height,bitsPerComponent:8,bitsPerPixel:32,bytesPerRow:width*4,space:CGColorSpace(name:CGColorSpace.sRGB)!,bitmapInfo:CGBitmapInfo(rawValue:CGImageAlphaInfo.premultipliedLast.rawValue),provider:CGDataProvider(data:Data(bytes) as CFData)!,decode:nil,shouldInterpolate:false,intent:.defaultIntent) else { throw ModelInstaller.fail("Could not render TruFor evidence") }
        return rendered
    }
}

@MainActor final class TruForPreviewModel:ObservableObject {
    @Published private(set) var photoURL:URL?
    @Published private(set) var rendered:NSImage?
    @Published private(set) var loading=false
    @Published private(set) var status=""
    private var ticket=UUID()
    private var renderTicket=UUID()
    private var evidence:TruForSpatial?
    private var original:CGImage?
    private var mode=TruForDisplay.combined
    func load(url:URL?,enabled:Bool,mode:TruForDisplay)async {
        let token=UUID();ticket=token;renderTicket=UUID();photoURL=url;rendered=nil;evidence=nil;original=nil;status="";loading=false;self.mode=mode
        guard enabled,WaifuDistribution.isNonCommercialOnly,let url,let item=PhotosMedia.shared.item(url),item.isStill,let key=item.assetCacheKey else { return }
        loading=true
        defer { if ticket==token { loading=false } }
        do {
            try await Task.sleep(for:.milliseconds(200));try Task.checkCancellation()
            status="TruFor · analyzing this photo…"
            let photo=TruForPhoto(url:url,assetKey:key)
            let image=try await PhotosMedia.shared.image(url,seconds:0)
            let png=try await Task.detached(priority:.utility){try SafireBackend.png(image)}.value
            let local=TruForConfiguration.load()
            let remote=RemoteInferencePool.shared.entries.map(\.client).first(where: \.offersTruFor)
            let map=try await SafireJobWait.run(onWaiting:{self.status="TruFor · waiting for another model job"}) {
                if local != nil {
                    let value=try await TruForBackend.shared.submit(images:[png],evidence:true,cacheKey:TruForConfiguration.scoreFingerprint+"|"+photo.contentIdentity)
                    guard case .evidence(let result)=value else { throw ModelInstaller.fail("Missing TruFor evidence") };return result
                }
                if let remote { return try await remote.truforEvidence(png) }
                throw ModelInstaller.fail("Set up TruFor in Settings → Models")
            }
            guard !Task.isCancelled,ticket==token,PhotosMedia.shared.item(url)?.assetCacheKey==key else { return }
            evidence=map;original=image
            await display(self.mode)
        } catch is CancellationError {} catch { if ticket==token { status=error.localizedDescription } }
    }
    func display(_ mode:TruForDisplay)async {
        self.mode=mode
        guard let evidence,let original else { return }
        let token=UUID();renderTicket=token
        do {
            let image=try await TruForEvidenceRenderer.shared.render(original:original,evidence:evidence,mode:mode)
            guard !Task.isCancelled,renderTicket==token else { return }
            rendered=NSImage(cgImage:image,size:NSSize(width:image.width,height:image.height))
            status="TruFor · detection score "+String(format:"%.3f",evidence.score)+(mode == .noiseprint ? " · contrast-scaled residual, not a probability" : mode == .combined ? " · orange: reliable anomaly; violet: uncertain anomaly" : mode == .outlines ? " · anomaly ≥ 0.5" : "")
        } catch { if renderTicket==token { status=error.localizedDescription } }
    }
}
