import AppKit
import CoreImage
import Combine

private func inspectPattern(shiftX: Int = 0, shiftY: Int = 0, change: Bool = false) -> CGImage {
    let width = 128, height = 96
    let c = CGContext(data: nil, width: width, height: height, bitsPerComponent: 8, bytesPerRow: 0,
        space: CGColorSpace(name: CGColorSpace.sRGB)!, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
    c.setFillColor(CGColor(gray: 0, alpha: 1)); c.fill(CGRect(x: 0, y: 0, width: width, height: height))
    c.translateBy(x: CGFloat(shiftX), y: CGFloat(shiftY))
    for y in 8..<88 {
        for x in 8..<120 {
            let v = CGFloat((x * 19 + y * 37 + x*y*7) % 251) / 255
            c.setFillColor(CGColor(red: v, green: CGFloat(x%17)/17, blue: CGFloat(y%13)/13, alpha: 1))
            c.fill(CGRect(x: x, y: y, width: 1, height: 1))
        }
    }
    if change { c.setFillColor(CGColor(gray: 1, alpha: 1)); c.fill(CGRect(x: 30, y: 40, width: 8, height: 6)) }
    return c.makeImage()!
}

func registerCrystalOrbInspectionTests() {
    let suite = "Crystal Orb inspection"
    Registry.shared.add(suite: suite, TestCase(name: "source comparison and viewport transforms round trip at both backing scales", knownBug: nil) {
        let geometry = CrystalOrbAlignment(target: CGSize(width: 4032, height: 3024), candidate: CGSize(width: 1920, height: 1080),
            registration: .init(x: 3.2, y: -1.5, scale: 1.02, degrees: 2))
        let source = CGPoint(x: 421.5, y: 711.5)
        let comparison = geometry.comparison(source, candidate: true)
        let inverse = geometry.source(comparison, candidate: true)!
        checkClose(inverse.x, source.x, tolerance: 0.00001, "source X round trip")
        checkClose(inverse.y, source.y, tolerance: 0.00001, "source Y round trip")
        for backing in [CGFloat(1), CGFloat(2)] {
            let viewport = CrystalOrbViewport(size: CGSize(width: 643, height: 311), center: CGPoint(x: 1700, y: 1400), scale: 1/backing)
            let result = viewport.comparison(viewport.screen(comparison))
            checkClose(result.x, comparison.x, tolerance: 0.00001, "Retina X uses points versus pixels explicitly")
            checkClose(result.y, comparison.y, tolerance: 0.00001, "top-left viewport maps to bottom-left pixels")
            checkClose(viewport.screen(CGPoint(x: 1701,y: 1400)).x - viewport.screen(CGPoint(x: 1700,y: 1400)).x, 1/backing, tolerance: 0.00001, "1:1 is one backing pixel")
        }
        let a = CrystalOrbViewport(size: CGSize(width: 300,height: 200), center: comparison, scale: 2)
        let b = CrystalOrbViewport(size: CGSize(width: 700,height: 160), center: comparison, scale: 2)
        checkEqual(a.comparison(CGPoint(x: 150,y: 100)), b.comparison(CGPoint(x: 350,y: 80)), "differently sized panes share the exact focal pixel")
    })
    Registry.shared.add(suite: suite, TestCase(name: "identical registered images have exact zero statistics and no hotspots", knownBug: nil) {
        let image = inspectPattern()
        let diff = try CrystalOrbDifferenceRenderer.render(target: image, candidate: image)
        let value = try await CrystalOrbAnalysis.shared.measure(diff, roi: nil)
        checkEqual(value.full.overlap, 128*96, "all opaque black and colored pixels remain valid")
        checkEqual(value.full.changed, 0, "identical images have no changes")
        checkEqual(value.full.maximum, 0, "raw maximum is zero")
        check(value.hotspots.isEmpty, "no false hotspots")
    })
    Registry.shared.add(suite: suite, TestCase(name: "translation registration recovers known offsets and preserves corresponding pixel samples", knownBug: nil) {
        let a = inspectPattern(), b = inspectPattern(shiftX: 3, shiftY: -2)
        let base = CrystalOrbAlignment(target: CGSize(width: 128,height: 96), candidate: CGSize(width: 128,height: 96))
        let frame = AnalyzedFrame(sampleIndex: 0, timeSeconds: 0, artScore: 0, faceScore: nil, faceCount: 0, faceAreaFraction: nil, featurePrint: nil)
        let result = try await CrystalOrbRegistrar.shared.align(target: a, candidate: b, geometry: base, query: UUID(), frame: frame.id)
        check(result != nil, "textured overlap produces a reliable estimate")
        guard let result else { return }
        checkClose(result.registration.x, -3, tolerance: 0.1, "candidate moves left by three")
        checkClose(result.registration.y, 2, tolerance: 0.1, "candidate moves up by two")
        let registered = CrystalOrbAlignment(target: base.targetSize,candidate: base.candidateSize,registration: result.registration)
        let unaligned = try await CrystalOrbAnalysis.shared.measure(CrystalOrbDifferenceRenderer.render(target: a,candidate: b), roi: nil)
        let diff = try CrystalOrbDifferenceRenderer.render(target: a,candidate: b,alignment: registered)
        let aligned = try await CrystalOrbAnalysis.shared.measure(diff, roi: nil)
        check(aligned.full.mean < unaligned.full.mean * 0.05, "correct alignment removes positional differences")
        let pixel = try await CrystalOrbPixelReader.shared.sample(diff, point: CGPoint(x: 45.5,y: 31.5))
        checkEqual(pixel.a,pixel.b,"registered samples correspond exactly")
        checkEqual(pixel.candidatePoint, CGPoint(x: 48.5,y: 29.5), "crosshair uses inverse source transform")
        checkEqual(pixel.delta, [0,0,0], "registered sample raw delta agrees")
    })
    Registry.shared.add(suite: suite, TestCase(name: "uniform images reject ambiguous automatic registration", knownBug: nil) {
        let image = try ImageFixture.cgImage(width: 64,height: 64,gray: 0.5)
        let geometry = CrystalOrbAlignment(target: CGSize(width: 64,height: 64), candidate: CGSize(width: 64,height: 64))
        let frame = AnalyzedFrame(sampleIndex: 1,timeSeconds: 1,artScore: 0,faceScore: nil,faceCount: 0,faceAreaFraction: nil, featurePrint: nil)
        let result = try await CrystalOrbRegistrar.shared.align(target: image,candidate: image,geometry: geometry,query: UUID(),frame: frame.id)
        check(result == nil,"textureless sources do not receive arbitrary offsets")
    })
    Registry.shared.add(suite: suite, TestCase(name: "tolerance changes significance while gain never changes raw measurements", knownBug: nil) {
        let a = try ImageFixture.cgImage(width: 32,height: 24,gray: 0.5)
        let b = try ImageFixture.cgImage(width: 32,height: 24,gray: 0.51)
        let original = try CrystalOrbDifferenceRenderer.render(target: a,candidate: b)
        let raw = try await CrystalOrbAnalysis.shared.measure(original,roi: nil)
        var settings = original.settings; settings.tolerance = 8; settings.gain = 20; settings.display = .heatmap
        let remapped = try CrystalOrbDifferenceRenderer.remap(original,settings: settings)
        let tolerant = try await CrystalOrbAnalysis.shared.measure(remapped,roi: nil)
        check(raw.full.changed > 0,"zero tolerance counts small encoding-scale variations")
        checkEqual(tolerant.full.changed,0,"tolerance suppresses small changes")
        checkEqual(tolerant.full.mean,raw.full.mean,"tolerance and gain cannot falsify raw mean")
        checkEqual(tolerant.full.maximum,raw.full.maximum,"raw max unchanged")
        let pixel = try await CrystalOrbPixelReader.shared.sample(remapped,point: CGPoint(x: 12,y: 10))
        check(!pixel.significant,"sample tolerance agrees with statistics")
    })
    Registry.shared.add(suite: suite, TestCase(name: "ROI metrics and connected hotspots locate localized changes", knownBug: nil) {
        let a = inspectPattern(), b = inspectPattern(change: true)
        let diff = try CrystalOrbDifferenceRenderer.render(target: a,candidate: b)
        let result = try await CrystalOrbAnalysis.shared.measure(diff,roi: CGRect(x: 30,y: 40,width: 8,height: 6))
        let display = CrystalOrbAnalysis.bytes(CIImage(cgImage: diff.difference), rect: CGRect(x: 30,y: 40,width: 1,height: 1))
        let overlay = CrystalOrbAnalysis.bytes(CIImage(cgImage: diff.redMask), rect: CGRect(x: 30,y: 40,width: 1,height: 1))
        check(display.prefix(3).contains { $0 > 0 }, "preview difference must visibly locate the changed pixels")
        check(overlay[3] > 0, "red overlay must visibly locate the changed pixels")
        var settings = diff.settings; settings.display = .heatmap
        let heatmap = try CrystalOrbDifferenceRenderer.remap(diff, settings: settings)
        let heat = CrystalOrbAnalysis.bytes(CIImage(cgImage: heatmap.difference), rect: CGRect(x: 30,y: 40,width: 1,height: 1))
        check(heat[0] > 0, "heatmap remapping retains the changed material")
        checkEqual(result.full.changed,48,"only exact changed pixels are counted")
        checkEqual(result.region?.changed,48,"ROI compares registered coordinates without cropping inputs")
        checkEqual(result.region?.overlap,48,"ROI valid area")
        checkEqual(result.hotspots.count,1,"one connected localized change")
        check(result.hotspots.first?.rect.contains(CGPoint(x: 34,y: 43)) == true,"hotspot contains the changed subject")
    })
    Registry.shared.add(suite: suite, TestCase(name: "transparent valid pixels and unmatched geometric areas stay distinct", knownBug: nil) {
        let c = CGContext(data: nil,width: 32,height: 32,bitsPerComponent: 8,bytesPerRow: 0,space: CGColorSpace(name: CGColorSpace.sRGB)!,bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
        let a = c.makeImage()!
        let geometry = CrystalOrbAlignment(target: CGSize(width: 32,height: 32),candidate: CGSize(width: 32,height: 32),registration: .init(x: 3))
        let diff = try CrystalOrbDifferenceRenderer.render(target: a,candidate: a,alignment: geometry)
        let stats = try await CrystalOrbAnalysis.shared.measure(diff,roi: nil)
        checkEqual(stats.full.changed,0,"transparent pixels compare over black and remain equal")
        checkEqual(stats.full.overlap,29*32,"only geometric coverage defines overlap")
        checkEqual(stats.full.covered-stats.full.overlap,3*32,"unmatched area excludes empty canvas")
        let pixel = try await CrystalOrbPixelReader.shared.sample(diff,point: CGPoint(x: 1,y: 10))
        check(pixel.candidatePoint == nil,"missing correspondence is explicit")
    })
    Registry.shared.add(suite: suite, TestCase(name: "known RGB changes channel selection and luma use documented units", knownBug: nil) {
        func flat(_ r: CGFloat,_ g: CGFloat,_ b: CGFloat) -> CGImage {
            let pixel: [UInt8] = [UInt8((r*255).rounded()),UInt8((g*255).rounded()),UInt8((b*255).rounded()),255]
            let bytes = Data((0..<(16*16)).flatMap { _ in pixel })
            return CGImage(width: 16,height: 16,bitsPerComponent: 8,bitsPerPixel: 32,bytesPerRow: 16*4,
                space: CGColorSpace(name: CGColorSpace.sRGB)!,bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue),
                provider: CGDataProvider(data: bytes as CFData)!,decode: nil,shouldInterpolate: false,intent: .defaultIntent)!
        }
        let diff = try CrystalOrbDifferenceRenderer.render(target: flat(100/255,120/255,140/255),candidate: flat(103/255,116/255,145/255))
        let sample = try await CrystalOrbPixelReader.shared.sample(diff,point: CGPoint(x: 8,y: 8))
        checkEqual(sample.delta,[3,4,5],"8-bit normalized sRGB channel deltas")
        for (channel,expected) in [(CrystalOrbDifferenceSettings.Channel.red,3.0),(.green,4.0),(.blue,5.0),(.rgb,5.0)] {
            var settings = diff.settings; settings.channel = channel
            let result = try await CrystalOrbAnalysis.shared.measure(CrystalOrbDifferenceRenderer.remap(diff,settings: settings),roi: nil)
            checkClose(result.full.mean,expected/255,tolerance: 0.0001,"per-channel measurement is accurate")
        }
    })
    Registry.shared.add(suite: suite, TestCase(name: "scaled native geometry and rotated correspondence use the same renderer transforms", knownBug: nil) {
        let large = try ImageFixture.cgImage(width: 256,height: 192,gray: 0.5)
        let small = try ImageFixture.cgImage(width: 64,height: 48,gray: 0.5)
        let diff = try CrystalOrbDifferenceRenderer.render(target: large,candidate: small)
        let p = try await CrystalOrbPixelReader.shared.sample(diff,point: CGPoint(x: 128.5,y: 96.5))
        checkEqual(p.candidatePoint,CGPoint(x: 32.125,y: 24.125),"source coordinates retain the native small-image resolution")
        let value = try await CrystalOrbAnalysis.shared.measure(diff,roi: nil)
        check(value.full.maximum <= 1/255,"resampling constant content does not fabricate differences")
        let rotated = CrystalOrbAlignment(target: CGSize(width: 256,height: 192),candidate: CGSize(width: 64,height: 48),registration: .init(x: 2,y: -3,scale: 1.1,degrees: 15))
        let source = CGPoint(x: 32.5,y: 24.5)
        let comparison = rotated.comparison(source,candidate: true)
        let inverse = rotated.source(comparison,candidate: true)!
        checkClose(inverse.x,source.x,tolerance: 0.000001,"rotation preserves exact inverse X")
        checkClose(inverse.y,source.y,tolerance: 0.000001,"rotation preserves exact inverse Y")
    })
    Registry.shared.add(suite: suite, TestCase(name: "viewport tile cache is bounded by visual identity and never returns another material", knownBug: nil) {
        let black = CIImage(color: .black).cropped(to: CGRect(x: 0,y: 0,width: 32,height: 32))
        let white = CIImage(color: .white).cropped(to: black.extent)
        let renderer = CrystalOrbViewportRenderer()
        let first = try await renderer.render(image: black,canvas: black.extent.size,viewport: CGSize(width: 16,height: 16),scale: 1,pan: .zero,backing: 2,identity: "black")
        let revisit = try await renderer.render(image: black,canvas: black.extent.size,viewport: CGSize(width: 16,height: 16),scale: 1,pan: .zero,backing: 2,identity: "black")
        check(first === revisit,"repeated native viewport reuses the cached raster")
        let other = try await renderer.render(image: white,canvas: black.extent.size,viewport: CGSize(width: 16,height: 16),scale: 1,pan: .zero,backing: 2,identity: "white")
        check(other !== first,"new identity cannot receive a stale material")
        let raw = CrystalOrbAnalysis.bytes(CIImage(cgImage: other),rect: CGRect(x: 0,y: 0,width: 32,height: 32))
        checkEqual(Array(raw.prefix(4)),[255,255,255,255],"the correct new pixels are present")
    })
    Registry.shared.add(suite: suite, TestCase(name: "difference auto stretch makes faint changes visible and keeps identical and noise pixels black", knownBug: nil) {
        let a = try ImageFixture.cgImage(width: 32,height: 24,gray: 0.5)
        let b = try ImageFixture.cgImage(width: 32,height: 24,gray: 0.52)
        let diff = try CrystalOrbDifferenceRenderer.render(target: a,candidate: b)
        let visible = CrystalOrbAnalysis.bytes(CIImage(cgImage:diff.difference),rect:CGRect(x:0,y:0,width:1,height:1))
        check((visible.prefix(3).max() ?? 0) >= 250,"faint differences stretch to full 255 brightness")
        let same = try CrystalOrbDifferenceRenderer.render(target: a,candidate: a)
        let equal = CrystalOrbAnalysis.bytes(CIImage(cgImage:same.difference),rect:CGRect(x:0,y:0,width:1,height:1))
        checkEqual(Array(equal.prefix(3)),[0,0,0],"identical images cannot amplify rounding noise")
        var settings=diff.settings; settings.tolerance=20
        let quiet=try CrystalOrbDifferenceRenderer.remap(diff,settings:settings)
        let noise=CrystalOrbAnalysis.bytes(CIImage(cgImage:quiet.difference),rect:CGRect(x:0,y:0,width:1,height:1))
        checkEqual(Array(noise.prefix(3)),[0,0,0],"sub-threshold noise stays black under auto stretch")
    })
    Registry.shared.add(suite: suite, TestCase(name: "gesture-rate navigation does not invalidate search metadata or the workspace shell", knownBug: nil) {
        await Task { @MainActor in
            let model=WaifuCrystalOrbModel()
            var shellUpdates=0, navigationUpdates=0
            let shell=model.objectWillChange.sink { shellUpdates += 1 }
            let navigation=model.navigation.objectWillChange.sink { navigationUpdates += 1 }
            for i in 0..<120 { model.center(on:CGPoint(x:i,y:i)); model.zoom=1+CGFloat(i)/120 }
            checkEqual(shellUpdates,0,"pan and zoom do not rebuild search or metadata")
            check(navigationUpdates>0,"image panes still receive responsive navigation updates")
            checkEqual(WaifuCrystalOrbModel.candidateLimit, 33, "gallery never exceeds thirty-three candidates")
            withExtendedLifetime((shell,navigation)) {}
        }.value
    })
    Registry.shared.add(suite: suite, TestCase(name: "candidate gallery keeps thirteen visible and trims by likeness up to the cap", knownBug: nil) {
        func probe(_ index: Int, _ distance: Double) -> CrystalOrbMatch {
            CrystalOrbMatch(frame: AnalyzedFrame(sampleIndex: index, timeSeconds: Double(index), artScore: 0,
                faceScore: nil, faceCount: 0, faceAreaFraction: nil, featurePrint: nil), distance: distance)
        }
        checkEqual(WaifuCrystalOrbModel.alikeMatches([]), [], "empty search yields no gallery")
        let six = (0..<6).map { probe($0, 0.30 + Double($0) * 0.001) }
        checkEqual(WaifuCrystalOrbModel.alikeMatches(six).count, 6, "short searches show every match found")
        // A best distance of 0.30 cuts the band at 0.30 × 1.35 + 0.02 = 0.425;
        // below the floor the strip still fills thirteen tiles.
        var mixed = (0..<5).map { probe($0, 0.30 + Double($0) * 0.03) }
        mixed += (5..<20).map { probe($0, 0.90) }
        let floored = WaifuCrystalOrbModel.alikeMatches(mixed)
        checkEqual(floored.count, WaifuCrystalOrbModel.minimumCandidates, "the floor keeps thirteen thumbnails visible")
        checkEqual(Array(floored.prefix(5)).map(\.id.sampleIndex), [0, 1, 2, 3, 4], "banded matches stay closest-first")
        // Beyond the floor only the likeness band trims; the distant ones drop.
        var banded = (0..<15).map { probe($0, 0.30 + Double($0) * 0.001) }
        banded += (15..<20).map { probe($0, 0.90) }
        checkEqual(WaifuCrystalOrbModel.alikeMatches(banded).count, 15, "beyond the floor only the likeness band trims")
        let dense = (0..<40).map { probe($0, 0.30 + Double($0) * 0.001) }
        checkEqual(WaifuCrystalOrbModel.alikeMatches(dense).count, WaifuCrystalOrbModel.candidateLimit, "a wide band still stops at the cap")
        checkEqual(WaifuCrystalOrbModel.alikeMatches(dense).last?.id.sampleIndex, 32, "the cap keeps the closest matches")
    })
    Registry.shared.add(suite: suite, TestCase(name: "comparison modes preserve text focus and hold-toggle ownership", knownBug: nil) {
        try await Task { @MainActor in
            let model = WaifuCrystalOrbModel()
            checkEqual(model.mode,.sideBySide,"Side by Side is the default")
            checkEqual(Set(WaifuCrystalOrbModel.Mode.allCases),Set([.sideBySide,.swipe,.toggle,.difference,.alpha,.blink]),"original and forensic comparison modes remain available")
            let session = WaifuSession(settingsURL: try Temporary.directory("orb-inspection-keys").appendingPathComponent("settings.json"))
            model.open(session: session)
            func key(_ characters: String,_ code: UInt16,_ type: NSEvent.EventType = .keyDown) -> NSEvent {
                NSEvent.keyEvent(with: type,location: .zero,modifierFlags: [],timestamp: 0,windowNumber: 0,context: nil,characters: characters,charactersIgnoringModifiers: characters,isARepeat: false,keyCode: code)!
            }
            check(model.handleKey(key("t",17)),"toggle shortcut")
            model.zoom = 3; model.navigationCenter = CGPoint(x: 42,y: 17)
            check(model.handleKey(key(" ",49)),"hold Space is owned in toggle")
            check(model.holdingToggle,"hold shows alternate source")
            check(model.handleKey(key(" ",49,.keyUp)),"release is owned")
            check(!model.holdingToggle,"release restores source")
            checkEqual(model.zoom,3,"toggle never changes zoom")
            checkEqual(model.navigationCenter,CGPoint(x: 42,y: 17),"toggle never changes focal coordinate")
            check(model.handleKey(key("s",1)),"Side by Side shortcut")
            checkEqual(model.mode,.sideBySide,"Side by Side shortcut selects the default mode")
            model.maximizedPane = "A"
            check(model.handleKey(key("d",2)),"difference shortcut works from a maximized source")
            check(model.maximizedPane == nil,"mode changes restore the correct comparison pane")
            model.mode = .sideBySide
            model.inspectionControlFocused = true
            check(!model.handleKey(key("s",1)),"focused analysis control owns keyboard input")
            checkEqual(model.mode,.sideBySide,"focused controls cannot accidentally switch modes")
            model.close()
        }.value
    })
}
