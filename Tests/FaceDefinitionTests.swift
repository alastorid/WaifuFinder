import CoreGraphics
import CoreImage
import Foundation
import ImageIO

private func definitionPattern(face: CGRect, blur: Double = 0) throws -> CGImage {
    let context = CGContext(data: nil, width: 384, height: 256, bitsPerComponent: 8, bytesPerRow: 1536,
                            space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
    context.translateBy(x: 0, y: 256); context.scaleBy(x: 1, y: -1)
    context.setFillColor(gray: 0.3, alpha: 1); context.fill(CGRect(x: 0, y: 0, width: 384, height: 256))
    context.setFillColor(gray: 0.7, alpha: 1); context.fill(face)
    context.setFillColor(gray: 0.1, alpha: 1)
    for offset in [0.25, 0.65] {
        context.fillEllipse(in: CGRect(x: face.minX + face.width * offset, y: face.minY + face.height * 0.3,
                                      width: face.width * 0.1, height: face.height * 0.12))
    }
    context.fill(CGRect(x: face.minX + face.width * 0.3, y: face.minY + face.height * 0.7,
                        width: face.width * 0.4, height: face.height * 0.04))
    for index in 0..<6 {
        context.fill(CGRect(x: face.minX + face.width * (0.2 + Double(index) * 0.1), y: face.minY + face.height * 0.15,
                            width: face.width * 0.02, height: face.height * 0.12))
    }
    let original = context.makeImage()!
    if blur == 0 { return original }
    let image = CIImage(cgImage: original).clampedToExtent().applyingFilter("CIGaussianBlur", parameters: [kCIInputRadiusKey: blur])
    return CIContext().createCGImage(image, from: CGRect(x: 0, y: 0, width: 384, height: 256))!
}

func registerFaceDefinitionTests() {
    let suite = "Face definition"
    if let path = ProcessInfo.processInfo.environment["WAIFU_TEST_FACE_FRAME"] {
        Registry.shared.add(suite: suite, TestCase(name: "supplied padded face survives initial scoring with capture quality and definition", knownBug: nil) {
            guard let source = CGImageSourceCreateWithURL(URL(fileURLWithPath: path) as CFURL, nil),
                  let decoded = CGImageSourceCreateImageAtIndex(source, 0, nil),
                  let picture = decoded.cropping(to: FramePictureArea.bounds(decoded)) else {
                check(false, "supplied frame decodes"); return
            }
            let scale = 512.0 / Double(max(picture.width, picture.height))
            let resized = CIImage(cgImage: picture).transformed(by: CGAffineTransform(scaleX: scale, y: scale))
            guard let image = CIContext().createCGImage(resized, from: resized.extent) else { check(false, "analysis input rasterizes"); return }
            let frame = SampledFrame(sampleIndex: 0, timeSeconds: 0, image: image, nativeImage: picture)
            let result = try await VisionFrameScorer().score(frame)
            check(result.faceCount > 0, "initial shared face request detects the subject")
            checkNotNil(result.faceScore, "capture quality passes visible-picture significance threshold")
            checkNotNil(result.faceDefinitionScore, "definition is already measured in the same scoring call")
            check((result.faceAreaFraction ?? 0) > 0.01, "large canvas no longer makes the face incidental")
            print("Supplied frame: crop \(picture.width)×\(picture.height), face area \(result.faceAreaFraction ?? 0), capture \(result.faceScore ?? 0), definition \(result.faceDefinitionScore ?? 0)")
        })
    }
    Registry.shared.add(suite: suite, TestCase(name: "large black canvas is cropped without trimming ordinary or completely black frames", knownBug: nil) {
        let context = CGContext(data: nil, width: 640, height: 360, bitsPerComponent: 8, bytesPerRow: 2560,
            space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
        context.setFillColor(gray: 0, alpha: 1); context.fill(CGRect(x: 0, y: 0, width: 640, height: 360))
        context.setFillColor(gray: 0.5, alpha: 1); context.fill(CGRect(x: 250, y: 80, width: 140, height: 200))
        let image = context.makeImage()!
        let picture = FramePictureArea.bounds(image)
        check(picture.width >= 140 && picture.width < 150, "picture width excludes the side canvas")
        check(picture.height >= 200 && picture.height < 210, "picture height excludes top and bottom padding")
        check(picture.contains(CGRect(x: 250, y: 80, width: 140, height: 200)), "native picture content is retained")
        let faceArea: Float = 39 * 39 / (640 * 360)
        check(faceArea < 0.01, "the old whole-canvas area rejects this face")
        check(faceArea / FramePictureArea.fraction(image) > 0.01, "visible-picture area admits it")
        for gray in [0.0, 0.02, 0.5] {
            let plain = try ImageFixture.cgImage(width: 320, height: 240, gray: gray)
            checkEqual(FramePictureArea.bounds(plain), CGRect(x: 0, y: 0, width: 320, height: 240), "uniform frames remain intact")
        }
        checkEqual(try JSONDecoder().decode(AnalysisMode.self, from: Data(#""faceDefinition""#.utf8)), .full, "old deferred-mode settings migrate to Full")
        checkEqual(AnalysisMode.allCases, [.quick, .keyframe, .full], "definition is a score, not a loading mode")
        checkEqual(AnalysisMode(rawValue: "keyframe"), .keyframe, "the keyframe pass persists under its own name")
        checkEqual(AnalysisMode.keyframe.frameStep(frameRate: 29.97), 1,
                   "keyframe keeps no ordinal grid — the sync flag is not a fixed step")
        checkEqual(AnalysisMode.keyframe.analysedFrameCount(frameCount: 84_067, frameRate: 29.97,
                                                           durationSeconds: 2_820),
                   1_410, "a keyframe pass seeds a two-second-GOP total it never displays")
    })
    Registry.shared.add(suite: suite, TestCase(name: "sampler crops black canvas once for both native face and scaled analysis inputs", knownBug: nil) {
        let directory = try Temporary.directory("picture-canvas")
        let source = directory.appendingPathComponent("padded.mp4")
        try VideoFixture.writeVideo(to: source, duration: 0.5, width: 640, height: 360, frameRate: 12,
                                    picture: CGRect(x: 250, y: 80, width: 140, height: 200))
        var iterator = AVFoundationFrameSampler().sample(source, analysisWidth: 64, mode: .full, onProgress: { _ in }).makeAsyncIterator()
        guard let frame = try await iterator.next(), let native = frame.nativeImage else { check(false, "native picture is supplied"); return }
        check(native.width >= 140 && native.width < 150, "native source is cropped before face detection")
        check(native.height >= 200 && native.height < 210, "native height retains visible content")
        check(frame.image.width < 64 && frame.image.height == 64, "portrait analysis stays bounded on its longest edge")
        let info = try await VideoProbe.probe(source)
        checkEqual(info.width, 640, "source geometry for preview/export stays unchanged")
        checkEqual(info.height, 360, "source canvas remains intact outside scoring")
    })
    Registry.shared.add(suite: suite, TestCase(name: "definition breaks face ties before art and filters independently of sort toggles", knownBug: nil) {
        func frame(_ index: Int, time: Double = 1, face: Float = 0.8, definition: Float, art: Float) -> AnalyzedFrame {
            AnalyzedFrame(sampleIndex: index, timeSeconds: time, artScore: art, faceScore: face,
                          faceCount: 1, faceAreaFraction: 0.1, featurePrint: nil, faceDefinitionScore: definition)
        }
        let detailed = frame(0, definition: 0.9, art: 0.1)
        let artistic = frame(1, definition: 0.2, art: 0.9)
        let betterFace = frame(2, face: 0.95, definition: 0.1, art: 0)
        let earlier = frame(3, time: 0, face: 0.1, definition: 0.1, art: 0)
        let input = [artistic, detailed, betterFace, earlier]
        func arranged(_ order: FrameOrder, limits: FrameLimits = .none) -> [FrameID] {
            FrameArrangement.arrange(input, order: order, faceFilter: .all, limits: limits,
                                     perGroupTopCount: 200, grouping: .default).visible.map(\.id)
        }
        checkEqual(arranged(.everything), [earlier.id, betterFace.id, detailed.id, artistic.id], "time and face win before definition, which wins before art")
        var order = FrameOrder.everything
        order.setEnabled(false, for: .definition)
        checkEqual(arranged(order), [earlier.id, betterFace.id, artistic.id, detailed.id], "turning definition off restores art tie breaking")
        var limits = FrameLimits.none
        limits.setLimit(FrameLimit(low: 0.8), for: .definition)
        checkEqual(arranged(order, limits: limits), [detailed.id], "definition range works even with its sort disabled")
    })
    Registry.shared.add(suite: suite, TestCase(name: "small sharp edge face beats large blurred centre face", knownBug: nil) {
        let small = CGRect(x: 0, y: 100, width: 40, height: 48)
        let large = CGRect(x: 120, y: 50, width: 120, height: 144)
        guard let sharp = checkNotNil(FaceDefinition.score(try definitionPattern(face: small), face: small), "small native crop scores"),
              let blurred = checkNotNil(FaceDefinition.score(try definitionPattern(face: large, blur: 6), face: large), "large crop scores") else { return }
        check(sharp > blurred * 3 && sharp > 0, "size/position must not outweigh detail: sharp \(sharp), blurred \(blurred)")
        let central = CGRect(x: 172, y: 100, width: 40, height: 48)
        guard let moved = FaceDefinition.score(try definitionPattern(face: central), face: central) else { check(false, "central crop scores"); return }
        checkClose(Double(moved), Double(sharp), tolerance: 0.001, "translation does not change definition")
    })
    Registry.shared.add(suite: suite, TestCase(name: "flat, tiny and clipped regions do not masquerade as detailed faces", knownBug: nil) {
        let image = try ImageFixture.cgImage(width: 128, height: 128, gray: 0.5)
        checkEqual(FaceDefinition.score(image, face: CGRect(x: 0, y: 0, width: 64, height: 64)), 0, "flat crop has no detail")
        checkNil(FaceDefinition.score(image, face: CGRect(x: 0, y: 0, width: 3, height: 3)), "insufficient native samples")
        checkNotNil(FaceDefinition.score(image, face: CGRect(x: 0, y: 0, width: 16, height: 16)), "small faces are not excluded by subject area")
        checkNotNil(FaceDefinition.score(image, face: CGRect(x: -0.00001, y: 0, width: 64, height: 64)), "subpixel edge rounding stays selectable")
        checkNil(FaceDefinition.score(image, face: CGRect(x: -5, y: 0, width: 64, height: 64)), "clipped face is not treated as fully visible")
    })
    Registry.shared.add(suite: suite, TestCase(name: "pixelation boundaries do not outrank continuous native detail", knownBug: nil) {
        func texture(block: Int) -> CGImage {
            let side = 64
            var pixels = [UInt8](repeating: 0, count: side * side)
            for y in 0..<side {
                for x in 0..<side {
                    let a = Double(x / block * block), b = Double(y / block * block)
                    let value = 120 + 45 * sin(a * 0.16) * cos(b * 0.14) + 12 * sin(a * 0.42 + b * 0.23)
                    pixels[y * side + x] = UInt8(max(0, min(255, Int(value))))
                }
            }
            return pixels.withUnsafeMutableBytes { bytes in
                CGContext(data: bytes.baseAddress, width: side, height: side, bitsPerComponent: 8,
                          bytesPerRow: side, space: CGColorSpaceCreateDeviceGray(),
                          bitmapInfo: CGImageAlphaInfo.none.rawValue)!.makeImage()!
            }
        }
        let face = CGRect(x: 0, y: 0, width: 64, height: 64)
        let continuous = FaceDefinition.score(texture(block: 1), face: face)!
        for block in [6, 8] {
            let pixelated = FaceDefinition.score(texture(block: block), face: face)!
            check(continuous > pixelated * 1.5, "continuous features beat sharp \(block)-pixel mosaic edges")
        }
    })
    Registry.shared.add(suite: suite, TestCase(name: "readable facial features beat sharper uncertain face detections", knownBug: nil) {
        let clear = FaceDefinition.verifiedScore(detail: 0.4, precisionEstimates: [0.012, 0.016])!
        let falsePositive = FaceDefinition.verifiedScore(detail: 0.8, precisionEstimates: [0.028, 0.040])!
        check(clear > falsePositive, "sharp false detections or mosaic edges must not outweigh reliable facial features")
        let softened = FaceDefinition.verifiedScore(detail: 0.2, precisionEstimates: [0.012, 0.016])!
        check(clear > softened, "texture still distinguishes equally recognizable faces")
        let uncertain = FaceDefinition.verifiedScore(detail: 0.4, precisionEstimates: [0.025, 0.035])!
        check(clear > uncertain, "uncertain facial features reduce definition at equal sharpness")
    })
    Registry.shared.add(suite: suite, TestCase(name: "missing or invalid landmark evidence cannot fabricate definition", knownBug: nil) {
        for estimates: [Float]? in [nil, [], [.nan], [.infinity], [-0.01]] {
            checkNil(FaceDefinition.verifiedScore(detail: 0.8, precisionEstimates: estimates), "invalid evidence stays absent")
        }
        checkNil(FaceDefinition.verifiedScore(detail: .nan, precisionEstimates: [0.01]), "invalid texture stays absent")
        checkEqual(FaceDefinition.verifiedScore(detail: 0, precisionEstimates: [0.01]), 0, "well-located flat pixels are still not detailed")
    })
    Registry.shared.add(suite: suite, TestCase(name: "definition sorts and filters small faces independently of capture quality and group cap", knownBug: nil) {
        let id = FrameGroupID(memberIDs: ["f000000", "f000001"])
        let small = AnalyzedFrame(sampleIndex: 0, timeSeconds: 0, artScore: 0, faceScore: nil,
                                  faceCount: 1, faceAreaFraction: 0.002, featurePrint: nil,
                                  groupID: id, faceDefinitionScore: 0.9)
        let large = AnalyzedFrame(sampleIndex: 1, timeSeconds: 1, artScore: 1, faceScore: 0.99,
                                  faceCount: 1, faceAreaFraction: 0.4, featurePrint: nil,
                                  groupID: id, faceDefinitionScore: 0.1)
        let visible = FrameArrangement.arrange([large, small], order: .definitionOnly, faceFilter: .withFaces,
                                               limits: .none, perGroupTopCount: 1, grouping: .default)
        checkEqual(visible.visible.map(\.id), [small.id], "sharp small face wins the group cap")
        var limits = FrameLimits.none
        limits.setLimit(FrameLimit(low: 0.8), for: .definition)
        check(limits.admits(small, grouping: .default), "definition threshold ignores subject area")
        check(!limits.admits(large, grouping: .default), "blurry high-capture-score face excluded")
        check(FrameFaceFilter.withFaces.admits(small, grouping: .default), "small measured face remains selectable")
        check(!FrameFaceFilter.withoutFaces.admits(small, grouping: .default), "filters remain complementary")
        var old = try JSONSerialization.jsonObject(with: JSONEncoder().encode(small)) as! [String: Any]
        old.removeValue(forKey: "faceDefinitionScore")
        let restored = try JSONDecoder().decode(AnalyzedFrame.self, from: JSONSerialization.data(withJSONObject: old))
        checkNil(restored.faceDefinitionScore, "old records still decode")
    })
    Registry.shared.add(suite: suite, TestCase(name: "initial full pass keeps native pixels, scaled thumbnails and every decoded ordinal", knownBug: nil) {
        let directory = try Temporary.directory("native-definition")
        let source = try VideoFixture.makeMP4(in: directory, durationSeconds: 1, width: 320, height: 240, frameRate: 12)
        var count = 0
        for try await frame in AVFoundationFrameSampler().sample(source, analysisWidth: 64, mode: .full, onProgress: { _ in }) {
            checkEqual(frame.sampleIndex, count, "no rare-frame skipping")
            checkEqual(frame.nativeImage?.width, 320, "native crop source retained")
            checkEqual(frame.nativeImage?.height, 240, "source geometry retained")
            checkEqual(frame.image.width, 64, "other models and thumbnails remain scaled")
            count += 1
        }
        checkEqual(count, 12, "all source frames checked")
        var quick = AVFoundationFrameSampler().sample(source, analysisWidth: 64, mode: .quick, onProgress: { _ in }).makeAsyncIterator()
        checkNotNil(try await quick.next()?.nativeImage, "Quick also supplies definition in its initial pass")
    })
    Registry.shared.add(suite: suite, TestCase(name: "native definition honors video rotation and detection failure remains optional", knownBug: nil) {
        let directory = try Temporary.directory("rotated-definition")
        let source = try VideoFixture.makeMP4(in: directory, durationSeconds: 1, width: 320, height: 240, frameRate: 12)
        let rotated = directory.appendingPathComponent("rotated.mp4")
        try await VideoFixture.rotate(source, to: rotated)
        var iterator = AVFoundationFrameSampler().sample(rotated, analysisWidth: 64, mode: .full, onProgress: { _ in }).makeAsyncIterator()
        guard let frame = try await iterator.next() else { check(false, "rotated fixture yields a frame"); return }
        checkEqual(frame.nativeImage?.width, 240, "native face coordinates use oriented geometry")
        checkEqual(frame.nativeImage?.height, 320, "native pixels retain source resolution after rotation")
        let empty = try await FaceDefinition.evaluate(try ImageFixture.cgImage(width: 128, height: 128, gray: 0.5))
        checkEqual(empty.count, 0, "uniform image is not a face")
        checkNil(empty.score, "no fabricated definition score without a detected face")
    })
}
