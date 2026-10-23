import Foundation
import ImageIO
import Photos

func registerPhotoMetadataTests() {
    let suite = "Photo metadata"

    Registry.shared.add(suite: suite, TestCase(name: "ImageIO reads real JPEG properties without decoding its pixels", knownBug: nil) {
        let directory = try Temporary.directory("photo-metadata-imageio")
        let url = directory.appendingPathComponent("native.jpg")
        guard let context = CGContext(data: nil, width: 48, height: 32,
                                      bitsPerComponent: 8, bytesPerRow: 0,
                                      space: CGColorSpaceCreateDeviceRGB(),
                                      bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue),
              let image = context.makeImage(),
              let destination = CGImageDestinationCreateWithURL(url as CFURL, "public.jpeg" as CFString, 1, nil) else {
            throw SkipTest(reason: "native JPEG fixture creation is unavailable")
        }
        let properties: [CFString: Any] = [
            kCGImagePropertyExifDictionary: [
                kCGImagePropertyExifDateTimeOriginal: "2025:02:03 04:05:06",
                kCGImagePropertyExifFNumber: 2.8,
                kCGImagePropertyExifISOSpeedRatings: [400]
            ] as [CFString: Any],
            kCGImagePropertyTIFFDictionary: [
                kCGImagePropertyTIFFMake: "Sony",
                kCGImagePropertyTIFFModel: "SONY ILCE-7SM2"
            ] as [CFString: Any]
        ]
        CGImageDestinationAddImage(destination, image, properties as CFDictionary)
        guard CGImageDestinationFinalize(destination),
              let source = CGImageSourceCreateWithURL(url as CFURL, [kCGImageSourceShouldCache: false] as CFDictionary),
              CGImageSourceGetCount(source) == 1,
              let extracted = CGImageSourceCopyPropertiesAtIndex(source, 0, [kCGImageSourceShouldCache: false] as CFDictionary)
                as? [CFString: Any] else {
            throw SkipTest(reason: "ImageIO could not write or inspect the native JPEG fixture")
        }
        let fileSize = (try FileManager.default.attributesOfItem(atPath: url.path)[.size] as? NSNumber)?.int64Value
        let parsed = PhotoMetadataParser.parse(properties: extracted,
            xmp: CGImageSourceCopyMetadataAtIndex(source, 0, nil), filename: url.lastPathComponent,
            fallbackWidth: 0, fallbackHeight: 0, fileSizeBytes: fileSize,
            photosCreationDate: nil, photosModificationDate: nil,
            sourceType: CGImageSourceGetType(source) as String?)

        checkEqual(parsed.fileType, "JPEG", "ImageIO type identifies the original JPEG")
        checkEqual(parsed.width, 48, "ImageIO returns native width without pixel decoding")
        checkEqual(parsed.height, 32, "ImageIO returns native height without pixel decoding")
        checkEqual(parsed.cameraModel, "SONY ILCE-7SM2", "TIFF camera model survives a native ImageIO round trip")
        checkEqual(parsed.aperture, 2.8, "EXIF aperture survives a native ImageIO round trip")
        checkEqual(parsed.iso, 400, "EXIF ISO survives a native ImageIO round trip")
        checkEqual(parsed.fileSizeBytes, fileSize, "the recorded source file size is reported exactly")
    })

    Registry.shared.add(suite: suite, TestCase(name: "native HEIC preserves dimensions, file size and embedded capture time", knownBug: nil) {
        let directory = try Temporary.directory("photo-metadata-heic")
        let url = directory.appendingPathComponent("iphone.heic")
        guard let context = CGContext(data: nil, width: 64, height: 40,
                                      bitsPerComponent: 8, bytesPerRow: 0,
                                      space: CGColorSpaceCreateDeviceRGB(),
                                      bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue),
              let image = context.makeImage(),
              let destination = CGImageDestinationCreateWithURL(url as CFURL, "public.heic" as CFString, 1, nil) else {
            throw SkipTest(reason: "native HEIC encoding is unavailable")
        }
        let properties: [CFString: Any] = [kCGImagePropertyExifDictionary: [
            kCGImagePropertyExifDateTimeOriginal: "2025:07:08 09:10:11",
            kCGImagePropertyExifOffsetTimeOriginal: "-04:00"
        ] as [CFString: Any]]
        CGImageDestinationAddImage(destination, image, properties as CFDictionary)
        guard CGImageDestinationFinalize(destination),
              let source = CGImageSourceCreateWithURL(url as CFURL, [kCGImageSourceShouldCache: false] as CFDictionary),
              let extracted = CGImageSourceCopyPropertiesAtIndex(source, 0, [kCGImageSourceShouldCache: false] as CFDictionary)
                as? [CFString: Any] else {
            throw SkipTest(reason: "ImageIO could not write or inspect the native HEIC fixture")
        }
        let fileSize = try photoFixtureFileSize(at: url)
        let parsed = PhotoMetadataParser.parse(properties: extracted,
            xmp: CGImageSourceCopyMetadataAtIndex(source, 0, nil), filename: url.lastPathComponent,
            fallbackWidth: 0, fallbackHeight: 0, fileSizeBytes: fileSize,
            photosCreationDate: nil, photosModificationDate: nil,
            sourceType: CGImageSourceGetType(source) as String?)

        checkEqual(parsed.fileType, "HEIC", "HEIC source type is identified")
        checkEqual(parsed.width, 64, "HEIC native width is available")
        checkEqual(parsed.height, 40, "HEIC native height is available")
        checkEqual(parsed.fileSizeBytes, fileSize, "HEIC original file size is exact")
        check(parsed.captureDate != nil, "HEIC embedded capture time is available")
        checkEqual(parsed.captureDateSource, .exifOriginal, "capture time is distinct from Photos library date")
    })

    Registry.shared.add(suite: suite, TestCase(name: "Sony EXIF is preserved and formatted as recorded", knownBug: nil) {
        let exif: [CFString: Any] = [
            kCGImagePropertyExifDateTimeOriginal: "2024:05:16 11:20:30",
            kCGImagePropertyExifOffsetTimeOriginal: "+02:00",
            kCGImagePropertyExifExposureTime: 1.0 / 125.0,
            kCGImagePropertyExifFNumber: 1.8,
            kCGImagePropertyExifISOSpeedRatings: [1600],
            kCGImagePropertyExifExposureBiasValue: -0.3,
            kCGImagePropertyExifExposureProgram: 3,
            kCGImagePropertyExifExposureMode: 0,
            kCGImagePropertyExifMeteringMode: 5,
            kCGImagePropertyExifFlash: 0,
            kCGImagePropertyExifWhiteBalance: 0,
            kCGImagePropertyExifFocalLength: 35.0,
            kCGImagePropertyExifFocalLenIn35mmFilm: 35,
            kCGImagePropertyExifLensMake: "Sony",
            kCGImagePropertyExifLensModel: "FE 35mm F1.8"
        ]
        let tiff: [CFString: Any] = [
            kCGImagePropertyTIFFMake: "SONY",
            kCGImagePropertyTIFFModel: "SONY ILCE-7SM2"
        ]
        let properties: [CFString: Any] = [
            kCGImagePropertyPixelWidth: 4240,
            kCGImagePropertyPixelHeight: 2832,
            kCGImagePropertyOrientation: 1,
            kCGImagePropertyProfileName: "sRGB IEC61966-2.1",
            kCGImagePropertyExifDictionary: exif,
            kCGImagePropertyTIFFDictionary: tiff
        ]
        let parsed = PhotoMetadataParser.parse(properties: properties, xmp: nil,
            filename: "DSC01234.ARW", fallbackWidth: 0, fallbackHeight: 0, fileSizeBytes: 18_000_000,
            photosCreationDate: nil, photosModificationDate: nil, sourceType: "com.sony.raw-image")

        checkEqual(parsed.filename, "DSC01234.ARW", "original Sony filename remains intact")
        checkEqual(parsed.fileType, "RAW", "Sony raw formats are identified as RAW")
        checkEqual(parsed.cameraMake, "SONY", "camera manufacturer is read from TIFF")
        checkEqual(parsed.cameraModel, "SONY ILCE-7SM2", "Sony model identifier is preserved verbatim")
        checkEqual(parsed.lensModel, "FE 35mm F1.8", "lens model is read from EXIF")
        checkEqual(parsed.focalLengthMM, 35, "focal length remains in millimeters")
        checkEqual(parsed.focalLength35MM, 35, "35mm-equivalent focal length is read")
        checkEqual(parsed.aperture, 1.8, "aperture is read without inference")
        checkEqual(parsed.exposureTime, 1.0 / 125.0, "exposure time is preserved")
        checkEqual(parsed.iso, 1600, "ISO is read from the recorded ratings")
        checkEqual(PhotoMetadataFormatter.aperture(parsed.aperture ?? 0), "ƒ/1.8", "aperture formatting")
        checkEqual(PhotoMetadataFormatter.shutterSpeed(parsed.exposureTime ?? 0), "1/125 s", "shutter formatting")
        checkEqual(PhotoMetadataFormatter.ev(parsed.exposureCompensationEV ?? 0), "-0.3 EV", "EV formatting")
        checkEqual(parsed.exposureProgram, "Aperture priority", "exposure program label")
        checkEqual(parsed.meteringMode, "Pattern", "metering mode label")
        checkEqual(parsed.whiteBalance, "Auto", "white balance label")
        checkEqual(parsed.flash, "Did not fire", "flash status is based on the recorded EXIF flag")
        checkEqual(parsed.width, 4240, "native source width")
        checkEqual(parsed.height, 2832, "native source height")
        let expectedCaptureDate = ISO8601DateFormatter().date(from: "2024-05-16T09:20:30Z")
        checkEqual(parsed.captureDate, expectedCaptureDate, "EXIF offset is applied to the original capture time")
        checkEqual(parsed.captureDateSource, .exifOriginal, "original capture timestamp takes precedence")
    })

    Registry.shared.add(suite: suite, TestCase(name: "IPTC and XMP keywords stay separate from unavailable Photos keywords", knownBug: nil) {
        let xmp = CGImageMetadataCreateMutable()
        check(CGImageMetadataRegisterNamespaceForPrefix(xmp,
            "http://purl.org/dc/elements/1.1/" as CFString, "dc" as CFString, nil), "register Dublin Core XMP namespace")
        check(CGImageMetadataSetValueWithPath(xmp, nil, "dc:subject[0]" as CFString, "sunset" as CFString), "add XMP subject keyword")
        check(CGImageMetadataSetValueWithPath(xmp, nil, "dc:title[0]" as CFString, "Summer coast" as CFString), "add XMP title")
        let iptc: [CFString: Any] = [
            kCGImagePropertyIPTCKeywords: ["travel", "beach"],
            kCGImagePropertyIPTCObjectName: "Coastline",
            kCGImagePropertyIPTCCaptionAbstract: "A quiet beach at sunset"
        ]
        let properties: [CFString: Any] = [
            kCGImagePropertyIPTCDictionary: iptc,
            kCGImagePropertyPixelWidth: 2400,
            kCGImagePropertyPixelHeight: 1600
        ]
        let parsed = PhotoMetadataParser.parse(properties: properties, xmp: xmp,
            filename: "coast.jpg", fallbackWidth: 1, fallbackHeight: 1, fileSizeBytes: 1_280_000,
            photosCreationDate: nil, photosModificationDate: nil, sourceType: "public.jpeg")

        checkEqual(parsed.embeddedKeywords, ["travel", "beach", "sunset"], "embedded keywords merge IPTC and XMP without duplicates")
        checkEqual(parsed.title, "Coastline", "IPTC object name takes priority over the XMP title")
        checkEqual(parsed.description, "A quiet beach at sunset", "IPTC caption is available")
        check(parsed.copyableText().contains("Photos.app library keywords: unavailable through public PhotoKit APIs"),
              "copied text does not imply embedded keywords came from the Photos library")
    })

    Registry.shared.add(suite: suite, TestCase(name: "missing EXIF keeps basic file information and invents no camera values", knownBug: nil) {
        let parsed = PhotoMetadataParser.parse(properties: [:], xmp: nil,
            filename: "stripped.heic", fallbackWidth: 3024, fallbackHeight: 4032, fileSizeBytes: 2_400_000,
            photosCreationDate: Date(timeIntervalSince1970: 1_700_000_000), photosModificationDate: nil,
            sourceType: "public.heic")
        checkEqual(parsed.fileType, "HEIC", "HEIC type uses the resource type")
        checkEqual(parsed.width, 3024, "PhotoKit fallback preserves native width")
        checkEqual(parsed.height, 4032, "PhotoKit fallback preserves native height")
        checkEqual(parsed.fileSizeBytes, 2_400_000, "the original byte count is retained")
        checkNil(parsed.captureDate, "Photos library date is not relabelled as EXIF capture date")
        checkNil(parsed.cameraMake, "camera make is not guessed")
        checkNil(parsed.cameraModel, "camera model is not guessed")
        checkNil(parsed.aperture, "aperture is not guessed")
        checkEqual(parsed.embeddedKeywords, [], "stripped metadata yields no fabricated keywords")
        checkEqual(parsed.photosCreationDate, Date(timeIntervalSince1970: 1_700_000_000), "library date remains separately available")
        checkEqual(PhotoMetadataFormatter.fileSize(12_800_000), "12.8 MB", "file size uses a readable decimal unit")
    })

    Registry.shared.add(suite: suite, TestCase(name: "source kind and original resource reads are cancellable", knownBug: nil) {
        check(PhotosMedia.isPhotos(URL(string: "waifu-photos://asset/example")!), "PhotoKit source scheme is recognized")
        check(!PhotosMedia.isPhotos(URL(fileURLWithPath: "/tmp/ordinary.jpg")), "ordinary files are not classified as PhotoKit assets")

        let directory = try Temporary.directory("photo-metadata-cancellation")
        let temporaryURL = directory.appendingPathComponent("resource")
        let probe = PhotoMetadataRequestProbe()
        let reader = PhotoResourceDataReader(temporaryURL: temporaryURL,
            startRequest: { _, _ in probe.started(); return 77 },
            cancelRequest: { id in probe.cancelled(id) })
        let task = Task { try await reader.read() }
        await probe.waitForStart()
        task.cancel()
        let error = await checkThrowsAsync("cancelling an obsolete metadata read") { try await task.value }
        check(error is CancellationError, "cancellation reaches the metadata waiter")
        checkEqual(probe.cancelledRequestID, 77, "the underlying PhotoKit request is cancelled")
        check(!FileManager.default.fileExists(atPath: temporaryURL.path), "partial original data is removed")
    })
}

private func photoFixtureFileSize(at url: URL) throws -> Int64? {
    (try FileManager.default.attributesOfItem(atPath: url.path)[.size] as? NSNumber)?.int64Value
}

private final class PhotoMetadataRequestProbe: @unchecked Sendable {
    private let lock = NSLock()
    private var hasStarted = false
    private var cancellation: Int32?

    var cancelledRequestID: Int32? { lock.withLock { cancellation } }

    func started() { lock.withLock { hasStarted = true } }
    func cancelled(_ id: Int32) { lock.withLock { cancellation = id } }

    func waitForStart() async {
        for _ in 0..<200 {
            if lock.withLock({ hasStarted }) { return }
            try? await Task.sleep(for: .milliseconds(2))
        }
    }
}
