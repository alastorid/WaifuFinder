// swift-tools-version: 6.0
import PackageDescription
import Foundation
let allowTruFor=ProcessInfo.processInfo.environment["WAIFU_COMMERCIAL_DISTRIBUTION"] != "1"
let package = Package(
    name: "WaifuRuntime",
    platforms: [.macOS(.v15)],
    products: (allowTruFor ? [.executable(name:"waifu-trufor",targets:["WaifuTruFor"])] : []) + [.executable(name:"waifu-safire",targets:["WaifuSAFIRE"]),.executable(name: "waifu-seedvr2", targets: ["WaifuSeedVR2"]),
               .executable(name: "waifu-flashvsr", targets: ["WaifuFlashVSR"]),
               .executable(name: "waifu-face-validation", targets: ["WaifuFaceValidation"]),
               .executable(name: "waifu-pgtformer", targets: ["WaifuPGTFormer"]),
               .executable(name: "waifu-wdtagger", targets: ["WaifuWDTagger"]),
               .executable(name: "waifu-wdtagger-ane-probe", targets: ["WaifuWDTaggerANEProbe"]),
               .executable(name: "waifu-wdtagger-validation", targets: ["WaifuWDTaggerValidation"])],
    dependencies: [.package(url: "https://github.com/ml-explore/mlx-swift", exact: "0.31.2")],
    targets: (allowTruFor ? [
        .target(name:"TruForMLX",dependencies:[.product(name:"MLX",package:"mlx-swift")]),
        .executableTarget(name:"WaifuTruFor",dependencies:["TruForMLX",.product(name:"MLX",package:"mlx-swift")])] : []) + [
        .target(name:"SAFIREMLX",dependencies:[.product(name:"MLX",package:"mlx-swift")]),
        .executableTarget(name:"WaifuSAFIRE",dependencies:["SAFIREMLX",.product(name:"MLX",package:"mlx-swift")]),
        .target(name: "WDTaggerMLX", dependencies: [.product(name: "MLX", package: "mlx-swift")],
                resources: [.copy("Metal")]),
        .target(name: "WDTaggerANE", path: "Sources/WDTaggerANE", publicHeadersPath: "include",
                cSettings: [.unsafeFlags(["-fobjc-arc"])],
                linkerSettings: [.linkedFramework("Foundation"), .linkedFramework("IOSurface")]),
        .executableTarget(name: "WaifuWDTaggerValidation", dependencies: ["WDTaggerMLX", .product(name: "MLX", package: "mlx-swift")]),
        .executableTarget(name: "WaifuWDTagger", dependencies: ["WDTaggerMLX", .product(name: "MLX", package: "mlx-swift")]),
        .executableTarget(name: "WaifuWDTaggerANEProbe", dependencies: [
            "WDTaggerANE", "WDTaggerMLX", .product(name: "MLX", package: "mlx-swift")]),
        .executableTarget(name: "WaifuPGTFormer", dependencies: ["NativeVideoIO"]),
        .target(name: "FaceRestorationMLX", dependencies: [.product(name: "MLX", package: "mlx-swift"),
            .product(name: "MLXNN", package: "mlx-swift")]),
        .executableTarget(name: "WaifuFaceValidation", dependencies: ["FaceRestorationMLX", .product(name: "MLX", package: "mlx-swift")]),
        .target(name: "SeedVR2MLX", dependencies: [
            .product(name: "MLX", package: "mlx-swift"), .product(name: "MLXNN", package: "mlx-swift"),
            .product(name: "MLXFast", package: "mlx-swift"), .product(name: "MLXRandom", package: "mlx-swift")]),
        .target(name: "NativeVideoIO", dependencies: [.product(name: "MLX", package: "mlx-swift")]),
        .executableTarget(name: "WaifuSeedVR2", dependencies: ["SeedVR2MLX", "NativeVideoIO", .product(name: "MLX", package: "mlx-swift")]),
        .target(name: "FlashVSRMLX", dependencies: [.product(name: "MLX", package: "mlx-swift"),
            .product(name: "MLXNN", package: "mlx-swift")]),
        .executableTarget(name: "WaifuFlashVSR", dependencies: ["FlashVSRMLX", "NativeVideoIO", .product(name: "MLX", package: "mlx-swift")])
    ],
    swiftLanguageModes: [.v5]
)
