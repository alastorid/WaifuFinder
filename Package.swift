// swift-tools-version: 5.10
// This manifest exists only for IDE support (Xcode, SourceKit-LSP). The actual
// build is performed by ./build.sh and the tests by ./run-tests.sh, both with
// swiftc directly — SwiftPM cannot build or test this package on the target
// machine (`swift package dump-package` fails to link `libPackageDescription`).
//
// The test target below exists for navigation only. The real suite is the hand-rolled harness under Tests/,
// compiled into one binary with the production sources — not a SwiftPM test
// target, and not XCTest.

import PackageDescription

let package = Package(
    name: "WaifuFinder",
    platforms: [.macOS("15.0")],
    products: [
        .executable(name: "WaifuFinder", targets: ["WaifuFinder"])
    ],
    targets: [
        .executableTarget(
            name: "WaifuFinder",
            path: "Sources/Waifu"
        ),
        .testTarget(
            name: "WaifuFinderTests",
            dependencies: ["WaifuFinder"],
            path: "Tests"
        )
    ]
)
