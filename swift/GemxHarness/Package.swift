// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "GemxHarness",
    platforms: [.macOS(.v14)],
    dependencies: [
        .package(url: "https://github.com/microsoft/onnxruntime-swift-package-manager.git", from: "1.21.0"),
    ],
    targets: [
        .executableTarget(
            name: "GemxHarness",
            dependencies: [.product(name: "onnxruntime", package: "onnxruntime-swift-package-manager")],
            path: "Sources/GemxHarness",
            swiftSettings: [.unsafeFlags(["-Ounchecked"])]
        ),
    ]
)
