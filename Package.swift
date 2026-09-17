// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "Yoob",
    platforms: [.iOS(.v17), .macOS(.v14)],
    products: [
        .library(name: "Yoob", targets: ["Yoob"]),
    ],
    targets: [
        .target(name: "Yoob", dependencies: ["YoobRealistic", "YoobAnime"],
                resources: [.copy("PrivacyInfo.xcprivacy")],
                swiftSettings: [.swiftLanguageMode(.v5)]),
        .target(name: "YoobRealistic", swiftSettings: [.swiftLanguageMode(.v5)]),
        .target(name: "YoobAnime", swiftSettings: [.swiftLanguageMode(.v5)]),
        .testTarget(name: "YoobTests", dependencies: ["Yoob", "YoobRealistic"]),
    ]
)
