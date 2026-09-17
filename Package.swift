// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "Yoob",
    platforms: [.iOS(.v17), .macOS(.v14)],
    products: [
        .library(name: "Yoob", targets: ["Yoob"]),
        // Optional: drives a Yoob character from a LiveKit voice agent. Only apps that add this product link LiveKit.
        .library(name: "YoobLiveKit", targets: ["YoobLiveKit"]),
    ],
    dependencies: [
        .package(url: "https://github.com/livekit/client-sdk-swift.git", exact: "2.17.0"),
    ],
    targets: [
        .target(name: "Yoob", dependencies: ["YoobRealistic", "YoobAnime"],
                resources: [.copy("PrivacyInfo.xcprivacy")],
                swiftSettings: [.swiftLanguageMode(.v5)]),
        .target(name: "YoobRealistic", swiftSettings: [.swiftLanguageMode(.v5)]),
        .target(name: "YoobAnime", swiftSettings: [.swiftLanguageMode(.v5)]),
        .target(name: "YoobLiveKit",
                dependencies: ["Yoob", .product(name: "LiveKit", package: "client-sdk-swift")],
                swiftSettings: [.swiftLanguageMode(.v5)]),
        .testTarget(name: "YoobTests", dependencies: ["Yoob", "YoobRealistic"]),
        .testTarget(name: "YoobLiveKitTests", dependencies: ["YoobLiveKit", "Yoob"]),
    ]
)
