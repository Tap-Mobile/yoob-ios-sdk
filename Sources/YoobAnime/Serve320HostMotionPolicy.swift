//
//  Serve320HostMotionPolicy.swift
//  Product policy for the full-frame host that sits behind speech-owned lips.
//

import Darwin
import Foundation
import CoreGraphics

enum Serve320HostMotionPolicy: Equatable, CustomStringConvertible {
    case bundleLoop
    case fixed(frame: Int)
    /// Forward and back over `count` host frames from `first`, holding each for `framesPerHost` speech frames. The call
    /// screen's anime idle frames come from the same stretch, so idle and speech share one head pose.
    case calm(first: Int, count: Int, framesPerHost: Int)

    /// Keep the 25 fps host loop on A17-class and newer devices. Older devices
    /// use a stable host so render skip-ahead cannot turn a missed frame into a
    /// visible head-pose jump. Mouth geometry still advances on every rendered
    /// speech frame; this policy changes only the host canvas and its matching
    /// conditioning row.
    static func productDefault(
        environment: [String: String] = ProcessInfo.processInfo.environment,
        machineIdentifier: String = hardwareMachineIdentifier()
    ) -> Serve320HostMotionPolicy {
        switch environment["AVATAR_SERVE320_HOST_MOTION"]?.lowercased() {
        case "stable", "fixed":
            return .fixed(frame: 0)
        case "loop":
            return .bundleLoop
        default:
            break
        }

        guard let platformMajor = iPhonePlatformMajor(machineIdentifier) else {
            // Simulator, Mac, and unknown future platforms retain the measured
            // reference path. An explicit environment override remains
            // available for device QA.
            return .bundleLoop
        }
        return platformMajor < 16 ? .fixed(frame: 0) : .bundleLoop
    }

    /// Maps a chunk-local speech frame onto the reply-global host timeline.
    /// The offset is essential for streamed replies: every geometry window is
    /// indexed from zero, but the head loop must not snap back at each window.
    func frameIndex(
        forLocalSpeechFrame localFrame: Int,
        replyFrameOffset: Int,
        idleFrameCount: Int
    ) -> Int {
        guard idleFrameCount > 0 else { return 0 }
        switch self {
        case .bundleLoop:
            let globalFrame = localFrame + replyFrameOffset
            return ((globalFrame % idleFrameCount) + idleFrameCount) % idleFrameCount
        case let .fixed(frame):
            return min(max(frame, 0), idleFrameCount - 1)
        case let .calm(first, count, framesPerHost):
            guard count > 1, first >= 0, first + count <= idleFrameCount else { return min(max(first, 0), idleFrameCount - 1) }
            let globalFrame = max(0, localFrame + replyFrameOffset), period = 2 * (count - 1)
            let position = (globalFrame / max(1, framesPerHost)) % period
            return first + (position < count ? position : period - position)
        }
    }

    var cachesRepeatedCanvasFrame: Bool {
        if case .fixed = self { return true }
        return false
    }

    var description: String {
        switch self {
        case .bundleLoop:
            return "loop"
        case let .fixed(frame):
            return "stable:\(frame)"
        case let .calm(first, count, framesPerHost):
            return "calm:\(first)+\(count)/\(framesPerHost)"
        }
    }

    static func iPhonePlatformMajor(_ identifier: String) -> Int? {
        guard identifier.hasPrefix("iPhone") else { return nil }
        guard let major = identifier.dropFirst("iPhone".count)
            .split(separator: ",", maxSplits: 1).first else { return nil }
        return Int(major)
    }

    private static func hardwareMachineIdentifier() -> String {
        var systemInfo = utsname()
        guard uname(&systemInfo) == 0 else { return "unknown" }
        return withUnsafePointer(to: &systemInfo.machine) { pointer in
            pointer.withMemoryRebound(to: CChar.self, capacity: 1) {
                String(cString: $0)
            }
        }
    }
}

/// Seam-free motion applied to the finished portrait while speech owns the
/// face. The neural mouth and its host are transformed together, so this cannot
/// create the cheek/chin handoff artifacts caused by a local face warp.
///
/// This is intentionally a slow whole-character sway, not a periodic nod. It
/// uses the reply-global frame index and incommensurate periods, which keeps
/// independently prepared Realtime windows continuous.
struct Serve320SpeechMotionPose: Equatable {
    let translationX: CGFloat
    let translationY: CGFloat
    let rotationRadians: CGFloat
    let scale: CGFloat
}

enum Serve320SpeechPresentationMotion {
    static let framesPerSecond = 25.0
    private static let rampSeconds = 0.6

    static func isEnabled(
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> Bool {
        switch environment["AVATAR_SPEECH_MOTION"]?.lowercased() {
        case "0", "false", "off", "disabled":
            return false
        default:
            return true
        }
    }

    static func pose(frameIndex: Int) -> Serve320SpeechMotionPose {
        let seconds = Double(max(0, frameIndex)) / framesPerSecond
        let ramp = smootherstep(min(1.0, seconds / rampSeconds))
        let tau = Double.pi * 2.0

        // Source-canvas pixels (1080x1920). A small translation plus rotation
        // about the lower torso reads as head/body motion on the phone without
        // bending facial pixels.
        let x = ramp * (
            6.0 * sin(tau * seconds / 7.3)
            + 2.0 * sin(tau * seconds / 11.9)
        )
        let y = ramp * 2.4 * sin(tau * seconds / 5.8)
        let degrees = ramp * (
            0.28 * sin(tau * seconds / 8.3)
            + 0.10 * sin(tau * seconds / 13.7)
        )

        return Serve320SpeechMotionPose(
            translationX: CGFloat(x),
            translationY: CGFloat(y),
            rotationRadians: CGFloat(degrees * Double.pi / 180.0),
            // Ramps with the motion so frame zero is identity. The overscan is
            // large enough to keep all four output corners covered at the
            // configured translation/rotation extrema.
            scale: CGFloat(1.0 + 0.032 * ramp)
        )
    }

    static func transform(frameIndex: Int, bounds: CGRect) -> CGAffineTransform {
        let pose = pose(frameIndex: frameIndex)
        // Lower-torso pivot: the body stays planted while the head travels more
        // than the hips. Core Image uses bottom-left image coordinates.
        let pivotX = bounds.midX
        let pivotY = bounds.minY + bounds.height * 0.38
        let coordinateScale = bounds.width / 1080.0
        let cosine = cos(pose.rotationRadians) * pose.scale
        let sine = sin(pose.rotationRadians) * pose.scale
        return CGAffineTransform(
            a: cosine,
            b: sine,
            c: -sine,
            d: cosine,
            tx: pivotX + pose.translationX * coordinateScale
                - cosine * pivotX + sine * pivotY,
            ty: pivotY + pose.translationY * coordinateScale
                - sine * pivotX - cosine * pivotY
        )
    }

    private static func smootherstep(_ value: Double) -> Double {
        let x = min(1.0, max(0.0, value))
        return x * x * x * (x * (x * 6.0 - 15.0) + 10.0)
    }
}
