import SwiftUI

/// Shows a `YoobAvatar`: its poster while it loads, idle frames while it is silent, and the rendered face while it speaks.
/// The idle and speaking layers cross-fade as one picture, so the switch never flashes the background.
public struct YoobAvatarView: View {
    private let avatar: YoobAvatar
    private let contentMode: ContentMode
    private let fade: Double

    /// - Parameters:
    ///   - contentMode: `.fill` crops to the view (the default, for full-screen characters); `.fit` letterboxes.
    ///   - fade: Seconds for the idle ↔ speaking cross-fade.
    public init(_ avatar: YoobAvatar, contentMode: ContentMode = .fill, fade: Double = 0.15) {
        self.avatar = avatar
        self.contentMode = contentMode
        self.fade = fade
    }

    public var body: some View {
        ZStack {
            IdleLayer(avatar: avatar, contentMode: contentMode)
                .compositingGroup()
                .opacity(avatar.isShowingSpeech ? 0 : 1)
            if let frame = avatar.speechFrame {
                picture(frame)
                    .compositingGroup()
                    .opacity(avatar.isShowingSpeech ? 1 : 0)
            }
        }
        .animation(.easeInOut(duration: fade), value: avatar.isShowingSpeech)
        .clipped()
        .accessibilityElement()
        .accessibilityLabel(avatar.manifest?.displayName ?? "Character")
        .accessibilityAddTraits(.isImage)
    }

    private func picture(_ image: CGImage) -> some View {
        Image(decorative: image, scale: 1)
            .resizable()
            .interpolation(.high)
            .aspectRatio(avatar.aspectRatio, contentMode: contentMode)
            .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}

private struct IdleLayer: View {
    let avatar: YoobAvatar
    let contentMode: ContentMode
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    var body: some View {
        let frames = avatar.idleFrames
        TimelineView(.periodic(from: .now, by: 1 / max(1, avatar.idleFramesPerSecond))) { timeline in
            if let image = frames.isEmpty ? avatar.poster : frames[index(at: timeline.date, count: frames.count)] {
                Image(decorative: image, scale: 1)
                    .resizable()
                    .interpolation(.high)
                    .aspectRatio(avatar.aspectRatio, contentMode: contentMode)
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                Color.clear
            }
        }
    }

    /// Forward and back: 0, 1, … n-1, n-2, … 1, so the loop has no seam.
    private func index(at date: Date, count: Int) -> Int {
        guard count > 1, !reduceMotion, !avatar.isShowingSpeech else { return 0 }
        let period = 2 * (count - 1)
        let step = Int(date.timeIntervalSinceReferenceDate * avatar.idleFramesPerSecond) % period
        return step < count ? step : period - step
    }
}
