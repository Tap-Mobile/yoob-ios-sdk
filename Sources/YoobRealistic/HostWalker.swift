import Foundation

/// Head motion while the companion speaks. Idle frames are rendered from the calm stretch of the host clip, so speech must be
/// back inside it whenever the voice stops. With speech still ahead the head walks further into the clip at its own speed
/// (one host frame per call frame); the reach is limited so the walk back always fits in the speech that is left, plus a
/// margin. Without enough speech ahead it sways inside the calm stretch at the idle loop's pace.
public struct HostWalker: Sendable, Equatable {
    public let window: AvatarPack.CalmHostWindow
    /// Call frames of speech kept in reserve after the walk back.
    public static let marginFrames = 10
    public private(set) var host: Int
    private var rising = true
    private var held = 0

    public init(window: AvatarPack.CalmHostWindow, startHost: Int) {
        self.window = window
        host = min(max(startHost, window.first), window.calmLast)
    }

    /// The host for the next call frame, given how many frames of speech are known to follow it (0 in silence).
    public mutating func next(speechAhead: Int) -> Int {
        let calmLast = window.calmLast
        let reach = max(0, speechAhead - Self.marginFrames)
        let upper = min(window.wideLast ?? calmLast, max(calmLast, calmLast + reach))
        if host > upper {
            // Not enough speech left to stay out here: head back one host frame per call frame.
            host -= 1; rising = false; held = 0
            return host
        }
        let fast = upper > calmLast
        held += 1
        if fast || held >= window.framesPerHost {
            held = 0
            if rising, host >= upper { rising = false }
            if !rising, host <= window.first { rising = true }
            host += rising ? 1 : -1
            host = min(max(host, window.first), upper)
        }
        return host
    }
}
