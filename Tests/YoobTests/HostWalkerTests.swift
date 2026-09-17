import XCTest
@testable import YoobRealistic

final class HostWalkerTests: XCTestCase {
    private let window = AvatarPack.CalmHostWindow(first: 1, count: 9, framesPerHost: 3, wideLast: 34)

    func testWithoutSpeechAheadTheHeadSwaysInsideTheCalmStretchAtTheIdlePace() {
        var walker = HostWalker(window: window, startHost: 1)
        let hosts = (0..<120).map { _ in walker.next(speechAhead: 0) }
        XCTAssertTrue(hosts.allSatisfy { (1...9).contains($0) })
        XCTAssertEqual(Set(hosts), Set(1...9), "the whole calm stretch is used")
        for (a, b) in zip(hosts, hosts.dropFirst()) { XCTAssertLessThanOrEqual(abs(a - b), 1) }
        // One host step every three call frames.
        XCTAssertEqual(zip(hosts, hosts.dropFirst()).filter { $0 != $1 }.count, 40)
    }

    func testLongSpeechWalksOutToTheWideStretch() {
        var walker = HostWalker(window: window, startHost: 5)
        let hosts = (0..<200).map { _ in walker.next(speechAhead: 120) }
        XCTAssertEqual(hosts.max(), 34)
        for (a, b) in zip(hosts, hosts.dropFirst()) { XCTAssertLessThanOrEqual(abs(a - b), 1, "no jumps") }
    }

    func testTheHeadIsBackInTheCalmStretchWhenSpeechEnds() {
        // Any amount of known speech, counting down to the end: whatever the walker did, it is calm at the end.
        for total in stride(from: 0, through: 150, by: 7) {
            var walker = HostWalker(window: window, startHost: 3)
            var hosts: [Int] = []
            for ahead in stride(from: total, through: 0, by: -1) { hosts.append(walker.next(speechAhead: ahead)) }
            XCTAssertLessThanOrEqual(hosts.last ?? 0, 9, "speech of \(total) frames ends calm")
            for (a, b) in zip(hosts, hosts.dropFirst()) { XCTAssertLessThanOrEqual(abs(a - b), 1) }
            if total >= 60 { XCTAssertGreaterThan(hosts.max() ?? 0, 20, "a long reply moves the head") }
        }
    }

    func testSpeechEndingSoonerThanExpectedStillWalksBackOneFramePerFrame() {
        var walker = HostWalker(window: window, startHost: 1)
        for _ in 0..<40 { _ = walker.next(speechAhead: 120) }
        let far = walker.host
        XCTAssertGreaterThan(far, 20)
        // The rest of the reply turns out to be short: the walk back starts at once.
        var hosts: [Int] = []
        for ahead in stride(from: 12, through: 0, by: -1) { hosts.append(walker.next(speechAhead: ahead)) }
        XCTAssertEqual(hosts.first, far - 1)
        for (a, b) in zip(hosts, hosts.dropFirst()) { XCTAssertEqual(b, a - 1) }
    }
}
