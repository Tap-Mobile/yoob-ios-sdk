//
//  Serve320Bundle.swift
//  Memory-mapped access to the m4ctrl320 serve bundle (see bundle/meta.json —
//  it documents every file/dtype/shape; built on b200 by build_bundle.py).
//
//  Serve lane ("Serve320"): canned features -> geometry CoreML -> pred6+contact
//  -> landmarks -> renderer CoreML (27ch) -> mouth crop -> QA9 compositor onto
//  idle-loop canvas. Mirrors arm_a/replay.py (serve path), arm_a/render_utils.py,
//  evaluation/per_frame_support.py and evaluation/canonical_composite.py from
//  talking-head-v3-ofir-v2 — python file/line cites inline where mirrored.
//
//  mmap pattern follows PrecomputedFaceBundle.swift (Data .mappedIfSafe).
//
//  The three u8 image arrays (idle_crops320 / idle_contours320 / ref_crops320)
//  may also ship as PNG-packed .pak twins (tools/bundle_pack/pack_bundle.py).
//  A .pak is preferred when present and decoded at load into the exact bytes
//  the .bin mmap yields; any pak problem falls back to the .bin (fail-closed).
//

import CoreGraphics
import Dispatch
import Foundation
import ImageIO

enum Serve320BundleError: Error {
    case missing(String)
    case badSize(String, Int, Int)
    case badNpy(String)
}

/// pca.json — contracts.load_pca_fit (arm_a/contracts.py:133-142):
/// reconstruct_geometry(scores6) = mean + (scores6 * score_std) @ components.
struct Serve320Pca: Decodable {
    let geometry_mean: [Double]      // (40,)
    let components: [[Double]]       // (6,40)
    let score_std: [Double]          // (6,)
}

/// ref_codebook.json — appearance_codebook_ref strata (arm_a/render_utils.py:32-61).
struct Serve320Codebook: Decodable {
    struct RefEntry: Decodable { let frame: Int; let row: Int }
    let strata: [String: [Int]]      // "closed"/"mid"/"wide" -> corpus frame ids
    let refs: [RefEntry]             // frame <-> row into ref_crops/ref_geom6/ref_anchors

    func rows(forStratum name: String) -> [Int] {
        let frames = strata[name] ?? []
        return frames.compactMap { f in refs.first(where: { $0.frame == f })?.row }
    }
}

/// Minimal meta.json surface the app depends on. n_frames.idle drives every
/// idle-host array size; the bundle refuses to load when it disagrees with the
/// shipped binaries (mmap size guards below fail closed).
private struct Serve320MetaFrames: Decodable {
    let idle: Int
    let features: Int
}
private struct Serve320MetaFile: Decodable {
    let n_frames: Serve320MetaFrames
}

final class Serve320Bundle {
    static let res = 320
    static let nFeat = 500
    static let nRefs = 30
    static let cropBytes = 3 * res * res          // u8 BGR planar per frame
    static let canvasW = 1080
    static let canvasH = 1920

    let root: URL
    /// Idle host-loop length, read from meta.json (n_frames.idle). Was a
    /// hard-coded 284 when the bundle only ever shipped the corpus-tail span.
    let nIdle: Int
    let pca: Serve320Pca
    let codebook: Serve320Codebook

    private let idleCrops: Data      // (284,3,320,320) u8 BGR planar CHW
    private let idleAnchors: [Float] // (284,4) [mid_x,mid_y,width,angle]; all == canon anchor
    private let stabBoxes: [Int32]   // (284,4) [x0,y0,x1,y1] on the 1080x1920 canvas
    private let idleContours: Data   // (284,320,320) u8 (value/255, binarize >= 128)
    private let idleMouthCenters: [Float] // (284,2) host mouth center in 320-crop coords
    private let hostLipY: [Float]    // (284,) host lower-lip y in 320-crop coords (QA8 cap floor)
    private let support320: [Float]  // (320,320) fixed v3.1 serve support
    private let inputMask: [Float]   // (320,320) 1 = mouth hole
    private let refCrops: Data       // (30,3,320,320) u8 BGR planar CHW
    private let refAnchors: [Float]  // (30,4)
    private let refGeom6: [Float]    // (30,6) standardized 6D
    private let featsWav: Data       // (500,1024) fp16
    private let featsPhon: Data      // (500,2048) fp16

    /// meta.json serve_config.r2b_gate: thr 0.70, gate_strength 0.9; closed_template_yoff
    /// exported from r2b_contact_wave3.pt (b200 ckpt metadata, 20 floats).
    static let r2bThreshold: Float = 0.70
    static let r2bStrength: Float = 0.9
    static let r2bTemplateYoff: [Float] = [
        -5.809554, -6.667094, -7.0917296, -7.0300374, -7.353298, -8.109091,
        -7.319757, 4.5667334, 14.743867, 19.228, 16.160421, 4.6456633,
        -4.4976387, -0.66066223, 1.155612, -1.8853236, -6.0676365, -0.9888773,
        2.4905596, 0.48984563,
    ]

    init(root: URL) throws {
        self.root = root
        func url(_ name: String) throws -> URL {
            let u = root.appendingPathComponent(name)
            guard FileManager.default.fileExists(atPath: u.path) else {
                throw Serve320BundleError.missing(name)
            }
            return u
        }
        func mmap(_ name: String, expectedBytes: Int) throws -> Data {
            let d = try Data(contentsOf: try url(name), options: .mappedIfSafe)
            guard d.count == expectedBytes else {
                throw Serve320BundleError.badSize(name, d.count, expectedBytes)
            }
            return d
        }
        func floats(_ name: String, count: Int) throws -> [Float] {
            let d = try mmap(name, expectedBytes: count * MemoryLayout<Float>.stride)
            return d.withUnsafeBytes { raw in
                Array(raw.bindMemory(to: Float.self))
            }
        }
        func ints32(_ name: String, count: Int) throws -> [Int32] {
            let d = try mmap(name, expectedBytes: count * MemoryLayout<Int32>.stride)
            return d.withUnsafeBytes { raw in
                Array(raw.bindMemory(to: Int32.self))
            }
        }

        /// Prefer the PNG-packed .pak twin when present (disk-size win; format in
        /// tools/bundle_pack/pack_bundle.py), decoded at load into the exact byte
        /// layout the .bin mmap yields — per-frame CRCs of the original .bin bytes
        /// are checked, so a wrong decode cannot be returned. Fail-closed: any pak
        /// problem (absent header field, decode error, CRC mismatch) falls through
        /// to the raw .bin, which still throws when missing.
        ///
        /// RAM note: decode-at-load trades disk size for the SAME steady-state
        /// footprint as the .bin path — with one honest difference: .bin pages are
        /// clean file-backed (evictable and refaultable under pressure) while
        /// decoded pak bytes are dirty anonymous memory that counts toward the
        /// jetsam limit. Lazy per-frame decode for crops (sketch, not implemented):
        /// keep the pak mmapped and decode inside idleCrop(_:) through a small LRU
        /// keyed by frame index; the accessor hands out interior pointers, so every
        /// LRU entry must outlive every pointer it produced (lifetime-contract
        /// change) — only worth it if crop RAM becomes the binding constraint.
        func packedOrRaw(_ binName: String, frames: Int, channels: Int) throws -> Data {
            let pakName = (binName as NSString).deletingPathExtension + ".pak"
            let pakURL = root.appendingPathComponent(pakName)
            if FileManager.default.fileExists(atPath: pakURL.path) {
                if let decoded = Serve320Pak.decodeVerified(url: pakURL,
                                                            expectCount: frames,
                                                            width: Self.res,
                                                            height: Self.res,
                                                            channels: channels) {
                    return decoded
                }
                print("Serve320Bundle: \(pakName) failed exact decode; falling back to .bin")
            }
            return try mmap(binName, expectedBytes: frames * channels * Self.res * Self.res)
        }

        let res = Self.res
        let meta = try JSONDecoder().decode(Serve320MetaFile.self,
                                            from: Data(contentsOf: try url("meta.json")))
        guard meta.n_frames.idle > 0, meta.n_frames.features == Self.nFeat else {
            throw Serve320BundleError.badNpy("meta.json n_frames \(meta.n_frames.idle)/\(meta.n_frames.features)")
        }
        nIdle = meta.n_frames.idle
        pca = try JSONDecoder().decode(Serve320Pca.self, from: Data(contentsOf: try url("pca.json")))
        codebook = try JSONDecoder().decode(Serve320Codebook.self,
                                            from: Data(contentsOf: try url("ref_codebook.json")))
        guard pca.geometry_mean.count == 40, pca.components.count == 6,
              pca.components.allSatisfy({ $0.count == 40 }), pca.score_std.count == 6 else {
            throw Serve320BundleError.badNpy("pca.json shape")
        }

        idleCrops = try packedOrRaw("idle_crops320.bin", frames: nIdle, channels: 3)
        idleAnchors = try floats("idle_anchors320.bin", count: nIdle * 4)
        stabBoxes = try ints32("idle_stab_boxes.bin", count: nIdle * 4)
        idleContours = try packedOrRaw("idle_contours320.bin", frames: nIdle, channels: 1)
        idleMouthCenters = try floats("idle_mouth_centers.bin", count: nIdle * 2)
        hostLipY = try floats("idle_host_lip_y320.bin", count: nIdle)
        support320 = try floats("support320.bin", count: res * res)
        inputMask = try floats("input_mask320.bin", count: res * res)
        refCrops = try packedOrRaw("ref_crops320.bin", frames: Self.nRefs, channels: 3)
        refAnchors = try floats("ref_anchors320.bin", count: Self.nRefs * 4)
        refGeom6 = try floats("ref_geom6.bin", count: Self.nRefs * 6)
        featsWav = try mmap("feats_wav2vec_l15.bin",
                            expectedBytes: Self.nFeat * 1024 * MemoryLayout<UInt16>.stride)
        featsPhon = try mmap("feats_phoneme_fa.bin",
                             expectedBytes: Self.nFeat * 2048 * MemoryLayout<UInt16>.stride)
    }

    /// Bundle.main locator (the bundle ships as a folder reference "bundle").
    static func defaultRoot() -> URL? {
        YoobResources.directory("bundle")
    }

    // MARK: - Typed accessors (slices valid while `self` is retained)

    /// Idle host crop (BGR u8, planar 3x320x320) for idle frame `idx`.
    func idleCrop(_ idx: Int) -> UnsafeRawBufferPointer {
        let off = idx * Self.cropBytes
        return idleCrops.withUnsafeBytes { raw in
            UnsafeRawBufferPointer(start: raw.baseAddress!.advanced(by: off),
                                   count: Self.cropBytes)
        }
    }

    /// Reference crop (BGR u8, planar 3x320x320) for codebook row `row`.
    func refCrop(_ row: Int) -> UnsafeRawBufferPointer {
        let off = row * Self.cropBytes
        return refCrops.withUnsafeBytes { raw in
            UnsafeRawBufferPointer(start: raw.baseAddress!.advanced(by: off),
                                   count: Self.cropBytes)
        }
    }

    /// Idle face contour (u8, 320x320; >= 128 == inside) — PRE-WARPED to stab space
    /// (meta.json contour_pipeline: erode-lower-2px@160 then raw_to_stab_contour; do not re-warp).
    func idleContour(_ idx: Int) -> UnsafeRawBufferPointer {
        let off = idx * Self.res * Self.res
        return idleContours.withUnsafeBytes { raw in
            UnsafeRawBufferPointer(start: raw.baseAddress!.advanced(by: off),
                                   count: Self.res * Self.res)
        }
    }

    func anchor(_ idx: Int) -> (midX: Float, midY: Float, width: Float, angle: Float) {
        let b = idx * 4
        return (idleAnchors[b], idleAnchors[b + 1], idleAnchors[b + 2], idleAnchors[b + 3])
    }

    func refAnchor(_ row: Int) -> (midX: Float, midY: Float, width: Float, angle: Float) {
        let b = row * 4
        return (refAnchors[b], refAnchors[b + 1], refAnchors[b + 2], refAnchors[b + 3])
    }

    func refGeom(_ row: Int) -> [Float] {
        let b = row * 6
        return Array(refGeom6[b..<(b + 6)])
    }

    /// Flat (30,6) ref geometry for the codebook retrieval loop.
    var refGeom6Flat: [Float] { refGeom6 }

    func stabBox(_ idx: Int) -> (x0: Int, y0: Int, x1: Int, y1: Int) {
        let b = idx * 4
        return (Int(stabBoxes[b]), Int(stabBoxes[b + 1]), Int(stabBoxes[b + 2]), Int(stabBoxes[b + 3]))
    }

    func hostLipY320(_ idx: Int) -> Float { hostLipY[idx] }
    func idleMouthCenter320(_ idx: Int) -> (x: Float, y: Float) {
        let b = idx * 2
        return (idleMouthCenters[b], idleMouthCenters[b + 1])
    }
    var support: [Float] { support320 }
    var hole: [Float] { inputMask }

    /// Idle span RAW mouth boxes (284,4 i32, K160 metadata) for the QA13 ROI
    /// (replay.py:469). Lazy-loaded; nil for bundles predating the file.
    func idleRawBoxes() -> [Int32]? {
        let u = root.appendingPathComponent("idle_raw_boxes.bin")
        guard let d = try? Data(contentsOf: u, options: .mappedIfSafe),
              d.count == nIdle * 4 * MemoryLayout<Int32>.stride else { return nil }
        return d.withUnsafeBytes { Array($0.bindMemory(to: Int32.self)) }
    }

    /// Canned clip per-frame RMS (feats_rms.bin, 500 f32) for the QA13 audio gate
    /// (replay.py:465-467). Lazy-loaded; nil if absent.
    func cannedRMS() -> [Float]? {
        let u = root.appendingPathComponent("feats_rms.bin")
        guard let d = try? Data(contentsOf: u, options: .mappedIfSafe),
              d.count == Self.nFeat * MemoryLayout<Float>.stride else { return nil }
        return d.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
    }

    /// Geometry CoreML input row `frame`: concat[wav2vec_l15 (1024), phoneme_fa (2048)]
    /// (meta.json serve_config.geometry_input; replay.py:419-421 val path concat order).
    /// fp16 on disk -> fp32 into `out` (must have 3072 elements).
    func featureRow(_ frame: Int, into out: UnsafeMutablePointer<Float>) {
        featsWav.withUnsafeBytes { raw in
            let src = raw.baseAddress!.advanced(by: frame * 1024 * 2)
                .bindMemory(to: UInt16.self, capacity: 1024)
            for j in 0..<1024 {
                out[j] = Float(Float16(bitPattern: src[j]))
            }
        }
        featsPhon.withUnsafeBytes { raw in
            let src = raw.baseAddress!.advanced(by: frame * 2048 * 2)
                .bindMemory(to: UInt16.self, capacity: 2048)
            for j in 0..<2048 {
                out[1024 + j] = Float(Float16(bitPattern: src[j]))
            }
        }
    }

    // MARK: - Parity .npy (bundle/parity/*) — minimal v1.0 reader, little-endian f4 only

    struct NpyArray {
        let shape: [Int]
        let values: [Float]
    }

    static func loadNpy(_ url: URL) throws -> NpyArray {
        let d = try Data(contentsOf: url)
        guard d.count > 10, d[0] == 0x93, d[1] == 0x4E, d[2] == 0x55, d[3] == 0x4D,
              d[4] == 0x50, d[5] == 0x59 else {
            throw Serve320BundleError.badNpy(url.lastPathComponent)
        }
        let major = d[6]
        let headerLen: Int
        let headerStart: Int
        if major == 1 {
            headerLen = Int(d[8]) | (Int(d[9]) << 8)
            headerStart = 10
        } else {
            guard d.count > 12 else { throw Serve320BundleError.badNpy(url.lastPathComponent) }
            headerLen = Int(d[8]) | (Int(d[9]) << 8) | (Int(d[10]) << 16) | (Int(d[11]) << 24)
            headerStart = 12
        }
        let dataStart = headerStart + headerLen
        guard d.count >= dataStart else {
            throw Serve320BundleError.badNpy(url.lastPathComponent)
        }
        let header = String(decoding: d[headerStart..<dataStart], as: UTF8.self)
        guard header.contains("'descr': '<f4'") || header.contains("\"descr\": \"<f4\""),
              header.contains("'fortran_order': False") || header.contains("\"fortran_order\": false") else {
            throw Serve320BundleError.badNpy("\(url.lastPathComponent): only C-order <f4 supported")
        }
        guard let shapeRange = header.range(of: #"\((\s*\d+\s*(,\s*\d+\s*)*,?)\)"#,
                                            options: .regularExpression) else {
            throw Serve320BundleError.badNpy("\(url.lastPathComponent): no shape")
        }
        let shape = header[shapeRange].split(whereSeparator: { "(), ".contains($0) })
            .compactMap { Int($0) }
        let count = shape.reduce(1, *)
        guard d.count >= dataStart + count * 4 else {
            throw Serve320BundleError.badNpy("\(url.lastPathComponent): truncated")
        }
        let values: [Float] = d.withUnsafeBytes { raw in
            let slice = UnsafeRawBufferPointer(rebasing: raw[dataStart...])
            return Array(slice.bindMemory(to: Float.self).prefix(count))
        }
        return NpyArray(shape: shape, values: values)
    }
}

// MARK: - S320PAK container (lossless PNG-per-frame packing of the u8 .bin arrays)
//
// Written by tools/bundle_pack/pack_bundle.py — that file documents the layout;
// summary (little-endian):
//   0   8  magic "S320PAK1"
//   8   4  version = 1
//  12   4  kind: 0 = gray8 (contours), 1 = BGR-planar crops stored as RGB8 PNGs
//  16   4  frame count            20/24  4+4  width/height (320)
//  28   4  channels (1|3)
//  32   count * { u64 pngOffset, u64 pngLength, u32 rawCRC32, u32 reserved }
//  ...  PNG streams (offsets absolute).
// rawCRC32 is the zlib CRC-32 of the frame's ORIGINAL .bin bytes (planar BGR
// for crops), so decode is proven byte-exact without the .bin present. The
// packer writes IHDR/IDAT/IEND-only PNGs (no gAMA/iCCP/sRGB), so ImageIO has
// nothing to color-manage; pixels come back verbatim from the data provider.
private enum Serve320Pak {
    private static let magic: [UInt8] = Array("S320PAK1".utf8)
    private static let headerSize = 32
    private static let indexEntrySize = 24

    /// One-shot failure flag for the concurrent decode below. Only ever moves
    /// false -> true; the lock keeps that transition free of data races so the
    /// whole decode stays deterministic (a bad frame fails the pak, never a
    /// partially-written buffer).
    private final class FailureFlag: @unchecked Sendable {
        private let lock = NSLock()
        private var tripped = false
        func trip() { lock.lock(); tripped = true; lock.unlock() }
        var isTripped: Bool { lock.lock(); defer { lock.unlock() }; return tripped }
    }

    /// Decode a .pak into the exact byte layout the raw .bin holds
    /// (count * channels * width * height, planar BGR CHW when channels == 3).
    /// Returns nil on ANY problem — caller falls back to the .bin.
    ///
    /// The index is parsed serially (cheap, ~24 B/frame) and the frames are then
    /// decoded with `concurrentPerform` into a pre-sized buffer: frame `i` owns
    /// exactly [i*frameBytes, (i+1)*frameBytes), so the writes are disjoint and
    /// the result is order-independent — byte-identical to the previous serial
    /// append. Frame decode is the cold-start cost that the .pak packing bought
    /// the install-size win with (156e747), so it is worth spreading over cores:
    /// measured on host (14 cores, real bundle) 307 ms -> 25.6 ms, output equal
    /// to both the old decoder and the raw .bin twins for all three arrays.
    static func decodeVerified(url: URL, expectCount: Int, width: Int, height: Int,
                               channels: Int) -> Data? {
        guard channels == 1 || channels == 3, expectCount > 0,
              let file = try? Data(contentsOf: url, options: .mappedIfSafe) else { return nil }
        let frameBytes = channels * width * height
        var out = Data(count: expectCount * frameBytes)
        let ok: Bool = file.withUnsafeBytes { (raw: UnsafeRawBufferPointer) -> Bool in
            guard raw.count >= headerSize + expectCount * indexEntrySize else { return false }
            let u8 = raw.bindMemory(to: UInt8.self)
            func u32(_ o: Int) -> UInt32 {
                UInt32(u8[o]) | UInt32(u8[o + 1]) << 8
                    | UInt32(u8[o + 2]) << 16 | UInt32(u8[o + 3]) << 24
            }
            func u64(_ o: Int) -> UInt64 { UInt64(u32(o)) | UInt64(u32(o + 4)) << 32 }
            guard (0..<8).allSatisfy({ u8[$0] == magic[$0] }) else { return false }
            let kind = u32(12)
            guard u32(8) == 1,
                  (kind == 0 && channels == 1) || (kind == 1 && channels == 3),
                  u32(16) == UInt32(expectCount),
                  u32(20) == UInt32(width), u32(24) == UInt32(height),
                  u32(28) == UInt32(channels) else { return false }

            var offsets = [Int](); offsets.reserveCapacity(expectCount)
            var lengths = [Int](); lengths.reserveCapacity(expectCount)
            var checksums = [UInt32](); checksums.reserveCapacity(expectCount)
            for i in 0..<expectCount {
                let e = headerSize + i * indexEntrySize
                let off = u64(e), len = u64(e + 8)
                guard len > 0, off <= UInt64(raw.count),
                      UInt64(raw.count) - off >= len else { return false }
                offsets.append(Int(off)); lengths.append(Int(len)); checksums.append(u32(e + 16))
            }

            guard let src = raw.baseAddress else { return false }
            let failure = FailureFlag()
            return out.withUnsafeMutableBytes { (dstRaw: UnsafeMutableRawBufferPointer) -> Bool in
                guard let dst = dstRaw.baseAddress, dstRaw.count == expectCount * frameBytes else {
                    return false
                }
                DispatchQueue.concurrentPerform(iterations: expectCount) { i in
                    if failure.isTripped { return }
                    guard let frame = decodeFrame(Data(bytes: src + offsets[i], count: lengths[i]),
                                                  width: width, height: height,
                                                  channels: channels),
                          frame.count == frameBytes else {
                        failure.trip()
                        return
                    }
                    frame.withUnsafeBufferPointer { fb in
                        guard let base = fb.baseAddress,
                              crc32(base, fb.count) == checksums[i] else {
                            failure.trip()
                            return
                        }
                        memcpy(dst + i * frameBytes, base, frameBytes)
                    }
                }
                return !failure.isTripped
            }
        }
        guard ok, out.count == expectCount * frameBytes else { return nil }
        return out
    }

    /// One PNG -> the frame's .bin byte layout: gray rows for channels == 1,
    /// planar BGR (CHW) for channels == 3 (inverse of the packer's BGR->RGB).
    private static func decodeFrame(_ png: Data, width: Int, height: Int,
                                    channels: Int) -> [UInt8]? {
        let opts = [kCGImageSourceShouldCache: false] as CFDictionary
        guard let src = CGImageSourceCreateWithData(png as CFData, opts),
              let img = CGImageSourceCreateImageAtIndex(src, 0, opts),
              img.width == width, img.height == height, img.bitsPerComponent == 8,
              let cf = img.dataProvider?.data, let base = CFDataGetBytePtr(cf) else {
            return nil
        }
        let bpp = img.bitsPerPixel / 8
        let bpr = img.bytesPerRow
        guard CFDataGetLength(cf) >= bpr * height, bpr >= width * bpp else { return nil }
        // ImageIO hands PNG8/PNG24 back unpromoted (bpp == channels); tolerate the
        // RGBX/XRGB promotions some OS builds do. A wrong guess here cannot ship —
        // the per-frame CRC in decodeVerified rejects it and the .bin path runs.
        var rOff = 0, gOff = 1, bOff = 2
        if channels == 1 {
            guard bpp == 1 else { return nil }
        } else if bpp == 4 {
            switch img.alphaInfo {
            case .noneSkipLast: break                              // R,G,B,X
            case .noneSkipFirst: rOff = 1; gOff = 2; bOff = 3      // X,R,G,B
            default: return nil
            }
            guard img.byteOrderInfo == .orderDefault
                    || img.byteOrderInfo == .order32Big else { return nil }
        } else {
            guard bpp == 3 else { return nil }
        }
        let wh = width * height
        var out = [UInt8](repeating: 0, count: channels * wh)
        out.withUnsafeMutableBufferPointer { dst in
            if channels == 1 {
                for y in 0..<height {
                    memcpy(dst.baseAddress! + y * width, base + y * bpr, width)
                }
            } else {
                for y in 0..<height {
                    let row = base + y * bpr
                    for x in 0..<width {
                        let s = x * bpp
                        let p = y * width + x
                        dst[p] = row[s + bOff]              // plane 0 = B
                        dst[wh + p] = row[s + gOff]         // plane 1 = G
                        dst[2 * wh + p] = row[s + rOff]     // plane 2 = R
                    }
                }
            }
        }
        return out
    }

    // zlib-compatible CRC-32 (poly 0xEDB88320), matches Python zlib.crc32.
    // Slice-by-8: eight 256-entry tables, table 0 being the classic
    // byte-at-a-time table. Same polynomial, same result — bit-identical to the
    // one-byte loop for every input (asserted over random lengths in the host
    // A/B harness and over the real .pak/.bin pairs). It exists because the
    // one-byte loop was ~40% of the pak decode: 70 MB of frame bytes are
    // checksummed on every cold start.
    private static let crcTables: [[UInt32]] = {
        var tables = [[UInt32]](repeating: [UInt32](repeating: 0, count: 256), count: 8)
        for n in 0..<256 {
            var c = UInt32(n)
            for _ in 0..<8 { c = (c & 1) != 0 ? 0xEDB8_8320 ^ (c >> 1) : c >> 1 }
            tables[0][n] = c
        }
        for n in 0..<256 {
            var c = tables[0][n]
            for k in 1..<8 {
                c = tables[0][Int(c & 0xFF)] ^ (c >> 8)
                tables[k][n] = c
            }
        }
        return tables
    }()

    private static func crc32(_ bytes: UnsafePointer<UInt8>, _ count: Int) -> UInt32 {
        let t = crcTables
        var c: UInt32 = ~0
        var i = 0
        while count - i >= 8 {
            let w0 = UInt32(bytes[i]) | UInt32(bytes[i + 1]) << 8
                | UInt32(bytes[i + 2]) << 16 | UInt32(bytes[i + 3]) << 24
            let w1 = UInt32(bytes[i + 4]) | UInt32(bytes[i + 5]) << 8
                | UInt32(bytes[i + 6]) << 16 | UInt32(bytes[i + 7]) << 24
            let a = c ^ w0
            c = t[7][Int(a & 0xFF)] ^ t[6][Int((a >> 8) & 0xFF)]
                ^ t[5][Int((a >> 16) & 0xFF)] ^ t[4][Int((a >> 24) & 0xFF)]
                ^ t[3][Int(w1 & 0xFF)] ^ t[2][Int((w1 >> 8) & 0xFF)]
                ^ t[1][Int((w1 >> 16) & 0xFF)] ^ t[0][Int((w1 >> 24) & 0xFF)]
            i += 8
        }
        while i < count {
            c = t[0][Int((c ^ UInt32(bytes[i])) & 0xFF)] ^ (c >> 8)
            i += 1
        }
        return ~c
    }
}
