//
//  Compositor.swift
//  Shared frame container for the Serve320 render path.
//
//  The legacy 160px mouth-local compositor that used to live here was removed
//  with the F4/160 lane; CompositedFrame is the surviving contract between
//  Serve320Player (producer) and MetalFrameView (consumer).
//

import Foundation
import CoreGraphics

/// Product-visible timing for one Serve320 frame. `readyAt` is the CPU handoff
/// point: either a composited CGImage is ready or the GPU producer command has
/// been submitted. `presentedAt` comes from CAMetalDrawable's actual display
/// callback rather than the earlier SwiftUI handoff.
struct Serve320PresentationTimeline: Sendable {
    let frameIndex: Int
    let renderStartedAt: CFTimeInterval
    let readyAt: CFTimeInterval
}

struct Serve320PresentationSample: Sendable {
    let frameIndex: Int
    let presentedAt: CFTimeInterval
    let prepareMs: Double
    let readyToDrawMs: Double
    let commandMs: Double
    let gpuMs: Double
    let endToPresentMs: Double
}

struct CompositedFrame: @unchecked Sendable {
    let source: CGImage?
    let overlay: CGImage?
    /// Optional GPU-resident full canvas. Serve320's experimental direct
    /// presentation path fills this texture in the compositor and lets
    /// MetalFrameView consume it without a GPU readback, CGContext copy, or
    /// second texture upload. The surface lease keeps the ring slot alive
    /// until Core Image has finished reading it.
    let metalSurface: Serve320MetalCompositor.PresentationSurface?
    let overlayRect: CGRect
    let pixelWidth: Int
    let pixelHeight: Int
    /// Reply-global 25 fps index. Streamed geometry windows start at zero, so
    /// presentation motion must use this value rather than a chunk-local row or
    /// the visible pose will snap at every preparation boundary.
    let speechFrameIndex: Int
    /// Nil for idle/reference frames. Serve320 product frames carry this token
    /// through SwiftUI so MetalFrameView can close the render-to-present trace.
    let presentationTimeline: Serve320PresentationTimeline?

    var sourceBounds: CGRect {
        CGRect(x: 0, y: 0, width: pixelWidth, height: pixelHeight)
    }

    func attachingPresentationTimeline(_ timeline: Serve320PresentationTimeline)
        -> CompositedFrame {
        CompositedFrame(
            source: source,
            overlay: overlay,
            metalSurface: metalSurface,
            overlayRect: overlayRect,
            pixelWidth: pixelWidth,
            pixelHeight: pixelHeight,
            speechFrameIndex: speechFrameIndex,
            presentationTimeline: timeline)
    }
}
