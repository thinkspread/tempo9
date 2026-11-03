// Copyright (c) 2026 Jiejing Zhang.
//
// The built-in page (GET /) and the watch gate's frame route, without a
// socket or a tower: the response bytes, the page's promise to load nothing
// from the network, and the pooling the gate's distances are computed on.

import Testing
import Foundation
import CoreGraphics
import ImageIO
@testable import Tempo9

@Suite("Built-in page and watch gate")
struct WebPageTests {
    static func split(_ response: Data) -> (head: String, body: Data) {
        let sep = response.range(of: Data("\r\n\r\n".utf8))!
        return (String(decoding: response[..<sep.lowerBound], as: UTF8.self),
                Data(response[sep.upperBound...]))
    }

    @Test func pageResponseIsTheLockedDownPage() {
        let (head, body) = Self.split(OpenAIServer.pageResponse())
        #expect(head.hasPrefix("HTTP/1.1 200 OK\r\n"))
        #expect(head.contains("Content-Type: text/html; charset=utf-8"))
        #expect(head.contains("Content-Length: \(body.count)"))
        #expect(head.contains("Content-Security-Policy: \(WebPage.contentSecurityPolicy)"))
        #expect(head.contains("Cache-Control: no-store"))
        #expect(body == Data(WebPage.html.utf8))
    }

    /// "Nothing leaves the machine" is the policy's claim, so the policy is
    /// held to it: no source but this origin, and nothing in the page that
    /// would need one.
    @Test func pageLoadsNothingFromTheNetwork() {
        let csp = WebPage.contentSecurityPolicy
        #expect(csp.contains("default-src 'none'"))
        #expect(csp.contains("connect-src 'self'"))
        #expect(!csp.contains("*") && !csp.contains("http"))
        let html = WebPage.html
        for needle in ["http://", "https://", "<link", "@import", "src=\"//"] {
            #expect(!html.contains(needle), "page references \(needle)")
        }
    }

    /// A route renamed here and not in the page is a page that silently
    /// stops working; this is the cross-reference.
    @Test func pageCallsRoutesTheServerHas() {
        for route in ["/v1/models", "/api/show", "/v1/chat/completions",
                      "/v1/vision/frame"] {
            #expect(WebPage.html.contains("\"\(route)\""), "page lost \(route)")
        }
    }

    /// The defaults are Eyes On's measured thresholds (thinkspread/eyes-on,
    /// Watchpoint.swift); changing one
    /// is a measurement, not an edit.
    @Test func gateDefaultsAreTheMeasuredOnes() {
        #expect(WebPage.html.contains(#"id="fire" value="0.025""#))
        #expect(WebPage.html.contains(#"id="settle" value="0.010""#))
        #expect(WebPage.html.contains(#"id="backstop" value="60""#))
    }

    /// Every key a label (data-t) or the script (t("…")) asks for must be in
    /// both string tables, or the page shows English holes in its Chinese.
    @Test func everyStringHasBothLanguages() throws {
        let html = WebPage.html
        var keys = Set<String>()
        for pattern in [#"data-t="([A-Za-z]+)""#, #"\bt\("([A-Za-z]+)"\)"#] {
            let re = try NSRegularExpression(pattern: pattern)
            for m in re.matches(in: html, range: NSRange(html.startIndex..., in: html)) {
                keys.insert(String(html[Range(m.range(at: 1), in: html)!]))
            }
        }
        #expect(keys.count > 40)
        for k in keys {
            let defs = html.components(separatedBy: " \(k): ").count - 1
            #expect(defs >= 2, "\(k) is defined \(defs) time(s); want en and zh")
        }
    }

    @Test func pooledUnitIsTheMeanAtUnitLength() {
        // Two tokens, three wide: mean (0.5, 0.5, 0) -> (1/sqrt2, 1/sqrt2, 0).
        let p = ImagePlacement(embedding: [1, 0, 0, 0, 1, 0], tokens: 2, hidden: 3,
                               gridH: 1, gridW: 2, contentKey: 0)
        let v = OpenAIServer.pooledUnit(p)
        #expect(v.count == 3)
        #expect(abs(v[0] - 0.70710677) < 1e-6 && abs(v[1] - 0.70710677) < 1e-6)
        #expect(v[2] == 0)
        // An empty or all-zero embedding is a zero vector, never NaN.
        let z = OpenAIServer.pooledUnit(ImagePlacement(
            embedding: [0, 0, 0], tokens: 1, hidden: 3, gridH: 1, gridW: 1,
            contentKey: 0))
        #expect(z == [0, 0, 0])
        // tokens claiming more rows than the buffer holds read only the buffer.
        let short = OpenAIServer.pooledUnit(ImagePlacement(
            embedding: [3, 4], tokens: 5, hidden: 2, gridH: 1, gridW: 1,
            contentKey: 0))
        #expect(abs(short[0] - 0.6) < 1e-6 && abs(short[1] - 0.8) < 1e-6)
    }

    @Test func frameJSONCarriesThePooledVectorAsFloat32() throws {
        let p = ImagePlacement(embedding: [3, 4, 0, 0], tokens: 2, hidden: 2,
                               gridH: 1, gridW: 2, contentKey: 7)
        let j = OpenAIServer.frameJSON(p, encodeMs: 12.34)
        #expect(j["object"] as? String == "vision.frame")
        #expect(j["dim"] as? Int == 2)
        #expect(j["tokens"] as? Int == 2)
        #expect(j["encode_ms"] as? Double == 12.3)
        let raw = try #require(Data(base64Encoded: j["embedding_f32"] as? String ?? ""))
        let floats = raw.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
        #expect(floats == OpenAIServer.pooledUnit(p))
    }

    @Test func frameDecodeRefusesWhatChatRefuses() throws {
        #expect(throws: OpenAIServer.ImagePartError.self) {
            try OpenAIServer.decodeDataURI("https://example.com/cat.png")
        }
        #expect(throws: OpenAIServer.ImagePartError.self) {
            try OpenAIServer.decodeDataURI("data:image/png,not-base64")
        }
        // A real 2x2 PNG decodes.
        let ctx = CGContext(data: nil, width: 2, height: 2, bitsPerComponent: 8,
                            bytesPerRow: 8, space: CGColorSpaceCreateDeviceRGB(),
                            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
        let png = NSMutableData()
        let dest = CGImageDestinationCreateWithData(png, "public.png" as CFString, 1, nil)!
        CGImageDestinationAddImage(dest, ctx.makeImage()!, nil)
        #expect(CGImageDestinationFinalize(dest))
        let img = try OpenAIServer.decodeDataURI(
            "data:image/png;base64," + (png as Data).base64EncodedString())
        #expect(img.width == 2 && img.height == 2)
    }
}
