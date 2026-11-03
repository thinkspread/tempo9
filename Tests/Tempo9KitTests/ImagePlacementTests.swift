// Copyright (c) 2026 Jiejing Zhang.
//
// ImagePlacement(_:) is the one way a host turns a tower encoding into what
// the engine reads.  The CLI used to copy the fields by hand and left out
// Qwen3-VL's DeepStack maps; the engine takes an empty plane as "text only"
// and says nothing, so the only guard is this test.

import Testing
@testable import VisionTowerKit
@testable import Tempo9

@Suite("Image placement")
struct ImagePlacementTests {
    @Test func placementFromEncodingKeepsDeepStack() {
        let e = Encoding(embedding: [1, 2, 3, 4],
                         deepstack: [5, 6, 7, 8, 9, 10, 11, 12],
                         numDeepstackLayers: 2, tokens: 2, outHidden: 2,
                         gridH: 2, gridW: 4, contentKey: 42, tier: .gpu,
                         cacheHit: false, preprocessSeconds: 0,
                         towerSeconds: 0, bucketN: 1024)
        let p = ImagePlacement(e)
        #expect(p.embedding == e.embedding)
        #expect(p.deepstack == e.deepstack)
        #expect(p.numDeepstackLayers == 2)
        #expect(p.tokens == 2 && p.hidden == 2)
        #expect(p.gridH == 2 && p.gridW == 4 && p.contentKey == 42)
    }
}
