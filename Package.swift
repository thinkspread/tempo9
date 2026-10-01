// swift-tools-version:5.9
// Copyright (c) 2026 Jiejing Zhang.

import PackageDescription
import Foundation

// Tempo9 on Apple platforms.
//
// The engine is C++ and mostly CUDA; this is the Swift side of it — enough
// to take a .gguf and an image or a sound and get tokens back, without a
// Python server in the middle.
//
// ONE package rather than four nested ones: a consumer depends on this
// repository by URL, and SwiftPM reads only the manifest at the root. The
// directory layout is kept as it was so history follows the files.
//
// The modules, bottom up. They are layered so that each depends only on what
// it needs, but only ONE of them is a product -- `Tempo9`; `products` below
// says why:
//
//   GGUFKit          read a .gguf, tokenize.  Foundation only.
//   ChatTemplateKit  render the model's own Jinja chat template.
//   VisionTowerKit   vision and audio front ends (Core ML, Metal, GGUF).
//   Tempo9Engine     the engine itself, in-process.
//   Tempo9           a session, and an OpenAI/Anthropic/Ollama-shaped server.
//
// GGUFKit is Foundation-only and ChatTemplateKit stays cross-platform; the
// towers and Tempo9 are Apple-only. CI asserts the first two.
/// The engine, as the linker needs to hear it.
///
/// This lives here now rather than in the demo app. Staging a Tempo9 build
/// for Swift consumption is the SDK's job: every consumer needs it, and the
/// app owning the only copy meant the SDK could not build its own tools --
/// which is how a benchmark once drifted onto a Python server and reported
/// those numbers as this one's.
///
/// -force_load on the engine only: models and operators register through
/// static initialisers, and the linker drops an archive member that nothing
/// references -- leaving an engine that links cleanly and knows no models.
let staticDir = Context.packageDirectory + "/staged-engine"

/// Is an engine build staged here?
///
/// The engine is not vendored: the maintainers' staging script copies its
/// archives into `staged-engine/`, which is gitignored, and until the engine
/// ships as a binary release nobody else has one. So a fresh clone has no
/// engine, and the targets that LINK one cannot be declared -- SwiftPM
/// builds every target of a bare `swift build`, and a declared-but-unlinkable
/// executable makes the whole command fail. That is what happened to CI: the
/// header claimed "everything here COMPILES without a Tempo9 build", which is
/// true, while the job ran `swift build` and `swift test`, which link.
///
/// Probing for the archive rather than reading a flag is deliberate, and is
/// the same rule the staging script applies to zmq: a flag can drift from
/// what was built, a missing file cannot.
///
/// Without an engine you still get: the `Tempo9` library (libraries do not
/// link), the portable tools, and every test -- linked against a stub of the
/// C ABI that fails every call (see `testEngine`).
let engineStaged = FileManager.default.fileExists(
    atPath: staticDir + "/liballspark_framework.a")
let engineLink: [LinkerSetting] = [
    .unsafeFlags([
        "-L\(staticDir)",
        "-lallspark_kernel", "-lggml", "-lggml-base", "-lggml-cpu",
        "-lxgrammar", "-lglog", "-lsmhasher", "-lipc", "-lgit_version",
        // No -lzmq: KV events are off on Apple (the engine build's
        // AS_KV_EVENTS), so the engine references no zmq_ symbols and the
        // staging script omits the archive.  Naming a library that is not
        // there fails the link outright -- which is the loud failure we
        // want over silently shipping an unusable MPL-2.0 dependency.
        "-lz",
        "-framework", "Accelerate", "-framework", "Metal",
        "-framework", "Foundation", "-framework", "CoreFoundation",
        "-Xlinker", "-force_load",
        "-Xlinker", "\(staticDir)/liballspark_framework.a",
    ]),
]

/// What the TEST targets link the C ABI against.
///
/// With an engine staged: the engine. Without one: CTempo9EngineStub, whose
/// every entry point fails with TE9_ERR_ENGINE. Most of the tests never call
/// into the engine at run time -- request parsing, refusals, the splitters,
/// the Ollama store -- and used to be skipped only because they could not
/// LINK; that left a fresh clone, and CI, running the ten GGUFKit tests and
/// nothing else. A test that does reach the engine now fails, loudly, rather
/// than passing against a fake.

/// The engine as a release ships it: packaging/make_xcframework.sh turns the
/// staged archives into staged-engine/Tempo9Engine.xcframework -- one
/// pre-linked object exporting only te9_*, plus a module map naming the system
/// frameworks it needs. When it is there it wins over the loose archives, and
/// the targets that link an engine are declared the way an outside app sees
/// them: a binary target, and no unsafe flags.
let engineXCFramework = FileManager.default.fileExists(
    atPath: staticDir + "/Tempo9Engine.xcframework/Info.plist")

/// The published engine: Tempo9Engine.xcframework.zip on a GitHub release,
/// and SwiftPM's checksum of it (packaging/release.sh prints both). Empty
/// until the first release. A release that does not change the engine keeps
/// pointing at the one that did: 1.1.0 changed only the Swift products and
/// kept the 1.0.0 asset; 1.1.1 ships a new engine (BF16 load and concurrent
/// decode fixes) and points at its own. When set, a checkout with no engine staged --
/// every outside developer's -- downloads it and links it like any binary
/// target; a staged engine still wins, so maintainers test what they build.
let releaseEngineURL =
    "https://github.com/thinkspread/tempo9/releases/download/v1.1.1/Tempo9Engine.xcframework.zip"
let releaseEngineChecksum =
    "c7a654c031b91792009eea4ecdda6ce078fd2a5f6c7a3fc1eaf66b7760d2a407"
let engineRelease = !engineXCFramework && !engineStaged && !releaseEngineURL.isEmpty

let engineAvailable = engineXCFramework || engineStaged || engineRelease
/// What an engine-linking target adds to its link: the flags above for loose
/// archives; nothing for an xcframework, local or downloaded (its module map
/// does it).
let engineLinkSettings: [LinkerSetting] =
    engineStaged && !engineXCFramework ? engineLink : []

let testEngine: (deps: [Target.Dependency], link: [LinkerSetting]) =
    engineAvailable ? ([], engineLinkSettings) : (["CTempo9EngineStub"], [])

/// The C ABI's module: the xcframework when there is one, otherwise a header
/// shim (include/tempo9_engine.h, kept in step with the engine's copy) whose
/// engine comes from the staged archives -- or, with neither, from the test
/// stub.
let engineModule: Target
if engineXCFramework {
    engineModule = .binaryTarget(name: "CTempo9Engine",
                                 path: "staged-engine/Tempo9Engine.xcframework")
} else if engineRelease {
    engineModule = .binaryTarget(name: "CTempo9Engine", url: releaseEngineURL,
                                 checksum: releaseEngineChecksum)
} else {
    engineModule = .systemLibrary(name: "CTempo9Engine",
                                  path: "Tempo9Kit/Sources/CTempo9Engine")
}

/// Maintainer tools are declared only where their source is, so a tree that
/// does not carry one never names a directory it does not have.
let hasASGraphDiff = FileManager.default.fileExists(
    atPath: Context.packageDirectory + "/Sources/asgraphdiff/main.swift")

// Built in steps with explicit types, not as one expression. The single
// `Package(...)` this replaced joined four target arrays with `+`, three of
// them behind ternaries: 5.5 ms to type-check with Xcode 27, and with the
// Xcode 16.4 on GitHub's macos-15 runners, "unable to type-check this
// expression in reasonable time" after two minutes. Keep it in statements.

// THREE public libraries.  An app developer writes `import Tempo9` and gets
// LocalSession -- chat, tool calls, vision, prefix cache.  GGUFKit reads a
// model file and tokenizes, on Foundation alone.  VisionTowerKit runs the
// vision and audio towers, and it is exported because LocalSession takes
// ImageEmbeddings and nothing else public could produce them: with Tempo9
// alone, an app outside this package could not show the model an image.
//
// The engine binding (Tempo9Engine) and the C shim (CTempo9Engine) stay
// internal: exporting them would make every symbol below the SDK a permanent
// API promise, and nobody outside needs to hold the engine directly -- the
// two facts an app does want from it, the build and the GEMM backend, are
// forwarded as Tempo9's EngineInfo.  ggufctl/vitctl/localctl (and
// asgraphdiff, where its source is) are development tools and are not
// shipped.
var products: [Product] = [
    .library(name: "Tempo9", targets: ["Tempo9"]),
    .library(name: "GGUFKit", targets: ["GGUFKit"]),
    .library(name: "VisionTowerKit", targets: ["VisionTowerKit"]),
]
if engineAvailable {
    // Product name is what users type; the TARGET must differ in more than
    // case from the "Tempo9" library -- macOS filesystems are
    // case-insensitive, so a target named "tempo9" collides with it and the
    // compiler reports it as "statements are not allowed at the top level"
    // in main.swift, which points nowhere near the cause.
    products.append(.executable(name: "tempo9", targets: ["Tempo9CLI"]))
}

let tempo9KitTestDeps: [Target.Dependency] =
    ["GGUFKit", "Tempo9", "Tempo9Engine"] + testEngine.deps
let tempo9TestDeps: [Target.Dependency] =
    ["Tempo9", "Tempo9Engine"] + testEngine.deps

var targets: [Target] = [
    // Foundation only, deliberately: a host that just needs to read a
    // model file and tokenize should not pull in a template engine.
    .target(name: "GGUFKit", path: "GGUFKit/Sources/GGUFKit"),
    .target(name: "ChatTemplateKit",
            dependencies: ["GGUFKit",
                           .product(name: "Jinja", package: "swift-jinja")],
            path: "GGUFKit/Sources/ChatTemplateKit"),

    .executableTarget(name: "ggufctl",
                      dependencies: ["GGUFKit", "ChatTemplateKit"],
                      path: "GGUFKit/Sources/ggufctl"),

    // No Python, numpy or coremltools: the point of the port is that a
    // host embeds the tower directly.
    .target(name: "VisionTowerKit", dependencies: ["GGUFKit"],
            path: "VisionTowerKit/Sources/VisionTowerKit"),
    .executableTarget(name: "vitctl", dependencies: ["VisionTowerKit"],
                      path: "VisionTowerKit/Sources/vitctl"),

    // The engine's C ABI; see `engineModule`. The engine is NOT vendored.
    engineModule,
    // GGUFKit for the tokenizer protocol the engine-side SentencePiece
    // tokenizer conforms to. The edge runs this way only -- GGUFKit
    // knows nothing about the engine -- so there is no cycle.
    .target(name: "Tempo9Engine", dependencies: ["CTempo9Engine", "GGUFKit"],
            path: "Tempo9Kit/Sources/Tempo9Engine"),

    .target(name: "Tempo9",
            dependencies: ["GGUFKit", "ChatTemplateKit",
                           "VisionTowerKit", "Tempo9Engine"],
            path: "Sources/Tempo9"),

    // Tempo9 is where two of the three regressions live, and testing it
    // without linking would only ever cover GGUFKit.
    .testTarget(name: "Tempo9KitTests",
                dependencies: tempo9KitTestDeps,
                path: "Tests/Tempo9KitTests",
                linkerSettings: testEngine.link),
    .testTarget(name: "GGUFKitTests",
                dependencies: ["GGUFKit", "ChatTemplateKit"],
                path: "GGUFKit/Tests/GGUFKitTests"),
    // Pure-Swift splitter logic; needs the C ABI linked because
    // Tempo9's other files call it. Discovered missing when
    // HarmonySplitterTests "passed" as 0 tests: this directory had never
    // been a test target, so ThinkSplitterTests had never run either.
    .testTarget(name: "Tempo9Tests",
                dependencies: tempo9TestDeps,
                path: "Tempo9Kit/Tests/Tempo9Tests",
                linkerSettings: testEngine.link),
]

if hasASGraphDiff {
    targets.append(.executableTarget(name: "asgraphdiff",
                                     path: "Sources/asgraphdiff"))
}

if engineAvailable {
    targets += [
        .executableTarget(name: "localctl",
                          dependencies: ["Tempo9", "Tempo9Engine",
                                         "VisionTowerKit"],
                          path: "Sources/localctl",
                          linkerSettings: engineLinkSettings),
        .executableTarget(name: "Tempo9CLI",
                          dependencies: ["Tempo9", "Tempo9Engine",
                                         "VisionTowerKit"],
                          path: "Sources/tempo9-cli",
                          linkerSettings: engineLinkSettings),
    ]
} else {
    // Tests only; no product depends on it. See `testEngine`.
    targets.append(.target(name: "CTempo9EngineStub",
                           path: "Tempo9Kit/Sources/CTempo9EngineStub",
                           cSettings: [.headerSearchPath("../CTempo9Engine/include")]))
}

let package = Package(
    name: "Tempo9Kit",
    // macOS 26, not 13. SwiftPM writes the declared platform into an
    // executable's LC_BUILD_VERSION as both minos AND sdk, and the Metal
    // runtime compiler resolves the Metal 4 headers the engine's kernels use
    // (metal_tensor, MetalPerformancePrimitives) by that sdk field. Declared
    // as .v13, the engine's tensor-API probe fails with "use of undeclared
    // identifier 'mpp'", the whole Metal backend turns itself off, and the
    // tempo9 CLI serves on the CPU without saying so. With Qwen3.5-9B
    // Q4_K_S on an M5 Pro (2026-09-29, three runs of each): a 1,004-token
    // prefill took 14.1-14.5 s that way against 0.79-0.85 s on Metal, and
    // decode ran at 23 tok/s against 52. The engine's Metal path needs
    // macOS 26 anyway.
    platforms: [.macOS("26.0"), .iOS(.v16)],
    products: products,
    dependencies: [
        // Only ChatTemplateKit depends on this. Chat templates are real
        // Jinja -- the Qwen3.5 one is 154 lines using macros, namespace(),
        // loop variables, tests, filters and raise_exception -- so writing a
        // "minimal subset" would mean writing an interpreter and then
        // discovering the next model's template needs the next corner of it.
        .package(url: "https://github.com/huggingface/swift-jinja.git",
                 from: "2.0.0"),
    ],
    targets: targets
)
