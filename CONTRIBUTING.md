# Contributing to Tempo9

Copyright (c) 2026 Jiejing Zhang.

Thank you for looking. This page says what this repository can take, how to
build and test it without the engine, and the one rule every commit has to
follow.

## What is here, and what is not

This repository is the Swift SDK (`import Tempo9`), the `tempo9` server CLI,
their tools, the manual and the examples, licensed Apache-2.0.

The inference engine is not here. It ships as a binary under its own licence
(see [NOTICE](NOTICE)), and changes to it cannot be taken through this
repository. If you hit an engine bug, open an issue with the model file, the
request and what came back; a reproduction through the CLI or the SDK is the
most useful form it can take.

## Sign your commits (DCO)

Every commit needs a `Signed-off-by:` line with your name and the email the
commit is authored with:

    Signed-off-by: Your Name <you@example.com>

`git commit -s` adds it. Signing off certifies the
[Developer Certificate of Origin 1.1](DCO): that you wrote the change, or
otherwise have the right to submit it under this repository's licence. A check
on every pull request refuses commits without it. Forgot? `git rebase
--signoff main`, then force-push the branch.

There is no separate contributor licence agreement. Contributions are taken
under the repository's licence (Apache-2.0), and you keep the copyright in
your work. A file you add gets a copyright line naming you:
`Copyright (c) 2026 Your Name.`

## How changes flow

Outside contributions are reviewed and merged here. The maintainers also
develop in a private tree that includes the engine: what is merged here is
carried into that tree with your authorship and sign-off intact, and the
maintainers' own changes arrive here as sync commits. Nothing about your pull
request has to change because of it.

## Build and test without the engine

    swift build
    swift test

A clone with no engine is a supported configuration. The libraries and tools
build, and the tests link against a stub of the engine's C ABI in which every
call fails (`Tempo9Kit/Sources/CTempo9EngineStub`). The tests that never call
into the engine at run time — request parsing, API refusals, the think and
harmony splitters, the Ollama store — run and must pass. A test that needs
the engine fails against the stub; it is never silently skipped.

XCTest and Swift Testing both print a green summary for zero tests, so CI
counts them and holds the totals to a floor. If you add tests, raise the
floors in `.github/workflows/ci.yml` in the same pull request.

You need macOS 26 or later with Xcode 26 or later; CI runs on `macos-26`.

## What makes a pull request easy to take

- One change per pull request, with a test that fails before it and passes
  after it.
- Comments that say why, not what. The code here explains its decisions where
  they are made; please keep that up.
- `GGUFKit` imports Foundation only, and `ChatTemplateKit` only cross-platform
  modules. CI enforces both.
- No new dependency without its reason in the pull request.

## Issues

Open issues for bugs, for API behaviour that differs from what OpenAI,
Anthropic or Ollama clients expect, and for model requests (which GGUF, which
quantization, what you want to run with it). Security problems go through
[SECURITY.md](SECURITY.md), not issues.
