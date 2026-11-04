// Copyright (c) 2026 Jiejing Zhang.
//
// Whitespace control must match transformers' apply_chat_template byte for
// byte. Every expected string below is what jinja2 renders with
// ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True), the
// environment transformers uses; none was written by hand.
//
// The cases that failed before: Unsloth's Qwen3.6 template (a trailing
// `{#- ... #}` left one extra newline at the end of every generation prompt)
// and Gemma 4 (block-tag newlines after <bos> and every turn, and a space
// after each `{` in its tool declarations).

import XCTest
@testable import ChatTemplateKit

final class ChatTemplateWhitespaceTests: XCTestCase {
    private func render(_ source: String) throws -> String {
        try ChatTemplate(source: source).render(contextJSON: "{}")
    }

    func testCommentLeftTrimEatsPrecedingNewline() throws {
        XCTAssertEqual(try render("a\n{#- c #}"), "a")
    }

    func testCommentTrimsBothSides() throws {
        XCTAssertEqual(try render("x  {#- c -#}\n  b"), "xb")
    }

    func testCommentRightTrim() throws {
        XCTAssertEqual(try render("a {# c -#}\n\n b"), "a b")
    }

    func testPlainCommentFollowsTrimBlocks() throws {
        XCTAssertEqual(try render("a\n{# c #}\nb"), "a\nb")
    }

    func testQwen36GenerationPromptTail() throws {
        let tail = "{%- if true %}{{- '<think>\\n\\n</think>\\n\\n' }}{%- endif %}\n"
            + "{#- Unsloth fixes #}"
        XCTAssertEqual(try render(tail), "<think>\n\n</think>\n\n")
    }

    func testCommentInsideStringLiteralIsText() throws {
        XCTAssertEqual(try render("{{ '{#- x #}' }}"), "{#- x #}")
    }

    func testLiteralBraceBeforeTrimmedComment() throws {
        XCTAssertEqual(try render("{\n{#- c #}{% if true %}x{% endif %}"), "{x")
    }

    func testLiteralBraceBeforeTrimmedExpressionKeepsNoSpace() throws {
        // Gemma 4's tool declarations: `properties:{ {{- ... -}}`.
        XCTAssertEqual(try render("properties:{ {{- 'city' -}} }"), "properties:{city}")
    }

    func testLiteralBraceBeforeTrimmedStatement() throws {
        XCTAssertEqual(try render("a:{\n  {%- if true %}b{% endif %}}"), "a:{b}")
    }

    func testBlockTagNewlineIsTrimmed() throws {
        XCTAssertEqual(try render("{% if true %}\nx\n{% endif %}\ny"), "x\ny")
    }

    func testLeadingSpacesBeforeBlockAreStripped() throws {
        XCTAssertEqual(try render("a\n    {% if true %}b{% endif %}"), "a\nb")
    }
}
