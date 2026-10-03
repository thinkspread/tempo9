// Copyright (c) 2026 Jiejing Zhang.
//
// A past call's arguments arrive from OpenAI clients as a JSON string, and
// chat templates iterate them as a mapping. Before decoding, swift-jinja
// rendered every historical call with no arguments at all.

import XCTest
@testable import ChatTemplateKit

final class ToolCallArgumentsTests: XCTestCase {
    /// The Qwen3.5 template's tool-call block, reduced to what matters here.
    private let qwenLike = """
        {%- for m in messages %}\
        {%- if m.tool_calls is defined %}\
        {%- for tool_call in m.tool_calls %}\
        {%- if tool_call.function is defined %}{%- set tool_call = tool_call.function %}{%- endif %}\
        <function={{ tool_call.name }}>\
        {%- if tool_call.arguments is defined %}\
        {%- for args_name, args_value in tool_call.arguments|items %}\
        <parameter={{ args_name }}>{{ args_value }}</parameter>\
        {%- endfor %}{%- endif %}</function>\
        {%- endfor %}{%- endif %}{%- endfor %}
        """

    private func call(_ arguments: String) -> String {
        """
        {"messages": [{"role": "assistant", "content": null, "tool_calls": [
          {"id": "c0", "type": "function",
           "function": {"name": "run", "arguments": \(arguments)}}]}],
         "add_generation_prompt": false}
        """
    }

    func testStringArgumentsAreRenderedInClientOrder() throws {
        let tmpl = try ChatTemplate(source: qwenLike)
        // A JSON string, as the OpenAI wire format sends it; "z" before "a"
        // so a sorted re-serialization would show.
        let out = try tmpl.render(contextJSON: call(#""{\"z\": \"ls\", \"a\": 2}""#))
        XCTAssertEqual(out,
            "<function=run><parameter=z>ls</parameter><parameter=a>2</parameter></function>")
    }

    func testObjectArgumentsAreUnchanged() throws {
        let tmpl = try ChatTemplate(source: qwenLike)
        let out = try tmpl.render(contextJSON: call(#"{"z": "ls", "a": 2}"#))
        XCTAssertEqual(out,
            "<function=run><parameter=z>ls</parameter><parameter=a>2</parameter></function>")
    }

    /// Not a JSON object: left exactly as sent, so a template that prints the
    /// raw string still gets it.
    func testNonObjectStringIsLeftAlone() throws {
        let raw = try ChatTemplate(source: """
            {%- for m in messages %}{%- for c in m.tool_calls %}\
            [{{ c.function.arguments }}]{%- endfor %}{%- endfor %}
            """)
        XCTAssertEqual(try raw.render(contextJSON: call(#""not json""#)), "[not json]")
        XCTAssertEqual(try raw.render(contextJSON: call(#""[1, 2]""#)), "[[1, 2]]")
    }

    /// The Swift-dictionary entry point goes through the same path.
    func testMessagesEntryPointDecodesToo() throws {
        let tmpl = try ChatTemplate(source: qwenLike)
        let msgs: [[String: Any]] = [[
            "role": "assistant", "content": NSNull(),
            "tool_calls": [["id": "c0", "type": "function",
                            "function": ["name": "run",
                                         "arguments": #"{"cmd": "pwd"}"#]]],
        ]]
        XCTAssertEqual(try tmpl.render(messages: msgs, addGenerationPrompt: false),
                       "<function=run><parameter=cmd>pwd</parameter></function>")
    }
}
