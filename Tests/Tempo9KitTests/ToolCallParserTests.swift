// Copyright (c) 2026 Jiejing Zhang.
//
// ToolCallParser.parse against the shapes models actually produce, in both
// dialects it reads: Qwen's XML (<tool_call><function=…><parameter=…>) and
// Gemma 4's (<|tool_call>call:NAME{k:v}<tool_call|>).  The two share one
// entry point and are told apart by their anchors -- their syntaxes have
// nothing in common, so the text says which one it is.
//
// These cases lived in the Eyes On app as a `tooltest` executable that
// printed pass or fail and always exited 0, so nothing ever failed on them.
// They moved here when the app became a consumer of this package: the parser
// is the SDK's, and so is the job of keeping it honest.
//
// Hermetic: no engine, no model.

import Testing
@testable import Tempo9

private func calls(_ text: String) -> [ToolCall] { ToolCallParser.parse(text) }

private func expect(_ text: String, _ want: [(String, [String: String])],
                    sourceLocation: SourceLocation = #_sourceLocation) {
    let got = calls(text)
    #expect(got.map(\.name) == want.map(\.0), sourceLocation: sourceLocation)
    #expect(got.map(\.arguments) == want.map(\.1),
            sourceLocation: sourceLocation)
}

// MARK: Qwen XML

@Test func xmlComplete() {
    expect("""
    <tool_call>
    <function=get_weather>
    <parameter=city>
    北京
    </parameter>
    </function>
    </tool_call>
    """, [("get_weather", ["city": "北京"])])
}

/// Seen in practice: the decoder had already stripped <tool_call>.
@Test func xmlWithoutToolCallWrapper() {
    expect("""
    <function=get_weather>
    <parameter=city>
    北京
    </parameter>
    """, [("get_weather", ["city": "北京"])])
}

@Test func xmlMissingParameterClose() {
    expect("""
    <tool_call><function=f>
    <parameter=a>
    1
    <parameter=b>
    2
    </parameter></function></tool_call>
    """, [("f", ["a": "1", "b": "2"])])
}

@Test func xmlSpacesInsideTags() {
    expect("<function = f><parameter = k >v</parameter></function>",
           [("f", ["k": "v"])])
}

/// Two calls in a row, the first never closed.
@Test func xmlTwoCallsBackToBack() {
    expect("""
    <tool_call><function=a><parameter=x>1</parameter></function>
    <function=b><parameter=y>2</parameter></function></tool_call>
    """, [("a", ["x": "1"]), ("b", ["y": "2"])])
}

@Test func xmlAmidProseWithStrayThinkClose() {
    expect("""
    好的,我来查。</think>
    <tool_call><function=get_weather><parameter=city>上海</parameter></function></tool_call>
    """, [("get_weather", ["city": "上海"])])
}

@Test func noCallAtAll() {
    expect("今天天气不错。", [])
}

// MARK: Gemma 4

@Test func gemmaMinimal() {
    expect("<|tool_call>call:get_weather{city:Paris}<tool_call|>",
           [("get_weather", ["city": "Paris"])])
}

/// The template documents strings wrapped in <|"|>; models often leave the
/// wrapper out when generating. Both must parse.
@Test func gemmaStringWrapper() {
    expect("<|tool_call>call:f{a:<|\"|>hello<|\"|>,b:2}<tool_call|>",
           [("f", ["a": "hello", "b": "2"])])
}

/// A comma inside a wrapped string must not split the argument.
@Test func gemmaCommaInsideString() {
    expect("<|tool_call>call:s{q:<|\"|>a,b,c<|\"|>}<tool_call|>",
           [("s", ["q": "a,b,c"])])
}

/// Depth has to be counted: the first } does not end the arguments.
@Test func gemmaNestedObjectAndArray() {
    expect("<|tool_call>call:f{o:{x:1,y:2},l:[1,2,3],z:true}<tool_call|>",
           [("f", ["o": "{x:1,y:2}", "l": "[1,2,3]", "z": "true"])])
}

@Test func gemmaTwoCalls() {
    expect("<|tool_call>call:a{x:1}<tool_call|> 然后 "
           + "<|tool_call>call:b{y:2}<tool_call|>",
           [("a", ["x": "1"]), ("b", ["y": "2"])])
}

@Test func gemmaAmidProse() {
    expect("好的,我查一下。<|tool_call>call:get_weather{city:上海}<tool_call|>",
           [("get_weather", ["city": "上海"])])
}

@Test func gemmaNoArguments() {
    expect("<|tool_call>call:now{}<tool_call|>", [("now", [:])])
}
