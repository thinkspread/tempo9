// Copyright (c) 2026 Jiejing Zhang.
//
// Tool calls across the local API's three protocols.
//
// Qwen3.5 is tool-trained: given OpenAI-style definitions through its chat
// template it emits
//
//     <tool_call>
//     {"name": "shell", "arguments": {"command": ["git", "status"]}}
//     </tool_call>
//
// in its output (the incremental decoder deliberately preserves those
// tags).  This file is the bridge between that wire-neutral fact and the
// three protocol dialects: parsing the model's calls out of the text,
// filtering them out of streamed text deltas, and converting each
// protocol's tool schema into the template's expected shape.

import Foundation

enum ToolCallBridge {
    struct Call {
        let name: String
        /// Raw JSON string of the arguments object.
        let argumentsJSON: String
    }

    /// Argument types, recovered from the tool schema.
    ///
    /// The model's XML dialect carries no types: `<parameter=side1>5</parameter>`
    /// yields the STRING "5", and ToolCall.arguments is [String: String] all
    /// the way through.  Emitting that as `{"side1": "5"}` breaks any tool
    /// that validates its input, and it is not a hypothetical -- the official
    /// BFCL evaluator scored `multiple` at 24% against this server, and the
    /// failures were almost entirely
    /// `Incorrect type for parameter 'side2'. Expected integer, got str.`
    ///
    /// This file's header used to say typing arguments "belongs with a caller
    /// that dispatches tools, and there is not one yet."  There is one now:
    /// this server, talking to real agent clients.
    ///
    /// The schema is the authority, never a guess at the value's shape.  A
    /// parameter DECLARED string stays a string even when it looks numeric --
    /// order ids and phone numbers are the cases that punishes.
    static func coerce(_ args: [String: String],
                       to types: [String: String]) -> [String: Any] {
        var out: [String: Any] = [:]
        for (k, v) in args {
            let t = types[k] ?? "string"
            let raw = v.trimmingCharacters(in: .whitespacesAndNewlines)
            switch t {
            case "integer":
                if let i = Int(raw) { out[k] = i } else { out[k] = v }
            case "number", "float", "double":
                if let d = Double(raw) { out[k] = d } else { out[k] = v }
            case "boolean":
                switch raw.lowercased() {
                case "true": out[k] = true
                case "false": out[k] = false
                default: out[k] = v
                }
            case "array", "object", "dict":
                if let d = raw.data(using: .utf8),
                   let j = try? JSONSerialization.jsonObject(
                       with: d, options: [.fragmentsAllowed]),
                   (j is [Any]) == (t == "array") {
                    out[k] = j
                } else {
                    out[k] = v          // unparseable: keep what the model said
                }
            default:
                out[k] = v              // declared string, or unknown: verbatim
            }
        }
        return out
    }

    /// name -> (parameter -> declared JSON type), from OpenAI-shaped tools.
    static func argTypes(_ tools: [[String: Any]]) -> [String: [String: String]] {
        var map: [String: [String: String]] = [:]
        for t in tools {
            let fn = (t["function"] as? [String: Any]) ?? t
            guard let name = fn["name"] as? String else { continue }
            let params = (fn["parameters"] as? [String: Any]) ?? [:]
            let props = (params["properties"] as? [String: Any]) ?? [:]
            var types: [String: String] = [:]
            for (k, v) in props {
                if let d = v as? [String: Any], let ty = d["type"] as? String {
                    types[k] = ty
                }
            }
            // The template flattens dots to underscores in function names;
            // accept both so a call parsed either way finds its schema.
            map[name] = types
            map[name.replacingOccurrences(of: ".", with: "_")] = types
        }
        return map
    }

    /// Split a finished reply into visible text and tool calls, typing the
    /// arguments against the tool schema the request supplied.
    ///
    /// A request that declared NO tools gets no tool calls: the reply is
    /// text, exactly as the model wrote it.  That is what every OpenAI-
    /// compatible server does, and the reason it matters is agent traffic
    /// replayed without its tool list (SiliconBench's agent split: BFCL /
    /// Hermes conversations whose history already contains <tool_call>s).
    /// Parsing anyway answered 30 of 100 of those with finish_reason
    /// "tool_calls" and an EMPTY content -- a client that sent no tools
    /// cannot act on a call, and a benchmark counts it as a 0-token reply.
    /// Where a call starts, in every dialect ToolCallParser reads: Qwen's
    /// wrapper and bare function tag, and Gemma 4's `<|tool_call>`.  ONE
    /// list for the visible-text cut and the stream filter both: the parser
    /// learned Gemma while these two places still knew only Qwen, so a
    /// declared-tools Gemma reply came back with the call in tool_calls AND
    /// its raw `<|tool_call>call:...<tool_call|>` text in content, streamed
    /// or not.
    static let callMarkers = ["<tool_call>", "<function=", "<|tool_call>"]

    /// Visible text = everything before the first call marker of any dialect.
    static func textBeforeCalls(_ text: String) -> String {
        var clean = text
        for marker in callMarkers {
            if let r = clean.range(of: marker) {
                clean = String(clean[..<r.lowerBound])
            }
        }
        return clean.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    static func parse(_ text: String, tools: [[String: Any]])
        -> (text: String, calls: [Call]) {
        guard !tools.isEmpty else {
            return (text.trimmingCharacters(in: .whitespacesAndNewlines), [])
        }
        let types = argTypes(tools)
        let xml = ToolCallParser.parse(text)
        guard !xml.isEmpty else { return parse(text) }
        let calls = xml.map { c -> Call in
            let typed = coerce(c.arguments, to: types[c.name] ?? [:])
            let data = (try? JSONSerialization.data(withJSONObject: typed))
                ?? Data("{}".utf8)
            return Call(name: c.name,
                        argumentsJSON: String(data: data, encoding: .utf8) ?? "{}")
        }
        return (textBeforeCalls(text), calls)
    }

    /// Split a finished reply into visible text and tool calls.
    ///
    /// Qwen3.5's template emits the XML form —
    /// <function=name><parameter=key>value</parameter> — with the
    /// <tool_call> wrapper OPTIONAL in practice.  ToolCallParser (tested in
    /// tooltest, used by localctl) owns that dialect; the JSON body form is
    /// kept as a fallback for models that emit it.
    static func parse(_ text: String) -> (text: String, calls: [Call]) {
        let xml = ToolCallParser.parse(text)
        if !xml.isEmpty {
            let calls = xml.map { c -> Call in
                let data = (try? JSONSerialization.data(
                    withJSONObject: c.arguments)) ?? Data("{}".utf8)
                return Call(name: c.name,
                            argumentsJSON: String(data: data,
                                encoding: .utf8) ?? "{}")
            }
            return (textBeforeCalls(text), calls)
        }
        return parseJSONForm(text)
    }

    private static func parseJSONForm(_ text: String)
        -> (text: String, calls: [Call]) {
        var calls: [Call] = []
        var clean = ""
        var rest = Substring(text)
        while let open = rest.range(of: "<tool_call>") {
            clean += rest[..<open.lowerBound]
            let after = rest[open.upperBound...]
            guard let close = after.range(of: "</tool_call>") else {
                // Unclosed tag (max_tokens hit mid-call): drop the fragment.
                rest = ""
                break
            }
            let body = after[..<close.lowerBound]
                .trimmingCharacters(in: .whitespacesAndNewlines)
            if let data = body.data(using: .utf8),
               let obj = try? JSONSerialization.jsonObject(with: data)
                   as? [String: Any],
               let name = obj["name"] as? String {
                let args = obj["arguments"] ?? [String: Any]()
                let argsData = (try? JSONSerialization.data(
                    withJSONObject: args)) ?? Data("{}".utf8)
                calls.append(Call(
                    name: name,
                    argumentsJSON: String(data: argsData, encoding: .utf8)
                        ?? "{}"))
            }
            rest = after[close.upperBound...]
        }
        clean += rest
        return (clean.trimmingCharacters(in: .whitespacesAndNewlines), calls)
    }

    /// Streams text deltas through, holding back anything that could be
    /// the start of a <tool_call> block.  The client sees clean prose; the
    /// calls surface as typed items when the reply finishes.
    /// Sequenced incremental tool-call emission for BLOCK protocols
    /// (Anthropic content blocks, Responses output items), where block k
    /// must close before block k+1 may open.  The XML stream satisfies
    /// this naturally -- call k's closer precedes call k+1's name -- and
    /// this driver enforces it against a malformed stream rather than
    /// trusting it: a name learned early waits in the queue until the
    /// previous call's arguments have been sent.
    ///
    /// open/args are invoked exactly once per call, in index order,
    /// args always after open of the same index.  finish() tops up from
    /// the final parse and reports (sent, final) so the caller can log a
    /// disagreement instead of double-sending.
    final class BlockToolDriver {
        private let filter: StreamFilter
        private var names: [String] = []
        private var opened = 0
        private var argsSent = 0
        init(filter: StreamFilter) { self.filter = filter }

        var callsOpened: Int { opened }
        var anyCalls: Bool { opened > 0 }

        func pump(tools: [[String: Any]],
                  open: (Int, String) -> Void,
                  args: (Int, String) -> Void) {
            names.append(contentsOf: filter.newCallNames())
            let done = filter.completedCalls(tools: tools)
            while true {
                if opened == argsSent, opened < names.count {
                    open(opened, names[opened]); opened += 1; continue
                }
                if argsSent < opened, argsSent < done.count {
                    args(argsSent, done[argsSent].argumentsJSON)
                    argsSent += 1; continue
                }
                break
            }
        }

        /// Emit whatever the final parse has that the stream did not
        /// certify.  Returns (sent, final) call counts.
        func finish(parsed: [Call],
                    open: (Int, String) -> Void,
                    args: (Int, String) -> Void) -> (sent: Int, final: Int) {
            var i = argsSent
            while i < parsed.count {
                if opened <= i { open(i, parsed[i].name); opened = i + 1 }
                args(i, parsed[i].argumentsJSON)
                i += 1
            }
            let sent = argsSent > parsed.count ? argsSent : i
            argsSent = sent
            return (sent, parsed.count)
        }
    }

    final class StreamFilter {
        /// No tools declared: nothing is a tool call, so nothing is held
        /// back (see parse(_:tools:)).  The argument is required on purpose
        /// -- every call site has to say which kind of request it serves.
        private let passThrough: Bool
        init(tools: [[String: Any]]) { passThrough = tools.isEmpty }

        private var pending = ""
        private var suppressing = false
        /// Everything from the first call marker onward.  The old filter
        /// DISCARDED this — the server re-parsed reply.text once generation
        /// finished — which is why a tool-call response had ttft ≈ wall and
        /// an agent client saw nothing until the whole call landed.  Keeping
        /// it is what lets the call be announced while it is still being
        /// written.
        private var callBuf = ""
        private var announced = 0
        /// Every dialect's call markers (ToolCallBridge.callMarkers): Qwen's
        /// wrapper AND bare function tag -- the wrapper is optional in what
        /// the model actually emits -- and Gemma 4's `<|tool_call>`.
        private static let tags = ToolCallBridge.callMarkers

        func feed(_ delta: String) -> String {
            if passThrough { return delta }
            if suppressing { callBuf += delta; return "" }
            pending += delta
            for tag in Self.tags {
                if let r = pending.range(of: tag) {
                    let out = String(pending[..<r.lowerBound])
                    suppressing = true
                    callBuf = String(pending[r.lowerBound...])
                    pending = ""
                    return out
                }
            }
            // Hold back a possible tag prefix at the tail; emit the rest.
            let keep = maxTagPrefix(of: pending)
            let out = String(pending.dropLast(keep))
            pending = String(pending.suffix(keep))
            return out
        }

        /// Whatever held-back text was NOT a tool call after all.
        func flush() -> String {
            defer { pending = "" }
            return suppressing ? "" : pending
        }

        /// Names of calls whose identity has become certain since the last
        /// ask, in order.  "Certain" means the delimiter after the name has
        /// arrived — a name is never announced from a partial token, because
        /// a client that has been told "shell" cannot be told "shell_exec"
        /// afterwards.
        func newCallNames() -> [String] {
            let names = Self.namesIn(callBuf)
            guard names.count > announced else { return [] }
            let fresh = Array(names[announced...])
            announced = names.count
            return fresh
        }

        /// How many calls have been announced to the client so far.
        var announcedCount: Int { announced }

        /// Calls that are CERTAINLY complete: parsed only from the prefix
        /// of callBuf ending at the last explicit closer.  ToolCallParser
        /// treats end-of-input as a parameter terminator (its paramRE ends
        /// on \z), so feeding it anything past the last closer would read
        /// a truncated value as a finished one -- "Par" for "Paris" -- and
        /// a streamed argument, once sent, cannot be unsent.  Calls whose
        /// dialect omits a closer (bare-brace Gemma) simply wait for the
        /// end-of-stream pass; late is recoverable, wrong is not.
        func completedCalls(tools: [[String: Any]]) -> [Call] {
            var last: String.Index? = nil
            for c in ["</tool_call>", "</function>", "<tool_call|>"] {
                var from = callBuf.startIndex
                while let r = callBuf.range(of: c,
                                            range: from..<callBuf.endIndex) {
                    if last == nil || r.upperBound > last! { last = r.upperBound }
                    from = r.upperBound
                }
            }
            guard let l = last else { return [] }
            return ToolCallBridge.parse(String(callBuf[..<l]),
                                        tools: tools).calls
        }

        /// Complete names visible in `s`, both dialects.  Only names whose
        /// terminator has arrived count.
        static func namesIn(_ s: String) -> [String] {
            var out: [String] = []
            var i = s.startIndex
            while i < s.endIndex {
                // Gemma 4: <|tool_call>call:NAME{  -- the name is certain
                // once its `{` has arrived.
                if let r = s.range(of: "<|tool_call>", range: i..<s.endIndex) {
                    var n = r.upperBound
                    if s[n...].hasPrefix("call:") {
                        n = s.index(n, offsetBy: 5)
                    }
                    guard let brace = s.range(of: "{", range: n..<s.endIndex)
                    else { break }          // name still arriving
                    out.append(String(s[n..<brace.lowerBound])
                        .trimmingCharacters(in: .whitespacesAndNewlines))
                    i = brace.upperBound
                    continue
                }
                if let r = s.range(of: "<function=", range: i..<s.endIndex) {
                    guard let end = s.range(of: ">",
                                            range: r.upperBound..<s.endIndex)
                    else { break }          // name still arriving
                    out.append(String(s[r.upperBound..<end.lowerBound]))
                    i = end.upperBound
                    continue
                }
                if let r = s.range(of: "\"name\"", range: i..<s.endIndex),
                   let c = s.range(of: ":", range: r.upperBound..<s.endIndex),
                   let q1 = s.range(of: "\"", range: c.upperBound..<s.endIndex),
                   let q2 = s.range(of: "\"", range: q1.upperBound..<s.endIndex) {
                    out.append(String(s[q1.upperBound..<q2.lowerBound]))
                    i = q2.upperBound
                    continue
                }
                break
            }
            return out
        }

        private func maxTagPrefix(of s: String) -> Int {
            var best = 0
            for tag in Self.tags {
                let maxLen = min(s.count, tag.count - 1)
                for len in stride(from: maxLen, through: 1, by: -1)
                    where tag.hasPrefix(String(s.suffix(len))) {
                    best = max(best, len)
                    break
                }
            }
            return best
        }
    }

    // MARK: - schema conversion (each dialect -> the template's shape)

    // Only function-shaped tools reach the model: the template renders
    // their schema and the reply is parsed for calls to them.  What arrives
    // in `tools[]` falls into three classes, and they are treated
    // differently on purpose (owner decision, 2026-09-06):
    //
    //  * a CUSTOM tool carries its own schema and passes through;
    //  * a CLIENT-executed Anthropic built-in -- bash_*, text_editor_*,
    //    computer_* -- carries no schema on the wire (Anthropic bakes it
    //    into the model), but the CLIENT runs it: the model emits tool_use,
    //    Claude Code executes.  All this server needs is the input schema,
    //    which Anthropic publishes per version; AnthropicBuiltin supplies
    //    it and the tool passes through under the client's name for it;
    //  * a SERVER-executed tool -- web_search_*, web_fetch_*,
    //    code_execution_*, tool_search_*, and OpenAI's web_search /
    //    file_search / code_interpreter -- is one the vendor's servers run
    //    BETWEEN generations: server_tool_use, then a
    //    web_search_tool_result block, then the model continues.  That is
    //    a real feature this server deliberately does not build now (the
    //    shape it would take is recorded in manual/claude-code.md).  Such a
    //    request is refused as a whole with a message that says which
    //    class the tool is, or -- only under TEMPO9_STRIP_UNSUPPORTED_TOOLS=1
    //    -- stripped with a warning in the log and the response.
    //
    // These used to be compactMapped away, and a client that asked for web
    // search got a reply with the model's guess.  Then, for one release,
    // every non-custom type was refused with "neither runs nor has a
    // schema for", which is false for the client-executed class.

    /// What a dialect's tool list became: the tools in the template's
    /// shape, plus one warning per tool that was stripped instead of
    /// refused (only under TEMPO9_STRIP_UNSUPPORTED_TOOLS; empty otherwise).
    struct Converted {
        var tools: [[String: Any]]
        var warnings: [String] = []
    }

    private static func function(name: String, description: String,
                                 parameters: Any) -> [String: Any] {
        ["type": "function", "function": [
            "name": name, "description": description,
            "parameters": parameters,
        ]]
    }

    /// The Anthropic-defined tools, by class, and the input schema of every
    /// client-executed version this server carries.
    ///
    /// Versions are the ones Claude Code 2.1.239 carries in its binary
    /// (`strings` over it: bash_20250124, text_editor_20250124/20250429/
    /// 20250728, computer_20251124) plus each family's other published
    /// versions.  Each schema is transcribed from Anthropic's public docs;
    /// the source and any inference is noted beside it.  None of these is
    /// executed here -- the schema exists so the template can show the
    /// model what the client will run.
    enum AnthropicBuiltin {
        /// Family -> versions with a schema below, oldest first.
        static let versions: [String: [String]] = [
            "bash": ["bash_20241022", "bash_20250124"],
            "text_editor": ["text_editor_20241022", "text_editor_20250124",
                            "text_editor_20250429", "text_editor_20250728"],
            "computer": ["computer_20250124", "computer_20251124"],
            "memory": ["memory_20250818"],
        ]

        /// Client-executed families the docs list but no schema is tabled
        /// for here: the client TOOLSETS (tool-reference, "Client
        /// toolsets") -- one entry with no name declaring a fixed set of
        /// member tools, each called by its own name with a `toolset_name`
        /// -- a different shape from a single built-in, and not added.
        /// Refused, but not as "server-side" -- that would be as false as
        /// the sentence this replaces.
        static let clientFamiliesWithoutTable: Set<String> =
            ["computer_toolset", "browser_toolset"]

        /// Server-executed families: the vendor runs them between
        /// generations.  Refused; strippable under the env switch.
        static let serverFamilies: Set<String> =
            ["web_search", "web_fetch", "code_execution",
             "tool_search_tool_regex", "tool_search_tool_bm25",
             "advisor", "mcp_toolset"]

        static let supported = "Supported tool types: custom (with "
            + "input_schema), bash_*, text_editor_*, computer_*, memory_* "
            + "(client-executed)."

        /// `web_search_20260209` -> `web_search`; a type with no date
        /// suffix is its own family.
        static func family(of type: String) -> String {
            type.replacingOccurrences(of: #"_\d{8}$"#, with: "",
                                      options: .regularExpression)
        }

        /// Every type with a schema, for the capabilities document.
        static var allVersions: [String] {
            versions.keys.sorted().flatMap { versions[$0]! }
        }

        /// The template-shaped description and parameters for a tabled
        /// version, or nil for a version this server does not know.
        static func schema(type: String, tool: [String: Any])
            -> (description: String, parameters: [String: Any])? {
            switch family(of: type) {
            case "bash": return bash(type)
            case "text_editor": return textEditor(type)
            case "computer": return computer(type, tool: tool)
            case "memory": return memory(type)
            default: return nil
            }
        }

        // memory_20250818: client-executed (tool-reference lists it under
        // "Client"; the memory-tool page: "Claude requests file
        // operations, and your application executes them").  Claude Code
        // 2.1.239 carries the type string in its binary.  Source for the
        // inputs: the SDK's six command types --
        // BetaMemoryTool20250818ViewCommand (command, path, view_range?),
        // ...CreateCommand (command, path, file_text),
        // ...StrReplaceCommand (command, path, old_str, new_str),
        // ...InsertCommand (command, path, insert_line: int, insert_text),
        // ...DeleteCommand (command, path),
        // ...RenameCommand (command, old_path, new_path)
        // (anthropic-sdk-python src/anthropic/types/beta/, the tool
        // entry itself being BetaMemoryTool20250818Param {name: "memory",
        // type: "memory_20250818"}) -- cross-checked against the docs'
        // "Tool commands" examples, which use the same names.  Two things
        // decided rather than read: `path` is not top-level required
        // because rename carries old_path/new_path instead; and the
        // memory-protocol instruction Anthropic's API adds to the SYSTEM
        // prompt when this tool is present ("ALWAYS VIEW YOUR MEMORY
        // DIRECTORY BEFORE DOING ANYTHING ELSE ...") is folded into the
        // tool description here, since this server injects no system
        // text for tools.  The docs also say new_str may be omitted on
        // str_replace (deletes old_str) where the SDK type has it
        // required; declared optional, as the docs describe the wire.
        private static func memory(_ type: String)
            -> (String, [String: Any])? {
            guard versions["memory"]!.contains(type) else { return nil }
            return ("Store and retrieve notes that persist across "
                    + "conversations, as files under /memories on the "
                    + "user's side. Every path starts with /memories; the "
                    + "/memories directory itself cannot be deleted or "
                    + "renamed. Commands: view (path, optional view_range) "
                    + "lists a directory or shows a file with line numbers; "
                    + "create (path, file_text) creates or overwrites a "
                    + "file; str_replace (path, old_str, new_str; old_str "
                    + "must appear exactly once, omit new_str to delete "
                    + "it); insert (path, insert_line, insert_text; text "
                    + "goes after line insert_line, 0 = top); delete "
                    + "(path); rename (old_path, new_path). Memory "
                    + "protocol: view /memories before starting a task to "
                    + "check for earlier progress, and record status and "
                    + "what you learn in memory as you go, since your "
                    + "context may be reset at any moment. Keep the "
                    + "directory organized; rename or delete files that "
                    + "are no longer relevant.",
                    ["type": "object", "required": ["command"],
                     "properties": [
                "command": ["type": "string",
                            "enum": ["view", "create", "str_replace",
                                     "insert", "delete", "rename"],
                            "description": "The operation to perform."],
                "path": ["type": "string",
                         "description": "Path under /memories (all "
                             + "commands except rename)."],
                "view_range": ["type": "array", "items": ["type": "integer"],
                               "minItems": 2, "maxItems": 2,
                               "description": "view, files only: "
                                   + "[start_line, end_line], 1-indexed; "
                                   + "-1 for end_line means to the end of "
                                   + "the file."],
                "file_text": ["type": "string",
                              "description": "create: the full content of "
                                  + "the file."],
                "old_str": ["type": "string",
                            "description": "str_replace: the exact text to "
                                + "replace; must appear exactly once."],
                "new_str": ["type": "string",
                            "description": "str_replace: the replacement "
                                + "text; omitted deletes old_str."],
                "insert_line": ["type": "integer",
                                "description": "insert: the line number "
                                    + "after which to insert (0 = top)."],
                "insert_text": ["type": "string",
                                "description": "insert: the text to insert."],
                "old_path": ["type": "string",
                             "description": "rename: the current path."],
                "new_path": ["type": "string",
                             "description": "rename: the new path; must "
                                 + "not already exist."],
            ]])
        }

        // bash_20250124 (current) and bash_20241022 (Sonnet 3.5, retired):
        // same inputs.  Source: platform.claude.com/docs/en/agents-and-tools/
        // tool-use/bash-tool "Parameters": `command` (string, required
        // unless `restart`), `restart` (boolean).  No tool-level fields.
        private static func bash(_ type: String)
            -> (String, [String: Any])? {
            guard versions["bash"]!.contains(type) else { return nil }
            return ("Run a shell command in a persistent bash session on "
                    + "the user's machine and return its output. State "
                    + "(working directory, environment) persists between "
                    + "commands. Send restart=true instead of a command to "
                    + "restart the session.",
                    ["type": "object", "properties": [
                        "command": ["type": "string",
                                    "description": "The bash command to run. "
                                        + "Required unless restart is true."],
                        "restart": ["type": "boolean",
                                    "description": "Set to true to restart "
                                        + "the bash session."],
                    ]])
        }

        // Source: platform.claude.com/docs/en/agents-and-tools/tool-use/
        // text-editor-tool, "Text editor tool commands" and "Change log".
        //   text_editor_20241022 / 20250124  name str_replace_editor
        //       commands view, create, str_replace, insert, undo_edit
        //   text_editor_20250429 / 20250728  name str_replace_based_edit_tool
        //       undo_edit removed; 20250728 adds tool-level max_characters
        // Inputs per command: view(path, view_range?), create(path,
        // file_text), str_replace(path, old_str, new_str), insert(path,
        // insert_line, insert_text), undo_edit(path).  The current page
        // documents insert's text as `insert_text`; the pre-2025 reference
        // implementation used `new_str` for it, so both are declared and
        // a client should accept either.  view_range is "an array of two
        // integers", insert_line "the line number" -> integer.
        private static func textEditor(_ type: String)
            -> (String, [String: Any])? {
            guard versions["text_editor"]!.contains(type) else { return nil }
            let undo = type == "text_editor_20241022"
                || type == "text_editor_20250124"
            var commands = ["view", "create", "str_replace", "insert"]
            if undo { commands.append("undo_edit") }
            var desc = "View, create and edit text files on the user's "
                + "machine. Commands: view (path, optional view_range), "
                + "create (path, file_text), str_replace (path, old_str, "
                + "new_str; old_str must match exactly once), insert "
                + "(path, insert_line, insert_text; 0 inserts at the top)"
            if undo { desc += ", undo_edit (path)" }
            desc += ". Paths are absolute."
            return (desc, ["type": "object",
                           "required": ["command", "path"],
                           "properties": [
                "command": ["type": "string", "enum": commands,
                            "description": "The operation to perform."],
                "path": ["type": "string",
                         "description": "Absolute path to the file or "
                             + "directory."],
                "view_range": ["type": "array", "items": ["type": "integer"],
                               "minItems": 2, "maxItems": 2,
                               "description": "view: [start, end] line "
                                   + "numbers, 1-indexed; -1 for end means "
                                   + "to the end of the file."],
                "file_text": ["type": "string",
                              "description": "create: the full content of "
                                  + "the new file."],
                "old_str": ["type": "string",
                            "description": "str_replace: the exact text to "
                                + "replace."],
                "new_str": ["type": "string",
                            "description": "str_replace: the replacement "
                                + "text."],
                "insert_line": ["type": "integer",
                                "description": "insert: the line number "
                                    + "after which to insert (0 = top)."],
                "insert_text": ["type": "string",
                                "description": "insert: the text to insert."],
            ]])
        }

        // computer_20251124 (Claude 4.5+, beta computer-use-2025-11-24) and
        // computer_20250124 (Claude 4, beta computer-use-2025-01-24).
        // Tool-level fields, from the SDK's BetaToolComputerUse20251124Param:
        // name "computer", display_width_px / display_height_px (required),
        // display_number (optional), enable_zoom (20251124 only, default
        // false).  Input fields: the docs' "Migrate from computer_20251124"
        // says the toolset members' inputs are the earlier version's
        // "with the action field removed", so the action list and inputs
        // below are that member table (platform.claude.com/docs/en/
        // agents-and-tools/tool-use/computer-use-tool "Available actions")
        // with `action` put back and `repeat` (added by the toolset) left
        // out; `zoom` + `region` are present only with enable_zoom.
        // Coordinates are [x, y] pixels from the top-left of the display.
        private static func computer(_ type: String, tool: [String: Any])
            -> (String, [String: Any])? {
            guard versions["computer"]!.contains(type) else { return nil }
            let zoom = type == "computer_20251124"
                && (tool["enable_zoom"] as? Bool ?? false)
            var actions = ["key", "hold_key", "type", "cursor_position",
                           "mouse_move", "left_mouse_down", "left_mouse_up",
                           "left_click", "left_click_drag", "right_click",
                           "middle_click", "double_click", "triple_click",
                           "scroll", "wait", "screenshot"]
            if zoom { actions.append("zoom") }
            var display = "the display"
            if let w = tool["display_width_px"] as? Int,
               let h = tool["display_height_px"] as? Int {
                display = "The display is \(w)x\(h) px"
                if let n = tool["display_number"] as? Int {
                    display += " (display number \(n))"
                }
            }
            var props: [String: Any] = [
                "action": ["type": "string", "enum": actions,
                           "description": "The action to perform."],
                "coordinate": ["type": "array", "items": ["type": "integer"],
                               "minItems": 2, "maxItems": 2,
                               "description": "[x, y] pixel position for "
                                   + "click, move, drag end and scroll."],
                "start_coordinate": ["type": "array",
                                     "items": ["type": "integer"],
                                     "minItems": 2, "maxItems": 2,
                                     "description": "left_click_drag: where "
                                         + "the drag starts."],
                "text": ["type": "string",
                         "description": "type: the text to type; key / "
                             + "hold_key: a key or +-joined combination "
                             + "(\"Return\", \"ctrl+s\"); clicks: modifier "
                             + "keys to hold."],
                "scroll_direction": ["type": "string",
                                     "enum": ["up", "down", "left", "right"],
                                     "description": "scroll: direction."],
                "scroll_amount": ["type": "integer",
                                  "description": "scroll: number of wheel "
                                      + "clicks."],
                "duration": ["type": "number",
                             "description": "hold_key / wait: seconds, up "
                                 + "to 300."],
            ]
            if zoom {
                props["region"] = ["type": "array",
                                   "items": ["type": "integer"],
                                   "minItems": 4, "maxItems": 4,
                                   "description": "zoom: [x0, y0, x1, y1] "
                                       + "of the area to capture."]
            }
            return ("Control the user's screen, keyboard and mouse. "
                    + "\(display); coordinates are [x, y] pixels from the "
                    + "top-left. Take a screenshot to see the screen; "
                    + "click, type and scroll at coordinates.",
                    ["type": "object", "required": ["action"],
                     "properties": props])
        }
    }

    /// OpenAI hosted tools: the vendor runs them server-side.  Anything
    /// else that is not "function" is simply not a tool this server has.
    static let openAIHostedTypes: Set<String> = [
        "web_search", "web_search_preview", "web_search_preview_2025_03_11",
        "web_search_2025_08_26", "file_search", "code_interpreter",
        "image_generation", "mcp",
    ]

    /// A refusal or a strip for an OpenAI-shaped non-function tool.
    private static func openAINonFunction(_ type: String, at: String,
                                          stripHosted: Bool,
                                          warnings: inout [String]) throws {
        let hosted = openAIHostedTypes.contains(type)
        if hosted && stripHosted {
            warnings.append("\(at) (\(type)) was stripped from tools: this "
                            + "server does not execute server-side tools "
                            + "(TEMPO9_STRIP_UNSUPPORTED_TOOLS=1)")
            return
        }
        throw RequestRefusal(
            param: "\(at).type",
            message: hosted
                ? "\(at) is a \(APIRequest.show(type)) tool; this server "
                  + "does not execute server-side tools -- remove it from "
                  + "tools. Supported tool types: \"function\"."
                : "\(at) is a \(APIRequest.show(type)) tool; only "
                  + "\"function\" tools are supported -- this server has "
                  + "nothing that runs a \(type)")
    }

    /// Template shape: OpenAI chat-completions style
    /// {type:"function", function:{name, description, parameters}}.
    static func fromChat(_ tools: [[String: Any]],
                         stripHosted: Bool = false) throws -> Converted {
        var out = Converted(tools: [])
        for (i, t) in tools.enumerated() {
            let type = t["type"] as? String ?? "function"
            guard type == "function" else {
                try openAINonFunction(type, at: "tools[\(i)]",
                                      stripHosted: stripHosted,
                                      warnings: &out.warnings)
                continue
            }
            guard (t["function"] as? [String: Any])?["name"] is String else {
                throw RequestRefusal(
                    param: "tools[\(i)].function.name",
                    message: "tools[\(i)].function.name is required")
            }
            out.tools.append(t)  // already the template's native shape
        }
        return out
    }

    /// Anthropic: {name, description, input_schema}; `type` is absent or
    /// "custom" for a client-defined tool and names a built-in otherwise.
    static func fromAnthropic(_ tools: [[String: Any]],
                              stripHosted: Bool = false) throws -> Converted {
        var out = Converted(tools: [])
        for (i, t) in tools.enumerated() {
            let at = "tools[\(i)]"
            let type = t["type"] as? String ?? "custom"
            func name() throws -> String {
                guard let n = t["name"] as? String else {
                    throw RequestRefusal(param: "\(at).name",
                                         message: "\(at).name is required")
                }
                return n
            }
            if type == "custom" {
                out.tools.append(function(
                    name: try name(),
                    description: t["description"] as? String ?? "",
                    parameters: t["input_schema"] ?? ["type": "object"]))
                continue
            }
            let family = AnthropicBuiltin.family(of: type)
            if let known = AnthropicBuiltin.versions[family] {
                guard let shape = AnthropicBuiltin.schema(type: type,
                                                          tool: t) else {
                    throw RequestRefusal(
                        param: "\(at).type",
                        message: "\(at) is a \(APIRequest.show(type)) tool; "
                               + "this server does not know that version's "
                               + "schema (the tool is client-executed, so "
                               + "only the schema is needed). Versions of "
                               + "\(family) it knows: "
                               + known.joined(separator: ", ") + ". "
                               + AnthropicBuiltin.supported)
                }
                // Client-executed: the name is the client's, so the reply's
                // tool_use carries exactly what the client dispatches on.
                out.tools.append(function(name: try name(),
                                          description: shape.description,
                                          parameters: shape.parameters))
                continue
            }
            if AnthropicBuiltin.clientFamiliesWithoutTable.contains(family) {
                throw RequestRefusal(
                    param: "\(at).type",
                    message: "\(at) is a \(APIRequest.show(type)) tool, a "
                           + "client-executed built-in whose schema this "
                           + "server does not carry; declare it as a "
                           + "custom tool with an input_schema, or remove "
                           + "it. " + AnthropicBuiltin.supported)
            }
            let hosted = AnthropicBuiltin.serverFamilies.contains(family)
            if hosted && stripHosted {
                out.warnings.append("\(at) (\(type)) was stripped from "
                                    + "tools: this server does not execute "
                                    + "server-side tools "
                                    + "(TEMPO9_STRIP_UNSUPPORTED_TOOLS=1)")
                continue
            }
            throw RequestRefusal(
                param: "\(at).type",
                message: "\(at) is a \(APIRequest.show(type)) tool; this "
                       + "server does not execute server-side tools"
                       + (hosted ? "" : " and does not know that tool type")
                       + " -- remove it from tools. "
                       + AnthropicBuiltin.supported)
        }
        return out
    }

    /// Responses API: flat {type:"function", name, description, parameters}.
    static func fromResponses(_ tools: [[String: Any]],
                              stripHosted: Bool = false) throws -> Converted {
        var out = Converted(tools: [])
        for (i, t) in tools.enumerated() {
            let type = t["type"] as? String ?? "?"
            guard type == "function" else {
                try openAINonFunction(type, at: "tools[\(i)]",
                                      stripHosted: stripHosted,
                                      warnings: &out.warnings)
                continue
            }
            guard let name = t["name"] as? String else {
                throw RequestRefusal(param: "tools[\(i)].name",
                                     message: "tools[\(i)].name is required")
            }
            out.tools.append(function(
                name: name,
                description: t["description"] as? String ?? "",
                parameters: t["parameters"] ?? ["type": "object"]))
        }
        return out
    }

    /// An assistant turn that made calls, in the template's message shape
    /// (content + OpenAI-style tool_calls array).
    static func assistantTurn(text: String, calls: [(id: String, name: String,
                                                     argumentsJSON: String)])
        -> [String: Any] {
        var m: [String: Any] = ["role": "assistant", "content": text]
        if !calls.isEmpty {
            m["tool_calls"] = calls.map { c in
                ["id": c.id, "type": "function",
                 "function": ["name": c.name, "arguments": c.argumentsJSON]]
            }
        }
        return m
    }

    /// A tool result, in the template's message shape.
    static func toolResultTurn(callId: String, content: String)
        -> [String: Any] {
        ["role": "tool", "tool_call_id": callId, "content": content]
    }
}
