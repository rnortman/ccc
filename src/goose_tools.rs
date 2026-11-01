//! `ccc goose-tools`: the MCP stdio servers a Goose run uses as its only
//! extensions. The protocol is newline-delimited JSON-RPC 2.0; each
//! `tools/call` runs on its own thread, and responses go out through one
//! locked writer.

#[cfg(target_os = "linux")]
pub mod boxed;
#[cfg(unix)]
pub mod host;
mod inner;

use std::io::{BufRead, Write};
use std::sync::{Arc, Mutex};

use anyhow::{bail, Result};
use serde_json::{json, Value};

use crate::config::GOOSE_MAX_TOOL_WAIT_S;
use crate::goose_policy::GoosePolicy;

/// Which server `ccc goose-tools` runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum Role {
    /// The `ccc_run`, `ccc_resume` and `ccc_wait` tools, on the host.
    Host,
    /// Builds the bubblewrap box and execs into it.
    #[value(name = "box")]
    Boxed,
    /// The file and shell tools, inside the box.
    BoxInner,
}

/// Every tool result is cut to this many bytes of text.
pub const RESULT_CAP: usize = 256 * 1024;

/// Runs the server for `role` on stdin/stdout until stdin closes. An
/// undecodable policy is an error before anything is read. `tool_wait_s` is
/// `box-inner`'s alone: `host` and `box` read `[goose] tool_wait_s`.
pub fn run(role: Role, policy_hex: &str, tool_wait_s: Option<u64>) -> Result<i32> {
    if role == Role::BoxInner {
        // Before anything else: no inherited descriptor may reach a program the
        // model runs.
        inner::close_inherited_fds();
    } else {
        // Goose runs with an empty home and an HTTPS sink; the host-side roles
        // read config and start children with the host's values.
        crate::harness::goose::restore_host_env()?;
    }
    let policy = GoosePolicy::from_hex(policy_hex)?;
    if role != Role::BoxInner && tool_wait_s.is_some() {
        bail!("--tool-wait-s is for --role box-inner only; host and box read [goose] tool_wait_s");
    }
    match role {
        Role::BoxInner => {
            let tool_wait_s = box_inner_wait(tool_wait_s)?;
            inner::check_in_box()?;
            let cwd = std::env::current_dir()?;
            let tools = inner::BoxTools::new(policy, cwd, tool_wait_s);
            let stdin = std::io::stdin();
            serve(Arc::new(tools), stdin.lock(), std::io::stdout())?;
            Ok(0)
        }
        #[cfg(target_os = "linux")]
        Role::Boxed => boxed::exec_box(&policy, &crate::config::ProcessEnv),
        #[cfg(not(target_os = "linux"))]
        Role::Boxed => bail!("the Goose box needs Linux"),
        #[cfg(unix)]
        Role::Host => host::serve_host(policy, &crate::config::ProcessEnv),
        #[cfg(not(unix))]
        Role::Host => bail!("the goose-tools host role needs Unix"),
    }
}

/// `box-inner`'s `--tool-wait-s`: required, within `1..=GOOSE_MAX_TOOL_WAIT_S`.
fn box_inner_wait(tool_wait_s: Option<u64>) -> Result<u64> {
    match tool_wait_s {
        Some(s) if (1..=GOOSE_MAX_TOOL_WAIT_S).contains(&s) => Ok(s),
        Some(_) => bail!("--tool-wait-s must be within 1..={GOOSE_MAX_TOOL_WAIT_S}"),
        None => bail!("--role box-inner needs --tool-wait-s"),
    }
}

/// A tool as `tools/list` describes it.
pub struct ToolDef {
    pub name: &'static str,
    pub description: String,
    /// JSON Schema of the tool's arguments.
    pub schema: Value,
}

/// A tool call's reply: text for the model, and whether it is an error.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolReply {
    pub text: String,
    pub is_error: bool,
}

impl ToolReply {
    /// A successful reply.
    pub fn ok(text: impl Into<String>) -> ToolReply {
        ToolReply {
            text: text.into(),
            is_error: false,
        }
    }

    /// An error reply.
    pub fn error(text: impl Into<String>) -> ToolReply {
        ToolReply {
            text: text.into(),
            is_error: true,
        }
    }
}

/// The tools one server role offers.
pub trait Toolset: Send + Sync + 'static {
    /// The tools `tools/list` reports.
    fn list(&self) -> Vec<ToolDef>;
    /// Runs tool `name`. Calls run concurrently, one thread each.
    fn call(&self, name: &str, args: &Value) -> ToolReply;
    /// Called once stdin has closed, before outstanding calls are joined.
    fn shutdown(&self) {}
}

/// Cuts `text` to at most `cap` bytes on a character boundary, saying so.
pub fn cap_text(mut text: String, cap: usize) -> String {
    if text.len() <= cap {
        return text;
    }
    let total = text.len();
    let mut end = cap;
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    text.truncate(end);
    text.push_str(&format!("\n[truncated: showing {end} of {total} bytes]"));
    text
}

type SharedOut = Arc<Mutex<Box<dyn Write + Send>>>;

fn send(out: &SharedOut, message: &Value) {
    let mut line = serde_json::to_vec(message).expect("a JSON value always serialises");
    line.push(b'\n');
    let mut out = out.lock().unwrap_or_else(|e| e.into_inner());
    // A closed stdout means Goose has gone; the read side sees EOF next.
    let _ = out.write_all(&line).and_then(|()| out.flush());
}

fn respond(out: &SharedOut, id: &Value, result: Value) {
    send(out, &json!({"jsonrpc": "2.0", "id": id, "result": result}));
}

fn respond_error(out: &SharedOut, id: &Value, code: i64, message: &str) {
    send(
        out,
        &json!({"jsonrpc": "2.0", "id": id, "error": {"code": code, "message": message}}),
    );
}

/// Serves MCP requests from `input` until it closes, then calls
/// [`Toolset::shutdown`] and joins every call still running.
pub fn serve(
    tools: Arc<dyn Toolset>,
    input: impl BufRead,
    output: impl Write + Send + 'static,
) -> Result<()> {
    let out: SharedOut = Arc::new(Mutex::new(Box::new(output)));
    let mut calls = Vec::new();
    for line in input.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let message: Value = match serde_json::from_str(&line) {
            Ok(m) => m,
            Err(e) => {
                respond_error(&out, &Value::Null, -32700, &format!("parse error: {e}"));
                continue;
            }
        };
        let method = message.get("method").and_then(Value::as_str);
        let id = message.get("id").cloned();
        let Some(method) = method else {
            // A response to nothing we sent, or garbage; neither needs a reply.
            if let Some(id) = id {
                respond_error(&out, &id, -32600, "invalid request: no method");
            }
            continue;
        };
        let Some(id) = id else {
            // Notifications: `initialized` and `cancelled` need nothing. A
            // cancel arrives only after Goose has stopped waiting for the call.
            continue;
        };
        match method {
            "initialize" => {
                let version = message
                    .pointer("/params/protocolVersion")
                    .and_then(Value::as_str)
                    .unwrap_or("2024-11-05");
                respond(
                    &out,
                    &id,
                    json!({
                        "protocolVersion": version,
                        "capabilities": {"tools": {}},
                        "serverInfo": {"name": "ccc", "version": env!("CARGO_PKG_VERSION")},
                    }),
                );
            }
            "ping" => respond(&out, &id, json!({})),
            "tools/list" => {
                let list: Vec<Value> = tools
                    .list()
                    .into_iter()
                    .map(|t| {
                        json!({"name": t.name, "description": t.description, "inputSchema": t.schema})
                    })
                    .collect();
                respond(&out, &id, json!({"tools": list}));
            }
            "tools/call" => {
                let Some(name) = message.pointer("/params/name").and_then(Value::as_str) else {
                    respond_error(&out, &id, -32602, "tools/call needs params.name");
                    continue;
                };
                let name = name.to_string();
                let args = message
                    .pointer("/params/arguments")
                    .cloned()
                    .unwrap_or_else(|| json!({}));
                let tools = Arc::clone(&tools);
                let out = Arc::clone(&out);
                calls.push(std::thread::spawn(move || {
                    // A panicking tool still answers, so Goose does not wait out
                    // its own call timeout.
                    let reply = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        tools.call(&name, &args)
                    }))
                    .unwrap_or_else(|_| ToolReply::error(format!("internal error in tool {name}")));
                    respond(
                        &out,
                        &id,
                        json!({
                            "content": [{"type": "text", "text": cap_text(reply.text, RESULT_CAP)}],
                            "isError": reply.is_error,
                        }),
                    );
                }));
                calls.retain(|c| !c.is_finished());
            }
            other => respond_error(&out, &id, -32601, &format!("method not found: {other}")),
        }
    }
    tools.shutdown();
    for call in calls {
        let _ = call.join();
    }
    Ok(())
}

/// A required string argument.
pub fn arg_str<'a>(args: &'a Value, key: &str) -> Result<&'a str, ToolReply> {
    args.get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| ToolReply::error(format!("missing string argument `{key}`")))
}

/// An optional string argument; a non-string value is an error.
pub fn arg_opt_str<'a>(args: &'a Value, key: &str) -> Result<Option<&'a str>, ToolReply> {
    match args.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(s)) => Ok(Some(s)),
        Some(_) => Err(ToolReply::error(format!(
            "argument `{key}` must be a string"
        ))),
    }
}

/// An optional non-negative integer argument.
pub fn arg_opt_u64(args: &Value, key: &str) -> Result<Option<u64>, ToolReply> {
    match args.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(v) => v.as_u64().map(Some).ok_or_else(|| {
            ToolReply::error(format!("argument `{key}` must be a non-negative integer"))
        }),
    }
}

/// An optional boolean argument.
pub fn arg_opt_bool(args: &Value, key: &str) -> Result<Option<bool>, ToolReply> {
    match args.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Bool(b)) => Ok(Some(*b)),
        Some(_) => Err(ToolReply::error(format!(
            "argument `{key}` must be a boolean"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    struct Echo;

    impl Toolset for Echo {
        fn list(&self) -> Vec<ToolDef> {
            vec![ToolDef {
                name: "echo",
                description: "echoes `text`".into(),
                schema: json!({"type": "object"}),
            }]
        }

        fn call(&self, name: &str, args: &Value) -> ToolReply {
            match (name, arg_str(args, "text")) {
                ("echo", Ok(t)) => ToolReply::ok(t),
                ("echo", Err(e)) => e,
                ("panic", _) => panic!("a tool bug"),
                _ => ToolReply::error(format!("unknown tool {name}")),
            }
        }
    }

    #[derive(Clone, Default)]
    struct Sink(Arc<Mutex<Vec<u8>>>);

    impl Write for Sink {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(buf);
            Ok(buf.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    /// Feeds `requests` to a server over `tools` and returns its replies in
    /// arrival order.
    pub(super) fn exchange(tools: Arc<dyn Toolset>, requests: &[Value]) -> Vec<Value> {
        let mut input = String::new();
        for r in requests {
            input.push_str(&r.to_string());
            input.push('\n');
        }
        let sink = Sink::default();
        serve(tools, Cursor::new(input), sink.clone()).unwrap();
        let bytes = sink.0.lock().unwrap().clone();
        String::from_utf8(bytes)
            .unwrap()
            .lines()
            .map(|l| serde_json::from_str(l).unwrap())
            .collect()
    }

    pub(super) fn reply_for(replies: &[Value], id: u64) -> &Value {
        replies
            .iter()
            .find(|r| r["id"] == json!(id))
            .unwrap_or_else(|| panic!("no reply for id {id} in {replies:?}"))
    }

    #[test]
    fn handshake_list_call_and_unknown_method() {
        let replies = exchange(
            Arc::new(Echo),
            &[
                json!({"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-03-26"}}),
                json!({"jsonrpc":"2.0","method":"notifications/initialized"}),
                json!({"jsonrpc":"2.0","id":2,"method":"tools/list"}),
                json!({"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"echo","arguments":{"text":"hi"}}}),
                json!({"jsonrpc":"2.0","id":4,"method":"resources/list"}),
                json!({"jsonrpc":"2.0","method":"notifications/cancelled","params":{"requestId":3}}),
                json!({"jsonrpc":"2.0","id":5,"method":"ping"}),
                json!({"jsonrpc":"2.0","id":6,"method":"tools/call","params":{"name":"echo","arguments":{}}}),
                json!({"jsonrpc":"2.0","id":7,"method":"tools/call","params":{"name":"panic"}}),
                json!({"jsonrpc":"2.0","id":8,"method":"tools/call","params":{}}),
                json!({"jsonrpc":"2.0","id":9,"result":{}}),
            ],
        );
        assert_eq!(replies.len(), 9, "{replies:?}");
        let init = reply_for(&replies, 1);
        assert_eq!(init["result"]["protocolVersion"], "2025-03-26");
        assert_eq!(init["result"]["capabilities"], json!({"tools": {}}));
        assert_eq!(reply_for(&replies, 2)["result"]["tools"][0]["name"], "echo");
        assert_eq!(
            reply_for(&replies, 2)["result"]["tools"][0]["inputSchema"],
            json!({"type": "object"})
        );
        let call = &reply_for(&replies, 3)["result"];
        assert_eq!(call["content"][0], json!({"type": "text", "text": "hi"}));
        assert_eq!(call["isError"], false);
        assert_eq!(reply_for(&replies, 4)["error"]["code"], -32601);
        assert_eq!(reply_for(&replies, 5)["result"], json!({}));
        let bad = &reply_for(&replies, 6)["result"];
        assert_eq!(bad["isError"], true);
        assert!(bad["content"][0]["text"]
            .as_str()
            .unwrap()
            .contains("`text`"));
        let panicked = &reply_for(&replies, 7)["result"];
        assert_eq!(panicked["isError"], true);
        assert_eq!(
            panicked["content"][0]["text"],
            "internal error in tool panic"
        );
        assert_eq!(reply_for(&replies, 8)["error"]["code"], -32602);
        assert_eq!(reply_for(&replies, 9)["error"]["code"], -32600);
    }

    #[test]
    fn unparseable_line_gets_a_parse_error_and_serving_continues() {
        let sink = Sink::default();
        let input = "not json\n{\"jsonrpc\":\"2.0\",\"id\":7,\"method\":\"ping\"}\n";
        serve(Arc::new(Echo), Cursor::new(input), sink.clone()).unwrap();
        let text = String::from_utf8(sink.0.lock().unwrap().clone()).unwrap();
        let lines: Vec<Value> = text
            .lines()
            .map(|l| serde_json::from_str(l).unwrap())
            .collect();
        assert_eq!(lines[0]["error"]["code"], -32700);
        assert_eq!(lines[1]["id"], 7);
    }

    #[test]
    fn cap_text_cuts_on_a_char_boundary_and_says_so() {
        assert_eq!(cap_text("short".into(), 10), "short");
        let capped = cap_text("aé".repeat(10), 4);
        assert!(
            capped.starts_with("aéa\n[truncated: showing 4 of 30 bytes]"),
            "{capped}"
        );
    }

    #[test]
    fn run_refuses_a_bad_policy_and_a_wait_for_host_or_box() {
        // Not BoxInner: that role closes every descriptor above 2 first.
        let err = run(Role::Host, "zz", None).unwrap_err();
        assert!(err.to_string().contains("hex"), "{err:#}");
        let policy = GoosePolicy {
            tools: [crate::goose_policy::GooseTool::Read].into(),
            agents: vec![],
            read_roots: vec![],
            write_roots: vec![],
            exec_roots: vec![],
        };
        for role in [Role::Host, Role::Boxed] {
            let err = run(role, &policy.to_hex(), Some(240)).unwrap_err();
            assert!(
                err.to_string()
                    .contains("--tool-wait-s is for --role box-inner only"),
                "{err:#}"
            );
        }
    }

    #[test]
    fn box_inner_wait_is_required_and_bounded() {
        assert_eq!(box_inner_wait(Some(1)).unwrap(), 1);
        assert_eq!(
            box_inner_wait(Some(GOOSE_MAX_TOOL_WAIT_S)).unwrap(),
            GOOSE_MAX_TOOL_WAIT_S
        );
        for bad in [0, GOOSE_MAX_TOOL_WAIT_S + 1] {
            let err = box_inner_wait(Some(bad)).unwrap_err();
            assert!(err.to_string().contains("must be within"), "{err:#}");
        }
        let err = box_inner_wait(None).unwrap_err();
        assert!(err.to_string().contains("needs --tool-wait-s"), "{err:#}");
    }
}
