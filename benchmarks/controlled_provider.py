"""Scripted OpenAI-compatible provider + local fixture server (loopback only).

This is a *controlled provider*: every chat completion is answered by a
deterministic, reactive script — no model, no network egress. The script
inspects the actual request (messages, advertised tools) and the actual tool
results Hermes sends back, so final answers are derived from real native tool
execution. It exists to measure wire bytes / rough token estimates / native
dispatch under identical scripted turns for Hermes profiles (baseline vs
context-mode MCP vs plugin). It is NOT an LLM benchmark.

Hermes's native progressive tool disclosure is respected as-is: when ctx tools
are deferred, the wire shows tool_search/tool_describe/tool_call and the script
walks that bridge (search -> describe -> call) exactly like a real model would;
when the schemas are inline, the script calls the ctx tools directly.
"""
from __future__ import annotations

import json
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

NONCE_RE = re.compile(r"CM-BENCH-NONCE-[0-9a-f]+")
PLUGIN_GUIDANCE_MARKER = "<context_window_protection>"
CTX_FILE_NAME_RE = re.compile(r"[A-Za-z0-9_.\-]*ctx_execute_file[A-Za-z0-9_.\-]*")

SERVER_MODEL_CONTEXT = 65_536  # advertised via /v1/models; Hermes requires >= 64_000
FULL_OUTPUT_PATH_RE = re.compile(
    r'(?:"full_output_path"\s*:\s*"([^"]+)"|Full output saved to:\s*([^\s\n]+))')
AGG_RE = re.compile(r"COUNT=(\d+)\s+SUM=(-?\d+)")
TRUNCATION_NOTE = "Output exceeded the capture window"
PERSISTED_OUTPUT_TAG = "<persisted-output>"
CTX_EXEC_NAME_RE = re.compile(r"[A-Za-z0-9_.\-]*ctx_execute[A-Za-z0-9_.\-]*")

AGG_UNWRAP = (
    "def parse(text):\n"
    "    return json.JSONDecoder(strict=False).raw_decode(text.strip())[0]\n"
    "data = parse(raw)\n"
    "if isinstance(data, dict) and 'records' not in data:\n"
    "    try:\n"
    "        data = parse(open(data['full_output_path']).read())\n"
    "    except Exception:\n"
    "        pass\n"
    "if isinstance(data, dict) and 'records' not in data:\n"
    "    data = parse(data['output'])\n"
    "recs = data['records']\n"
    "print('COUNT=%d SUM=%d' % (len(recs), sum(r['value'] for r in recs)))\n"
)

AGG_CODE_FROM_FILE_CONTENT = (
    "import json\n"
    "raw = FILE_CONTENT\n"
    + AGG_UNWRAP
)


def _agg_code_open(path: str) -> str:
    return (
        "import json\n"
        f"raw = open({path!r}).read()\n"
        + AGG_UNWRAP
    )


def _agg_code_fetch(url: str) -> str:
    return (
        "import urllib.request, json\n"
        f"raw = urllib.request.urlopen({url!r}, timeout=30).read().decode()\n"
        + AGG_UNWRAP
    )


AGG_SHELL_BODY = (
    "import json\n"
    "p=lambda t: json.JSONDecoder(strict=False).raw_decode(t.strip())[0]\n"
    "d=p(raw)\n"
    "d=p(open(d['full_output_path']).read()) if isinstance(d,dict) and 'records' not in d "
    "and 'full_output_path' in d else d\n"
    "d=p(d['output']) if isinstance(d,dict) and 'records' not in d else d\n"
    "r=d['records'];print('COUNT=%d SUM=%d'%(len(r),sum(x['value'] for x in r)))"
)


def _agg_shell(path: str) -> str:
    return "python3 -c " + _shell_quote(
        f"import json\nraw=open({path!r}).read()\n" + AGG_SHELL_BODY)


def _agg_fetch_shell(url: str) -> str:
    body = "import json\nraw=open('fixture.json').read()\n" + AGG_SHELL_BODY
    return f"curl -sS {url} -o fixture.json && python3 -c " + _shell_quote(body)


def _shell_quote(body: str) -> str:
    return "'" + body.replace("'", "'\\''") + "'"


UNTRUSTED_CONTEXT = {"session_id": "untrusted-bench-metadata", "cwd": "/nonexistent-bench",
                     "project_dir": "/nonexistent-bench"}

@dataclass
class RunParams:
    key: str
    arm: str                     # baseline | mcp | plugin
    workload: str                # small | large
    model: str
    nonce: str
    fixture_url: str
    fixture_count: int
    fixture_sum: int
    probe_untrusted_context: bool = False
    script_log: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class _RunState:
    params: RunParams
    scripted_index: int = 0       # index among scripted (tool-bearing) requests
    started: float = field(default_factory=time.monotonic)
    spill_path: Optional[str] = None      # terminal spill file from the curl result
    found_ctx_tool: Optional[str] = None
    bridge_stage: Optional[str] = None    # None | "search" | "describe"
    recovering: bool = False              # ctx rejected -> native fallback issued


class _Script:
    """Reactive turn script: each reply is chosen from the request's actual
    messages/tools, so the same script adapts to whatever each Hermes arm
    actually advertises (inline ctx tools, the deferred-tool bridge, plugin
    blocks, native terminal spill files)."""

    def __init__(self, state: _RunState) -> None:
        self.state = state
        self.p = state.params

    @property
    def spill_path(self) -> Optional[str]:
        return self.state.spill_path

    @spill_path.setter
    def spill_path(self, value: Optional[str]) -> None:
        self.state.spill_path = value

    @property
    def found_ctx_tool(self) -> Optional[str]:
        return self.state.found_ctx_tool

    @found_ctx_tool.setter
    def found_ctx_tool(self, value: Optional[str]) -> None:
        self.state.found_ctx_tool = value

    @property
    def bridge_stage(self) -> Optional[str]:
        return self.state.bridge_stage

    @bridge_stage.setter
    def bridge_stage(self, value: Optional[str]) -> None:
        self.state.bridge_stage = value

    @property
    def recovering(self) -> bool:
        return self.state.recovering

    @recovering.setter
    def recovering(self, value: bool) -> None:
        self.state.recovering = value

    # -- helpers ---------------------------------------------------------
    @staticmethod
    def _last_tool_content(messages: list[dict]) -> str:
        chunks: list[str] = []
        for msg in reversed(messages):
            if msg.get("role") == "tool":
                content = msg.get("content")
                if isinstance(content, str):
                    chunks.append(content)
                elif content is not None:
                    chunks.append(json.dumps(content))
                if len(chunks) >= 3:
                    break
        return "\n".join(reversed(chunks))

    @staticmethod
    def _tool_names(tools: Optional[list]) -> list[str]:
        names = []
        for tool in tools or []:
            fn = tool.get("function") if isinstance(tool, dict) else None
            name = fn.get("name") if isinstance(fn, dict) else tool.get("name")
            if isinstance(name, str):
                names.append(name)
        return names

    @staticmethod
    def _bridge_active(names: list[str]) -> bool:
        return "tool_call" in names and "tool_search" in names

    def _ctx_tool(self, names: list[str], needle: str) -> Optional[str]:
        exact = [n for n in names if n.split("__")[-1] == needle]
        if exact:
            return exact[0]
        contains = [n for n in names if needle in n]
        return contains[0] if contains else None

    def _log(self, event: str, **detail: Any) -> None:
        self.p.script_log.append({"event": event, **detail})

    def _untrusted(self, args: dict[str, Any]) -> dict[str, Any]:
        if self.p.probe_untrusted_context:
            args["_hermes_context"] = dict(UNTRUSTED_CONTEXT)
        return args

    # -- entry ------------------------------------------------------------
    def reply(self, body: dict) -> dict:
        if self.p.workload == "small":
            return self._small(body)
        return self._large(body)

    def _small(self, body: dict) -> dict:
        messages = body.get("messages") or []
        if self.state.scripted_index == 0:
            self.state.scripted_index += 1
            self._log("tool_call", tool="terminal", command=f"printf '{self.p.nonce}'")
            return _tool_call_reply("terminal", {"command": f"printf '{self.p.nonce}'"})
        content = self._last_tool_content(messages)
        self.state.scripted_index += 1
        match = NONCE_RE.search(content)
        if match:
            self._log("final", nonce=match.group(0), source="tool_result")
            return _final_reply(f"NONCE={match.group(0)}")
        self._log("final_missing_nonce", tool_result_head=content[:400])
        return _final_reply("NONCE=MISSING")

    # -- large workload -----------------------------------------------------
    def _large(self, body: dict) -> dict:
        messages = body.get("messages") or []
        names = self._tool_names(body.get("tools"))
        idx = self.state.scripted_index
        self.state.scripted_index += 1

        if idx == 0:
            self._log("tool_call", tool="terminal", command=f"curl -sS {self.p.fixture_url}")
            return _tool_call_reply("terminal", {"command": f"curl -sS {self.p.fixture_url}"})

        if self.bridge_stage == "search":
            return self._bridge_describe(messages)
        if self.bridge_stage == "describe":
            return self._bridge_call()

        if idx == 1:
            return self._aggregate_step(messages, names)

        content = self._last_tool_content(messages)
        match = AGG_RE.search(content)
        if match:
            self._log("final", count=int(match.group(1)), sum=int(match.group(2)), source="tool_result")
            return _final_reply(f"COUNT={match.group(1)} SUM={match.group(2)}")
        if self.recovering:
            self.recovering = False
            self._log("final_missing_aggregate_after_recovery", tool_result_head=content[:400])
            return _final_reply("AGGREGATE=MISSING")
        if "_hermes_context" in content or ("ctx_" in content and "rejected" in content):
            # The ctx path was rejected (e.g. lifecycle hook not injected yet).
            # Recover like a real model: fall back to the native aggregation and
            # record the rejection — the report shows it either way.
            self.recovering = True
            self._log("ctx_call_rejected_fallback_terminal",
                      tool_result_head=content[:300])
            command = (_agg_shell(self.spill_path) if self.spill_path
                       else _agg_fetch_shell(self.p.fixture_url))
            return _tool_call_reply("terminal", {"command": command})
        self._log("final_missing_aggregate", tool_result_head=content[:400])
        return _final_reply("AGGREGATE=MISSING")

    def _aggregate_step(self, messages: list[dict], names: list[str]) -> dict:
        content = self._last_tool_content(messages)
        spill_match = FULL_OUTPUT_PATH_RE.search(content)
        spill = (spill_match.group(1) or spill_match.group(2)) if spill_match else None
        if spill:
            self.spill_path = spill
        blocked = any(marker in content.lower() for marker in ("denied", "blocked", "context-mode"))
        self._log("saw_tool_result", truncated=TRUNCATION_NOTE in content or
                  PERSISTED_OUTPUT_TAG in content, spill_path=spill,
                  result_chars=len(content), blocked_markers=blocked and not bool(spill))
        ctx_file = self._ctx_tool(names, "ctx_execute_file")
        ctx_exec = self._ctx_tool(names, "ctx_execute")
        if spill and (ctx_file or ctx_exec):
            return self._ctx_direct(ctx_file, ctx_exec)
        if self._bridge_active(names):
            # Native progressive disclosure deferred the ctx tools: walk the
            # bridge (search -> describe -> call) like a real model.
            self.bridge_stage = "search"
            self._log("tool_search", note="ctx tools deferred; discovering via native bridge")
            return _tool_call_reply("tool_search", {
                "queries": ["context execute python code file json aggregate"],
                "limit": 8})
        if spill:
            self._log("tool_call", tool="terminal", path=spill, note="no ctx tools advertised")
            return _tool_call_reply("terminal", {"command": _agg_shell(spill)})
        # No spill handle, no bridge: refetch to disk, then aggregate natively.
        self._log("no_spill_path", tool_result_head=content[:400])
        self._log("tool_call", tool="terminal", note="fallback fetch+aggregate")
        return _tool_call_reply("terminal", {"command": _agg_fetch_shell(self.p.fixture_url)})

    def _ctx_direct(self, ctx_file: Optional[str], ctx_exec: Optional[str]) -> dict:
        assert self.spill_path
        if ctx_file:
            tool, args = ctx_file, {"path": self.spill_path, "language": "python",
                                    "code": AGG_CODE_FROM_FILE_CONTENT}
        else:
            tool, args = ctx_exec or "", {"language": "python",
                                          "code": _agg_code_open(self.spill_path)}
        self._log("tool_call", tool=tool, path=self.spill_path,
                  untrusted_context=self.p.probe_untrusted_context)
        return _tool_call_reply(tool, self._untrusted(args))

    # -- bridge steps -------------------------------------------------------
    def _bridge_describe(self, messages: list[dict]) -> dict:
        content = self._last_tool_content(messages)
        found = CTX_FILE_NAME_RE.search(content) or CTX_EXEC_NAME_RE.search(content)
        if not found:
            self.bridge_stage = None
            self._log("tool_search_found_nothing", search_result_head=content[:300])
            return _tool_call_reply("terminal",
                                    {"command": _agg_fetch_shell(self.p.fixture_url)
                                     if not self.spill_path else _agg_shell(self.spill_path)})
        self.found_ctx_tool = found.group(0)
        self.bridge_stage = "describe"
        self._log("tool_describe", tool=self.found_ctx_tool, via="native bridge")
        return _tool_call_reply("tool_describe", {"names": [self.found_ctx_tool]})

    def _bridge_call(self) -> dict:
        self.bridge_stage = None
        tool = self.found_ctx_tool or ""
        if "ctx_execute_file" in tool and self.spill_path:
            args: dict[str, Any] = {"path": self.spill_path, "language": "python",
                                    "code": AGG_CODE_FROM_FILE_CONTENT}
        elif self.spill_path:
            args = {"language": "python", "code": _agg_code_open(self.spill_path)}
        else:
            args = {"language": "python", "code": _agg_code_fetch(self.p.fixture_url)}
        self._log("bridge_tool_call", tool=tool, via="tool_call bridge",
                  untrusted_context=self.p.probe_untrusted_context)
        return _tool_call_reply("tool_call", {"calls": [{"name": tool,
                                                         "arguments": self._untrusted(args)}]})


# ─── OpenAI wire shapes ────────────────────────────────────────────────────────

def _tool_call_reply(tool: str, arguments: dict[str, Any]) -> dict:
    return {"kind": "tool_calls", "tool": tool, "arguments": arguments}


def _final_reply(text: str) -> dict:
    return {"kind": "final", "text": text}


def _sse_chunks(model: str, reply: dict, call_id: str) -> bytes:
    def frame(delta: dict, finish: Optional[str] = None) -> str:
        chunk = {"id": "chatcmpl-bench", "object": "chat.completion.chunk", "created": 1,
                 "model": model,
                 "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
        return f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

    parts = [frame({"role": "assistant", "content": ""})]
    if reply["kind"] == "tool_calls":
        parts.append(frame({"tool_calls": [
            {"index": 0, "id": call_id, "type": "function",
             "function": {"name": reply["tool"],
                          "arguments": json.dumps(reply["arguments"], ensure_ascii=False)}}]}))
        parts.append(frame({}, finish="tool_calls"))
    else:
        parts.append(frame({"content": reply["text"]}))
        parts.append(frame({}, finish="stop"))
    parts.append("data: [DONE]\n\n")
    return "".join(parts).encode("utf-8")


def _json_response(model: str, reply: dict, call_id: str) -> bytes:
    message: dict[str, Any] = {"role": "assistant"}
    finish = "stop"
    if reply["kind"] == "tool_calls":
        message["content"] = None
        message["tool_calls"] = [{"id": call_id, "type": "function",
                                  "function": {"name": reply["tool"],
                                               "arguments": json.dumps(reply["arguments"],
                                                                       ensure_ascii=False)}}]
        finish = "tool_calls"
    else:
        message["content"] = reply["text"]
    payload = {"id": "chatcmpl-bench", "object": "chat.completion", "created": 1, "model": model,
               "choices": [{"index": 0, "message": message, "finish_reason": finish}],
               "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
               "bench_controlled_provider": True}
    return json.dumps(payload, ensure_ascii=False).encode("utf-8")


def _models_payload(models: list[str]) -> bytes:
    data = [{"id": name, "object": "model", "created": 1, "owned_by": "bench",
             "context_length": SERVER_MODEL_CONTEXT,
             "context_window": SERVER_MODEL_CONTEXT,
             "max_context_length": SERVER_MODEL_CONTEXT} for name in models]
    return json.dumps({"object": "list", "data": data}).encode("utf-8")


def _summarize(body: dict, *, auxiliary: bool) -> dict:
    messages = body.get("messages") or []
    tools = body.get("tools") or []
    tool_result_chars = 0
    tool_messages = 0
    for msg in messages:
        if isinstance(msg, dict) and msg.get("role") == "tool":
            tool_messages += 1
            content = msg.get("content")
            if isinstance(content, str):
                tool_result_chars += len(content)
            elif content is not None:
                tool_result_chars += len(json.dumps(content))
    names = []
    for tool in tools:
        fn = tool.get("function") if isinstance(tool, dict) else None
        name = fn.get("name") if isinstance(fn, dict) else tool.get("name")
        if isinstance(name, str):
            names.append(name)
    joined = "\n".join(m.get("content") if isinstance(m.get("content"), str) else ""
                       for m in messages if isinstance(m, dict))
    return {
        "auxiliary": auxiliary,
        "model": body.get("model"),
        "n_messages": len(messages),
        "roles": [m.get("role") for m in messages if isinstance(m, dict)],
        "tool_messages": tool_messages,
        "tool_result_chars": tool_result_chars,
        "n_tools": len(names),
        "tool_names": names,
        "bridge_active": "tool_call" in names and "tool_search" in names,
        "stream": bool(body.get("stream")),
        "truncation_notes": joined.count(TRUNCATION_NOTE) + joined.count(PERSISTED_OUTPUT_TAG),
        "spill_handles": joined.count("Full output saved to") + joined.count("full_output_path"),
        "plugin_guidance_marker": PLUGIN_GUIDANCE_MARKER in joined,
    }


class ControlledProvider:
    """Threaded loopback HTTP server: /v1/models, /v1/chat/completions, /fixture.json."""

    def __init__(self, fixture: bytes, capture_dir: Path) -> None:
        self.fixture = fixture
        self.capture_dir = capture_dir
        self.capture_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._runs: dict[str, _RunState] = {}
        self._models: set[str] = set()
        self.records: list[dict[str, Any]] = []
        provider = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *_args: Any) -> None:
                pass

            def _send(self, status: int, body: bytes, content_type: str) -> None:
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:  # noqa: N802
                path = self.path.split("?")[0]
                if path == "/fixture.json":
                    self._send(200, provider.fixture, "application/json")
                    provider._record("GET", path, 200, 0, None)
                    return
                if path in ("/v1/models", "/models"):
                    self._send(200, _models_payload(provider.models), "application/json")
                    provider._record("GET", path, 200, 0, None)
                    return
                self._send(404, b'{"error": {"message": "bench: unknown GET path"}}',
                           "application/json")
                provider._record("GET", path, 404, 0, None)

            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length) if length else b""
                path = self.path.split("?")[0]
                if path not in ("/v1/chat/completions", "/chat/completions"):
                    self._send(404, b'{"error": {"message": "bench: unknown POST path"}}',
                               "application/json")
                    provider._record("POST", path, 404, len(raw), None)
                    return
                try:
                    body = json.loads(raw.decode("utf-8")) if raw else {}
                except ValueError:
                    self._send(400, b'{"error": {"message": "bench: invalid JSON"}}',
                               "application/json")
                    provider._record("POST", path, 400, len(raw), None)
                    return
                auth = self.headers.get("Authorization") or ""
                key = auth.removeprefix("Bearer ").strip()
                state = provider._runs.get(key)
                if state is None:
                    self._send(401, b'{"error": {"message": "bench: unknown run key"}}',
                               "application/json")
                    provider._record("POST", path, 401, len(raw), _summarize(body, auxiliary=True))
                    return
                scripted = bool(body.get("tools"))
                reply = (_Script(state).reply(body) if scripted
                         else _final_reply("(bench auxiliary response)"))
                if not scripted:
                    state.params.script_log.append({"event": "auxiliary_reply"})
                call_id = f"call_bench_{len(provider.records)}"
                model = str(body.get("model") or "cm-bench")
                if bool(body.get("stream")):
                    payload, content_type = _sse_chunks(model, reply, call_id), "text/event-stream"
                else:
                    payload, content_type = _json_response(model, reply, call_id), "application/json"
                self._send(200, payload, content_type)
                provider._record("POST", path, 200, len(raw), _summarize(body, auxiliary=not scripted),
                                 key=key, scripted=scripted, reply_kind=reply["kind"],
                                 reply_tool=reply.get("tool"))
                if scripted:
                    provider._dump_body(key, state, raw)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.httpd.serve_forever,
                                       name="bench-provider", daemon=True)

    # -- API for the orchestrator ----------------------------------------
    @property
    def base_url(self) -> str:
        host, port = self.httpd.server_address[:2]
        return f"http://{host}:{port}/v1"

    @property
    def fixture_url(self) -> str:
        host, port = self.httpd.server_address[:2]
        return f"http://{host}:{port}/fixture.json"

    @property
    def models(self) -> list[str]:
        with self._lock:
            return sorted(self._models)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()

    def register_run(self, params: RunParams) -> None:
        with self._lock:
            self._runs.pop(params.key, None)
            self._runs[params.key] = _RunState(params=params)
            self._models.add(params.model)

    def drop_run(self, key: str) -> None:
        with self._lock:
            self._runs.pop(key, None)

    def script_log(self, key: str) -> list[dict[str, Any]]:
        state = self._runs.get(key)
        return list(state.params.script_log) if state else []

    def _record(self, method: str, path: str, status: int, body_bytes: int,
                summary: Optional[dict], **extra: Any) -> None:
        with self._lock:
            self.records.append({"method": method, "path": path, "status": status,
                                 "body_bytes": body_bytes, "ts": time.time(),
                                 "summary": summary, **extra})

    def _dump_body(self, key: str, state: _RunState, raw: bytes) -> None:
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", key)
        (self.capture_dir / f"{safe}.scripted.req{state.scripted_index - 1}.json").write_bytes(raw)


def new_nonce() -> str:
    return f"CM-BENCH-NONCE-{uuid.uuid4().hex[:16]}"
