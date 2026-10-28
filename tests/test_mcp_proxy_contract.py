"""Behavior tests for the stdio MCP proxy at the standard MCP client seam.

The proxy is launched exactly the way the Hermes installer configures it —
``python -m context_mode_hermes.mcp_proxy --engine-bin ...`` — and spoken to
with the official MCP SDK client over stdio, mirroring Hermes's native MCP
Python handler. A scripted upstream engine records what each worker process
actually observed (environment, working directory, verbatim arguments), so the
tests assert proxy behavior without touching engine internals.
"""
from __future__ import annotations

import asyncio
import json
import os
import stat
import sys
import textwrap
import time
from pathlib import Path

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_STUB_ENGINE = textwrap.dedent(
    """
    import json
    import os
    import signal
    import uuid

    import anyio
    from mcp import types
    from mcp.server.lowlevel import Server
    from mcp.server.stdio import stdio_server

    WORKER = uuid.uuid4().hex[:8]

    TOOLS = [
        types.Tool(
            name="ctx_execute",
            description="stub execution tool",
            input_schema={
                "type": "object",
                "properties": {
                    "language": {"type": "string"},
                    "code": {"type": "string"},
                    "cwd": {"type": "string"},
                    "timeout": {"type": "number"},
                    "intent": {"type": "string"},
                },
                "required": ["language", "code"],
            },
        ),
        types.Tool(
            name="ctx_index",
            description="stub indexing tool",
            input_schema={
                "type": "object",
                "properties": {
                    "content": {"type": "string"},
                    "path": {"type": "string"},
                    "source": {"type": "string"},
                },
                "required": ["content"],
            },
        ),
        types.Tool(
            name="ctx_slow",
            description="stub slow tool",
            input_schema={
                "type": "object",
                "properties": {"seconds": {"type": "number"}},
            },
        ),
        types.Tool(
            name="ctx_fail",
            description="stub failing tool",
            input_schema={"type": "object", "properties": {}},
        ),
    ]

    RECORD = os.environ["CM_STUB_RECORD"]
    PIDFILE = os.environ["CM_STUB_PIDFILE"]

    def record(name, arguments, phase="start"):
        entry = {
            "worker": WORKER,
            "pid": os.getpid(),
            "name": name,
            "phase": phase,
            "arguments": arguments,
            "cwd": os.getcwd(),
            "env": {
                key: os.environ.get(key)
                for key in (
                    "CLAUDE_SESSION_ID",
                    "CLAUDE_PROJECT_DIR",
                    "CONTEXT_MODE_PROJECT_DIR",
                    "CONTEXT_MODE_DIR",
                    "HERMES_HOME",
                )
            },
        }
        with open(RECORD, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry) + "\\n")

    async def main() -> None:
        with open(f"{PIDFILE}.{os.getpid()}", "w", encoding="utf-8") as handle:
            handle.write(str(os.getpid()))

        async def on_list_tools(_ctx, _params):
            return types.ListToolsResult(tools=TOOLS)

        async def on_call_tool(_ctx, params):
            name = params.name
            arguments = dict(params.arguments or {})
            if name == "ctx_slow":
                record(name, arguments, "start")
                await anyio.sleep(float(arguments.get("seconds", 0)))
                record(name, arguments, "end")
                text = f"slept worker={WORKER}"
            elif name == "ctx_fail":
                record(name, arguments)
                return types.CallToolResult(
                    is_error=True,
                    content=[types.TextContent(type="text", text="stub failure")],
                    structured_content={"fail": True},
                )
            else:
                record(name, arguments)
                text = f"ok worker={WORKER} name={name}"
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=text)],
                structured_content={"worker": WORKER, "name": name},
            )

        server = Server(
            "context-mode",
            version="stub-1",
            on_list_tools=on_list_tools,
            on_call_tool=on_call_tool,
        )
        async def serve() -> None:
            async with stdio_server() as (read, write):
                await server.run(read, write, server.create_initialization_options())

        if os.environ.get("CM_STUB_HANG"):
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            # Pathological worker: serves calls normally, but the process
            # outlives stdin EOF, so the proxy's close() must bound its wait.
            async with anyio.create_task_group() as tg:
                tg.start_soon(serve)
                tg.start_soon(anyio.sleep_forever)
        else:
            await serve()

    anyio.run(main)
    """
)


def _hermes_context(session_id: str, cwd: str, project_dir: str) -> dict:
    return {"session_id": session_id, "cwd": cwd, "project_dir": project_dir}


def _write_stub_engine(tmp_path: Path) -> Path:
    stub = tmp_path / "stub_engine.py"
    stub.write_text(f"#!{sys.executable}\n" + _STUB_ENGINE, encoding="utf-8")
    stub.chmod(stub.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return stub


def _proxy_parameters(
    tmp_path: Path, stub: Path, max_workers: int, hang_on_close: bool = False
) -> StdioServerParameters:
    home = tmp_path / "home"
    (home / "context-mode").mkdir(parents=True, exist_ok=True)
    return StdioServerParameters(
        command=sys.executable,
        args=[
            "-m", "context_mode_hermes.mcp_proxy",
            "--engine-bin", str(stub),
            "--max-workers", str(max_workers),
        ],
        env={
            **({"CM_STUB_HANG": "1"} if hang_on_close else {}),
            "HERMES_HOME": str(home),
            "CONTEXT_MODE_DIR": str(home / "context-mode"),
            "CM_STUB_RECORD": str(tmp_path / "record.jsonl"),
            "CM_STUB_PIDFILE": str(tmp_path / "pidfile"),
            "PATH": os.environ.get("PATH", ""),
            "HOME": os.environ.get("HOME", str(home)),
        },
        cwd=str(tmp_path),
    )


def _records(tmp_path: Path) -> list[dict]:
    record = tmp_path / "record.jsonl"
    if not record.exists():
        return []
    return [json.loads(line) for line in record.read_text(encoding="utf-8").splitlines()]


def _stub_pids(tmp_path: Path) -> list[int]:
    return [int(p.read_text()) for p in sorted(tmp_path.glob("pidfile.*"))]


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _workspace(tmp_path: Path, name: str, sub: str | None = None) -> Path:
    path = tmp_path / name
    if sub:
        path = path / sub
    path.mkdir(parents=True, exist_ok=True)
    return path


async def _open_session(params: StdioServerParameters) -> tuple:
    import contextlib

    stack = contextlib.AsyncExitStack()
    read, write = await stack.enter_async_context(stdio_client(params))
    session = await stack.enter_async_context(ClientSession(read, write))
    await session.initialize()
    return stack, session




def test_call_without_metadata_fails_closed(tmp_path: Path) -> None:
    stub = _write_stub_engine(tmp_path)

    async def scenario() -> dict:
        stack, session = await _open_session(_proxy_parameters(tmp_path, stub, 4))
        try:
            result = await session.call_tool(
                "ctx_execute", arguments={"language": "bash", "code": "ls"}
            )
            return {"is_error": result.is_error, "text": result.content[0].text}
        finally:
            await stack.aclose()

    outcome = asyncio.run(scenario())
    assert outcome["is_error"] is True
    assert _records(tmp_path) == [], "rejected calls must never reach an engine worker"


def test_call_with_incomplete_metadata_fails_closed(tmp_path: Path) -> None:
    stub = _write_stub_engine(tmp_path)

    async def scenario() -> bool:
        stack, session = await _open_session(_proxy_parameters(tmp_path, stub, 4))
        try:
            result = await session.call_tool(
                "ctx_index",
                arguments={
                    "content": "x",
                    "_hermes_context": {"session_id": "", "cwd": "/tmp", "project_dir": "/tmp"},
                },
            )
            return result.is_error
        finally:
            await stack.aclose()

    assert asyncio.run(scenario()) is True
    assert _records(tmp_path) == []




def test_forged_metadata_cannot_move_workspace(tmp_path: Path) -> None:
    stub = _write_stub_engine(tmp_path)
    ws_a = _workspace(tmp_path, "wsA")
    ws_b = _workspace(tmp_path, "wsB")

    async def scenario() -> list[dict]:
        stack, session = await _open_session(_proxy_parameters(tmp_path, stub, 4))
        try:
            await session.call_tool(
                "ctx_execute",
                arguments={
                    "language": "bash",
                    "code": "pwd",
                    "_hermes_context": _hermes_context("s1", str(ws_a), str(ws_a)),
                },
            )
            forged = _hermes_context("s1", str(ws_b), str(ws_b))
            await session.call_tool(
                "ctx_execute",
                arguments={"language": "bash", "code": "pwd", "_hermes_context": forged},
            )
        finally:
            await stack.aclose()
        return _records(tmp_path)

    records = asyncio.run(scenario())
    assert len(records) == 2
    assert all(r["env"]["CLAUDE_PROJECT_DIR"] == str(ws_a) for r in records)
    assert all(r["env"]["CONTEXT_MODE_PROJECT_DIR"] == str(ws_a) for r in records)
    assert records[0]["worker"] == records[1]["worker"], "same session keeps one worker"


def test_sessions_use_separate_workers(tmp_path: Path) -> None:
    stub = _write_stub_engine(tmp_path)
    ws_a = _workspace(tmp_path, "wsA")
    ws_b = _workspace(tmp_path, "wsB")

    async def scenario() -> list[dict]:
        stack, session = await _open_session(_proxy_parameters(tmp_path, stub, 4))
        try:
            for sid, ws in (("s1", ws_a), ("s2", ws_b)):
                await session.call_tool(
                    "ctx_execute",
                    arguments={
                        "language": "bash",
                        "code": "pwd",
                        "_hermes_context": _hermes_context(sid, str(ws), str(ws)),
                    },
                )
        finally:
            await stack.aclose()
        return _records(tmp_path)

    records = asyncio.run(scenario())
    assert [r["env"]["CLAUDE_SESSION_ID"] for r in records] == ["s1", "s2"]
    assert records[0]["worker"] != records[1]["worker"]
    assert records[0]["env"]["CLAUDE_PROJECT_DIR"] == str(ws_a)
    assert records[1]["env"]["CLAUDE_PROJECT_DIR"] == str(ws_b)


def test_calls_serialize_per_worker(tmp_path: Path) -> None:
    stub = _write_stub_engine(tmp_path)
    ws = _workspace(tmp_path, "wsA")
    meta = _hermes_context("s1", str(ws), str(ws))

    async def scenario() -> list[dict]:
        stack, session = await _open_session(_proxy_parameters(tmp_path, stub, 4))
        try:
            import time

            start = time.monotonic()
            await asyncio.gather(
                session.call_tool("ctx_slow", arguments={"seconds": 0.4, "_hermes_context": meta}),
                session.call_tool("ctx_slow", arguments={"seconds": 0.4, "_hermes_context": meta}),
            )
            assert time.monotonic() - start >= 0.7, "calls on one worker must not overlap"
        finally:
            await stack.aclose()
        return _records(tmp_path)

    records = asyncio.run(scenario())
    assert len([r for r in records if r["phase"] == "start"]) == 2
    assert len({r["worker"] for r in records}) == 1


async def _two_slow_calls(session, ws1: str, ws2: str) -> None:
    await asyncio.gather(
        session.call_tool(
            "ctx_slow",
            arguments={"seconds": 1.0, "_hermes_context": _hermes_context("s1", ws1, ws1)},
        ),
        session.call_tool(
            "ctx_slow",
            arguments={"seconds": 1.0, "_hermes_context": _hermes_context("s2", ws2, ws2)},
        ),
    )


def test_pool_bounds_busy_workers(tmp_path: Path) -> None:
    stub = _write_stub_engine(tmp_path)

    async def scenario() -> str:
        stack, session = await _open_session(_proxy_parameters(tmp_path, stub, 2))
        try:
            ws1, ws2, ws3 = (
                str(_workspace(tmp_path, name)) for name in ("ws1", "ws2", "ws3")
            )
            slow = asyncio.ensure_future(_two_slow_calls(session, ws1, ws2))
            await asyncio.sleep(0.8)
            result = await session.call_tool(
                "ctx_execute",
                arguments={
                    "language": "bash",
                    "code": "pwd",
                    "_hermes_context": _hermes_context("s3", ws3, ws3),
                },
            )
            await slow
            return result.content[0].text
        finally:
            await stack.aclose()

    text = asyncio.run(scenario())
    assert "limit" in text or "busy" in text, text
    assert not any(r["env"]["CLAUDE_SESSION_ID"] == "s3" for r in _records(tmp_path))


def test_lru_eviction_and_respawn(tmp_path: Path) -> None:
    stub = _write_stub_engine(tmp_path)

    async def scenario() -> list[str]:
        stack, session = await _open_session(_proxy_parameters(tmp_path, stub, 2))
        try:
            async def call(sid: str, ws: str) -> str:
                result = await session.call_tool(
                    "ctx_execute",
                    arguments={
                        "language": "bash",
                        "code": "pwd",
                        "_hermes_context": _hermes_context(sid, ws, ws),
                    },
                )
                return result.structured_content["worker"]

            ws1, ws2, ws3 = (
                str(_workspace(tmp_path, name)) for name in ("ws1", "ws2", "ws3")
            )
            workers = [await call("s1", ws1), await call("s2", ws2), await call("s3", ws3)]
            workers.append(await call("s1", ws1))
            return workers
        finally:
            await stack.aclose()

    workers = asyncio.run(scenario())
    assert len(set(workers[:3])) == 3, "each session got a fresh worker"
    assert workers[3] != workers[0], "evicted session must respawn a new worker"


def test_stuck_worker_close_does_not_stall_eviction_or_leak_the_pool(tmp_path: Path) -> None:
    """A worker whose process ignores stdin EOF cannot stall pool eviction:
    closing the stuck worker stays bounded, the pool slot is released, and
    every later session is still served (max_workers=1, so each new session
    must evict the previous one)."""
    import signal

    stub = _write_stub_engine(tmp_path)

    async def scenario() -> list[str]:
        stack, session = await _open_session(_proxy_parameters(tmp_path, stub, 1, hang_on_close=True))
        try:
            async def call(sid: str, ws: str) -> str:
                result = await session.call_tool(
                    "ctx_execute",
                    arguments={
                        "language": "bash",
                        "code": "pwd",
                        "_hermes_context": _hermes_context(sid, ws, ws),
                    },
                )
                live = [pid for pid in _stub_pids(tmp_path) if _pid_alive(pid)]
                assert len(live) == 1, f"Evicted workers survived beyond the pool limit: {live}"
                return result.structured_content["worker"]

            ws1, ws2, ws3 = (str(_workspace(tmp_path, name)) for name in ("ws1", "ws2", "ws3"))
            return [await call("s1", ws1), await call("s2", ws2), await call("s3", ws3)]
        finally:
            await stack.aclose()

    try:
        workers = asyncio.run(scenario())
    finally:
        # The pathological workers must never outlive the test.
        for pid in _stub_pids(tmp_path):
            try:
                os.kill(pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
    assert len(set(workers)) == 3, "every session must be served despite the stuck close"




def test_eof_shutdown_reaps_workers(tmp_path: Path) -> None:
    stub = _write_stub_engine(tmp_path)
    ws = _workspace(tmp_path, "wsA")

    async def scenario() -> None:
        stack, session = await _open_session(_proxy_parameters(tmp_path, stub, 4))
        try:
            await session.call_tool(
                "ctx_execute",
                arguments={
                    "language": "bash",
                    "code": "pwd",
                    "_hermes_context": _hermes_context("s1", str(ws), str(ws)),
                },
            )
        finally:
            await stack.aclose()

    asyncio.run(scenario())
    pids = _stub_pids(tmp_path)
    assert pids, "workers must have announced themselves"
    for _ in range(100):
        if not any(_pid_alive(pid) for pid in pids):
            break
        time.sleep(0.1)
    assert not any(_pid_alive(pid) for pid in pids), f"orphaned engine workers: {pids}"


def _real_engine_binary() -> str | None:
    binary = os.environ.get("CONTEXT_MODE_TEST_BIN")
    return binary if binary and Path(binary).is_file() else None


def test_real_engine_roundtrip_and_workspace_isolation(tmp_path: Path) -> None:
    binary = _real_engine_binary()
    if binary is None:
        pytest.skip("Set CONTEXT_MODE_TEST_BIN to a pinned context-mode install to exercise the real engine")
    ws_a = _workspace(tmp_path, "wsA")
    ws_b = _workspace(tmp_path, "wsB")
    marker = "llama-quasar-7741"

    async def scenario() -> dict:
        stack, session = await _open_session(_proxy_parameters(tmp_path, binary, 4))
        try:
            await session.call_tool(
                "ctx_index",
                arguments={
                    "content": f"# {marker}\nDeployment requires forty-six ceramic shields.",
                    "_hermes_context": _hermes_context("s1", str(ws_a), str(ws_a)),
                },
            )
            found = await session.call_tool(
                "ctx_search",
                arguments={
                    "queries": [marker],
                    "limit": 3,
                    "_hermes_context": _hermes_context("s1", str(ws_a), str(ws_a)),
                },
            )
            other = await session.call_tool(
                "ctx_search",
                arguments={
                    "queries": [marker],
                    "limit": 3,
                    "_hermes_context": _hermes_context("s2", str(ws_b), str(ws_b)),
                },
            )
            return {
                "found": found.content[0].text,
                "other": other.content[0].text,
            }
        finally:
            await stack.aclose()

    outcome = asyncio.run(scenario())
    assert "forty-six ceramic shields" in outcome["found"]
    assert "forty-six ceramic shields" not in outcome["other"], "Other workspaces must not see this content"
