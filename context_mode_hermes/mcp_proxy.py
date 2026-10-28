"""Stdio MCP proxy that gives every Hermes session its own engine worker.

Hermes v0.22.5 launches MCP servers as plain stdio subprocesses and calls
``call_tool(name, arguments=...)`` with no session or workspace context, while
the upstream context-mode engine keys its knowledge base off the process
environment. This proxy sits between them and speaks nothing but standard MCP:

* ``tools/list``   -> schemas discovered from a retained bootstrap engine
  worker (its own scratch context below the profile root), each schema
  extended with an optional ``_hermes_context`` object (``session_id``,
  ``cwd``, ``project_dir``) that Hermes-side hooks and commands inject into
  every call. That worker's successful MCP initialize() — reached only via a
  real client handshake — is also what publishes the profile's readiness
  sentinel in ``CONTEXT_MODE_MCP_SENTINEL_DIR``; it counts toward the pool
  bound and closes with the pool, so an un-handshaken proxy never counts as
  ready.
* ``tools/call``   -> ``_hermes_context`` is stripped before the call crosses
  the proxy. The session anchor from ``Engine.context`` (first cwd of the
  Hermes session, persisted by the adapter) decides which worker serves the
  call, so model-supplied metadata cannot relocate a session's storage. One
  engine worker process exists per ``(session_id, project_dir)`` with the
  anchored ``CLAUDE_PROJECT_DIR``/``CONTEXT_MODE_PROJECT_DIR`` environment and
  ``cwd``, while relative ``path`` arguments and shell ``cwd`` resolve against
  the live Hermes ``cwd``.

Workers (the bootstrap included) are bounded by an LRU pool; calls on one
worker serialize and distinct workers run concurrently. Each worker's MCP
client stack is opened and closed inside its own task, and every worker lives
in the single task group that owns the stdio server, so parent EOF or SIGTERM
unwinds the whole tree without orphans.
"""
from __future__ import annotations

import argparse
import contextlib
import logging
import os
import signal
import sys
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import anyio
from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server

from context_mode_hermes.engine import Engine, EngineError

logger = logging.getLogger("context_mode_hermes.mcp_proxy")

HERMES_CONTEXT = "_hermes_context"
METADATA_FIELDS = ("session_id", "cwd", "project_dir")
DEFAULT_MAX_WORKERS = 4
START_TIMEOUT = 60.0
CLOSE_TIMEOUT = 5.0
SHELL_CWD_TOOLS = frozenset({"ctx_execute", "ctx_batch_execute"})
PATH_ARGUMENT_TOOLS = frozenset({"ctx_index", "ctx_execute_file"})

SENTINEL_PREFIX = "context-mode-mcp-ready-"


def _publish_sentinel(engine: Engine) -> Path | None:
    """Publish the profile's MCP readiness sentinel (upstream convention).

    The pinned engine's hooks decide redirect-vs-passthrough by scanning
    ``CONTEXT_MODE_MCP_SENTINEL_DIR`` for ``context-mode-mcp-ready-<pid>``
    files and probing the PID (hooks/core/mcp-ready.mjs). The file names the
    proxy — the profile's live MCP server process — but it is only written
    AFTER a genuine upstream engine worker has completed its MCP initialize()
    and is retained in the bounded pool (see McpProxy.list_tools), so a
    spawned-but-never-handshaken proxy is honestly NOT ready. Always in the
    profile-owned dir, never machine-global /tmp, which foreign agents also
    write.
    """
    directory = engine.environment().get("CONTEXT_MODE_MCP_SENTINEL_DIR")
    if not directory:
        return None
    try:
        path = Path(directory) / f"{SENTINEL_PREFIX}{os.getpid()}"
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        path.write_text(str(os.getpid()), encoding="utf-8")
        return path
    except OSError:
        return None


def _retract_sentinel(path: Path | None) -> None:
    if path is None:
        return
    try:
        if path.read_text(encoding="utf-8").strip() == str(os.getpid()):
            path.unlink()
    except OSError:
        pass


class ProxyError(RuntimeError):
    """A condition the proxy reports to the caller as a tool-level error."""


def _augmented_context_property() -> dict[str, Any]:
    return {
        "type": "object",
        "description": "Runtime context injected by the Hermes context-mode plugin.",
        "properties": {
            "session_id": {"type": "string"},
            "cwd": {"type": "string"},
            "project_dir": {"type": "string"},
        },
        "required": list(METADATA_FIELDS),
        "additionalProperties": False,
    }


def _augment_tool_schema(tool: types.Tool) -> types.Tool:
    schema = dict(tool.input_schema or {})
    properties = dict(schema.get("properties") or {})
    properties[HERMES_CONTEXT] = _augmented_context_property()
    schema["properties"] = properties
    schema["required"] = [
        name for name in (schema.get("required") or []) if name != HERMES_CONTEXT
    ]
    return tool.model_copy(update={"input_schema": schema})


def _absolute(base: str, value: str) -> str:
    expanded = os.path.expanduser(value)
    if os.path.isabs(expanded):
        return os.path.normpath(expanded)
    return os.path.normpath(os.path.join(base, expanded))


def _rewrite_arguments(tool_name: str, arguments: dict[str, Any], cwd: str) -> dict[str, Any]:
    """Anchor path-like arguments to the session's live cwd before forwarding."""
    if tool_name in SHELL_CWD_TOOLS:
        value = arguments.get("cwd")
        arguments["cwd"] = _absolute(cwd, value) if isinstance(value, str) and value.strip() else cwd
    elif tool_name in PATH_ARGUMENT_TOOLS:
        value = arguments.get("path")
        if isinstance(value, str) and value.strip():
            arguments["path"] = _absolute(cwd, value)
    return arguments


def _metadata_problem(raw: Any) -> str | None:
    if not isinstance(raw, dict):
        return (
            f"is missing; the Hermes context-mode plugin must inject "
            f"{HERMES_CONTEXT}({', '.join(METADATA_FIELDS)})"
        )
    for name in METADATA_FIELDS:
        value = raw.get(name)
        if not isinstance(value, str) or not value.strip():
            return f".{name} must be a non-empty string"
    return None


def _tool_error(message: str) -> types.CallToolResult:
    return types.CallToolResult(
        is_error=True,
        content=[types.TextContent(type="text", text=message)],
    )


@dataclass
class _Request:
    name: str = ""
    arguments: dict[str, Any] | None = None
    list_tools: bool = False
    done: anyio.Event = field(default_factory=anyio.Event)
    result: types.CallToolResult | None = None
    tools: list[types.Tool] | None = None
    error: BaseException | None = None


class _Worker:
    """One upstream engine subprocess owned by a single supervisor task.

    The MCP client stack (``stdio_client`` + ``ClientSession``) is opened and
    closed inside :meth:`run`, so its cancel scopes never cross tasks. Callers
    enqueue :class:`_Request` items; the supervisor executes them strictly in
    order, which serializes calls per worker while distinct workers run
    concurrently. ``None`` on the inbox is the shutdown sentinel.
    """

    def __init__(self, key: tuple[str, str], params: StdioServerParameters) -> None:
        self.key = key
        self._params = params
        self._inbox_send, self._inbox = anyio.create_memory_object_stream["_Request | None"](0)
        self.started = anyio.Event()
        self.closed = anyio.Event()
        self.startup_error: BaseException | None = None
        self._scope: anyio.CancelScope | None = None

    async def send(self, request: _Request) -> None:
        try:
            await self._inbox_send.send(request)
        except (anyio.BrokenResourceError, anyio.ClosedResourceError) as exc:
            raise ProxyError(
                f"context-mode engine worker for {self.key[0]} exited before the call; retry"
            ) from exc

    async def request(self, request: _Request) -> None:
        await self.send(request)
        await request.done.wait()

    async def close(self) -> None:
        if self.closed.is_set():
            return
        with contextlib.suppress(anyio.BrokenResourceError, anyio.ClosedResourceError):
            with anyio.move_on_after(1.0):
                await self._inbox_send.send(None)
        with anyio.move_on_after(CLOSE_TIMEOUT):
            await self.closed.wait()
        if not self.closed.is_set() and self._scope is not None:
            self._scope.cancel()
            await self.closed.wait()

    async def run(self) -> None:
        # Own cancellation keeps an unresponsive evicted worker inside the
        # same task that opened its MCP SDK cancel scopes.
        with anyio.CancelScope() as scope:
            self._scope = scope
            await self._run()

    async def _run(self) -> None:
        try:
            cancelled_exc = anyio.get_cancelled_exc_class()
            async with contextlib.AsyncExitStack() as stack:
                read, write = await stack.enter_async_context(stdio_client(self._params))
                session = await stack.enter_async_context(ClientSession(read, write))
                with anyio.fail_after(START_TIMEOUT):
                    await session.initialize()
                self.started.set()
                while True:
                    item = await self._inbox.receive()
                    if item is None:
                        break
                    try:
                        if item.list_tools:
                            item.tools = await self._paginate_tools(session)
                        else:
                            item.result = await session.call_tool(
                                item.name, arguments=item.arguments
                            )
                    except cancelled_exc:
                        raise
                    except BaseException as exc:  # session died mid-call
                        item.error = exc
                        item.done.set()
                        break
                    item.done.set()
        except BaseException as exc:
            if not isinstance(exc, cancelled_exc):
                if not self.started.is_set():
                    self.startup_error = exc
                logger.error("context-mode worker %s failed: %r", self.key[0], exc)
            raise
        finally:
            self.started.set()
            self.closed.set()

    async def _paginate_tools(self, session: ClientSession) -> list[types.Tool]:
        page = await session.list_tools()
        tools = list(page.tools)
        while page.next_cursor:
            page = await session.list_tools(
                params=types.PaginatedRequestParams(cursor=page.next_cursor)
            )
            tools.extend(page.tools)
        return tools


class _WorkerPool:
    """Bounded LRU registry of live workers keyed by (session_id, project_dir).

    A session's call counts as busy from before its worker is acquired until
    its result is delivered, so a worker still starting up is never evicted.
    When the pool is full, the least-recently-used idle worker is closed (its
    session respawns on demand from persisted storage); if every worker is
    busy the call fails with a clear error instead of killing an active call.
    """

    def __init__(self, engine: Engine, limit: int, group: anyio.abc.TaskGroup) -> None:
        self._engine = engine
        self._limit = max(1, limit)
        self._group = group
        self._workers: OrderedDict[tuple[str, str], _Worker] = OrderedDict()
        self._pending: dict[tuple[str, str], int] = {}
        self._mutex = anyio.Lock()

    async def _spawn(self, key: tuple[str, str]) -> _Worker:
        session_id, project_dir = key
        env = self._engine.environment()
        env["CLAUDE_SESSION_ID"] = session_id
        env["CLAUDE_PROJECT_DIR"] = project_dir
        env["CONTEXT_MODE_PROJECT_DIR"] = project_dir
        params = StdioServerParameters(
            command=self._engine.binary, args=[], env=env, cwd=project_dir
        )
        worker = _Worker(key, params)
        self._group.start_soon(worker.run)
        return worker

    async def acquire(self, key: tuple[str, str], *,
                      spawn: Any = None) -> _Worker:
        """Get (or start) the worker for ``key``. ``spawn`` overrides the
        per-key factory — used for the retained bootstrap worker, which still
        counts toward the pool bound and LRU policy like any other worker."""
        while True:
            async with self._mutex:
                worker = self._workers.get(key)
                if worker is None or worker.closed.is_set():
                    self._workers.pop(key, None)
                    if len(self._workers) >= self._limit:
                        victim_key = next(
                            (
                                candidate
                                for candidate, w in self._workers.items()
                                if not w.closed.is_set() and not self._pending.get(candidate)
                            ),
                            None,
                        )
                        if victim_key is None:
                            raise ProxyError(
                                f"context-mode worker limit reached ({self._limit} Hermes "
                                "sessions busy); retry when a session goes idle"
                            )
                        victim = self._workers.pop(victim_key)
                        await victim.close()
                    worker = await (spawn or self._spawn)(key)
                    self._workers[key] = worker
                self._workers.move_to_end(key)
            await worker.started.wait()
            if worker.startup_error is not None:
                if self._workers.get(key) is worker:
                    self._workers.pop(key, None)
                raise ProxyError(
                    "context-mode engine worker failed to start: "
                    f"{worker.startup_error!r}"
                ) from worker.startup_error
            if not worker.closed.is_set():
                return worker

    async def call(self, key: tuple[str, str], name: str, arguments: dict[str, Any]) -> types.CallToolResult:
        self._pending[key] = self._pending.get(key, 0) + 1
        try:
            worker = await self.acquire(key)
            request = _Request(name=name, arguments=arguments)
            await worker.request(request)
            if request.error is not None:
                raise ProxyError(
                    f"context-mode engine call failed: {request.error!r}"
                ) from request.error
            assert request.result is not None
            return request.result
        finally:
            remaining = self._pending.get(key, 1) - 1
            if remaining > 0:
                self._pending[key] = remaining
            else:
                self._pending.pop(key, None)

    async def aclose(self) -> None:
        for worker in list(self._workers.values()):
            await worker.close()


class McpProxy:
    """Standard-MCP server face: schema discovery plus routed tool calls."""

    def __init__(self, engine: Engine, limit: int, group: anyio.abc.TaskGroup) -> None:
        self._engine = engine
        self._pool = _WorkerPool(engine, limit, group)
        self._group = group
        self._schema_mutex = anyio.Lock()
        self._tools: list[types.Tool] | None = None
        self._ready_sentinel: Path | None = None

    def _bootstrap_key(self) -> tuple[str, str]:
        return ("__discovery__", str(self._engine.host_root / "discovery"))

    async def _spawn_bootstrap(self, key: tuple[str, str]) -> _Worker:
        """The bootstrap worker: a genuine upstream engine server under a
        private context below the profile root.

        Schema discovery only needs the handshake; it must not touch any real
        session's knowledge base or stats, so it runs with its own scratch
        CONTEXT_MODE_DIR and project directory. Unlike a throwaway probe, the
        worker is RETAINED in the bounded pool while the proxy serves (it
        counts toward the max-worker bound and closes with the pool): its
        successful initialize() is what proves genuine engine readiness, and
        only then does the proxy publish the profile sentinel."""
        scratch = self._engine.host_root / "discovery"
        (scratch / "sessions").mkdir(parents=True, exist_ok=True, mode=0o700)
        env = self._engine.environment()
        env.update(
            {
                "CONTEXT_MODE_DIR": str(scratch / "sessions"),
                "CLAUDE_SESSION_ID": "hermes-mcp-proxy-discovery",
                "CLAUDE_PROJECT_DIR": str(scratch),
                "CONTEXT_MODE_PROJECT_DIR": str(scratch),
            }
        )
        params = StdioServerParameters(
            command=self._engine.binary, args=[], env=env, cwd=str(scratch)
        )
        worker = _Worker(key, params)
        self._group.start_soon(worker.run)
        return worker

    async def list_tools(self, _ctx: Any, _params: Any = None) -> types.ListToolsResult:
        async with self._schema_mutex:
            if self._tools is None:
                # Real MCP handshake path: a client's initialize() reaches
                # tools/list, the pool starts the genuine bootstrap worker, and
                # ONLY a successfully initialized upstream worker publishes the
                # profile's readiness sentinel. A proxy spawned but never
                # handshaken is honestly NOT ready.
                worker = await self._pool.acquire(
                    self._bootstrap_key(), spawn=self._spawn_bootstrap)
                self._ready_sentinel = _publish_sentinel(self._engine)
                request = _Request(list_tools=True)
                await worker.request(request)
                if request.error is not None:
                    raise RuntimeError(
                        f"context-mode tool discovery failed: {request.error!r}"
                    ) from request.error
                self._tools = [_augment_tool_schema(tool) for tool in request.tools or []]
        return types.ListToolsResult(tools=self._tools)

    async def call_tool(self, _ctx: Any, params: types.CallToolRequestParams) -> types.CallToolResult:
        arguments = dict(params.arguments) if isinstance(params.arguments, dict) else {}
        raw = arguments.pop(HERMES_CONTEXT, None)
        problem = _metadata_problem(raw)
        if problem is not None:
            return _tool_error(
                f"context-mode {params.name} call rejected: {HERMES_CONTEXT} {problem}"
            )
        assert isinstance(raw, dict)
        try:
            context = self._engine.context(raw["session_id"], raw["cwd"])
        except (EngineError, OSError) as exc:
            return _tool_error(
                f"context-mode {params.name} call rejected: cannot resolve Hermes "
                f"session context: {exc}"
            )
        key = (context["session_id"], context["project_dir"])
        arguments = _rewrite_arguments(params.name, arguments, context["cwd"])
        try:
            result = await self._pool.call(key, params.name, arguments)
        except ProxyError as exc:
            return _tool_error(str(exc))
        payload: dict[str, Any] = {"content": result.content}
        if result.structured_content is not None:
            payload["structured_content"] = result.structured_content
        if result.is_error is not None:
            payload["is_error"] = result.is_error
        return types.CallToolResult(**payload)

    async def aclose(self) -> None:
        """Retract the readiness sentinel, then close every live worker — the
        retained bootstrap worker included (serve() drives this at shutdown)."""
        _retract_sentinel(self._ready_sentinel)
        self._ready_sentinel = None
        await self._pool.aclose()


async def _exit_on_signal(group: anyio.abc.TaskGroup) -> None:
    with anyio.open_signal_receiver(signal.SIGTERM, signal.SIGINT) as signals:
        async for _ in signals:
            group.cancel_scope.cancel()
            return


def _engine_version(engine: Engine) -> str:
    try:
        return engine.version
    except (EngineError, OSError, ValueError):
        return "unknown"


async def serve(engine: Engine, max_workers: int) -> None:
    async with anyio.create_task_group() as group:
        group.start_soon(_exit_on_signal, group)
        proxy = McpProxy(engine, max_workers, group)
        server = Server(
            "context-mode",
            version=_engine_version(engine),
            on_list_tools=proxy.list_tools,
            on_call_tool=proxy.call_tool,
        )
        try:
            async with stdio_server() as (read, write):
                await server.run(read, write, server.create_initialization_options())
        finally:
            # Retract readiness on EOF, errors and signals, not only normal exit.
            # Shield cleanup from the cancellation that brought us here.
            with anyio.CancelScope(shield=True):
                with anyio.move_on_after(CLOSE_TIMEOUT):
                    await proxy.aclose()
            group.cancel_scope.cancel()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="context-mode-hermes-mcp-proxy",
        description="Hermes-facing stdio MCP proxy for the context-mode engine",
    )
    parser.add_argument(
        "--engine-bin",
        default=None,
        help="absolute path to the context-mode executable (default: profile install)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=DEFAULT_MAX_WORKERS,
        help=f"maximum concurrent engine worker processes (default {DEFAULT_MAX_WORKERS})",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=os.environ.get("CONTEXT_MODE_LOG_LEVEL", "WARNING"), stream=sys.stderr
    )
    try:
        engine = Engine(binary=args.engine_bin)
    except EngineError as exc:
        print(f"context-mode MCP proxy cannot start: {exc}", file=sys.stderr)
        return 1
    if not engine.available:
        print(
            f"context-mode MCP proxy cannot start: executable unavailable at {engine.binary}",
            file=sys.stderr,
        )
        return 1
    anyio.run(serve, engine, max(1, args.max_workers))
    return 0


if __name__ == "__main__":
    sys.exit(main())
