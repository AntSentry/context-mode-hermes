"""Hermes slash commands (/ctx-*) backed by the real context-mode MCP tools.

Dispatch contract (pinned Hermes v0.22.5, ``hermes_cli.plugins.PluginContext``):
  * ``ctx.register_command(name, handler, description, args_hint, argument_mode)`` registers
    an in-session slash command; the handler is ``fn(raw_args: str, **session_context)`` and
    may declare any of the session-context keywords (``source``, ``session_id``,
    ``session_key``, ``platform`` — ``hermes_cli.plugin_command_dispatch``).
  * ``ctx.call_mcp(server, tool, arguments, timeout)`` calls a tool on an MCP server through
    Hermes' native client. It is default-deny: the operator grants per-server access via
    ``plugins.entries.context-mode.mcp_allowlist: [context-mode]`` in config.yaml (the
    installer configures this). The result is the ``_mcp_envelope`` shape
    ``{"ok": bool, "result" | "error", "truncated"?}``.

Engine operations use Hermes' native MCP client. Upgrades install the packaged
dependency lock into an isolated staging prefix and validate execution plus
SQLite/FTS retrieval before a reversible swap. The upstream global/Claude
upgrade script is never executed.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
import logging
import os
import re
import shlex
import shutil
import time
import tempfile
from pathlib import Path
from typing import Any, Callable, Optional

from .engine import Engine, EngineError
from .install import ENGINE_VERSION as TESTED_ENGINE_VERSION
from .install import install_runtime

logger = logging.getLogger(__name__)

#: MCP server name configured by the installer (mcp: + mcp_allowlist entry).
MCP_SERVER = "context-mode"

_STATS_TIMEOUT = 60
_DOCTOR_TIMEOUT = 120
_INDEX_TIMEOUT = 300

class _Usage(Exception):
    """Usage/validation error; rendered as guidance text, never raised to Hermes."""


# ─── Hermes MCP envelope rendering ─────────────────────────────────────────────

def _mcp_call(ctx: Any, engine: Any, tool: str, arguments: dict, timeout: float = _STATS_TIMEOUT,
              session: Optional[dict] = None) -> str:
    """Call a real context-mode MCP tool via Hermes' public ``ctx.call_mcp``.

    ``session`` (the handler's Hermes dispatch context) injects the trusted
    ``_hermes_context`` identity — session id from Hermes, live session cwd, project anchor
    from ``engine.context()`` — mirroring the pre-tool hook's injection; never user input.
    Returns displayable text; raises :class:`_Usage` with actionable guidance when the host
    denies access or the call fails (Hermes propagates handler exceptions, so callers wrap).
    """
    if not getattr(engine, "available", False):
        raise _Usage(
            "context-mode engine is unavailable (binary not found). "
            "Install context-mode or point CONTEXT_MODE_BIN at the binary."
        )
    call_mcp = getattr(ctx, "call_mcp", None)
    if call_mcp is None:
        raise _Usage(
            "This Hermes build does not expose ctx.call_mcp; this plugin is verified "
            "against Hermes v0.22.5 only."
        )
    payload = dict(arguments)
    if session is not None:
        payload["_hermes_context"] = _hermes_context(ctx, engine, session)
    try:
        response = call_mcp(MCP_SERVER, tool, payload, timeout)
    except PermissionError as exc:
        raise _Usage(
            f"Hermes denied MCP access: {exc}\n"
            f"Grant it in config.yaml by adding {MCP_SERVER!r} to "
            f"plugins.entries.context-mode.mcp_allowlist (default is no MCP access)."
        ) from None
    except Exception as exc:  # host timeouts, transport failures, tool errors
        raise _Usage(f"MCP call {tool} failed: {exc}") from None
    rendered = _render_envelope(response, tool)
    if (tool == "ctx_index" and session is not None
            and isinstance(response, dict) and response.get("ok") is True):
        # Operator commands bypass Hermes' model-tool hooks. Capture this real
        # index result through the same upstream hook as model-issued indexing:
        # project-scoped retrieval joins indexed chunks to captured session ids.
        # Without this, an existing/emptied session DB hides operator-only indexes.
        from . import CC_PLUGIN_TOOL_PREFIX

        context = payload["_hermes_context"]
        try:
            engine.hook("posttooluse", {
                "tool_name": CC_PLUGIN_TOOL_PREFIX + tool,
                "tool_input": arguments,
                "tool_response": rendered,
                "session_id": context["session_id"],
                "cwd": context["cwd"],
            })
        except EngineError as exc:
            raise _Usage(
                f"Content was indexed, but its session attribution failed: {exc}. "
                "Project-scoped retrieval may not find it."
            ) from None
    return rendered


def _render_envelope(response: Any, tool: str) -> str:
    """Render the pinned ``_mcp_envelope`` contract ({"ok", "result"|"error", "truncated"?})."""
    if not isinstance(response, dict):
        return str(response)
    if not response.get("ok"):
        error = response.get("error", "unknown error")
        if isinstance(error, dict):  # JSON-RPC error objects: surface the real message
            error = error.get("message") or json.dumps(error, ensure_ascii=False)
        return f"context-mode {tool} error:\n{error}"
    result = response.get("result")
    if isinstance(result, str) and result.strip():
        text = result
    elif response.get("structuredContent") is not None:
        text = json.dumps(response["structuredContent"], indent=2, ensure_ascii=False)
    elif result is not None:
        text = str(result)
    else:
        text = "(no output)"
    if response.get("truncated"):
        text += "\n… [output truncated by the Hermes MCP cap]"
    return text.rstrip()


# ─── Argument parsing helpers ──────────────────────────────────────────────────

def _tokens(raw_args: str) -> list[str]:
    try:
        return shlex.split(raw_args or "")
    except ValueError as exc:
        raise _Usage(f"Could not parse arguments ({exc}). Quote paths and text containing spaces.") from None


def _value(tokens: list[str], index: int, flag: str) -> str:
    if index + 1 >= len(tokens):
        raise _Usage(f"{flag} needs a value.")
    return tokens[index + 1]


def _int_value(raw: str, flag: str) -> int:
    try:
        return int(raw)
    except ValueError:
        raise _Usage(f"{flag} needs a whole number, got {raw!r}.") from None


def _session_id(context: dict) -> str:
    return str(context.get("session_id") or context.get("session_key") or "").strip()


def _safe(handler: Callable[[str], str]) -> Callable[[str], str]:
    """Slash-command wrapper: no exception ever reaches the Hermes dispatcher."""

    def wrapped(raw_args: str, **session_context: Any) -> str:
        try:
            return handler(raw_args, **session_context)
        except _Usage as exc:
            return f"Error: {exc}"
        except Exception as exc:  # never crash the host session
            logger.warning("[context-mode] command failed: %s", exc, exc_info=True)
            return f"Error: context-mode command failed: {exc}"

    wrapped.__name__ = handler.__name__
    return wrapped

def _agent_cwd(ctx: Any, session: dict) -> str:
    """The session's live working directory: ctx.session_cwd() -> agent.runtime_cwd -> cwd."""
    getter = getattr(ctx, "session_cwd", None)
    if getter is not None:
        try:
            cwd = getter()
        except Exception:  # degrade along the chain, never fail the command on this
            cwd = None
        if cwd:
            return str(cwd)
    agent = (session or {}).get("agent")
    runtime_cwd = getattr(agent, "runtime_cwd", None)
    if runtime_cwd:
        return str(runtime_cwd)
    return os.getcwd()


def _hermes_context(ctx: Any, engine: Any, session: dict) -> dict:
    """Trusted per-session identity for MCP calls — never derived from user input.

    Mirrors the pre-tool hook's ``action: modify`` injection: the session id comes from
    Hermes' dispatch context, the cwd from the session's live working directory, and the
    persistent project anchor from ``engine.context()`` (first workspace wins).
    """
    session_id = _session_id(session)
    if not session_id:
        raise _Usage(
            "Hermes did not provide a session id; context-mode commands require one."
        )
    try:
        return engine.context(session_id, _agent_cwd(ctx, session))
    except EngineError as exc:
        raise _Usage(f"cannot anchor the session workspace: {exc}") from None
    except Exception as exc:
        raise _Usage(f"cannot build the Hermes session context: {exc}") from None


# ─── /ctx-stats and /ctx-doctor ────────────────────────────────────────────────

_STATS_HINT = (
    "Engine/profile-server context statistics (process-wide counters; "
    "not scoped to the current Hermes conversation)"
)


def _stats_command(ctx: Any, engine: Any) -> Callable[[str], str]:
    @_safe
    def handler(raw_args: str, **session: Any) -> str:
        return (
            "context-mode engine stats (profile-server counters; "
            "not current-conversation-scoped):\n"
            + _mcp_call(ctx, engine, "ctx_stats", {}, timeout=_STATS_TIMEOUT, session=session)
        )
    return handler


_DOCTOR_HINT = (
    "Diagnose context-mode: upstream engine report plus real Hermes integration validation"
)


def _hermes_integration_lines(engine: Any) -> list[str]:
    """Real Hermes-side validation: engine metadata plus the actual Hermes MCP config.

    Reads Hermes' own config through ``hermes_cli.config.read_raw_config`` when running
    inside Hermes; otherwise reports that this validation is unavailable here — never a
    fabricated pass.
    """
    lines = []
    installed = _installed_version(engine)
    if installed == "unknown":
        lines.append(f"[FAIL] Engine runtime metadata unreadable at {engine.binary!r}")
    else:
        match = "[OK]" if installed == TESTED_ENGINE_VERSION else "[FAIL]"
        lines.append(
            f"{match} Engine runtime v{installed} at {engine.binary} "
            f"(plugin-tested: {TESTED_ENGINE_VERSION})"
        )
    try:
        from hermes_cli.config import read_raw_config
        config = read_raw_config() or {}
    except Exception as exc:
        lines.append(f"[n/a] Hermes MCP config not readable here: {exc}")
        return lines
    server = (config.get("mcp_servers") or {}).get(MCP_SERVER)
    command = server.get("command") if isinstance(server, dict) else None
    lines.append(
        f"[{'OK' if server else 'FAIL'}] Hermes mcp_servers[{MCP_SERVER!r}] configured"
        + (f" ({command})" if command else "")
    )
    entry = ((config.get("plugins") or {}).get("entries") or {}).get("context-mode") or {}
    allowlist = entry.get("mcp_allowlist") or []
    lines.append(
        f"[{'OK' if MCP_SERVER in allowlist else 'FAIL'}] "
        f"plugins.entries.context-mode.mcp_allowlist grants {MCP_SERVER!r}"
    )
    return lines


def _doctor_command(ctx: Any, engine: Any) -> Callable[[str], str]:
    @_safe
    def handler(raw_args: str, **session: Any) -> str:
        report = _mcp_call(ctx, engine, "ctx_doctor", {}, timeout=_DOCTOR_TIMEOUT, session=session)
        settings_note = ""
        if "settings.json" in report and "[FAIL]" in report:
            settings_note = (
                "\n[note] The settings.json hook-wiring check above validates Claude Code "
                "settings (engine protocol); Hermes registers the same hooks through its "
                "plugin API (ctx.register_hook) and is not represented by that file.\n"
            )
        return (
            "context-mode /ctx-doctor — upstream engine report, verbatim, then real "
            "Hermes integration validation.\n"
            "\n[Upstream engine report]\n"
            + report
            + settings_note
            + "\n[Hermes integration]\n"
            + "\n".join(_hermes_integration_lines(engine))
        )
    return handler


# ─── /ctx-index ────────────────────────────────────────────────────────────────

_INDEX_USAGE = """Index content into the searchable knowledge base.

Usage:
  /ctx-index <path> [--source LABEL] [directory options]
  /ctx-index --text "inline content" --source LABEL

Options:
  -s, --source LABEL        source label used to scope later ctx-search calls
  -t, --text TEXT           index inline text instead of a file
  --include GLOB            (directory) only index matching files (repeatable)
  --exclude GLOB            (directory) skip matching files (repeatable)
  --extensions LIST         (directory) comma-separated extensions, e.g. md,txt,py
  --max-depth N             (directory) recursion depth cap
  --max-files N             (directory) file count cap
  --no-gitignore            (directory) index gitignored files too
  --follow-symlinks         (directory) follow symbolic links

Provide either a path or --text, not both."""


def _index_command(ctx: Any, engine: Any) -> Callable[[str], str]:
    @_safe
    def handler(raw_args: str, **session: Any) -> str:
        tokens = _tokens(raw_args)
        path: Optional[str] = None
        text: Optional[str] = None
        source: Optional[str] = None
        include: list[str] = []
        exclude: list[str] = []
        extensions: Optional[list[str]] = None
        max_depth: Optional[int] = None
        max_files: Optional[int] = None
        respect_gitignore: Optional[bool] = None
        follow_symlinks = False

        i = 0
        while i < len(tokens):
            token = tokens[i]
            if token in ("-h", "--help"):
                return _INDEX_USAGE
            if token in ("-s", "--source"):
                source = _value(tokens, i, token); i += 2; continue
            if token in ("-t", "--text"):
                text = _value(tokens, i, token); i += 2; continue
            if token == "--include":
                include.append(_value(tokens, i, token)); i += 2; continue
            if token == "--exclude":
                exclude.append(_value(tokens, i, token)); i += 2; continue
            if token == "--extensions":
                raw = _value(tokens, i, token)
                extensions = [ext.strip() for ext in raw.split(",") if ext.strip()]
                if not extensions:
                    raise _Usage("--extensions needs a comma-separated list, e.g. md,txt,py")
                i += 2; continue
            if token == "--max-depth":
                max_depth = _int_value(_value(tokens, i, token), token); i += 2; continue
            if token == "--max-files":
                max_files = _int_value(_value(tokens, i, token), token); i += 2; continue
            if token == "--no-gitignore":
                respect_gitignore = False; i += 1; continue
            if token == "--follow-symlinks":
                follow_symlinks = True; i += 1; continue
            if token.startswith("-") and token != "-":
                raise _Usage(f"Unknown option {token!r}.\n{_INDEX_USAGE}")
            if path is not None:
                raise _Usage(f"Unexpected extra argument {token!r} (only one path is indexed).\n{_INDEX_USAGE}")
            path = token
            i += 1

        if text is not None and path is not None:
            raise _Usage("Provide either a path or --text, not both.\n" + _INDEX_USAGE)
        if text is None and path is None:
            raise _Usage(_INDEX_USAGE)

        arguments: dict[str, Any] = {}
        if text is not None:
            arguments["content"] = text
        else:
            resolved = Path(path).expanduser()
            if not resolved.is_absolute():
                resolved = Path(_agent_cwd(ctx, session)) / resolved
            if not resolved.exists():
                raise _Usage(f"Path not found: {resolved}")
            is_dir = resolved.is_dir()
            if not is_dir and (include or exclude or extensions or max_depth is not None
                               or max_files is not None or respect_gitignore is not None
                               or follow_symlinks):
                raise _Usage(
                    f"{resolved} is a file; --include/--exclude/--extensions/--max-depth/"
                    "--max-files/--no-gitignore/--follow-symlinks apply to directories only."
                )
            arguments["path"] = str(resolved)
        if source:
            arguments["source"] = source
        if include:
            arguments["include"] = include
        if exclude:
            arguments["exclude"] = exclude
        if extensions:
            arguments["extensions"] = extensions
        if max_depth is not None:
            arguments["maxDepth"] = max_depth
        if max_files is not None:
            arguments["maxFiles"] = max_files
        if respect_gitignore is not None:
            arguments["respectGitignore"] = respect_gitignore
        if follow_symlinks:
            arguments["followSymlinks"] = True

        return _mcp_call(ctx, engine, "ctx_index", arguments, timeout=_INDEX_TIMEOUT, session=session)

    return handler


# ─── /ctx-search ───────────────────────────────────────────────────────────────

_SEARCH_USAGE = """Search the unified knowledge base (indexed content + session memory).

Usage:
  /ctx-search <query> [options]
  /ctx-search -q "first question" -q "second question" [options]

Options:
  -q, --query TEXT           add a query (repeatable — batch related questions in one call)
  -s, --source LABEL         restrict to one indexed source (partial match)
  -l, --limit N              results per query (default 3)
  -c, --content-type TYPE    filter by content shape: code | prose
      --sort MODE            relevance (default, current session) | timeline (chronological:
                             current + prior sessions and persistent auto-memory in this
                             profile's shared knowledge base)

At least one query is required."""


def _search_command(ctx: Any, engine: Any) -> Callable[[str], str]:
    @_safe
    def handler(raw_args: str, **session: Any) -> str:
        tokens = _tokens(raw_args)
        queries: list[str] = []
        source: Optional[str] = None
        limit: Optional[int] = None
        content_type: Optional[str] = None
        sort_mode: Optional[str] = None
        positional: list[str] = []

        i = 0
        while i < len(tokens):
            token = tokens[i]
            if token in ("-h", "--help"):
                return _SEARCH_USAGE
            if token in ("-q", "--query"):
                queries.append(_value(tokens, i, token)); i += 2; continue
            if token in ("-s", "--source"):
                source = _value(tokens, i, token); i += 2; continue
            if token in ("-l", "--limit"):
                limit = _int_value(_value(tokens, i, token), token)
                if limit < 1:
                    raise _Usage(f"--limit must be >= 1, got {limit}.")
                i += 2; continue
            if token in ("-c", "--content-type"):
                content_type = _value(tokens, i, token)
                if content_type not in ("code", "prose"):
                    raise _Usage(f"--content-type must be 'code' or 'prose', got {content_type!r}.")
                i += 2; continue
            if token == "--sort":
                sort_mode = _value(tokens, i, token)
                if sort_mode not in ("relevance", "timeline"):
                    raise _Usage(f"--sort must be 'relevance' or 'timeline', got {sort_mode!r}.")
                i += 2; continue
            if token.startswith("-") and token != "-":
                raise _Usage(f"Unknown option {token!r}.\n{_SEARCH_USAGE}")
            positional.append(token)
            i += 1

        if positional and not queries:
            queries.append(" ".join(positional))
        elif positional:
            raise _Usage("Pass queries either as positional text or via -q, not both.\n" + _SEARCH_USAGE)
        if not queries:
            raise _Usage(_SEARCH_USAGE)

        arguments: dict[str, Any] = {"queries": queries}
        if source:
            arguments["source"] = source
        if limit is not None:
            arguments["limit"] = limit
        if content_type:
            arguments["contentType"] = content_type
        if sort_mode:
            arguments["sort"] = sort_mode

        return _mcp_call(ctx, engine, "ctx_search", arguments, timeout=_STATS_TIMEOUT, session=session)

    return handler


# ─── /ctx-upgrade — profile-owned staged update, never the upstream global script ──

_UPGRADE_HINT = (
    "Update the profile-owned context-mode runtime to the plugin-tested version "
    "(preview first; add `confirm` to install)"
)



_RESTART_NOTE = "Restart required: restart this profile's Hermes gateway (or reload MCP and start a new session) to load the update."


def _version_tuple(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in re.findall(r"\d+", version)[:3])


def _probe_runtime(binary: str, env: dict) -> tuple[str, Optional[str]]:
    """Validate execution and SQLite/FTS round-trip before replacing a runtime."""
    async def probe() -> str:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        with tempfile.TemporaryDirectory(prefix="hermes-engine-probe-") as directory:
            isolated = {**env,
                "CONTEXT_MODE_DIR": str(Path(directory) / "data"),
                "CLAUDE_CONFIG_DIR": str(Path(directory) / "host"),
                "XDG_CONFIG_HOME": str(Path(directory) / "xdg"),
                "CLAUDE_PROJECT_DIR": directory,
                "CONTEXT_MODE_PROJECT_DIR": directory,
                "CLAUDE_SESSION_ID": "upgrade-probe",
            }
            parameters = StdioServerParameters(command=binary, env=isolated, cwd=directory)
            async with asyncio.timeout(60):
                async with stdio_client(parameters) as (reader, writer):
                    async with ClientSession(reader, writer) as client:
                        await client.initialize()
                        async def call(name: str, args: dict) -> str:
                            result = await client.call_tool(name, arguments=args)
                            text = "\n".join(getattr(item, "text", "") for item in result.content)
                            if result.is_error:
                                raise RuntimeError(text or f"{name} failed")
                            return text
                        executed = await call("ctx_execute", {
                            "language": "python", "code": "print(sum([19, 23]))",
                        })
                        if not re.search(r"(?m)^42\s*$", executed):
                            raise RuntimeError("Staged runtime did not execute Python correctly")
                        marker = f"engineprobe{time.time_ns()}"
                        await call("ctx_index", {
                            "content": f"# {marker}\nThe round-trip value is forty-two.",
                            "source": "engine-upgrade-probe",
                        })
                        found = await call("ctx_search", {"queries": [marker], "source": "engine-upgrade-probe"})
                        if "round-trip value is forty-two" not in found:
                            raise RuntimeError("Staged runtime did not preserve indexed content")
        return "v" + Engine(binary=binary).version

    try:
        # Slash handlers can run on the gateway's event loop. Own the async
        # client's entire lifetime in one worker, rather than nesting run().
        with ThreadPoolExecutor(max_workers=1) as executor:
            return executor.submit(lambda: asyncio.run(probe())).result(), None
    except Exception as exc:
        return "unknown", str(exc)

def _installed_version(engine: Any) -> str:
    """Installed runtime version from the Engine's own package.json metadata."""
    version = str(getattr(engine, "version", "") or "").strip().lower().lstrip("v")
    return version or "unknown"


def _runtime_prefix(engine: Any) -> Path:
    """Return only the installer-owned runtime, never a shared/global npm prefix."""
    managed_path = Path(engine.home) / "context-mode-runtime"
    if managed_path.is_symlink():
        raise _Usage("Refusing to update a symlinked runtime prefix")
    managed = managed_path.resolve()
    candidate = Path(engine.binary).resolve()
    for path in (candidate, *candidate.parents):
        if path.name == "node_modules":
            if path.parent.resolve() == managed:
                return managed
            break
    raise _Usage(
        "Refusing to update a shared or external engine. Only this profile's "
        f"managed runtime can be replaced: {managed}"
    )



def _swap_runtime(prefix: Path, staging: Path) -> tuple[bool, str]:
    """Swap the staged runtime into place, keeping a rollback backup; (ok, message)."""
    backup = prefix.parent / f"{prefix.name}-backup-{time.time_ns()}"
    os.rename(prefix, backup)
    try:
        os.rename(staging, prefix)
    except OSError as exc:
        os.rename(backup, prefix)  # roll back — the session keeps its current engine
        return False, f"staged swap failed ({exc}); rolled back, current runtime untouched."
    return True, f"previous runtime kept at {backup} for rollback"


def _upgrade_command(ctx: Any, engine: Any) -> Callable[[str], str]:
    @_safe
    def handler(raw_args: str, **session: Any) -> str:
        tokens = _tokens(raw_args)
        if tokens not in ([], ["confirm"]):
            raise _Usage("Usage: /ctx-upgrade [confirm] (target is the plugin-tested engine version).")
        confirmed = tokens == ["confirm"]
        target = TESTED_ENGINE_VERSION

        installed = _installed_version(engine)
        if installed == "unknown":
            return (
                f"Error: cannot determine the installed engine version from "
                f"{engine.binary!r}; refusing to update the runtime."
            )
        v_installed, v_target = f"v{installed}", f"v{target}"


        if installed == target:
            return (
                f"context-mode is already current.\n"
                f"Installed: {v_installed} — the plugin-tested version ({v_target}).\n"
                f"Nothing to install; no command was executed.\n"
            )
        if _version_tuple(installed) > _version_tuple(target):
            return (
                f"Runtime {v_installed} is NEWER than the plugin-tested version {v_target}.\n"
                f"This plugin has not been validated against {v_installed}; refusing to touch it.\n"
            )

        prefix = _runtime_prefix(engine)
        staging_prefix = f"{prefix.name}-staged-"
        if not confirmed:
            return (
                f"Runtime update plan (profile-owned, staged):\n"
                f"  1. npm-install context-mode {v_target} with --install-strategy=nested into a fresh directory beside {prefix}\n"
                f"  2. validate execution plus SQLite/FTS indexing and retrieval in isolation\n"
                f"  3. swap it into {prefix}, keeping the current runtime as backup\n"
                f"  4. restart this profile's Hermes gateway (or reload MCP and start a new session)\n"
                f"Installed: {v_installed}  ->  Tested target: {v_target}\n"
                f"Re-run as `/ctx-upgrade confirm` to execute the plan."
            )

        staging = Path(tempfile.mkdtemp(prefix=staging_prefix, dir=prefix.parent))
        try:
            installed = install_runtime(staging, env=engine.environment())
            if installed.returncode:
                return (
                    f"Error: staged install failed (exit {installed.returncode}); current runtime untouched.\n"
                    f"{installed.stdout}{installed.stderr}"
                )
            staged_binary = staging / "node_modules" / ".bin" / "context-mode"
            staged_version, staged_error = _probe_runtime(str(staged_binary), engine.environment())
            if staged_error or staged_version != v_target:
                return (
                    f"Error: staged engine probe failed "
                    f"({staged_error or f'version {staged_version}, expected {target}'}); "
                    f"current runtime untouched."
                )
            swapped, swap_message = _swap_runtime(prefix, staging)
            if not swapped:
                return f"Error: {swap_message}"
            return (
                f"context-mode runtime updated: {v_installed} -> {staged_version}\n"
                f"{swap_message}.\n"
                f"{_RESTART_NOTE}"
            )
        finally:
            # The path was uniquely allocated by this invocation; never remove a
            # predictable existing directory beside a user's runtime.
            if staging.exists():
                shutil.rmtree(staging)

    return handler


# ─── /ctx-purge ────────────────────────────────────────────────────────────────

_PURGE_USAGE = """Purge context-mode data. Irreversible.

Usage:
  /ctx-purge session confirm            purge this session's captured events
  /ctx-purge session <id> confirm       purge an explicit session id (plain string;
                                        Hermes ids are not UUIDs)
  /ctx-purge project confirm            purge project-wide stats and indexed content

Scopes: session or project. The literal word `confirm` is required — without it
nothing is deleted."""



def _purge_command(ctx: Any, engine: Any) -> Callable[[str], str]:
    @_safe
    def handler(raw_args: str, **session: Any) -> str:
        tokens = _tokens(raw_args)
        if tokens and tokens[0] in ("-h", "--help"):
            return _PURGE_USAGE
        scope = tokens[0] if tokens else ""
        rest = tokens[1:]
        confirmed = "confirm" in rest
        operands = [token for token in rest if token != "confirm"]
        if any(token.startswith("-") for token in operands) or len(operands) > (1 if scope == "session" else 0):
            raise _Usage(f"Ambiguous purge arguments.\n{_PURGE_USAGE}")
        if scope not in ("session", "project"):
            raise _Usage(f"Unknown scope {scope!r}.\n{_PURGE_USAGE}")
        if not confirmed:
            return (
                f"Purge scope `{scope}` staged, nothing deleted.\n"
                f"This is irreversible. Re-run as `/ctx-purge {scope} confirm` to delete."
            )

        arguments: dict[str, Any] = {"confirm": True, "scope": scope}
        if scope == "session":
            session_id = operands[0] if operands else _session_id(session)
            if not session_id.strip():
                raise _Usage(
                    "Hermes did not provide a session id for this command; pass one "
                    "explicitly: `/ctx-purge session <id> confirm`, or purge everything "
                    "with `/ctx-purge project confirm`."
                )
            arguments["sessionId"] = session_id

        return _mcp_call(ctx, engine, "ctx_purge", arguments, timeout=_STATS_TIMEOUT, session=session)

    return handler


# ─── /ctx-insight (hosted dashboard, explicit opt-in) ──────────────────────────

_INSIGHT_HINT = "Open the hosted Insight analytics dashboard (explicit opt-in)"
_INSIGHT_EXPLANATION = (
    "context-mode Insight is a hosted analytics dashboard: https://context-mode.com/insight\n"
    "Launching it opens your browser and works with data sent to the hosted Insight layer,\n"
    "so it is strictly opt-in.\n\n"
    "To launch: /ctx-insight open"
)


def _insight_command(ctx: Any, engine: Any) -> Callable[[str], str]:
    @_safe
    def handler(raw_args: str, **session: Any) -> str:
        tokens = _tokens(raw_args)
        if tokens not in ([], ["open"]):
            raise _Usage("Usage: /ctx-insight [open]; opening the hosted dashboard requires explicit opt-in.")
        if tokens != ["open"]:
            return _INSIGHT_EXPLANATION
        return _mcp_call(ctx, engine, "ctx_insight", {}, timeout=_STATS_TIMEOUT, session=session)

    return handler


# ─── Registration entry point ──────────────────────────────────────────────────

def register_commands(ctx: Any, engine: Any) -> None:
    """Register the /ctx-* slash commands on the Hermes plugin context.

    Called by the plugin's ``register(ctx)`` with the shared Engine. Commands only work when
    the engine is available; registration is skipped (with a warning) otherwise so Hermes
    never carries dead handlers.
    """
    if engine is None or not getattr(engine, "available", False):
        logger.warning("[context-mode] engine unavailable — /ctx-* commands not registered")
        return

    commands: list[tuple[str, Callable[[str], str], str, str]] = [
        ("ctx-stats", _stats_command(ctx, engine), _STATS_HINT, ""),
        ("ctx-doctor", _doctor_command(ctx, engine), _DOCTOR_HINT, ""),
        ("ctx-index", _index_command(ctx, engine),
         "Index a file, directory, or inline text into the searchable knowledge base",
         "<path> [--source LABEL]"),
        ("ctx-search", _search_command(ctx, engine),
         "Search indexed knowledge and session memory (BM25 + timeline)",
         "<query> [--source LABEL]"),
        ("ctx-upgrade", _upgrade_command(ctx, engine), _UPGRADE_HINT, "[confirm]"),
        ("ctx-purge", _purge_command(ctx, engine),
         "Purge context-mode data for a scope (session|project) — irreversible; requires `confirm`",
         "<session [id]|project> [confirm]"),
        ("ctx-insight", _insight_command(ctx, engine), _INSIGHT_HINT, "[open]"),
    ]
    for name, handler, description, args_hint in commands:
        ctx.register_command(name, handler, description=description, args_hint=args_hint)
        logger.debug("[context-mode] registered /%s", name)
