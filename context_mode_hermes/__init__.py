"""context-mode Hermes plugin — Hermes-native lifecycle bridge.

Fail-open bridge between Hermes lifecycle hooks (pinned v0.22.5) and the
context-mode hook CLI (pinned engine 1.0.169) via :mod:`context_mode_hermes.engine`.
Engine wraps the hook subprocess protocol; this module owns Hermes-side semantics:

┌────────────────────────┬────────────────────────────────────────────────────┐
│ Hermes hook            │ bridge behavior                                    │
├────────────────────────┼────────────────────────────────────────────────────┤
│ pre_tool_call          │ routed tools → engine ``pretooluse``; Claude-Code   │
│                        │ decisions mapped to Hermes directives: deny→block, │
│                        │ ask→approve (human gate), updatedInput→modify,      │
│                        │ additionalContext→deferred guidance.                │
│                        │ ``mcp__context_mode__ctx_*`` calls fail closed on   │
│                        │ ask (or an engine-check error): Hermes' approval    │
│                        │ path forwards RAW arguments (plugin args merge      │
│                        │ only for modify), and the proxy anchors storage on  │
│                        │ ``_hermes_context``. On allow, every ctx_* call     │
│                        │ carries the trusted identity (engine.context)      │
│                        │ injected via modify — LLM-supplied values are       │
│                        │ overwritten. Admin tools (ctx_purge/upgrade/        │
│                        │ insight) are operator-only: blocked with /ctx-*     │
│                        │ guidance.                                           │
│ post_tool_call         │ captured tools → engine ``posttooluse`` (session    │
│                        │ memory; the per-project SessionDB is the store).    │
│ transform_tool_result  │ appends deferred guidance to the tool result.       │
│ pre_llm_call           │ session-start/compact/resume injection via engine   │
│                        │ ``sessionstart`` (once per boundary); the user      │
│                        │ prompt → engine ``userpromptsubmit`` once per turn  │
│                        │ (subagent/fork turns are not user prompts).         │
│ post_llm_call          │ turn end → engine ``stop`` (turn_end capture).      │
│ on_session_start       │ marks the session for injection (startup vs resume  │
│                        │ resolved from the first turn's history).            │
│ post_compaction        │ committed summarize/provider-native compaction →   │
│                        │ engine ``precompact`` (resume snapshot from the     │
│                        │ already-captured SessionDB), then queue the compact │
│                        │ re-injection. Hermes has no precompaction hook.     │
│ on_session_reset       │ drop bridge state for the old/new session (fresh    │
│                        │ session after /new; fresh injection follows).       │
│ on_session_delete      │ drop bridge state + upstream /tmp markers.          │
│ on_session_finalize    │ real session teardown (on_session_end fires every   │
│                        │ turn and must NOT drop capture/injection state).    │
└────────────────────────┴────────────────────────────────────────────────────┘

One :class:`ContextModeBridge` per ``register(ctx)`` call holds the engine and all
per-session state, so a PluginManager that loads this entry point once across
profiles never leaks one profile's callbacks or captures into another. Redirect-
shaped decisions additionally require the plugin's own Hermes MCP server to be
available (Hermes-scoped status, never the machine-global /tmp sentinels, which
also see foreign agents' servers — the engine's own hook oracle reads the
profile-owned sentinel dir exported by :meth:`Engine.environment` instead).
Security-policy denies never depend on MCP availability. Hooks do not execute
the proposed tool call — they only advise; Hermes' own approval/trust controls
stay in charge.
"""

from __future__ import annotations

import functools
import glob
import json
import logging
import os
import re
import tempfile
import time
from typing import Any, Callable, Optional

from context_mode_hermes.commands import register_commands
from context_mode_hermes.engine import Engine, EngineError

__version__ = "1.4.0.dev0"

logger = logging.getLogger(__name__)

# Hermes MCP wire names for the plugin's ``context-mode`` server vs the names the
# pinned engine's claude-code protocol emits (hooks/core/tool-naming.mjs).
CC_PLUGIN_TOOL_PREFIX = "mcp__plugin_context-mode_context-mode__"
HERMES_TOOL_PREFIX = "mcp__context_mode__"

# Hermes tool name → Claude-Code name (upstream routing/extraction branch on CC names).
HERMES_TO_CC = {
    "terminal": "Bash",
    "read_file": "Read",
    "write_file": "Write",
    "patch": "Edit",
    "search_files": "Grep",
    "file_operations_search": "Glob",
    "todo_list": "TodoWrite",
    "web_extract": "WebFetch",
    "skill_manage": "Skill",
    "skill_view": "Skill",
    "skills_list": "Skill",
}
# Tools upstream's PreToolUse matcher set routes (Bash|WebFetch|Read|Grep|mcp__).
ROUTED_TOOLS = frozenset({"terminal", "read_file", "search_files", "web_extract"})
# Tools upstream's PostToolUse matcher set captures into session memory.
CAPTURED_TOOLS = frozenset(HERMES_TO_CC) | {"web_extract"}
# File tools whose relative ``path`` must resolve against the runtime cwd.
FILE_PATH_TOOLS = frozenset({"read_file", "write_file", "patch"})

# Model-facing MCP admin tools → operator slash command (commands.py owns the
# confirmed operator surface).
ADMIN_CTX_COMMANDS = {
    "ctx_purge": "ctx-purge",
    "ctx_upgrade": "ctx-upgrade",
    "ctx_insight": "ctx-insight",
}
SECURITY_DENY_PREFIX = "Blocked by security policy"
MCP_SERVER_NAME = "context-mode"
_MCP_READY_STATUSES = frozenset({"connected", "lazy"})

# Bounded per-registration state (long-lived processes; see _drop_session / caps).
_MAX_PENDING_SESSIONS = 64
_MAX_INJECTED_SESSIONS = 256
_MAX_PROMPT_TURNS = 512
_MAX_GUIDANCE = 64
_GUIDANCE_TTL_SECONDS = 120.0



# ─── Tool-name / payload translation ───────────────────────────────────────────

def _cc_tool_name(tool_name: str, routed: bool) -> Optional[str]:
    """Translate a Hermes tool name to the engine's Claude-Code name.

    Our own MCP server's tools are re-prefixed so upstream matches them; other
    servers already use the shared ``mcp__<server>__<tool>`` convention.
    """
    if tool_name.startswith(HERMES_TOOL_PREFIX):
        return CC_PLUGIN_TOOL_PREFIX + tool_name[len(HERMES_TOOL_PREFIX):]
    if tool_name.startswith("mcp__"):
        return tool_name
    if tool_name in (ROUTED_TOOLS if routed else CAPTURED_TOOLS):
        return HERMES_TO_CC[tool_name]
    return None


def _cc_tool_input(tool_name: str, args: Any, cwd: str) -> dict[str, Any]:
    """Shape Hermes args for the engine's readers.

    File tools carry ``path``; a relative path resolves against the ACTUAL
    runtime cwd so capture records <runtime>/subdir/file.py, not the session's
    initial project anchor (Engine.hook anchors payload cwd per session).
    Upstream readers accept path|file_path; its extractor reads file_path.
    WebFetch gets a single url. Everything else passes through untouched.
    """
    if not isinstance(args, dict):
        return {}
    if tool_name in FILE_PATH_TOOLS:
        path = args.get("path")
        if isinstance(path, str) and path.strip():
            resolved = path if os.path.isabs(path) else os.path.abspath(os.path.join(cwd, path))
            return {**args, "path": resolved, "file_path": resolved}
        return args
    if tool_name != "web_extract":
        return args
    urls = args.get("urls")
    if isinstance(urls, str):
        url = urls
    elif isinstance(urls, list) and urls:
        url = str(urls[0])
    else:
        url = ""
    return {**args, "url": url}


def _remap_names(text: str) -> str:
    """Rewrite engine tool names to this plugin's Hermes MCP wire names."""
    return text.replace(CC_PLUGIN_TOOL_PREFIX, HERMES_TOOL_PREFIX)


def _remap_input(updated: dict[str, Any]) -> dict[str, Any]:
    return {key: _remap_names(value) if isinstance(value, str) else value
            for key, value in updated.items()}


# ─── Routing-block surgery for this engine's real capabilities ─────────────────

_ROUTING_BLOCK_RE = re.compile(
    r"<context_window_protection>.*?</context_window_protection>", re.S
)
_HERMES_ROUTING = """<context_window_protection>
Use context-mode for large-output analysis and indexed retrieval; return only relevant findings.
Discover ctx_execute, ctx_index and ctx_search with Hermes tool_search when needed.
Native Hermes tools remain appropriate for small results and durable edits.
ctx_execute runs real host code, not an OS security sandbox; filesystem writes persist.
Administrative actions require the user's /ctx-purge, /ctx-upgrade or /ctx-insight command.
</context_window_protection>"""


def _hermes_block(text: str) -> str:
    """Replace generic client boilerplate, preserving actual restored session knowledge."""
    return _ROUTING_BLOCK_RE.sub(_HERMES_ROUTING, _remap_names(text))


def _hook_safe(default: Any = None) -> Callable:
    """Swallow every failure: Hermes fails ``pre_tool_call`` CLOSED (veto) when a
    callback raises, so bridge errors must degrade to a no-op instead."""
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return fn(*args, **kwargs)
            except Exception:
                logger.warning("[context-mode] %s failed", fn.__name__,
                               exc_info=logger.isEnabledFor(logging.DEBUG))
                return default
        return wrapper
    return decorator


# ─── Per-registration bridge ───────────────────────────────────────────────────

class ContextModeBridge:
    """One bridge per ``register(ctx)``: engine + per-session state stay on the
    instance so multiple profile managers in one process never share captures,
    callbacks, or injection state."""

    def __init__(self, ctx: Any, engine: Engine) -> None:
        self._engine = engine
        self._register_cwd = self._capture_session_cwd(ctx)
        self._pending_injection: dict[str, str] = {}
        self._injected_sessions: dict[str, None] = {}
        self._prompt_turns_seen: dict[tuple[str, str], None] = {}
        self._guidance_pending: dict[tuple[str, str], tuple[float, str]] = {}

    # -- context ---------------------------------------------------------------

    @staticmethod
    def _capture_session_cwd(ctx: Any) -> str:
        getter = getattr(ctx, "session_cwd", None)
        if callable(getter):
            try:
                return str(getter() or "")
            except Exception:
                return ""
        return ""

    def _runtime_cwd(self) -> str:
        """The agent's logical cwd (session override / terminal scope / launch
        dir), never the plugin process's incidental cwd."""
        try:
            from agent.runtime_cwd import resolve_agent_cwd

            return str(resolve_agent_cwd() or "")
        except Exception:
            pass
        return self._register_cwd or os.getcwd()

    def _mcp_ready(self) -> bool:
        """Whether THIS Hermes process can call the plugin's ctx_* MCP tools.

        Reads Hermes' own cached, read-only MCP status (never connects) — this
        profile's real readiness oracle, never a stub. Engine hooks use their
        own oracle: the profile-owned sentinel dir selected by
        ``CONTEXT_MODE_MCP_SENTINEL_DIR`` (machine-global ``/tmp`` sentinels
        also see foreign agents' servers and are never consulted by the
        bridge).
        """
        try:
            from tools.mcp_tool_discovery import get_mcp_status

            entries = get_mcp_status() or []
        except Exception:
            return False  # fail-closed for redirects only; security denies still pass
        return any(
            isinstance(entry, dict)
            and entry.get("name") == MCP_SERVER_NAME
            and entry.get("status") in _MCP_READY_STATUSES
            for entry in entries
        )

    # -- bounded state ----------------------------------------------------------

    @staticmethod
    def _cap(mapping: dict, limit: int) -> None:
        while len(mapping) > limit:
            mapping.pop(next(iter(mapping)))

    def _remember_guidance(self, session_id: str, tool_call_id: str, text: str) -> None:
        now = time.monotonic()
        stale = [key for key, (seen_at, _) in self._guidance_pending.items()
                 if now - seen_at > _GUIDANCE_TTL_SECONDS]
        for key in stale:
            del self._guidance_pending[key]
        self._guidance_pending[(session_id, tool_call_id)] = (now, text)
        self._cap(self._guidance_pending, _MAX_GUIDANCE)

    def _take_guidance(self, session_id: str, tool_call_id: str) -> str:
        entry = self._guidance_pending.pop((session_id, tool_call_id), None)
        return entry[1] if entry else ""

    def _cleanup_markers(self, session_id: str) -> None:
        """Remove upstream's cross-hook /tmp markers for one session (best-effort)."""
        safe = re.sub(r"[^a-zA-Z0-9_-]", "_", session_id)
        for pattern in (f"context-mode-*{glob.escape(session_id)}*", f"context-mode-*{glob.escape(safe)}*"):
            for path in glob.glob(os.path.join(tempfile.gettempdir(), pattern)):
                try:
                    os.unlink(path)
                except OSError:
                    pass

    def _drop_session(self, session_id: str) -> None:
        self._pending_injection.pop(session_id, None)
        self._injected_sessions.pop(session_id, None)
        for key in [key for key in self._prompt_turns_seen if key[0] == session_id]:
            del self._prompt_turns_seen[key]
        for key in [key for key in self._guidance_pending if key[0] == session_id]:
            del self._guidance_pending[key]
        self._cleanup_markers(session_id)

    # -- hook callbacks ---------------------------------------------------------

    @_hook_safe()
    def pre_tool_call(self, *, tool_name: str = "", args: Optional[dict] = None,
                      session_id: str = "", tool_call_id: str = "", task_id: str = "",
                      **_kwargs: Any) -> Optional[dict]:
        engine = self._engine
        if not engine.available or not session_id:
            return None
        runtime_cwd = self._runtime_cwd()

        own_ctx_tool = tool_name.startswith(HERMES_TOOL_PREFIX)
        cc_name = _cc_tool_name(tool_name, routed=True)
        if cc_name is None:
            return None

        # Operator-only admin tools: never let the model purge/upgrade/launch Insight.
        if own_ctx_tool:
            leaf = tool_name[len(HERMES_TOOL_PREFIX):]
            slash = ADMIN_CTX_COMMANDS.get(leaf)
            if slash:
                return {"action": "block",
                        "message": (f"context-mode: {leaf} is an operator command and is not "
                                    f"authorized for model-side calls. Ask the user to run /{slash}.")}

        try:
            output = engine.hook("pretooluse", {
                "tool_name": cc_name,
                "tool_input": _cc_tool_input(tool_name, args, runtime_cwd),
                "session_id": session_id,
                "cwd": runtime_cwd,
            })
        except EngineError:
            # Our MCP tools fail closed on a broken engine check: proceeding
            # would forward the RAW LLM arguments, and a forged
            # ``_hermes_context`` re-anchors the proxy's per-session storage.
            # Every other tool keeps the documented fail-open behavior.
            if own_ctx_tool:
                return {"action": "block",
                        "message": ("context-mode: the engine check is unavailable; "
                                    "MCP context tools are blocked for this call.")}
            return None
        spec = output.get("hookSpecificOutput") or {}
        decision = spec.get("permissionDecision")
        message = _remap_names(str(spec.get("permissionDecisionReason") or ""))

        if decision == "deny":
            if not message:
                return None
            # Redirect-shaped denies point the model at our MCP tools; they are only
            # honest when this Hermes process can actually call them. Security-policy
            # denies never depend on MCP availability.
            if not self._mcp_ready() and not message.startswith(SECURITY_DENY_PREFIX):
                return None
            return {"action": "block", "message": message}
        if decision == "ask":
            if own_ctx_tool:
                # Hermes' approval path forwards the RAW LLM arguments when a
                # call is approved — plugin args are only merged for
                # ``action: modify`` dispatches (hermes_cli.plugins), and the
                # proxy anchors per-session storage on ``_hermes_context``.
                # Our MCP tools therefore fail closed on ask: they only ever
                # proceed through the identity-modify below.
                return {"action": "block",
                        "message": message or (
                            "context-mode: this call needs an approval that cannot carry the "
                            "bridge-owned session identity; ask the operator to run the "
                            "equivalent /ctx-* command.")}
            result: dict[str, Any] = {"action": "approve"}
            if message:
                result["message"] = message
            return result

        # Every call into our MCP server carries bridge-owned identity; the proxy
        # strips it before upstream. LLM-supplied values are overwritten.
        if own_ctx_tool:
            merged: dict[str, Any] = dict(args or {})
            updated = spec.get("updatedInput")
            if isinstance(updated, dict) and updated:
                merged.update(_remap_input(updated))
            merged["_hermes_context"] = engine.context(session_id, runtime_cwd)
            return {"action": "modify", "args": merged}

        updated = spec.get("updatedInput")
        if isinstance(updated, dict) and updated:
            if not self._mcp_ready():
                return None  # redirect-shaped modify (needs our MCP tools)
            return {"action": "modify", "args": _remap_input(updated)}

        context = spec.get("additionalContext")
        if isinstance(context, str) and context.strip():
            self._remember_guidance(session_id, tool_call_id, _remap_names(context))
        return None

    @_hook_safe()
    def post_tool_call(self, *, tool_name: str = "", args: Optional[dict] = None,
                       result: Any = None, session_id: str = "", **_kwargs: Any) -> None:
        engine = self._engine
        if not engine.available or not session_id:
            return
        cc_name = _cc_tool_name(tool_name, routed=False)
        if cc_name is None:
            return
        tool_input = _cc_tool_input(tool_name, args, self._runtime_cwd())
        if tool_name.startswith(HERMES_TOOL_PREFIX):
            tool_input.pop("_hermes_context", None)  # bridge-internal, not capture data
        engine.hook("posttooluse", {
            "tool_name": cc_name,
            "tool_input": tool_input,
            "tool_response": result if isinstance(result, str)
            else (json.dumps(result) if result is not None else ""),
            "session_id": session_id,
            "cwd": self._runtime_cwd(),
        })

    @_hook_safe()
    def transform_tool_result(self, *, result: Any = None, session_id: str = "",
                              tool_call_id: str = "", **_kwargs: Any) -> Optional[str]:
        """Append deferred PreToolUse guidance to the tool result — the channel
        Hermes offers at the tool boundary for Claude-Code ``additionalContext``."""
        guidance = self._take_guidance(session_id, tool_call_id)
        if not guidance:
            return None
        base = result if isinstance(result, str) else json.dumps(result)
        return f"{base}\n\n{guidance}"

    @_hook_safe()
    def pre_llm_call(self, *, session_id: str = "", turn_id: str = "", user_message: Any = None,
                     is_first_turn: bool = False, parent_session_id: str = "",
                     **_kwargs: Any) -> Optional[dict[str, str]]:
        engine = self._engine
        if not engine.available or not session_id:
            return None
        runtime_cwd = self._runtime_cwd()

        context: Optional[str] = None
        source = self._pending_injection.pop(session_id, None)
        if source is None and is_first_turn and session_id not in self._injected_sessions:
            source = "startup"
        if source:
            if source == "auto":  # on_session_start: fresh vs resumed is the history
                source = "startup" if is_first_turn else "resume"
            self._injected_sessions[session_id] = None
            self._cap(self._injected_sessions, _MAX_INJECTED_SESSIONS)
            try:
                output = engine.hook("sessionstart", {
                    "source": source, "session_id": session_id, "cwd": runtime_cwd,
                })
                spec = output.get("hookSpecificOutput") or {}
                block = spec.get("additionalContext")
                if isinstance(block, str) and block.strip():
                    context = _hermes_block(block)
            except EngineError:
                logger.warning("[context-mode] sessionstart (%s) failed; skipping injection", source,
                               exc_info=True)

        # One user-prompt capture per assistant turn; delegated goals are not user
        # prompts (matching Claude Code, where UserPromptSubmit never fires for
        # subagents).
        if not parent_session_id and turn_id and user_message is not None:
            key = (session_id, turn_id)
            if key not in self._prompt_turns_seen:
                self._prompt_turns_seen[key] = None
                self._cap(self._prompt_turns_seen, _MAX_PROMPT_TURNS)
                try:
                    engine.hook("userpromptsubmit", {
                        "prompt": user_message if isinstance(user_message, str) else "",
                        "session_id": session_id,
                        "cwd": runtime_cwd,
                    })
                except EngineError:
                    # A failed capture must not cost the already-built injection.
                    logger.warning("[context-mode] userpromptsubmit capture failed", exc_info=True)

        return {"context": context} if context else None

    @_hook_safe()
    def post_llm_call(self, *, session_id: str = "", assistant_response: Any = None,
                      **_kwargs: Any) -> None:
        engine = self._engine
        if not engine.available or not session_id:
            return
        engine.hook("stop", {
            "session_id": session_id,
            "stop_hook_active": False,
            "last_assistant_message": (assistant_response if isinstance(assistant_response, str) else "")[:2000],
            "cwd": self._runtime_cwd(),
        })

    @_hook_safe()
    def on_session_start(self, *, session_id: str = "", **_kwargs: Any) -> None:
        if session_id:
            self._pending_injection[session_id] = "auto"
            self._cap(self._pending_injection, _MAX_PENDING_SESSIONS)

    @_hook_safe()
    def post_compaction(self, *, session_id: str = "", kind: Any = None,
                        **_kwargs: Any) -> None:
        """Hermes has no precompaction hook; on a committed history rewrite the
        bridge snapshots the already-captured SessionDB (engine ``precompact``,
        which builds the ≤2KB resume snapshot and bumps compact_count exactly
        once per real compaction) and queues the compact re-injection.

        Only ``summarize`` (manual /compact + automatic) and ``provider_native``
        compactions break continuity — micro-summarize and tool-result prunes
        are bounded housekeeping and never bump the compact count.
        """
        engine = self._engine
        if not session_id or kind not in ("summarize", "provider_native"):
            return
        try:
            engine.hook("precompact", {"session_id": session_id, "cwd": self._runtime_cwd()})
        except EngineError:
            logger.warning("[context-mode] precompact snapshot failed", exc_info=True)
        self._pending_injection[session_id] = "compact"
        self._cap(self._pending_injection, _MAX_PENDING_SESSIONS)

    @_hook_safe()
    def on_session_reset(self, *, session_id: str = "", old_session_id: str = "",
                         new_session_id: str = "", **_kwargs: Any) -> None:
        for sid in {sid for sid in (session_id, old_session_id, new_session_id) if sid}:
            self._drop_session(sid)

    @_hook_safe()
    def on_session_delete(self, *, session_id: str = "", **_kwargs: Any) -> None:
        # No upstream delete event exists; bridge state and /tmp markers go, the
        # knowledge base is wiped via the /ctx-purge operator command.
        if session_id:
            self._drop_session(session_id)

    @_hook_safe()
    def on_session_finalize(self, *, session_id: str = "", **_kwargs: Any) -> None:
        """Real session teardown. ``on_session_end`` fires after EVERY turn and
        is deliberately not registered: dropping capture/injection state there
        would re-inject the routing block and duplicate captures each turn."""
        if session_id:
            self._drop_session(session_id)

    # -- registration -----------------------------------------------------------

    def register_hooks(self, ctx: Any) -> None:
        for name, callback in (
            ("pre_tool_call", self.pre_tool_call),
            ("post_tool_call", self.post_tool_call),
            ("transform_tool_result", self.transform_tool_result),
            ("pre_llm_call", self.pre_llm_call),
            ("post_llm_call", self.post_llm_call),
            ("on_session_start", self.on_session_start),
            ("post_compaction", self.post_compaction),
            ("on_session_reset", self.on_session_reset),
            ("on_session_delete", self.on_session_delete),
            ("on_session_finalize", self.on_session_finalize),
        ):
            ctx.register_hook(name, callback)
        logger.info("[context-mode] lifecycle bridge active (engine %s, storage %s)",
                    self._engine.version, self._engine.storage_root)


# ─── Entry point ───────────────────────────────────────────────────────────────

def _plugin_setting(ctx: Any, key: str) -> Optional[str]:
    """String plugin setting (installer-written), never a mock/garbled value."""
    getter = getattr(ctx, "get_config", None)
    if not callable(getter):
        return None
    try:
        value = getter(key, None)
    except Exception:
        return None
    if isinstance(value, (str, os.PathLike)) and str(value).strip():
        return str(value)
    return None


def register(ctx: Any) -> None:
    """Entry point called by the Hermes plugin system."""
    binary = _plugin_setting(ctx, "engine_binary")
    storage = _plugin_setting(ctx, "storage_dir")
    try:
        engine = Engine(binary=binary, storage_root=storage) if binary and storage else None
    except EngineError as exc:
        # Bad settings disable the plugin visibly — never fall back to ambient
        # env paths, which could capture into another profile's store.
        logger.warning("[context-mode] invalid engine settings (engine_binary=%r, storage_dir=%r): %s "
                       "— plugin disabled", binary, storage, exc)
        engine = None

    register_commands(ctx, engine)

    if engine is None or not engine.available:
        logger.warning("[context-mode] engine unavailable (binary=%r) — lifecycle hooks disabled",
                       getattr(engine, "binary", None))
        return

    ContextModeBridge(ctx, engine).register_hooks(ctx)
