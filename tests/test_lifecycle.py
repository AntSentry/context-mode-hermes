"""Behavioral tests for the context-mode Hermes plugin (pinned Hermes v0.22.5).

Consumer-visible seams only:
- ``context_mode_hermes.register(ctx)`` on a real ``hermes_cli.plugins.PluginContext``
  backed by ``PluginManager``, with configuration produced by the real installer
  (``context_mode_hermes.install.configure`` writes
  ``plugins.entries.context-mode.settings.engine_binary`` AND ``...storage_dir``;
  there is no ambient env fallback).
- Hermes' own dispatch: ``hermes_cli.plugins._dispatch_pre_tool_call_hooks``
  (block/modify decisions) and ``hermes_cli.lifecycle.invoke_hook`` for
  observer/injection hooks, plus ``agent.compaction_events.fire_post_compaction``
  for compaction boundaries.
- Ground truth for capture continuity is the engine's per-project SessionDB
  under the profile's storage dir, read read-only via sqlite.

The tests are wording-agnostic on purpose: routing guidance prose belongs to
the engine and may be reworded or shortened at any time, so these tests assert
decisions (deny/pass/modify), isolation, and lifecycle boundaries — never
exact message strings.

Requirements (skip otherwise):
- pinned Hermes v0.22.5 importable (``hermes_cli``, ``agent.compaction_events``);
- ``CONTEXT_MODE_TEST_BIN`` pointing at the pinned context-mode 1.0.169 binary
  from a nested npm install (CI installs it with ``context-mode-hermes
  --install-engine``, which uses the packaged package-lock.json).

MCP readiness is REAL everywhere — no stubbed seam. Engine hooks decide
redirect-vs-passthrough by scanning the profile-owned sentinel dir
(``CONTEXT_MODE_MCP_SENTINEL_DIR``, created under the profile's host dir), and
the bridge asks Hermes' own cached MCP status. Tests that need a ready profile
start the profile's real MCP server through Hermes' native discovery, so a
foreign agent's machine-global /tmp sentinel can never mask a fresh host.
"""

from __future__ import annotations

import json
import os
import sqlite3
import uuid
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

TEST_BIN = os.environ.get("CONTEXT_MODE_TEST_BIN", "")


def _sid():
    """Fresh session id (the engine names per-session markers after it in /tmp)."""
    return str(uuid.uuid4())


def _pinned_engine() -> Path:
    if not TEST_BIN:
        pytest.skip("Set CONTEXT_MODE_TEST_BIN to the pinned context-mode 1.0.169 binary to run real-engine tests")
    binary = Path(TEST_BIN)
    if not binary.is_file():
        pytest.skip(f"CONTEXT_MODE_TEST_BIN does not point at a file: {binary}")
    return binary


@pytest.fixture
def cm(tmp_path, monkeypatch):
    """An isolated Hermes profile home with no ambient engine state."""
    home = tmp_path / "hermes-home"
    home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("CONTEXT_MODE_BIN", raising=False)
    monkeypatch.delenv("CONTEXT_MODE_DIR", raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(home / "context-mode-host" / "xdg-config"))
    return SimpleNamespace(home=home, store=home / "context-mode", host=home / "context-mode-host")


def _write_plugin_config(cm, *, engine_binary=None, storage_dir=None):
    settings = {}
    if engine_binary is not None:
        settings["engine_binary"] = engine_binary
    if storage_dir is not None:
        settings["storage_dir"] = storage_dir
    cm.home.joinpath("config.yaml").write_text(
        "plugins:\n"
        "  enabled:\n"
        "    - context-mode\n"
        "  entries:\n"
        "    context-mode:\n"
        "      settings:\n"
        f"        {json.dumps(settings)}\n",
        encoding="utf-8",
    )


@pytest.fixture
def plugin(cm, monkeypatch):
    """A fully configured profile: real installer config, real register(ctx).

    The profile's MCP server is NOT started here — this is the honest cold
    state every fresh host begins in (no owned sentinel, no connection)."""
    try:
        import hermes_cli.plugins as hp
    except ImportError as exc:
        pytest.skip(f"hermes_cli not importable (requires pinned Hermes v0.22.5): {exc}")
    pytest.importorskip("mcp")
    from context_mode_hermes.install import configure

    binary = _pinned_engine()
    # The real installer path: engine_binary AND storage_dir settings, the MCP
    # server entry, and the allowlist — never ambient auto-enable.
    configure(cm.home, binary)

    manager = hp.PluginManager(scope_key=str(cm.home))
    monkeypatch.setattr(hp, "_plugin_manager", manager, raising=False)
    manager.discover_and_load()

    bridge = manager._hooks["pre_tool_call"][0].__self__  # the registered ContextModeBridge
    cm.store = Path(bridge._engine.storage_root)
    yield SimpleNamespace(
        manager=manager, cm=cm,
        bridge=bridge, engine=bridge._engine,
    )


def _sentinel_dir(cm) -> Path:
    return cm.host / "mcp-sentinels"


def _await_profile_sentinel(cm, timeout: float = 60.0) -> None:
    """Wait until the profile's own MCP server has published its readiness
    sentinel in the profile-owned dir (upstream mcp-ready.mjs convention)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if any(_sentinel_dir(cm).glob("context-mode-mcp-ready-*")):
            return
        time.sleep(0.05)
    raise AssertionError(
        f"the profile's MCP server never published a sentinel in {_sentinel_dir(cm)}")


@pytest.fixture
def live_plugin(plugin):
    """Plugin plus the profile's own MCP server, started through Hermes'
    native server startup (``register_mcp_servers`` — the same public seam
    ``hermes chat`` uses): the bridge's oracle (Hermes' cached MCP status)
    and the engine hooks' sentinel scan both see the REAL server — readiness
    is never stubbed."""
    from hermes_cli.config import load_config
    from tools import mcp_tool_lifecycle
    from tools.mcp_tool_discovery import register_mcp_servers

    try:
        register_mcp_servers(load_config()["mcp_servers"])
        _await_profile_sentinel(plugin.cm)
        yield plugin
    finally:
        mcp_tool_lifecycle.shutdown_mcp_servers()


_NATIVE_SERVER_CHILD = textwrap.dedent("""
    import signal, sys, time

    from hermes_cli.config import load_config
    from tools import mcp_tool_lifecycle
    from tools.mcp_tool_discovery import register_mcp_servers

    register_mcp_servers(load_config()["mcp_servers"])  # real handshake: initialize + tools/list
    print("NATIVE-READY", flush=True)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    try:
        time.sleep(600)
    finally:
        mcp_tool_lifecycle.shutdown_mcp_servers()
""")


def _spawn_native_profile_server(cm):
    """Hermes' own native MCP startup, run in a helper process against the
    profile: register_mcp_servers performs the real handshake (initialize +
    tools/list), so the proxy retains its genuine bootstrap worker and
    publishes the profile sentinel — while THIS process's Hermes registry
    stays cold. The honest engine-ready / Hermes-cold cross-oracle case."""
    env = {**os.environ, "HERMES_HOME": str(cm.home)}
    proc = subprocess.Popen(
        [sys.executable, "-c", _NATIVE_SERVER_CHILD], env=env,
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True,
        cwd=str(Path(__file__).resolve().parent.parent),
    )
    first = proc.stdout.readline().strip()
    if first != "NATIVE-READY":
        proc.terminate()
        detail = proc.stderr.read()[-2000:] if proc.stderr else ""
        raise AssertionError(f"native profile server failed to start: {first!r} {detail}")
    return proc


def _dispatch_pre(**kwargs):
    """The exact pre_tool_call seam Hermes' tool loop calls."""
    from hermes_cli.plugins import _dispatch_pre_tool_call_hooks

    return _dispatch_pre_tool_call_hooks(**kwargs)


def _invoke(hook_name, **kwargs):
    from hermes_cli.lifecycle import invoke_hook

    return invoke_hook(hook_name, **kwargs)


def _session_rows(cm):
    """All captured session events across the isolated profile's per-project DBs."""
    rows = []
    for db in cm.store.glob("sessions/*.db"):
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        try:
            for row in con.execute(
                "SELECT session_id, type, category, data, project_dir FROM session_events"
            ):
                rows.append(row)
        finally:
            con.close()
    return rows


# ─── Registration gates ────────────────────────────────────────────────────────

def test_unconfigured_profile_never_routes_or_captures(cm, monkeypatch):
    """Without complete installer settings the plugin stays down: no hooks, no
    commands, no routing, and nothing captured — an unconfigured profile must
    never route or write into unrelated stores."""
    pytest.importorskip("hermes_cli.plugins")
    import hermes_cli.plugins as hp

    cases = [
        {},  # never configured
        # Engine binary gone: bad settings never fall back to ambient env engines.
        {"engine_binary": str(cm.home / "missing-context-mode"), "storage_dir": str(cm.store)},
    ]
    if TEST_BIN:
        # Partial config: engine present but no storage_dir — must not activate
        # and route captures into an engine-chosen default store.
        cases.append({"engine_binary": TEST_BIN})
    for settings in cases:
        _write_plugin_config(cm, **settings)
        manager = hp.PluginManager(scope_key=str(cm.home))
        monkeypatch.setattr(hp, "_plugin_manager", manager, raising=False)
        manager.discover_and_load()
        block, modified = _dispatch_pre(
            tool_name="terminal",
            args={"command": "curl https://example.com/huge"},
            session_id=_sid(), task_id="t", tool_call_id="c", turn_id="u", api_request_id="",
        )
        assert block is None and modified is None, f"routing must stay down for settings {settings}"
        results = _invoke(
            "pre_llm_call", session_id=_sid(), task_id="task", turn_id="t1", user_message="hello",
            conversation_history=[], is_first_turn=True, model="m", platform="cli",
            parent_session_id="", sender_id="user",
        )
        assert not [r for r in results if isinstance(r, dict) and r.get("context")]
        assert _session_rows(cm) == [], f"no session events may be written for settings {settings}"


# ─── PreToolUse routing (decisions, not wording) ───────────────────────────────

def test_terminal_stdout_flood_is_blocked(live_plugin):
    """With the profile's own server live (real sentinel, real status), curl
    without file output is denied."""
    block, modified = _dispatch_pre(
        tool_name="terminal",
        args={"command": "curl https://example.com/huge"},
        session_id=_sid(), task_id="t", tool_call_id="c1", turn_id="u1", api_request_id="",
    )
    assert block, "stdout-flood command must be denied"
    assert modified is None


def test_fresh_profile_passes_unredirected_commands_without_any_server(plugin):
    """Fresh-host cold start: the profile's server has never run, so its owned
    sentinel dir is empty and the engine's mcpRedirect guard passes commands
    through instead of denying them. This is the deterministic fresh-host
    truth — a foreign machine-global /tmp sentinel cannot mask it."""
    assert not list(_sentinel_dir(plugin.cm).glob("context-mode-mcp-ready-*"))
    for tool_name, args in (
        ("terminal", {"command": "curl https://example.com/huge"}),
        ("web_extract", {"urls": ["https://example.com/article"]}),
    ):
        block, _ = _dispatch_pre(
            tool_name=tool_name, args=args,
            session_id=_sid(), task_id="t", tool_call_id="c", turn_id="u", api_request_id="",
        )
        assert block is None, f"{tool_name} must pass through on a fresh host"


def test_silent_file_output_and_bounded_commands_pass(live_plugin):
    """Silent curl -o and structurally bounded commands are not blocked even
    with the profile's server live."""
    for command in ("curl -s -o /tmp/page.html https://example.com/x", "pwd", "git status"):
        block, _ = _dispatch_pre(
            tool_name="terminal", args={"command": command},
            session_id=_sid(), task_id="t", tool_call_id="c", turn_id="u", api_request_id="",
        )
        assert block is None, f"{command!r} must pass"


def test_web_extract_is_denied(live_plugin):
    """web_extract (urls list) maps to WebFetch and is denied."""
    block, _ = _dispatch_pre(
        tool_name="web_extract", args={"urls": ["https://example.com/article"]},
        session_id=_sid(), task_id="t", tool_call_id="c", turn_id="u", api_request_id="",
    )
    assert block, "web extraction must be denied"


def test_admin_mcp_tools_are_operator_only(plugin):
    """Model calls to purge/upgrade/insight are blocked; read-only passes."""
    for leaf in ("ctx_purge", "ctx_upgrade", "ctx_insight"):
        block, _ = _dispatch_pre(
            tool_name=f"mcp__context_mode__{leaf}",
            args={"confirm": True} if leaf == "ctx_purge" else {},
            session_id=_sid(), task_id="t", tool_call_id="c", turn_id="u", api_request_id="",
        )
        assert block, f"{leaf} must be blocked for the model"
    # Ordinary read-only diagnostics still pass.
    block, _ = _dispatch_pre(
        tool_name="mcp__context_mode__ctx_stats", args={},
        session_id=_sid(), task_id="t", tool_call_id="c", turn_id="u", api_request_id="",
    )
    assert block is None


def test_security_policy_deny_survives_mcp_unready_while_redirects_suppress(plugin):
    """The real cross-oracle mismatch: the profile's server is ALIVE (the
    engine hooks see its owned sentinel, so redirect-shaped denies are
    emitted) while THIS Hermes process has not connected it (Hermes' cached
    status is not ready) → redirect blocks are suppressed; the security-policy
    gate still blocks. No readiness is stubbed in either direction."""
    from tools import mcp_tool_lifecycle

    mcp_tool_lifecycle.shutdown_mcp_servers()  # this process stays honestly cold
    host = plugin.cm.host
    host.mkdir(parents=True, exist_ok=True)
    (host / "settings.json").write_text(
        # Claude permission-rule syntax: prefix rules are "<tool>(<prefix>:*)".
        # The denied command is one the engine itself would allow, so any block
        # here can only come from the security policy gate.
        json.dumps({"permissions": {"deny": ["Bash(git status:*)"]}}), encoding="utf-8"
    )
    server = _spawn_native_profile_server(plugin.cm)
    try:
        _await_profile_sentinel(plugin.cm)
        assert not plugin.bridge._mcp_ready(), "Hermes' own status must still be cold"
        block, _ = _dispatch_pre(
            tool_name="terminal", args={"command": "git status"},
            session_id=_sid(), task_id="t", tool_call_id="c", turn_id="u", api_request_id="",
        )
        assert block, "security-policy deny must be enforced with this process's MCP down"
        # Redirect flavor (points the model at the ctx_* MCP tools): suppressed
        # when this Hermes process cannot actually call them.
        block2, _ = _dispatch_pre(
            tool_name="web_extract", args={"urls": ["https://example.com/huge"]},
            session_id=_sid(), task_id="t", tool_call_id="c2", turn_id="u2", api_request_id="",
        )
        assert block2 is None, "redirect-shaped blocks are suppressed when MCP is down"
    finally:
        server.terminate()
        try:
            server.wait(timeout=15)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait(timeout=5)


def test_own_ctx_ask_executes_under_trusted_identity_after_human_approval(plugin, monkeypatch):
    """Engine 'ask' on our MCP tools escalates to the NATIVE human-approval
    gate — not a bridge-side block — and the human-approved call EXECUTES with
    the bridge-owned _hermes_context.

    Hermes aggregates every pre_tool_call callback: plugin args merge only
    from action=modify results, and the accumulated args ride the approve
    directive (hermes_cli.plugins). The bridge's mandatory identity callback
    therefore stamps the trusted identity as a modify, so approval can never
    forward the LLM-forged metadata the proxy would otherwise anchor storage
    on. The human decision is driven through the REAL approval gate: the
    session allowlist (the gate's own '[a]lways' store, seeded via
    tools.approval.approve_session) approves the exact plugin rule."""
    from tools.approval import approve_session
    from tools.approval_context import get_current_session_key
    import tools.approval as approval_state

    def ask(_event, _payload):
        return {"hookSpecificOutput": {"permissionDecision": "ask",
                                       "permissionDecisionReason": "needs confirmation"}}

    monkeypatch.setattr(plugin.engine, "hook", ask)
    sid = _sid()
    forged = {"session_id": "forged", "cwd": "/etc", "project_dir": "/etc"}
    tool_name = "mcp__context_mode__ctx_execute"
    args = {"language": "javascript", "code": "1", "_hermes_context": forged}
    session_key = get_current_session_key()
    pattern_key = f"plugin_rule:{tool_name}"  # _dispatch passes rule_key=tool_name

    # Unapproved (fresh session store): the gate itself fails closed — the
    # operation does NOT execute and nothing dispatches.
    block, modified = _dispatch_pre(
        tool_name=tool_name, args=args,
        session_id=sid, task_id="t", tool_call_id="c0", turn_id="u", api_request_id="",
    )
    assert block, "an unapproved escalation must not execute"
    assert "forged" not in json.dumps(modified or {}), "no forged metadata anywhere"

    # The human (via the gate's session allowlist) approves this exact rule:
    # the call proceeds — with trusted identity, forged values gone.
    try:
        approve_session(session_key, pattern_key)
        block2, modified2 = _dispatch_pre(
            tool_name=tool_name, args=dict(args),
            session_id=sid, task_id="t", tool_call_id="c1", turn_id="u", api_request_id="",
        )
        assert block2 is None, f"approved call must execute, got: {block2}"
        assert modified2 is not None, "the executor dispatches modified args, not raw args"
        assert modified2["_hermes_context"] == plugin.engine.context(sid, os.getcwd())
        assert modified2["_hermes_context"] != forged
    finally:
        approval_state._session_approved.get(session_key, set()).discard(pattern_key)

    # Non-own tools keep the same ask→human-gate mapping at the hook level.
    results = _invoke(
        "pre_tool_call", tool_name="terminal", args={"command": "rm -rf /tmp/x"},
        session_id=sid, task_id="t", tool_call_id="c2", turn_id="u2", api_request_id="",
    )
    decision = next(r for r in results if isinstance(r, dict) and r.get("action") == "approve")
    assert "needs confirmation" in decision["message"]


def test_own_ctx_missing_identity_fails_closed(plugin, monkeypatch):
    """If the trusted identity cannot be produced at all, own tools fail
    closed; other tools are untouched (no identity is involved for them)."""
    from context_mode_hermes.engine import EngineError

    def broken_identity(_session_id, _cwd):
        raise EngineError("identity unavailable")

    monkeypatch.setattr(plugin.engine, "context", broken_identity)
    block, _ = _dispatch_pre(
        tool_name="mcp__context_mode__ctx_stats",
        args={"_hermes_context": {"session_id": "forged", "cwd": "/etc", "project_dir": "/etc"}},
        session_id=_sid(), task_id="t", tool_call_id="c", turn_id="u", api_request_id="",
    )
    assert block, "own-ctx calls must fail closed when identity cannot be produced"


def test_own_ctx_engine_error_fails_closed_while_other_tools_stay_fail_open(plugin, monkeypatch):
    """An engine check that errors must not let our MCP tools run on raw LLM
    arguments (same forged-_hermes_context risk); every other tool keeps the
    documented fail-open bridge behavior."""
    from context_mode_hermes.engine import EngineError

    def broken(_event, _payload):
        raise EngineError("engine down")

    monkeypatch.setattr(plugin.engine, "hook", broken)
    forged = {"session_id": "forged", "cwd": "/etc", "project_dir": "/etc"}
    block, _ = _dispatch_pre(
        tool_name="mcp__context_mode__ctx_search",
        args={"queries": ["x"], "_hermes_context": forged},
        session_id=_sid(), task_id="t", tool_call_id="c", turn_id="u", api_request_id="",
    )
    assert block, "own-ctx calls must fail closed when the engine check errors"
    block2, _ = _dispatch_pre(
        tool_name="terminal", args={"command": "pwd"},
        session_id=_sid(), task_id="t", tool_call_id="c2", turn_id="u2", api_request_id="",
    )
    assert block2 is None, "non-own tools stay fail-open"




def test_trusted_context_overwrites_llm_metadata_on_every_ctx_call(plugin):
    """_hermes_context is bridge-owned on every ctx tool call; LLM-supplied
    values are overwritten, never trusted."""
    engine = plugin.engine
    for tool, args in [
        ("mcp__context_mode__ctx_execute", {"language": "javascript", "code": "1"}),
        ("mcp__context_mode__ctx_search", {"queries": ["x"]}),
        ("mcp__context_mode__ctx_stats", {}),
    ]:
        sid = _sid()
        _, modified = _dispatch_pre(
            tool_name=tool,
            args={**args, "_hermes_context": {"session_id": "forged", "cwd": "/", "project_dir": "/etc"}},
            session_id=sid, task_id="t", tool_call_id="c", turn_id="u", api_request_id="",
        )
        assert modified, f"{tool} must always carry bridge context"
        assert modified["_hermes_context"] == engine.context(sid, os.getcwd())





# ─── Injection & continuity ────────────────────────────────────────────────────

def _first_turn(plugin, sid, message="hello", history=(), first=True, turn="t1"):
    results = _invoke(
        "pre_llm_call", session_id=sid, task_id="task", turn_id=turn, user_message=message,
        conversation_history=list(history), is_first_turn=first, model="m", platform="cli",
        parent_session_id="", sender_id="user",
    )
    contexts = [r.get("context") for r in results if isinstance(r, dict) and r.get("context")]
    assert len(contexts) <= 1
    return contexts[0] if contexts else None


def test_first_turn_injects_once_per_session(plugin):
    """The first LLM call of a session gets the injected context block; later
    turns never get it again (exact prose belongs to the engine)."""
    sid = _sid()
    block = _first_turn(plugin, sid)
    assert block, "first turn must inject a context block"
    assert _first_turn(plugin, sid, message="second", turn="t2", first=False) is None


def test_resumed_session_injects_via_on_session_start(plugin):
    """A resumed session (history present) still gets the block."""
    sid = _sid()
    _invoke("on_session_start", session_id=sid, model="m", platform="cli")
    block = _first_turn(plugin, sid, history=[{"role": "user", "content": "earlier"}], first=False)
    assert block, "resumed sessions must still be injected"


def test_post_compaction_snapshots_and_reinjects_after_summarize_not_micro(plugin):
    from agent.compaction_events import fire_post_compaction

    sid = _sid()
    marker = f"quartermaster-compact-{sid[:8]}"
    _first_turn(plugin, sid, message=f"Always deploy to {marker}; never use the default region.")
    # Micro compaction is bounded housekeeping: no snapshot, no re-inject.
    fire_post_compaction(session_id=sid, old_session_id="", kind="micro_summarize", reason="turn_start_threshold",
                         before_tokens=1000, after_tokens=900, messages_removed=2)
    assert _first_turn(plugin, sid, message="after micro", turn="t2", first=False) is None
    # A committed summarize compaction breaks continuity: the bridge snapshots the
    # already-captured SessionDB (engine precompact), then the next turn re-injects
    # the routing block WITH that resume-snapshot region — not an empty re-send.
    fire_post_compaction(session_id=sid, old_session_id="", kind="summarize", reason="manual",
                         before_tokens=90000, after_tokens=8000, messages_removed=40)
    block = _first_turn(plugin, sid, message="after compact", turn="t3", first=False)
    assert marker in block, "Committed compaction must restore the captured deployment decision"


# ─── Capture continuity (SessionDB is ground truth) ────────────────────────────

def test_user_prompt_captured_once_per_turn(plugin):
    sid = _sid()
    marker = f"capture-once-{sid[:8]}"
    _first_turn(plugin, sid, message=marker, turn="t1")
    _first_turn(plugin, sid, message=marker, turn="t1", first=False)  # same turn, second LLM call
    rows = [r for r in _session_rows(plugin.cm) if r[0] == sid and r[1] == "user_prompt"]
    assert len(rows) == 1, f"expected exactly one user_prompt capture, got {len(rows)}"
    assert marker in rows[0][3]
    # Subagent/fork turns (parent_session_id) are not user prompts.
    _invoke(
        "pre_llm_call", session_id=sid, task_id="task", turn_id="t9",
        user_message="delegated goal", conversation_history=[], is_first_turn=False,
        model="m", platform="cli", parent_session_id="parent-1", sender_id="agent",
    )
    rows = [r for r in _session_rows(plugin.cm) if r[0] == sid and r[1] == "user_prompt"]
    assert len(rows) == 1


def test_terminal_result_captured_under_runtime_session_and_project(plugin):
    sid = _sid()
    marker = f"terminal-capture-{sid[:8]}"
    _invoke(
        "post_tool_call", tool_name="terminal", args={"command": "make"},
        result=json.dumps({"output": marker, "error": "Error: compilation failed"}),
        duration_ms=7, status="error", error_type="BuildError",
        error_message="Error: compilation failed", task_id="t", session_id=sid,
        tool_call_id="c", turn_id="u", api_request_id="", middleware_trace=[],
    )
    rows = [r for r in _session_rows(plugin.cm) if r[0] == sid]
    assert rows, "post_tool_call capture missing from the per-project SessionDB"
    assert all(r[4] == os.getcwd() for r in rows), rows
    assert any("compilation failed" in r[3] for r in rows), rows


def test_relative_file_paths_recorded_against_runtime_cwd(plugin, tmp_path, monkeypatch):
    """After the agent's logical cwd moves into a subdir, a relative write
    records <runtime>/sub/x.py, not the session's initial project anchor.
    The bridge reads agent.runtime_cwd.resolve_agent_cwd(), never the process cwd."""
    sub = tmp_path / "sub"
    sub.mkdir()
    runtime_cwd = pytest.importorskip("agent.runtime_cwd")
    monkeypatch.setattr(runtime_cwd, "resolve_agent_cwd", lambda: str(sub))
    sid = _sid()
    _invoke(
        "post_tool_call", tool_name="write_file", args={"path": "x.py", "content": "print(1)"},
        result="File written successfully", duration_ms=3, status="ok", error_type=None,
        error_message=None, task_id="t", session_id=sid, tool_call_id="c", turn_id="u",
        api_request_id="", middleware_trace=[],
    )
    rows = [r for r in _session_rows(plugin.cm) if r[0] == sid and r[1] == "file_write"]
    assert rows, "file_write capture missing"
    assert str(sub / "x.py") in rows[0][3], rows[0][3]




# ─── Session boundaries ────────────────────────────────────────────────────────

def test_session_reset_and_end_clean_plugin_state_and_markers(plugin, tmp_path):
    sid = _sid()
    marker = Path(os.path.join(tempfile_dir(), f"context-mode-latency-{sid}-terminal.txt"))
    marker.write_text("123", encoding="utf-8")
    # Per-turn on_session_end must NOT clean state (not even registered); the
    # real teardown seam is on_session_finalize.
    _invoke("on_session_end", session_id=sid, completed=True, failed=False, interrupted=False)
    assert marker.exists(), "on_session_end fires every turn; it must not tear down state"
    _invoke("on_session_finalize", session_id=sid)
    assert not marker.exists(), "session finalize must clean upstream /tmp markers"
    # Fresh session after /new gets a fresh injection (old injected state dropped).
    new_sid = _sid()
    _invoke("on_session_reset", session_id=new_sid, reason="new_session", platform="cli",
            old_session_id=sid, new_session_id=new_sid)
    block = _first_turn(plugin, new_sid)
    assert block, "fresh session after /new must be injected again"


def test_on_session_delete_cleans_state(plugin):
    sid = _sid()
    from agent.compaction_events import fire_post_compaction

    _first_turn(plugin, sid)
    fire_post_compaction(session_id=sid, old_session_id="", kind="summarize", reason="manual",
                         before_tokens=900, after_tokens=90, messages_removed=9)
    _invoke("on_session_delete", session_id=sid, reason="delete", profile="default")
    assert _first_turn(plugin, sid, message="later", turn="t5", first=False) is None


def tempfile_dir():
    import tempfile

    return tempfile.gettempdir()
