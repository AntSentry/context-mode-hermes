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

The only stubbed seam is ``bridge._mcp_ready`` (host state): whether THIS Hermes
process can call its ``context-mode`` MCP server depends on machine-global
state in the real host, not on this profile; tests pin it explicitly instead of
letting ambient host state decide block-vs-suppress outcomes.
"""

from __future__ import annotations

import json
import os
import sqlite3
import uuid
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
    """A fully configured profile: real installer config, real register(ctx)."""
    try:
        import hermes_cli.plugins as hp
    except ImportError as exc:
        pytest.skip(f"hermes_cli not importable (requires pinned Hermes v0.22.5): {exc}")
    from context_mode_hermes.install import configure

    binary = _pinned_engine()
    # The real installer path: engine_binary AND storage_dir settings, the MCP
    # server entry, and the allowlist — never ambient auto-enable.
    configure(cm.home, binary)

    manager = hp.PluginManager(scope_key=str(cm.home))
    monkeypatch.setattr(hp, "_plugin_manager", manager, raising=False)
    manager.discover_and_load()

    bridge = manager._hooks["pre_tool_call"][0].__self__  # the registered ContextModeBridge
    monkeypatch.setattr(bridge, "_mcp_ready", lambda: True)
    cm.store = Path(bridge._engine.storage_root)
    yield SimpleNamespace(
        manager=manager, cm=cm,
        bridge=bridge, engine=bridge._engine,
    )


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

def test_terminal_stdout_flood_is_blocked(plugin):
    """curl without file output is denied."""
    block, modified = _dispatch_pre(
        tool_name="terminal",
        args={"command": "curl https://example.com/huge"},
        session_id=_sid(), task_id="t", tool_call_id="c1", turn_id="u1", api_request_id="",
    )
    assert block, "stdout-flood command must be denied"
    assert modified is None


def test_silent_file_output_and_bounded_commands_pass(plugin):
    """Silent curl -o and structurally bounded commands are not blocked."""
    for command in ("curl -s -o /tmp/page.html https://example.com/x", "pwd", "git status"):
        block, _ = _dispatch_pre(
            tool_name="terminal", args={"command": command},
            session_id=_sid(), task_id="t", tool_call_id="c", turn_id="u", api_request_id="",
        )
        assert block is None, f"{command!r} must pass"


def test_web_extract_is_denied(plugin):
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


def test_security_policy_deny_survives_mcp_unready_while_redirects_suppress(plugin, monkeypatch):
    """Unready Hermes MCP suppresses redirects, never the security policy gate."""
    host = plugin.cm.host
    host.mkdir(parents=True, exist_ok=True)
    (host / "settings.json").write_text(
        # Claude permission-rule syntax: prefix rules are "<tool>(<prefix>:*)".
        # The denied command is one the engine itself would allow, so any block
        # here can only come from the security policy gate.
        json.dumps({"permissions": {"deny": ["Bash(git status:*)"]}}), encoding="utf-8"
    )
    monkeypatch.setattr(plugin.bridge, "_mcp_ready", lambda: False)
    block, _ = _dispatch_pre(
        tool_name="terminal", args={"command": "git status"},
        session_id=_sid(), task_id="t", tool_call_id="c", turn_id="u", api_request_id="",
    )
    assert block, "security-policy deny must be enforced with MCP down"
    # Redirect flavor (points the model at the ctx_* MCP tools): suppressed
    # when this Hermes process cannot call them.
    block2, _ = _dispatch_pre(
        tool_name="web_extract", args={"urls": ["https://example.com/huge"]},
        session_id=_sid(), task_id="t", tool_call_id="c2", turn_id="u2", api_request_id="",
    )
    assert block2 is None, "redirect-shaped blocks are suppressed when MCP is down"




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
