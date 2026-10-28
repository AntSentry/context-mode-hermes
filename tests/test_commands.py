"""Consumer behavior through real Hermes command dispatch and real MCP workers.

Set CONTEXT_MODE_TEST_BIN to the pinned nested engine installation. No cloud or
LLM is used; every indexed/retrieved/purged value crosses the actual native MCP
client and this adapter's stdio transport.
"""
import os
from pathlib import Path
import shlex
import time
from types import SimpleNamespace
import uuid

import pytest


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    pytest.importorskip("hermes_cli.plugins")
    pytest.importorskip("mcp")
    binary = os.environ.get("CONTEXT_MODE_TEST_BIN")
    if not binary:
        pytest.skip("Set CONTEXT_MODE_TEST_BIN for real Hermes/context-mode integration")
    from agent.runtime_cwd import reset_session_cwd, set_session_cwd
    from context_mode_hermes.engine import Engine
    from context_mode_hermes.install import configure
    from hermes_cli.config import load_config
    from hermes_cli.plugin_command_dispatch import invoke_plugin_command
    from hermes_cli.plugins import get_plugin_commands
    from tools.mcp_tool_discovery import register_mcp_servers
    from tools.mcp_tool_lifecycle import shutdown_mcp_servers

    home, workspace = tmp_path / "profile", tmp_path / "workspace"
    workspace.mkdir()
    # Install the browser guard BEFORE spawning the proxy; it captures its env.
    launcher_dir = tmp_path / "browser-bin"
    launcher_dir.mkdir()
    marker = tmp_path / "browser-request"
    launcher = launcher_dir / "xdg-open"
    launcher.write_text("#!/bin/sh\nprintf '%s' \"$1\" > " + shlex.quote(str(marker)) + "\n", encoding="utf-8")
    launcher.chmod(0o700)
    monkeypatch.setenv("PATH", str(launcher_dir) + os.pathsep + os.environ["PATH"])
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("CONTEXT_MODE_DIR", raising=False)
    configure(home, Path(binary))
    cwd_token = set_session_cwd(str(workspace))
    sid = str(uuid.uuid4())
    try:
        register_mcp_servers(load_config()["mcp_servers"])
        commands = get_plugin_commands()
        def run(name, args="", session_id=sid):
            return invoke_plugin_command(commands[name]["handler"], args,
                source="cli", session_id=session_id, session_key="cli:" + session_id, platform="cli")
        yield SimpleNamespace(run=run, home=home, workspace=workspace, session_id=sid,
            engine=Engine(binary=binary, storage_root=str(home / "context-mode")), browser_marker=marker)
    finally:
        try:
            shutdown_mcp_servers()
        finally:
            reset_session_cwd(cwd_token)


def test_relative_file_index_search_uses_the_session_workspace(runtime):
    source = runtime.workspace / "notes with spaces.md"
    source.write_text("# Pressure regulator\nThe service interval requires fourteen seals.\n", encoding="utf-8")
    runtime.run("ctx-index", f"{shlex.quote(source.name)} --source regulator")
    found = runtime.run("ctx-search", "regulator --source regulator")
    assert "fourteen seals" in found


def test_source_filter_excludes_another_indexed_source(runtime):
    runtime.run("ctx-index", '--text "# Turbine\nThe turbine uses copper bearings." --source turbine')
    runtime.run("ctx-index", '--text "# Valve\nThe valve uses platinum seats." --source valve')
    found = runtime.run("ctx-search", "uses --source turbine")
    assert "copper bearings" in found
    assert "platinum seats" not in found


@pytest.mark.parametrize("purge_args", ["project", "project unwanted confirm", "session first second confirm", "project --yes"])
def test_purge_requires_unambiguous_explicit_confirmation(runtime, purge_args):
    runtime.run("ctx-index", '--text "# Retention marker\nThe archive contains six irreplaceable gaskets." --source retain')
    runtime.run("ctx-purge", purge_args)
    found = runtime.run("ctx-search", "archive --source retain")
    assert "six irreplaceable gaskets" in found


def test_confirmed_project_purge_really_removes_content(runtime):
    runtime.run("ctx-index", '--text "# Purge marker\nThe obsolete packet contains twenty-seven disks." --source erase')
    assert "twenty-seven disks" in runtime.run("ctx-search", "packet --source erase")
    runtime.run("ctx-purge", "project confirm")
    assert "twenty-seven disks" not in runtime.run("ctx-search", "packet --source erase")


def test_session_purge_preserves_sibling_session_decisions(runtime):
    other = str(uuid.uuid4())
    for sid, phrase in [(runtime.session_id, "Always use the amber deployment region."),
                        (other, "Always retain the violet encryption key.")]:
        payload = {"session_id": sid, "cwd": str(runtime.workspace)}
        runtime.engine.hook("userpromptsubmit", {**payload, "prompt": phrase})
        runtime.engine.hook("precompact", payload)
        runtime.engine.hook("sessionstart", {**payload, "source": "compact"})
    assert "amber deployment region" in runtime.run("ctx-search", "amber --sort timeline")
    assert "violet encryption key" in runtime.run("ctx-search", "violet --sort timeline", session_id=other)
    runtime.run("ctx-purge", "session confirm")
    assert "amber deployment region" not in runtime.run("ctx-search", "amber --sort timeline")
    assert "violet encryption key" in runtime.run("ctx-search", "violet --sort timeline", session_id=other)

def test_operator_can_index_and_retrieve_again_after_session_purge(runtime):
    runtime.engine.hook("userpromptsubmit", {
        "session_id": runtime.session_id,
        "cwd": str(runtime.workspace),
        "prompt": "Prior maintenance decision for this session.",
    })
    runtime.run("ctx-index", '--text "# Prior order\nReplace nineteen bronze couplings." --source prior')
    assert "nineteen bronze couplings" in runtime.run("ctx-search", "couplings --source prior")
    runtime.run("ctx-purge", "session confirm")
    assert "nineteen bronze couplings" not in runtime.run("ctx-search", "couplings --source prior")
    runtime.run("ctx-index", '--text "# New order\nRetain twenty-three ceramic sleeves." --source renewal')
    assert "twenty-three ceramic sleeves" in runtime.run("ctx-search", "sleeves --source renewal")



def test_malformed_index_request_does_not_replace_existing_source(runtime):
    runtime.run("ctx-index", '--text "# Existing\nThe retained invoice totals forty-three credits." --source invoice')
    runtime.run("ctx-index", 'missing.md --text "replacement" --source invoice')
    assert "forty-three credits" in runtime.run("ctx-search", "invoice --source invoice")


def test_current_upgrade_does_not_mutate_the_installed_runtime(runtime):
    metadata = Path(runtime.engine.binary).resolve().parent / "package.json"
    before = metadata.read_bytes()
    output = runtime.run("ctx-upgrade", "confirm")
    assert runtime.engine.version in output
    assert metadata.read_bytes() == before
    assert not list(runtime.home.glob("context-mode-runtime-staged-*"))


def test_insight_launch_requires_explicit_operator_opt_in(runtime):
    # Only the external browser is intercepted; dispatch and MCP remain real.
    marker = runtime.browser_marker
    output = runtime.run("ctx-insight")
    assert "https://context-mode.com/insight" in output
    assert not marker.exists()
    runtime.run("ctx-insight", "open")
    deadline = time.monotonic() + 3
    while not marker.exists() and time.monotonic() < deadline:
        time.sleep(0.02)
    assert marker.exists(), "Explicit /ctx-insight open did not request a browser launch"
    assert marker.read_text(encoding="utf-8").startswith("https://context-mode.com/insight")
