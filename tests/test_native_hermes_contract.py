"""Real Hermes dispatch + real context-mode MCP: no mock tool results or model calls."""
import json
import os
from pathlib import Path
import uuid

import pytest


def test_native_compaction_restores_decision_and_mcp_can_retrieve_it(tmp_path, monkeypatch):
    pytest.importorskip("hermes_cli.plugins")
    pytest.importorskip("mcp")
    binary = os.environ.get("CONTEXT_MODE_TEST_BIN")
    if not binary:
        pytest.skip("Set CONTEXT_MODE_TEST_BIN to exercise pinned Hermes + context-mode")
    from context_mode_hermes.install import configure
    from context_mode_hermes.engine import Engine
    from agent.runtime_cwd import reset_session_cwd, set_session_cwd
    from hermes_cli.config import load_config
    from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest, invoke_hook
    from tools.mcp_tool_discovery import register_mcp_servers
    from tools.mcp_tool_lifecycle import shutdown_mcp_servers

    home = tmp_path / "profile"
    workspace = tmp_path / "different-runtime-workspace"
    workspace.mkdir()
    configure(home, Path(binary))
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("CONTEXT_MODE_DIR", raising=False)
    sid = str(uuid.uuid4())
    region = "NATIVE-REGION-" + uuid.uuid4().hex[:10]
    prompt = f"Always deploy in {region} and reject region default."
    ctx = PluginContext(PluginManifest(name="context-mode", source="entrypoint"), PluginManager())
    cwd_token = set_session_cwd(str(workspace))
    try:
        register_mcp_servers(load_config()["mcp_servers"])
        payload = {"session_id": sid, "cwd": str(workspace), "task_id": "continuity-turn", "turn_id": "1"}
        invoke_hook("on_session_start", **payload, model="validation", platform="cli")
        invoke_hook("pre_llm_call", **payload, user_message=prompt, is_first_turn=True,
                    model="validation", platform="cli")
        invoke_hook("post_llm_call", **payload, assistant_response="I will use the selected region.")
        invoke_hook("on_session_end", **payload, completed=True, failed=False, interrupted=False)
        ordinary = invoke_hook("pre_llm_call", **{**payload, "turn_id": "2"},
                               user_message="Keep working.", is_first_turn=False,
                               model="validation", platform="cli")
        assert not ordinary, "Ordinary turns must not repeatedly inject session-start guidance"
        invoke_hook("post_compaction", **payload, kind="summarize", before_tokens=48000, after_tokens=5000)
        restored = invoke_hook("pre_llm_call", **payload, user_message="Continue the deployment.",
                               is_first_turn=False, model="validation", platform="cli")
        assert region in json.dumps(restored), "Committed compaction lost the deployment decision"
        metadata = Engine(binary=binary).context(sid, str(workspace))
        found = ctx.call_mcp("context-mode", "ctx_search", {
            "queries": [region], "sort": "timeline", "limit": 5,
            "_hermes_context": metadata,
        }, timeout=60)
        assert found["ok"], found
        # The query itself is echoed even on an empty result: assert independent stored content.
        assert "reject region default" in json.dumps(found), found
    finally:
        try:
            shutdown_mcp_servers()
        finally:
            reset_session_cwd(cwd_token)
