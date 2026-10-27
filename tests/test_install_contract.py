"""Installation must preserve unrelated Hermes configuration and profile isolation."""
import json
import os
from pathlib import Path

import pytest


def test_install_keeps_existing_model_plugins_and_other_profile(tmp_path, monkeypatch):
    pytest.importorskip("hermes_cli.config")
    from context_mode_hermes.install import configure
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    from hermes_cli.config import read_raw_config

    binary = os.environ.get("CONTEXT_MODE_TEST_BIN")
    if not binary:
        pytest.skip("Set CONTEXT_MODE_TEST_BIN to the real pinned engine")
    home = tmp_path / "selected"
    home.mkdir()
    config_path = home / "config.yaml"
    original = {
        "model": {"provider": "custom", "default": "existing-local-model", "base_url": "http://127.0.0.1:8080/v1"},
        "plugins": {"enabled": ["existing-plugin"], "disabled": ["context-mode", "still-off"],
                    "entries": {"existing-plugin": {"settings": {"keep": 42}}}},
        "mcp_servers": {"existing": {"command": "existing-server"}},
    }
    config_path.write_text(json.dumps(original), encoding="utf-8")
    other = tmp_path / "other"
    other.mkdir()
    (other / "config.yaml").write_text("model: untouched\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(other))

    configure(home, Path(binary))
    configure(home, Path(binary))
    token = set_hermes_home_override(home)
    try:
        configured = read_raw_config()
    finally:
        reset_hermes_home_override(token)
    assert configured["model"] == original["model"]
    assert configured["plugins"]["enabled"] == ["existing-plugin", "context-mode"]
    assert configured["plugins"]["disabled"] == ["still-off"], "Installation must override only this plugin's previous disable"
    assert configured["plugins"]["entries"]["existing-plugin"] == original["plugins"]["entries"]["existing-plugin"]
    assert configured["mcp_servers"]["existing"] == original["mcp_servers"]["existing"]
    assert (other / "config.yaml").read_text(encoding="utf-8") == "model: untouched\n"
    assert configured["mcp_servers"]["context-mode"]["env"]["CONTEXT_MODE_DIR"] == str(home / "context-mode")
    assert configured["mcp_servers"]["context-mode"]["env"]["CONTEXT_MODE_MCP_SENTINEL_DIR"] == (
        str(home / "context-mode-host" / "mcp-sentinels")), (
        "the profile's MCP server must publish readiness to the profile-owned sentinel dir")
