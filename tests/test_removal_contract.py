"""Removal contract for ``context-mode-hermes --remove`` (reversible, profile-scoped).

Public seams only:
- ``context_mode_hermes.install.main(argv)`` — the console-script entry point,
  asserting exit statuses, stdout/stderr contracts and ``--help``;
- Hermes' real configuration read/write path under an isolated ``--home``
  (``hermes_cli.config.save_config`` / ``require_readable_config_before_write``);
- for reversibility only, the real pinned engine binary via
  ``context_mode_hermes.engine.Engine`` (skipped without ``CONTEXT_MODE_TEST_BIN``).

Contract under test: removal is configuration-only for one explicit ``--home``.
It never spawns subprocesses, needs no npm/Node/engine, keeps the current
(later-edited) config rather than rolling back to the pre-install snapshot,
retains engine runtime/indexed content/captured sessions, refuses malformed
config without partial changes, and treats an unconfigured profile as success.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def _cli(home: Path, *extra: str) -> int:
    from context_mode_hermes import install

    return install.main(["--home", str(home), *extra])


def _read(home: Path, monkeypatch) -> dict:
    from hermes_cli.config import read_raw_config

    monkeypatch.setenv("HERMES_HOME", str(home))
    return read_raw_config()


def _installed_config(home: Path) -> dict:
    """Exactly the registration ``install.configure`` writes, plus unrelated
    settings and post-install edits an operator may have made meanwhile."""
    binary = str(home / "context-mode-runtime" / "node_modules" / ".bin" / "context-mode")
    return {
        "model": {"provider": "custom", "default": "newer-model", "base_url": "http://127.0.0.1:8080/v1"},
        "plugins": {
            "enabled": ["existing-plugin", "context-mode", "added-later"],
            # the operator disabled the plugin after installing it; removal must
            # strip only context-mode's mention and keep unrelated ones
            "disabled": ["context-mode", "still-off"],
            "entries": {
                "existing-plugin": {"settings": {"keep": 42}},
                "context-mode": {
                    "settings": {"engine_binary": binary, "storage_dir": str(home / "context-mode")},
                    "mcp_allowlist": ["context-mode"],
                },
            },
        },
        "mcp_servers": {
            "existing": {"command": "existing-server"},
            "later": {"command": "later-server"},
            "context-mode": {
                "command": sys.executable,
                "args": ["-m", "context_mode_hermes.mcp_proxy", "--engine-bin", binary],
                "env": {"HERMES_HOME": str(home), "CONTEXT_MODE_DIR": str(home / "context-mode")},
                "timeout": 300, "connect_timeout": 60,
            },
        },
    }


def _result_json(capsys) -> dict:
    lines = capsys.readouterr().out.splitlines()
    return json.loads("\n".join(lines[: lines.index("}") + 1]))


@pytest.fixture
def configured_home(tmp_path, monkeypatch) -> Path:
    pytest.importorskip("hermes_cli.config")
    home = tmp_path / "profile"
    home.mkdir()
    (home / "config.yaml").write_text(json.dumps(_installed_config(home)), encoding="utf-8")
    # retained data an operator must never lose to --remove
    for retained in ("context-mode", "context-mode-runtime", "context-mode-host"):
        (home / retained).mkdir()
        (home / retained / "retained.db").write_text("data", encoding="utf-8")
    (home / "config.yaml.before-context-mode").write_text("older snapshot", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


def test_help_documents_removal(capsys):
    from context_mode_hermes import install

    with pytest.raises(SystemExit) as done:
        install.main(["--help"])
    assert done.value.code == 0
    assert "--remove" in capsys.readouterr().out


def test_remove_requires_explicit_home():
    from context_mode_hermes import install

    with pytest.raises(SystemExit) as done:
        install.main(["--remove"])
    assert done.value.code == 2


@pytest.mark.parametrize("flag", [["--install-engine"], ["--engine-bin", "/somewhere/context-mode"]])
def test_remove_is_exclusive_with_install_and_engine_selection(configured_home, capsys, flag):
    before = (configured_home / "config.yaml").read_bytes()
    with pytest.raises(SystemExit) as done:
        _cli(configured_home, "--remove", *flag)
    assert done.value.code == 2
    assert (configured_home / "config.yaml").read_bytes() == before


def test_remove_unregisters_only_integration_and_preserves_later_settings(configured_home, capsys, monkeypatch):
    from context_mode_hermes import install

    # removal is configuration-only: any subprocess or npm call is a contract break
    monkeypatch.setattr(install.subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("removal must not spawn subprocesses")))
    monkeypatch.setattr(install.shutil, "which", lambda name: None)

    assert _cli(configured_home, "--remove") == 0
    out = capsys.readouterr().out
    lines = out.splitlines()
    result = json.loads("\n".join(lines[: lines.index("}") + 1]))
    assert result["changed"] is True
    assert result["removed"] == {"enabled": True, "disabled": True, "plugin_entry": True, "mcp_server": True}
    assert "Restart" in out  # running gateways need a restart; new sessions are native

    configured = _read(configured_home, monkeypatch)
    # integration registration gone everywhere
    assert configured["plugins"]["enabled"] == ["existing-plugin", "added-later"]
    assert configured["plugins"]["disabled"] == ["still-off"]
    assert set(configured["plugins"]["entries"]) == {"existing-plugin"}
    assert configured["plugins"]["entries"]["existing-plugin"] == {"settings": {"keep": 42}}
    assert set(configured["mcp_servers"]) == {"existing", "later"}
    # model/provider and every later edit survive untouched
    assert configured["model"] == {"provider": "custom", "default": "newer-model",
                                   "base_url": "http://127.0.0.1:8080/v1"}
    # retained data: storage, runtime, host dirs, pre-install backup, adapter package
    for retained in ("context-mode", "context-mode-runtime", "context-mode-host"):
        assert (configured_home / retained / "retained.db").read_text(encoding="utf-8") == "data"
    assert (configured_home / "config.yaml.before-context-mode").read_text(encoding="utf-8") == "older snapshot"
    assert result["retained"]["storage_dir"] == str(configured_home / "context-mode")


def test_remove_needs_no_engine_or_node(configured_home, monkeypatch):
    from context_mode_hermes import install

    # engine runtime absent, npm/node unfindable, nothing deletable: removal still works
    monkeypatch.setattr(install.subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("removal must not spawn subprocesses")))
    monkeypatch.setattr(install.shutil, "which", lambda name: None)
    monkeypatch.setattr(install.shutil, "rmtree", lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("removal must not delete data")))
    assert _cli(configured_home, "--remove") == 0
    assert "context-mode" not in _read(configured_home, monkeypatch)["plugins"]["entries"]


def test_remove_without_prior_integration_is_noop_success(tmp_path, capsys):
    pytest.importorskip("hermes_cli.config")
    home = tmp_path / "plain"
    home.mkdir()
    (home / "config.yaml").write_text(
        json.dumps({"model": {"default": "m"}, "plugins": {"enabled": ["other"]}}), encoding="utf-8")
    before = (home / "config.yaml").read_bytes()

    assert _cli(home, "--remove") == 0
    result = _result_json(capsys)
    assert result["changed"] is False
    assert result["removed"] == {"enabled": False, "disabled": False, "plugin_entry": False, "mcp_server": False}
    assert (home / "config.yaml").read_bytes() == before  # no rewrite, no backup manufactured


def test_remove_of_missing_config_is_noop_success(tmp_path, capsys):
    pytest.importorskip("hermes_cli.config")
    home = tmp_path / "never-configured"
    assert _cli(home, "--remove") == 0
    assert _result_json(capsys)["changed"] is False
    assert not (home / "config.yaml").exists()  # no-op removal writes nothing
    assert not home.exists()


def test_remove_refuses_malformed_config_without_changes(tmp_path, capsys):
    pytest.importorskip("hermes_cli.config")
    home = tmp_path / "broken"
    home.mkdir()
    broken = home / "config.yaml"
    broken.write_text("model: {unclosed", encoding="utf-8")
    before = broken.read_bytes()

    assert _cli(home, "--remove") == 1
    captured = capsys.readouterr()
    assert "context-mode removal failed" in captured.err
    assert "Refusing to remove" in captured.err  # refuses rather than silently treating it as removed
    assert broken.read_bytes() == before  # recoverable file left exactly as it was


@pytest.mark.parametrize("shape", [
    {"plugins": {"enabled": {"context-mode": True}}},
    {"plugins": {"enabled": ["context-mode"], "entries": "not-a-mapping"}},
    {"plugins": {"enabled": ["context-mode"]}, "mcp_servers": ["context-mode"]},
])
def test_remove_refuses_malformed_plugin_shapes_without_changes(configured_home, capsys, shape):
    configured_home.joinpath("config.yaml").write_text(json.dumps(shape), encoding="utf-8")
    before = (configured_home / "config.yaml").read_bytes()

    assert _cli(configured_home, "--remove") == 1
    assert "removal failed" in capsys.readouterr().err
    assert (configured_home / "config.yaml").read_bytes() == before  # no partial destructive edits


@pytest.mark.skipif(not TEST_BIN, reason="Set CONTEXT_MODE_TEST_BIN to exercise install/remove/re-enable with the real engine")
def test_remove_then_offline_reenable_retains_searchable_content(tmp_path, monkeypatch, capsys):
    from context_mode_hermes.engine import Engine

    binary = Path(TEST_BIN)
    home = tmp_path / "profile"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("CONTEXT_MODE_DIR", raising=False)

    # 1. install into this profile only (existing engine, no npm install)
    assert _cli(home, "--engine-bin", str(binary)) == 0
    marker = "REGION-KEEP-99117"
    session = {"session_id": "removal-reversibility", "cwd": str(workspace)}
    capture = Engine(binary=str(binary), storage_root=str(home / "context-mode"))
    capture.hook("userpromptsubmit", {**session, "prompt": f"Always deploy in {marker}, never elsewhere."})
    capture.hook("precompact", session)

    # 2. remove: configuration-only, storage and the installed engine must survive untouched
    assert _cli(home, "--remove") == 0
    assert (home / "context-mode").is_dir()
    assert binary.is_file()
    assert "context-mode" not in _read(home, monkeypatch)["mcp_servers"]

    # 3. re-enable offline: the retained engine is reused, npm must not run
    from context_mode_hermes import install

    monkeypatch.setattr(install, "install_runtime", lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("re-enable must reuse the retained engine, not reinstall")))
    assert _cli(home, "--engine-bin", str(binary)) == 0
    assert _result_json(capsys)["engine_version"] == "1.0.169"
    reconfigured = _read(home, monkeypatch)
    assert "context-mode" in reconfigured["plugins"]["enabled"]
    assert reconfigured["mcp_servers"]["context-mode"]["env"]["CONTEXT_MODE_DIR"] == str(home / "context-mode")

    # 4. retained content is reachable without re-indexing
    restored = Engine(binary=str(binary), storage_root=str(home / "context-mode")).hook(
        "sessionstart", {"session_id": "removal-reversibility", "cwd": str(workspace), "source": "compact"})
    assert marker in json.dumps(restored)
