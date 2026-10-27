"""Behavior at the upstream hook seam; needs an explicitly selected real engine."""
import json
import os

import pytest


def engine_for(monkeypatch, home):
    from context_mode_hermes.engine import Engine

    binary = os.environ.get("CONTEXT_MODE_TEST_BIN")
    if not binary:
        pytest.skip("Set CONTEXT_MODE_TEST_BIN to exercise the real pinned context-mode engine")
    monkeypatch.setenv("CONTEXT_MODE_BIN", binary)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("CONTEXT_MODE_DIR", raising=False)
    return Engine()


def test_captured_context_is_restored_only_inside_its_profile(monkeypatch, tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    payload = {"session_id": "same-session-id", "cwd": str(workspace)}
    first = engine_for(monkeypatch, tmp_path / "first")
    first.hook("userpromptsubmit", {**payload, "prompt": "Always use region DEPLOY-EU-48321, not the default region."})
    first.hook("precompact", payload)
    restored = first.hook("sessionstart", {**payload, "source": "compact"})
    assert "DEPLOY-EU-48321" in json.dumps(restored)

    second = engine_for(monkeypatch, tmp_path / "second")
    unrelated = second.hook("sessionstart", {**payload, "source": "compact"})
    assert "DEPLOY-EU-48321" not in json.dumps(unrelated)
    assert not (tmp_path / "first" / "context-mode-host" / "settings.json").exists()


def test_unknown_hook_is_rejected_instead_of_silently_losing_capture(monkeypatch, tmp_path):
    from context_mode_hermes.engine import EngineError

    engine = engine_for(monkeypatch, tmp_path / "profile")
    with pytest.raises(EngineError, match="Unsupported"):
        engine.hook("sessionllmstart", {"session_id": "session", "cwd": str(tmp_path)})


def test_session_decisions_survive_working_directory_changes_and_restart(monkeypatch, tmp_path):
    workspace = tmp_path / "workspace"
    subdir = workspace / "subdir"
    subdir.mkdir(parents=True)
    engine = engine_for(monkeypatch, tmp_path / "profile")
    engine.hook("userpromptsubmit", {
        "session_id": "directory-change", "cwd": str(workspace),
        "prompt": "Always deploy in REGION-CD-67512, never the default region.",
    })
    engine.hook("userpromptsubmit", {
        "session_id": "directory-change", "cwd": str(subdir),
        "prompt": "Continue working from this subdirectory.",
    })
    engine.hook("precompact", {"session_id": "directory-change", "cwd": str(subdir)})
    restarted = engine_for(monkeypatch, tmp_path / "profile")
    restored = restarted.hook("sessionstart", {
        "session_id": "directory-change", "cwd": str(subdir), "source": "compact",
    })
    assert "REGION-CD-67512" in json.dumps(restored)
