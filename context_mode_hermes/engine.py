"""Bounded subprocess interface to context-mode's supported Claude hook protocol.

Hermes keeps its own profile and native MCP client. Only hook payloads use the
Claude protocol; neither a Claude installation nor its user configuration is read.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile


class EngineError(RuntimeError):
    """The configured engine could not execute a supported hook."""


class Engine:
    EVENTS = frozenset({
        "pretooluse", "posttooluse", "userpromptsubmit", "sessionstart",
        "precompact", "stop",
    })
    HOOK_TIMEOUT = 8

    def __init__(self, *, binary: str | None = None, storage_root: str | None = None):
        self._env = dict(os.environ)
        try:
            from hermes_constants import get_hermes_home
            self.home = get_hermes_home().expanduser().resolve()
        except ImportError:
            self.home = Path(self._env.get("HERMES_HOME", "~/.hermes")).expanduser().resolve()
        configured = storage_root or self._env.get("CONTEXT_MODE_DIR", "").strip()
        if configured and not Path(configured).is_absolute():
            raise EngineError("CONTEXT_MODE_DIR must be an absolute isolated storage path")
        self.storage_root = Path(configured) if configured else self.home / "context-mode"
        self.storage_root = self.storage_root.resolve()
        self.host_root = self.home / "context-mode-host"
        self.sentinel_root = self.host_root / "mcp-sentinels"
        selected = binary or self._env.get("CONTEXT_MODE_BIN", "").strip()
        known = self.home / "context-mode-runtime" / "node_modules" / ".bin" / "context-mode"
        self.binary = selected or (str(known) if known.is_file() else None) or (
            shutil.which("context-mode") or "context-mode"
        )
        resolved = shutil.which(self.binary)
        if resolved:
            self.binary = resolved

    @property
    def available(self) -> bool:
        return os.path.isfile(self.binary) and os.access(self.binary, os.X_OK)

    @property
    def version(self) -> str:
        try:
            metadata = Path(self.binary).resolve().parent / "package.json"
            return str(json.loads(metadata.read_text(encoding="utf-8"))["version"])
        except (OSError, ValueError, KeyError):
            return "unknown"

    def environment(self) -> dict[str, str]:
        env = dict(self._env)
        # Hooks can inspect several client identifiers even when CLI dispatch
        # explicitly names claude-code. Never inherit another agent's paths.
        for name in (
            "CLAUDE_PLUGIN_ROOT", "CLAUDE_CONFIG_DIR", "CLAUDE_PROJECT_DIR",
            "CLAUDE_SESSION_ID", "CLAUDE_CODE_ENTRYPOINT", "PI_CODING_AGENT_DIR",
            "PI_PROJECT_DIR", "OMP_PLUGIN_ROOT", "CONTEXT_MODE_PLATFORM",
        ):
            env.pop(name, None)
        env.update({
            "HERMES_HOME": str(self.home),
            "CONTEXT_MODE_DIR": str(self.storage_root),
            "CLAUDE_CONFIG_DIR": str(self.host_root),
            "XDG_CONFIG_HOME": str(self.host_root / "xdg-config"),
            "CONTEXT_MODE_PLATFORM": "claude-code",
            "CLAUDE_CODE_ENTRYPOINT": "hermes-context-mode-bridge",
        })
        # Upstream's isMCPReady() (hooks/core/mcp-ready.mjs) scans this dir when
        # set, otherwise machine-global /tmp — which foreign agents also write.
        # Every process spawned with this env (hook CLI and MCP workers) then
        # sees THIS profile's readiness only. Best-effort: on an unwritable
        # home, readiness fails closed and spawning still proceeds.
        try:
            self.sentinel_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        except OSError:
            pass
        env["CONTEXT_MODE_MCP_SENTINEL_DIR"] = str(self.sentinel_root)
        return env

    def context(self, session_id: str, cwd: str) -> dict[str, str]:
        """Anchor storage to the session's first workspace; retain its live cwd.

        Both hooks and MCP workers use this identity. A directory change must
        not silently move the conversation's captured decisions to another DB.
        The tiny record also survives gateway restarts and resumed sessions.
        """
        if not session_id:
            raise EngineError("A Hermes session_id is required for context-mode operations")
        current = str(Path(cwd).expanduser().resolve())
        state = self.host_root / "session-roots"
        state.mkdir(parents=True, exist_ok=True, mode=0o700)
        record = state / (hashlib.sha256(session_id.encode()).hexdigest() + ".json")
        if not record.exists():
            with tempfile.NamedTemporaryFile("w", dir=state, encoding="utf-8", delete=False) as out:
                temporary = Path(out.name)
                json.dump({"session_id": session_id, "project_dir": current}, out)
            try:
                try:
                    os.link(temporary, record)  # atomic first-writer-wins, even across processes
                except FileExistsError:
                    pass
            finally:
                temporary.unlink()
        try:
            saved = json.loads(record.read_text(encoding="utf-8"))
            project = saved["project_dir"]
            if saved["session_id"] != session_id or not Path(project).is_absolute():
                raise ValueError("invalid session workspace record")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise EngineError(f"Cannot restore Hermes session workspace: {record}") from exc
        return {"session_id": session_id, "cwd": current, "project_dir": project}

    def hook(self, event: str, payload: dict) -> dict:
        if event not in self.EVENTS:
            # Upstream's unknown-hook dispatch can exit zero without doing work.
            raise EngineError(f"Unsupported context-mode hook: {event}")
        if not self.available:
            raise EngineError("context-mode executable unavailable; configure CONTEXT_MODE_BIN")
        cwd = str(Path(payload.get("cwd") or os.getcwd()).expanduser().resolve())
        context = self.context(str(payload.get("session_id", "")), cwd)
        env = self.environment()
        env["CLAUDE_PROJECT_DIR"] = context["project_dir"]
        env["CLAUDE_SESSION_ID"] = context["session_id"]
        for directory in (self.storage_root, self.host_root):
            directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        try:
            result = subprocess.run(
                [self.binary, "hook", "claude-code", event],
                input=json.dumps({**payload, "cwd": context["project_dir"]}),
                cwd=cwd, env=env, capture_output=True, text=True,
                timeout=self.HOOK_TIMEOUT, check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise EngineError(f"context-mode {event} failed: {exc}") from exc
        if result.returncode:
            raise EngineError(f"context-mode {event} exited {result.returncode}: {result.stderr[-1000:]}")
        output = result.stdout.strip()
        if not output:
            return {}
        try:
            value = json.loads(output)
        except ValueError as exc:
            raise EngineError(f"context-mode {event} returned invalid hook JSON") from exc
        if not isinstance(value, dict):
            raise EngineError(f"context-mode {event} returned a non-object hook result")
        return value
