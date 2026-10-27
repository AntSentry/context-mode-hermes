"""Install the pinned engine and configure only an explicitly selected Hermes home."""
from __future__ import annotations

import argparse
import copy
from importlib.resources import files
import json
from pathlib import Path
import shutil
import subprocess
import sys

from .engine import Engine, EngineError

ENGINE_VERSION = "1.0.169"


def install_runtime(prefix: Path, *, env: dict | None = None) -> subprocess.CompletedProcess:
    """Install exactly the dependency tree exercised by this adapter's tests."""
    if prefix.is_symlink():
        raise EngineError("Refusing to replace a symlinked runtime prefix")
    manifest = prefix / "package.json"
    if manifest.exists():
        try:
            existing = json.loads(manifest.read_text(encoding="utf-8"))
        except ValueError as exc:
            raise EngineError(f"Cannot read existing runtime manifest: {manifest}") from exc
        dependencies = existing.get("dependencies") if isinstance(existing, dict) else None
        if not isinstance(dependencies, dict) or set(dependencies) != {"context-mode"} or any(
            key in existing for key in ("scripts", "devDependencies", "workspaces")
        ):
            raise EngineError(f"Refusing to overwrite a shared/custom npm prefix: {prefix}")
    npm = shutil.which("npm")
    if not npm:
        raise EngineError("npm is required; Node >=22.5 is required at runtime")
    prefix.mkdir(parents=True, exist_ok=True, mode=0o700)
    resources = files("context_mode_hermes").joinpath("runtime")
    for name in ("package.json", "package-lock.json"):
        (prefix / name).write_text(resources.joinpath(name).read_text(encoding="utf-8"), encoding="utf-8")
    # The lock was generated in nested mode. Hoisting triggers this engine
    # pin's automatic npm repair on every hook invocation.
    return subprocess.run([npm, "ci", "--prefix", str(prefix), "--install-strategy=nested"],
                          env=env, capture_output=True, text=True, timeout=300)


def configure(home: Path, binary: Path) -> dict:
    """Merge context-mode configuration using Hermes's atomic configuration writer."""
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    from hermes_cli.config import read_raw_config, save_config
    from hermes_cli.plugins import PluginContext, VALID_HOOKS

    required = {"pre_tool_call", "post_tool_call", "pre_llm_call", "post_llm_call",
                "post_compaction", "on_session_end", "on_session_finalize"}
    missing = required - VALID_HOOKS
    if missing or not hasattr(PluginContext, "register_command") or not hasattr(PluginContext, "call_mcp"):
        raise EngineError(f"Hermes lacks required plugin interfaces: {sorted(missing)}; use tested v0.22.5")
    home = home.expanduser().resolve()
    binary = binary.expanduser().absolute()
    token = set_hermes_home_override(home)
    try:
        engine = Engine(binary=str(binary), storage_root=str(home / "context-mode"))
        if not engine.available or engine.version != ENGINE_VERSION:
            raise EngineError(f"Requires context-mode {ENGINE_VERSION}; found {engine.version} at {binary}")
        package = Path(engine.binary).resolve().parent
        if not (package / "node_modules" / "better-sqlite3").is_dir():
            raise EngineError(
                "Engine dependencies are hoisted: this pin retries npm repair on every hook. "
                "Install into a fresh prefix with --install-strategy=nested."
            )
        original = read_raw_config()
        config = copy.deepcopy(original)
        plugins = config.setdefault("plugins", {})
        enabled = plugins.setdefault("enabled", [])
        if not isinstance(enabled, list):
            raise EngineError("plugins.enabled must be a list; existing configuration was not changed")
        if "context-mode" not in enabled:
            enabled.append("context-mode")
        disabled = plugins.get("disabled", [])
        if not isinstance(disabled, list):
            raise EngineError("plugins.disabled must be a list; existing configuration was not changed")
        if "context-mode" in disabled:
            plugins["disabled"] = [name for name in disabled if name != "context-mode"]
        entry = plugins.setdefault("entries", {}).setdefault("context-mode", {})
        settings = entry.setdefault("settings", {})
        settings.update({"engine_binary": str(binary), "storage_dir": str(engine.storage_root)})
        allowlist = entry.setdefault("mcp_allowlist", [])
        if not isinstance(allowlist, list):
            raise EngineError("context-mode mcp_allowlist must be a list")
        if "context-mode" not in allowlist:
            allowlist.append("context-mode")
        server = config.setdefault("mcp_servers", {}).setdefault("context-mode", {})
        child_env = engine.environment()
        isolated = {key: child_env[key] for key in (
            "HERMES_HOME", "CONTEXT_MODE_DIR", "CLAUDE_CONFIG_DIR", "CONTEXT_MODE_PLATFORM",
            "CLAUDE_CODE_ENTRYPOINT", "XDG_CONFIG_HOME",
        )}
        server.update({
            "command": sys.executable,
            "args": ["-m", "context_mode_hermes.mcp_proxy", "--engine-bin", str(binary)],
        })
        server.setdefault("env", {}).update(isolated)
        server.setdefault("timeout", 300)
        server.setdefault("connect_timeout", 60)
        changed = config != original
        backup = home / "config.yaml.before-context-mode"
        if changed:
            home.mkdir(parents=True, exist_ok=True, mode=0o700)
            source = home / "config.yaml"
            if source.exists() and not backup.exists():
                shutil.copy2(source, backup)
            save_config(config, strip_defaults=False)
        return {"home": str(home), "engine": str(binary), "engine_version": engine.version,
                "changed": changed, "backup": str(backup) if backup.exists() else None}
    finally:
        reset_hermes_home_override(token)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--home", required=True, type=Path, help="Explicit target Hermes profile directory")
    parser.add_argument("--engine-bin", type=Path, help="Existing context-mode 1.0.169 executable")
    parser.add_argument("--install-engine", action="store_true", help="Install the pinned npm engine inside this profile")
    args = parser.parse_args(argv)
    home = args.home.expanduser().resolve()
    prefix = home / "context-mode-runtime"
    binary = args.engine_bin or prefix / "node_modules" / ".bin" / "context-mode"
    try:
        if args.install_engine:
            if args.engine_bin:
                parser.error("--engine-bin and --install-engine are mutually exclusive")
            installed = install_runtime(prefix)
            if installed.returncode:
                raise EngineError(installed.stdout + installed.stderr)
        result = configure(home, binary)
    except (EngineError, OSError, subprocess.SubprocessError, ImportError) as exc:
        print(f"context-mode installation failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    print("Restart this profile's Hermes gateway. Validate /ctx-doctor, routing and compaction before relying on it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
