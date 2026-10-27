#!/usr/bin/env python
"""Reproducible controlled-provider benchmark for the context-mode Hermes adapter.

Two Hermes profiles are built from scratch in a temp root:

* ``baseline`` — plain profile: no plugins, no MCP servers.
* ``plugin``   — full installer.configure() output: MCP server + plugin hooks.

All arms drive the REAL pinned Hermes AIAgent (fresh child process per run,
native plugin manager, native MCP discovery, native ContextCompressor, default
tool output caps/permissions except approvals=off because the run is
unattended) against a scripted OpenAI-compatible provider on 127.0.0.1 that
answers with native tool calls and derives its final answers from the actual
tool results Hermes sends back.

Workloads:
* ``small`` — one fixed-output command (printf with a fresh random nonce), then
  report it. Expected: ctx schemas are pure overhead here.
* ``large`` — curl a local fixture (default 256 KiB JSON; the terminal cap
  spills it), then aggregate COUNT/SUM from the real data via whatever the arm
  actually advertises (ctx_execute_file / ctx_execute when present, otherwise
  the same native python3 aggregation the baseline gets). Expected: the plugin
  arm may win on later-turn request bytes if compaction pays off; report the
  actual result either way — a negative net is a valid result.

The JSON report is machine-readable and self-labels as a controlled-provider
measurement — NOT an LLM quality or speed benchmark.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from benchmarks.controlled_provider import (ControlledProvider, RunParams, SERVER_MODEL_CONTEXT,
                                            new_nonce)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ENGINE_BIN = Path(os.environ.get("CONTEXT_MODE_TEST_BIN") or shutil.which("context-mode") or "context-mode")
DEFAULT_PYTHON = Path(sys.executable)
ARMS = ("baseline", "plugin")

BASE_CONFIG = """\
# context-mode benchmark base profile (identical for all arms)
model:
  default: cm-bench-large
  provider: custom
approvals:
  mode: off
"""

FAIRNESS_NOTES = [
    "All arms run the real pinned Hermes AIAgent with the native ContextCompressor enabled, "
    "default tool output caps and native terminal spill files; nothing is disabled to force a win.",
    "The provider is a scripted, deterministic OpenAI-compatible fake on 127.0.0.1. It is NOT a "
    "model: no LLM quality or speed claims are possible from this benchmark.",
    "The same scripted turn logic runs in every arm and reacts only to actual request content "
    "(advertised tools, actual tool results). The baseline aggregation is equally efficient "
    "native python3 — no strawman 'baseline cannot aggregate' setup.",
    "Wire bytes and rough token estimates include tool schemas and any injected plugin guidance, "
    "for every scripted request; auxiliary requests are counted separately.",
    "estimate_request_tokens_rough is Hermes's own explicit rough estimate (not real model tokens).",
    "Approvals are 'off' in ALL arms because the benchmark is unattended; the only non-default "
    "setting and identical across arms.",
    "Native progressive tool disclosure (tool_search/tool_describe/tool_call bridge) stays at its "
    "defaults in every arm; when Hermes defers the ctx tools the scripted provider walks the "
    "bridge exactly like a real model would, so schema costs are measured as actually shipped.",
    "Numbers are reported as measured. A negative net result for the ctx arms (schema/guidance "
    "overhead exceeding compaction savings) is a valid outcome, never adjusted post hoc.",
]


def build_fixture(target_bytes: int) -> tuple[bytes, int, int]:
    """Deterministic JSON fixture padded to ~target_bytes. Returns (bytes, count, sum)."""
    rng = random.Random(0xC0FFEE)
    records: list[dict] = []
    value_sum = 0
    skeleton = len(json.dumps({"bench": "context-mode-controlled", "records": records},
                              separators=(",", ":")))
    total = 0
    while total + skeleton + 64 < target_bytes:
        value = rng.randint(1, 1000)
        value_sum += value
        record = {"id": len(records), "value": value, "label": "x" * 24}
        records.append(record)
        total += len(json.dumps(record, separators=(",", ":")))
    return (json.dumps({"bench": "context-mode-controlled", "records": records},
                       separators=(",", ":")).encode(), len(records), value_sum)


def sweep_strays(root: Path) -> list[dict]:
    """Kill anything whose command line references the benchmark temp root (MCP
    server children, engine workers, leftover gateways). This process's own
    argv never contains the root path."""
    victims: list[dict] = []
    try:
        out = subprocess.run(["ps", "-eo", "pid=,args="], capture_output=True, text=True,
                             timeout=15).stdout
    except Exception:
        return victims
    mine = {os.getpid(), os.getppid()}
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        pid_text, _, cmd_args = line.partition(" ")
        try:
            pid = int(pid_text)
        except ValueError:
            continue
        if pid in mine or str(root) not in cmd_args:
            continue
        victims.append({"pid": pid, "args": cmd_args[:160]})
        try:
            os.kill(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    return victims


def rough_tokens(body: dict) -> int | None:
    try:
        from agent.model_metadata import estimate_request_tokens_rough
        return int(estimate_request_tokens_rough(body.get("messages") or [],
                                                 tools=body.get("tools") or None))
    except Exception:
        return None


def _hermes_config_op(home: Path, code: str) -> dict:
    """Run a config read/modify through Hermes's own config API in a subprocess
    bound to ``home`` (returns parsed JSON printed by ``code``)."""
    script = ("import sys,json;from pathlib import Path;import hermes_constants;"
              "hermes_constants.set_hermes_home_override(Path(sys.argv[1]));"
              + code)
    proc = subprocess.run([sys.executable, "-c", script, str(home)],
                          capture_output=True, text=True, timeout=120, check=True)
    return json.loads(proc.stdout) if proc.stdout.strip() else {}


def profile_summary(home: Path) -> dict:
    summary: dict = {"home": str(home)}
    try:
        config = _hermes_config_op(home, "from hermes_cli.config import read_raw_config;"
                                        "print(json.dumps(read_raw_config()))")
    except Exception as exc:
        return {**summary, "config_error": f"{type(exc).__name__}: {exc}"[:300]}
    plugins = config.get("plugins", {}) or {}
    servers = config.get("mcp_servers", {}) or {}
    entry = ((plugins.get("entries", {}) or {}).get("context-mode", {}) or {})
    summary.update({
        "plugins_enabled": plugins.get("enabled", []),
        "context_mode_settings": entry.get("settings", {}),
        "context_mode_allowlist": entry.get("mcp_allowlist", []),
        "mcp_servers": {name: {"command": srv.get("command"), "args": srv.get("args")}
                        for name, srv in servers.items()},
    })
    return summary




def run_arm(provider: ControlledProvider, *, arm: str, workload: str, run_id: str, home: Path,
            workdir: Path, python: Path, logs_dir: Path, timeout_s: int, nonce: str, model: str,
            capture_dir: Path, probe_untrusted_context: bool, fixture_truth: dict,
            temp_root: Path) -> dict:
    key = f"{run_id}.{arm}.{workload}"
    session_id = f"cm-bench-{run_id}-{arm}-{workload}"
    out_path = logs_dir / f"{key}.child.json"
    provider.register_run(RunParams(
        key=key, arm=arm, workload=workload, model=model, nonce=nonce,
        fixture_url=provider.fixture_url, fixture_count=fixture_truth["count"],
        fixture_sum=fixture_truth["sum"], probe_untrusted_context=probe_untrusted_context))
    env = {k: v for k, v in os.environ.items()
           if k not in ("HERMES_HOME", "CLAUDE_CONFIG_DIR", "CLAUDE_PROJECT_DIR", "CLAUDE_SESSION_ID",
                        "CONTEXT_MODE_DIR", "CONTEXT_MODE_BIN", "CONTEXT_MODE_PLATFORM",
                        "XDG_CONFIG_HOME", "OMP_PLUGIN_ROOT", "CLAUDE_PLUGIN_ROOT")}
    env["HERMES_HOME"] = str(home)
    env["PYTHONPATH"] = str(REPO_ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    cmd = [str(python), "-m", "benchmarks.arm_runner",
           "--arm", arm, "--workload", workload, "--home", str(home),
           "--base-url", provider.base_url, "--api-key", key, "--model", model,
           "--workdir", str(workdir), "--session-id", session_id,
           "--nonce", nonce, "--fixture-url", provider.fixture_url,
           "--out", str(out_path)]
    started = time.monotonic()
    proc = subprocess.Popen(cmd, cwd=str(REPO_ROOT), env=env, start_new_session=True,
                            stdout=(logs_dir / f"{key}.stdout.log").open("w"),
                            stderr=(logs_dir / f"{key}.stderr.log").open("w"))
    timed_out = False
    try:
        proc.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        timed_out = True
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            proc.kill()
        proc.wait(timeout=30)
    child_wall = round(time.monotonic() - started, 3)
    strays = sweep_strays(temp_root)
    child: dict = {}
    if out_path.exists():
        try:
            child = json.loads(out_path.read_text(encoding="utf-8"))
        except ValueError:
            child = {"ok": False, "error": "child result JSON unparsable"}
    child["timed_out"] = timed_out
    child["child_wall_s"] = child_wall
    if timed_out:
        child.setdefault("error", f"arm process exceeded {timeout_s}s and was killed")
    if strays:
        child["strays_killed"] = strays

    scripted = [r for r in provider.records if r.get("key") == key and r.get("scripted")]
    auxiliary = [r for r in provider.records if r.get("key") == key and not r.get("scripted")
                 and r["method"] == "POST"]
    per_request: list[dict] = []
    total_bytes = 0
    total_tokens = 0
    for index, record in enumerate(scripted):
        total_bytes += record["body_bytes"]
        tokens = None
        dump = capture_dir / f"{_safe(key)}.scripted.req{index}.json"
        if dump.exists():
            try:
                tokens = rough_tokens(json.loads(dump.read_text(encoding="utf-8")))
            except ValueError:
                tokens = None
            if tokens is not None:
                total_tokens += tokens
        summary = record.get("summary") or {}
        per_request.append({
            "body_bytes": record["body_bytes"],
            "rough_tokens": tokens,
            "n_tools": summary.get("n_tools"),
            "tool_names": summary.get("tool_names"),
            "n_messages": summary.get("n_messages"),
            "tool_result_chars": summary.get("tool_result_chars"),
            "truncation_notes": summary.get("truncation_notes"),
            "spill_handles": summary.get("spill_handles"),
            "plugin_guidance_marker": summary.get("plugin_guidance_marker"),
            "reply_kind": record.get("reply_kind"),
        })
    run_report = {
        "arm": arm,
        "workload": workload,
        "run_key": key,
        "session_id": session_id,
        "child_ok": bool(child.get("ok")),
        "child_error": child.get("error"),
        "child_traceback": child.get("traceback"),
        "completed": child.get("completed"),
        "wall_s": child.get("wall_s"),
        "child_wall_s": child.get("child_wall_s"),
        "final_text": child.get("final_text"),
        "tool_calls": child.get("tool_calls"),
        "n_agent_tools": child.get("n_agent_tools"),
        "ctx_tools_advertised": child.get("ctx_tools_advertised"),
        "effective_context_length": child.get("effective_context_length"),
        "mcp_discovery": child.get("mcp_discovery"),
        "cleanup_errors": child.get("cleanup_errors"),
        "requests_scripted": len(scripted),
        "requests_auxiliary": len(auxiliary),
        "auxiliary_body_bytes": sum(r["body_bytes"] for r in auxiliary),
        "total_body_bytes": total_bytes,
        "total_rough_tokens": total_tokens,
        "per_request": per_request,
        "provider_script_log": provider.script_log(key),
        "probe_untrusted_context": probe_untrusted_context,
    }
    if workload == "small":
        run_report["correct"] = bool(nonce and nonce in (child.get("final_text") or ""))
    else:
        text = child.get("final_text") or ""
        run_report["correct"] = (f"COUNT={fixture_truth['count']}" in text
                                 and f"SUM={fixture_truth['sum']}" in text)
        run_report["expected"] = fixture_truth
    provider.drop_run(key)
    return run_report


def _safe(key: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in key)


def _delta(a: dict, b: dict) -> dict:
    def num(field):
        left, right = a.get(field), b.get(field)
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            return round(left - right, 3)
        return None
    return {"total_body_bytes": num("total_body_bytes"),
            "total_rough_tokens": num("total_rough_tokens"),
            "requests_scripted": num("requests_scripted"),
            "wall_s": num("wall_s")}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine-bin", type=Path, default=DEFAULT_ENGINE_BIN,
                        help="Pinned context-mode 1.0.169 binary (or CONTEXT_MODE_TEST_BIN)")
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON,
                        help="Interpreter with pinned Hermes + this adapter installed (editable)")
    parser.add_argument("--output", type=Path, default=None,
                        help="Report JSON path (default: benchmarks/results/bench-<run>.json)")
    parser.add_argument("--workloads", default="small,large", help="Comma-separated subset")
    parser.add_argument("--arms", default=",".join(ARMS), help="Comma-separated arm subset")
    parser.add_argument("--fixture-bytes", type=int, default=262_144)
    parser.add_argument("--arm-timeout", type=int, default=420)
    parser.add_argument("--probe-untrusted-context", action="store_true",
                        help="ctx calls carry deliberately bogus _hermes_context to verify "
                             "hooks/proxy overwrite untrusted metadata")
    parser.add_argument("--keep", action="store_true", help="Keep the temp benchmark root")
    args = parser.parse_args()

    workloads = [w.strip() for w in args.workloads.split(",") if w.strip()]
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    unknown = [a for a in arms if a not in ARMS]
    if unknown:
        parser.error(f"unknown arms {unknown}; choose from {ARMS}")
    run_id = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S}-{uuid.uuid4().hex[:6]}"
    results_dir = REPO_ROOT / "benchmarks" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    output = args.output or (results_dir / f"bench-{run_id}.json")

    temp_root = Path(tempfile.mkdtemp(prefix=f"cm-bench-{run_id}-"))
    logs_dir = temp_root / "logs"
    capture_dir = temp_root / "captured-requests"
    logs_dir.mkdir(parents=True)
    all_killed: list[dict] = []
    provider: ControlledProvider | None = None
    try:
        fixture, count, value_sum = build_fixture(args.fixture_bytes)
        fixture_truth = {"count": count, "sum": value_sum, "bytes": len(fixture)}

        provider = ControlledProvider(fixture, capture_dir)
        provider.start()

        homes: dict[str, Path] = {}
        for arm in arms:
            home = temp_root / f"home-{arm}"
            home.mkdir(parents=True)
            (home / "config.yaml").write_text(BASE_CONFIG, encoding="utf-8")
            homes[arm] = home

        install_report: dict = {"configured": False}
        if "plugin" in arms:
            from context_mode_hermes.install import configure
            install_report = configure(homes["plugin"], args.engine_bin.expanduser())
            install_report["configured"] = True

        report: dict = {
            "label": "context-mode Hermes controlled-provider benchmark",
            "disclaimer": ("Scripted OpenAI-compatible fake provider on loopback; deterministic "
                           "scripted turns; measures wire request bytes, rough token estimates and "
                           "native dispatch. NOT an LLM benchmark; no quality or speed claims."),
            "run_id": run_id,
            "generated": datetime.now(timezone.utc).isoformat(),
            "environment": {
                "hermes_python": str(args.python.expanduser()),
                "engine_bin": str(args.engine_bin.expanduser()),
                "engine_version": install_report.get("engine_version"),
                "node_version": _node_version(),
                "fixture": fixture_truth,
                "workloads": workloads,
                "arms": arms,
                "approvals_mode": "off (all arms, unattended benchmark)",
                "compressor": "native default (all arms)",
                "probe_untrusted_context": args.probe_untrusted_context,
                "provider_base_url": provider.base_url,
                "advertised_context_length": SERVER_MODEL_CONTEXT,
            },
            "fairness": FAIRNESS_NOTES,
            "profiles": {arm: profile_summary(home) for arm, home in homes.items()},
            "install": install_report,
            "runs": {},
        }

        for workload in workloads:
            cell: dict = {"workload": workload}
            for arm in arms:
                workdir = temp_root / f"work-{arm}-{workload}"
                workdir.mkdir()
                cell[arm] = run_arm(
                    provider, arm=arm, workload=workload, run_id=run_id, home=homes[arm],
                    workdir=workdir, python=args.python.expanduser(), logs_dir=logs_dir,
                    timeout_s=args.arm_timeout, nonce=new_nonce(),
                    model=f"cm-bench-{workload}", capture_dir=capture_dir,
                    probe_untrusted_context=args.probe_untrusted_context and arm == "plugin",
                    fixture_truth=fixture_truth, temp_root=temp_root)
                all_killed.extend(cell[arm].get("strays_killed", []))
            if "plugin" in cell and "baseline" in cell:
                cell["delta_plugin_minus_baseline"] = _delta(cell["plugin"], cell["baseline"])
            report["runs"][workload] = cell

        report["cleanup"] = {"temp_root": str(temp_root), "kept": bool(args.keep),
                             "strays_killed": all_killed}
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        _print_summary(report)
        print(f"\nreport: {output}")
        valid = all(cell[arm]["child_ok"] and cell[arm]["correct"]
                    for cell in report["runs"].values() for arm in arms)
        return 0 if valid else 1
    finally:
        if provider is not None:
            provider.stop()
        sweep_strays(temp_root)
        if not args.keep:
            shutil.rmtree(temp_root, ignore_errors=True)


def _node_version() -> str:
    try:
        return subprocess.run(["node", "--version"], capture_output=True, text=True,
                              timeout=10).stdout.strip()
    except Exception:
        return "unavailable"


def _print_summary(report: dict) -> None:
    print(report["label"])
    print(report["disclaimer"])
    for workload, cell in report["runs"].items():
        print(f"\n== workload: {workload} ==")
        for arm, run in ((k, v) for k, v in cell.items() if k in ARMS):
            status = "ok" if run["child_ok"] else f"FAILED: {run['child_error']}"
            print(f"  {arm:9} {status} | requests={run['requests_scripted']} "
                  f"bytes={run['total_body_bytes']} rough_tokens={run['total_rough_tokens']} "
                  f"wall={run.get('wall_s')}s correct={run.get('correct')} "
                  f"tools={run.get('n_agent_tools')} ctx={len(run.get('ctx_tools_advertised') or [])}")
        for name in ("delta_plugin_minus_baseline",):
            if name in cell:
                d = cell[name]
                print(f"  {name}: bytes={d['total_body_bytes']} tokens={d['total_rough_tokens']} "
                      f"requests={d['requests_scripted']}")


if __name__ == "__main__":
    raise SystemExit(main())
