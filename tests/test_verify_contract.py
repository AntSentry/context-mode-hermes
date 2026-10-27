"""Behavior tests for the safe verification smoke (``context_mode_hermes.verify``).

The smoke is exercised end-to-end at its public seams: the CLI, the PTY the
interactive Hermes REPL runs on, the loopback provider's HTTP socket and the
profile filesystem. ``--hermes`` points at a scripted REPL that speaks the same
interactive contract as ``hermes chat --cli`` (never ``-q``), and the pinned
Hermes config/plugin API is provided by a stub package so the installer
subprocess runs for real against an owned temporary profile. No model is ever
called: the loopback provider rejects inference, and the smoke treats any
inference attempt as a failure with retained evidence, never as success.

Safety behaviors under test: owned-profile isolation, ambient ``HERMES_HOME``
scrubbing, default-profile preservation, interruption cleanup that only ever
signals identity-verified owned processes (a decoy "hermes" process in a
different profile must survive), bounded escalation for unresponsive owned
children, refusal of unpinned Hermes/engine versions, and credential redaction
in retained provider request logs.
"""
from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

REPO_ROOT = Path(__file__).resolve().parent.parent
VERIFY_MODULE = "context_mode_hermes.verify"

FAKE_REPL = textwrap.dedent(
    '''\
    import json
    import os
    import re
    import signal
    import subprocess
    import sys
    import time
    import urllib.request

    MODE = os.environ.get("FAKE_MODE", "ok")
    HERMES_HOME = os.environ["HERMES_HOME"]
    INDEXED = {"nonce": "nothing-indexed-yet"}


    def read_config_text():
        with open(os.path.join(HERMES_HOME, "config.yaml"), "r", encoding="utf-8") as fh:
            return fh.read()


    def base_url_and_key():
        text = read_config_text()
        url = re.search(r"http://127\\.0\\.0\\.1:\\d+/v1", text).group(0)
        key = re.search(r"cm-verify-[0-9a-f]+", text).group(0)
        return url, key


    def capability_probe():
        """Mirror Hermes startup: one authenticated GET /v1/models."""
        url, key = base_url_and_key()
        request = urllib.request.Request(url + "/models", headers={"Authorization": "Bearer " + key})
        try:
            urllib.request.urlopen(request, timeout=10)
        except Exception:
            pass


    def infer_once():
        url, key = base_url_and_key()
        request = urllib.request.Request(
            url + "/chat/completions",
            data=json.dumps({"model": "verify-model",
                             "messages": [{"role": "user", "content": "hi"}]}).encode(),
            headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
            method="POST")
        try:
            urllib.request.urlopen(request, timeout=10)
        except Exception:
            pass


    def answer(line):
        if MODE == "slow-first-command" and line == "/ctx-stats":
            time.sleep(600)
        if line == "/ctx-stats":
            if MODE == "silent-stats":
                print("context-mode engine stats: (quiet)")
            else:
                print("context-mode engine stats:")
                print("No tool calls yet.")
                print("v1.0.169")
        elif line == "/ctx-doctor":
            print("[Hermes integration]")
            print("[OK] Engine runtime v1.0.169 at /engine (plugin-tested: 1.0.169)")
            print("[OK] Hermes mcp_servers['context-mode'] configured")
        elif line.startswith("/ctx-index"):
            if MODE == "inference":
                infer_once()
            INDEXED["nonce"] = re.search(r"CM-VERIFY-NONCE-[0-9a-f]+", line).group(0)
            text = re.search(r'--text "(.*)" --source', line).group(1)
            store_dir = os.path.join(HERMES_HOME, "context-mode")
            os.makedirs(store_dir, exist_ok=True)
            with open(os.path.join(store_dir, "verify-store.jsonl"), "a",
                      encoding="utf-8") as fh:
                fh.write(json.dumps({"content": text, "source": "cm-verify-smoke"}) + "\\n")
            print("Indexed 1 sections (0 with code) from: cm-verify-smoke")
        elif line.startswith("/ctx-search"):
            if MODE == "stuck":
                signal.signal(signal.SIGTERM, signal.SIG_IGN)
                time.sleep(600)
            print(INDEXED["nonce"])
        elif line == "/exit":
            return "exit"
        else:
            print("unknown command")
        return None


    def main():
        capability_probe()
        if MODE == "startup-typeahead-flush":
            import termios
            time.sleep(0.5)
            termios.tcflush(sys.stdin.fileno(), termios.TCIFLUSH)
        print("Hermes Agent 0.22.5 (scripted verification REPL) ready")
        # A foreign profile may inherit our group; only PID identity proves ownership.
        if os.environ.get("FAKE_DECOY_PIDFILE"):
            decoy = subprocess.Popen(
                [sys.executable, "-c",
                 "import os, signal, time\\n"
                 "signal.signal(signal.SIGHUP, signal.SIG_IGN)\\n"
                 "with open(os.environ['FAKE_DECOY_PIDFILE'], 'w') as f: f.write(str(os.getpid()))\\n"
                 "time.sleep(600)\\n"],
                cwd="/tmp",
                env={**os.environ, "HERMES_HOME": os.environ["FAKE_DECOY_HOME"]},
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("❯ ", flush=True)
        while True:
            line = sys.stdin.readline()
            if not line:
                return 0
            line = line.strip()
            if not line:
                continue
            if answer(line) == "exit":
                if MODE == "stuck":
                    signal.signal(signal.SIGTERM, signal.SIG_IGN)
                    time.sleep(600)
                return 0


    sys.exit(main())
    '''
)

HERMES_LAUNCHER = textwrap.dedent(
    """\
    #!{python}
    import os
    import sys

    args = sys.argv[1:]
    if args[:1] == ["--version"]:
        print(os.environ.get("FAKE_HERMES_VERSION", "hermes-agent 0.22.5 (scripted)"))
        raise SystemExit(0)
    if args[:2] == ["chat", "--cli"]:
        if "-q" in args or "--query" in args or "--oneshot" in args:
            sys.stderr.write("fake hermes: one-shot mode forbidden in smoke\\n")
            raise SystemExit(9)
        raise SystemExit(os.system("{python} \\"{repl}\\"") >> 8)
    sys.stderr.write("fake hermes: unsupported arguments %r\\n" % (args,))
    raise SystemExit(3)
    """
)

FAKE_ENGINE_MCP = r"""#!__PYTHON__
import json
import os
import sys

STORE = os.path.join(
    os.environ.get("CONTEXT_MODE_DIR",
                   os.path.join(os.environ.get("HERMES_HOME", ""), "context-mode")),
    "verify-store.jsonl")


def send(payload):
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()


for raw in sys.stdin:
    raw = raw.strip()
    if not raw:
        continue
    try:
        msg = json.loads(raw)
    except ValueError:
        continue
    method = msg.get("method", "")
    if method == "notifications/initialized":
        continue
    if method == "initialize":
        send({"jsonrpc": "2.0", "id": msg.get("id"), "result": {
            "protocolVersion": msg.get("params", {}).get("protocolVersion",
                                                         "2024-11-05"),
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "context-mode-fake", "version": "1.0.169"}}})
        continue
    if method == "tools/list":
        send({"jsonrpc": "2.0", "id": msg.get("id"), "result": {
            "tools": [{"name": "ctx_search",
                       "inputSchema": {"type": "object", "properties": {}}}]}})
        continue
    if method == "tools/call":
        name = msg.get("params", {}).get("name")
        args = msg.get("params", {}).get("arguments", {})
        if name == "ctx_index":
            os.makedirs(os.path.dirname(STORE), exist_ok=True)
            with open(STORE, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"content": args.get("content", ""),
                                     "source": args.get("source", "")}) + "\n")
            send({"jsonrpc": "2.0", "id": msg.get("id"), "result": {
                "content": [{"type": "text", "text": "indexed"}], "isError": False}})
            continue
        if name == "ctx_search":
            queries = args.get("queries", [])
            hits = []
            try:
                with open(STORE, encoding="utf-8") as fh:
                    for line in fh:
                        record = json.loads(line)
                        if args.get("source") and record.get("source") != args.get("source"):
                            continue
                        if any(q in record.get("content", "") for q in queries):
                            hits.append(record["content"])
            except OSError:
                pass
            send({"jsonrpc": "2.0", "id": msg.get("id"), "result": {
                "content": [{"type": "text", "text": "\n".join(hits)}],
                "isError": False}})
            continue
        send({"jsonrpc": "2.0", "id": msg.get("id"), "result": {
            "content": [{"type": "text", "text": name or ""}], "isError": False}})
        continue
    if msg.get("id") is not None:
        send({"jsonrpc": "2.0", "id": msg.get("id"), "result": {}})
"""


STUB_HERMES_CONSTANTS = textwrap.dedent(
    """\
    import os
    from pathlib import Path

    _override = None


    def set_hermes_home_override(home):
        global _override
        token = ("stub", _override)
        _override = Path(home)
        return token


    def reset_hermes_home_override(token):
        global _override
        _override = token[1]


    def get_hermes_home():
        if _override is not None:
            return _override
        env = os.environ.get("HERMES_HOME")
        return Path(env).expanduser() if env else Path.home() / ".hermes"


    def display_hermes_home(home=None):
        return str(home or get_hermes_home())
    """
)

STUB_HERMES_CONFIG = textwrap.dedent(
    """\
    import json

    def _home():
        from hermes_constants import get_hermes_home
        return get_hermes_home()


    def read_raw_config():
        path = _home() / "config.yaml"
        if not path.exists():
            return {}
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)


    def save_config(config, strip_defaults=True):
        path = _home() / "config.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(config, fh, indent=2)
    """
)

STUB_HERMES_PLUGINS = textwrap.dedent(
    """\
    VALID_HOOKS = {
        "pre_tool_call", "post_tool_call", "pre_llm_call", "post_llm_call",
        "post_compaction", "on_session_end", "on_session_finalize",
    }


    class PluginContext:
        register_command = staticmethod(lambda *a, **k: None)
        call_mcp = staticmethod(lambda *a, **k: None)
    """
)


class VerifyRun:
    def __init__(self, proc: subprocess.Popen, evidence_dir: Path) -> None:
        self.proc = proc
        self.evidence_dir = evidence_dir
        self.stdout, self.stderr = proc.communicate(timeout=300)
        self.returncode = proc.returncode
        self.report = None
        report_path = evidence_dir / "verify-report.json"
        if report_path.exists():
            self.report = json.loads(report_path.read_text(encoding="utf-8"))

    @property
    def status(self) -> str | None:
        return (self.report or {}).get("status")

    def provider_records(self) -> list[dict]:
        records = []
        for path in sorted((self.evidence_dir / "provider").glob("*.json")):
            records.append(json.loads(path.read_text(encoding="utf-8")))
        return records

    def all_evidence_text(self) -> str:
        parts = []
        for path in sorted(self.evidence_dir.rglob("*")):
            if path.is_file():
                parts.append(path.read_text(encoding="utf-8", errors="replace"))
        return "\n".join(parts)


@pytest.fixture()
def fake_stack(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    """Owned workdir with a scripted hermes, a stub Hermes API and a pinned fake engine."""
    work = tmp_path / "stack"
    work.mkdir()
    python = sys.executable

    repl = work / "fake_hermes_repl.py"
    repl.write_text(FAKE_REPL, encoding="utf-8")

    hermes = work / "hermes"
    hermes.write_text(HERMES_LAUNCHER.format(python=python, repl=repl), encoding="utf-8")
    hermes.chmod(0o755)

    stub = work / "stub"
    (stub / "hermes_cli").mkdir(parents=True)
    (stub / "hermes_constants.py").write_text(STUB_HERMES_CONSTANTS, encoding="utf-8")
    (stub / "hermes_cli" / "__init__.py").write_text("", encoding="utf-8")
    (stub / "hermes_cli" / "config.py").write_text(STUB_HERMES_CONFIG, encoding="utf-8")
    (stub / "hermes_cli" / "plugins.py").write_text(STUB_HERMES_PLUGINS, encoding="utf-8")

    engine_pkg = work / "engine-runtime" / "context-mode-stub"
    engine_pkg.mkdir(parents=True)
    cli = engine_pkg / "cli"
    cli.write_text(FAKE_ENGINE_MCP.replace("__PYTHON__", python), encoding="utf-8")
    cli.chmod(0o755)
    (engine_pkg / "package.json").write_text(json.dumps({"version": "1.0.169"}), encoding="utf-8")
    # configure() refuses engines with hoisted deps; fake a nested prefix.
    (engine_pkg / "node_modules" / "better-sqlite3").mkdir(parents=True)
    engine_bin = work / "engine-runtime" / "node_modules" / ".bin" / "context-mode"
    engine_bin.parent.mkdir(parents=True)
    engine_bin.symlink_to(cli)

    default_home = tmp_path / "default-home"
    default_home.mkdir()
    (default_home / "config.yaml").write_text("model: untouched\n", encoding="utf-8")

    monkeypatch.setenv("HERMES_HOME", str(default_home))
    monkeypatch.setenv("CM_VERIFY_EXTRA_PYTHONPATH", str(stub))
    return {
        "work": work,
        "hermes": hermes,
        "engine_bin": engine_bin,
        "default_home": default_home,
        "evidence_root": tmp_path / "evidence",
        "profile_root": tmp_path / "profiles",
    }


def run_verify(fake_stack: dict, *extra: str, env: dict | None = None) -> VerifyRun:
    evidence = fake_stack["evidence_root"] / f"ev{time.monotonic_ns()}"
    cmd = [
        sys.executable, "-m", VERIFY_MODULE,
        "--hermes", str(fake_stack["hermes"]),
        "--engine-bin", str(fake_stack["engine_bin"]),
        "--evidence-dir", str(evidence),
        "--profile-root", str(fake_stack["profile_root"]),
        *extra,
    ]
    merged = {k: v for k, v in os.environ.items() if k != "PYTEST_CURRENT_TEST"}
    if env:
        merged.update(env)
    proc = subprocess.Popen(cmd, cwd=str(REPO_ROOT), env=merged,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return VerifyRun(proc, evidence)


@pytest.mark.parametrize("mode", ["ok", "startup-typeahead-flush"])
def test_passing_smoke_end_to_end(fake_stack: dict, mode: str) -> None:
    run = run_verify(fake_stack, "--command-timeout", "8", env={"FAKE_MODE": mode})

    assert run.returncode == 0, (run.stdout, run.stderr, run.report)
    assert run.status == "pass"

    report = run.report
    for check in ("hermes_pin", "engine_pin", "owned_profile_isolated",
                  "installer_configured_owned_profile", "dispatch_stats", "dispatch_doctor",
                  "dispatch_index", "dispatch_search", "mcp_stored_content",
                  "no_inference", "child_exited",
                  "default_profile_unchanged", "no_owned_survivors", "telemetry_opted_out"):
        assert report["checks"][check]["ok"], (check, report["checks"][check])
    assert report["provider"]["inference_requests"] == 0
    assert report["owned_home"]["removed"] is True
    assert not Path(report["owned_home"]["path"]).exists()

    # Provider log retained: the capability GET is recorded with a complete raw record.
    records = run.provider_records()
    assert any(r["method"] == "GET" and r["path"] == "/v1/models" for r in records)

    # The default profile (the ambient HERMES_HOME) is untouched.
    assert (fake_stack["default_home"] / "config.yaml").read_text(encoding="utf-8") == "model: untouched\n"


def test_inference_attempt_is_failure_with_retained_request_log(fake_stack: dict) -> None:
    run = run_verify(fake_stack, env={"FAKE_MODE": "inference"})

    assert run.returncode == 2
    assert run.status == "refused"
    assert run.report["provider"]["inference_requests"] >= 1
    assert run.report["provider"]["inference_attempts"] >= 1
    assert run.report["checks"]["no_inference"]["ok"] is False
    names = [d["name"] for d in run.report["dispatches"]]
    assert names[-1] == "index" and "search" not in names, names

    posts = [r for r in run.provider_records()
             if r["method"] == "POST" and r["path"] == "/v1/chat/completions"]
    assert posts, "the rejected inference request must be retained as evidence"
    post = posts[0]
    assert post["response_status"] == 503
    assert "hi" in post["body"], "the complete raw request body is kept"
    assert post["headers"].get("Authorization") == "REDACTED"
    leaks = re.findall(r"cm-verify-[0-9a-f]+", run.all_evidence_text())
    assert leaks == [], "the provider API key must never appear in evidence"


def test_missing_marker_fails_and_retains_transcript(fake_stack: dict) -> None:
    run = run_verify(fake_stack, *("--command-timeout", "8"),
                     env={"FAKE_MODE": "silent-stats"})

    assert run.returncode == 1
    assert run.status == "fail"
    assert run.report["checks"]["dispatch_stats"]["ok"] is False


def test_refuses_unpinned_hermes(fake_stack: dict) -> None:
    run = run_verify(fake_stack, env={"FAKE_HERMES_VERSION": "hermes-agent 0.22.4 (scripted)"})

    assert run.returncode == 2
    assert run.status == "refused"
    assert run.report["checks"]["hermes_pin"]["ok"] is False
    assert run.report["owned_home"]["path"] is not None
    assert run.report["owned_home"]["removed"] is True
    assert run.report["provider"]["inference_attempts"] == 0, (
        "refused runs keep the same provider summary shape as completed runs")
    assert run.report["provider"]["base_url"] is None
    assert not (run.evidence_dir / "provider").exists()


def test_refuses_wrong_engine_version(fake_stack: dict) -> None:
    pkg = fake_stack["engine_bin"].resolve().parent
    (pkg / "package.json").write_text(json.dumps({"version": "1.0.168"}), encoding="utf-8")
    run = run_verify(fake_stack)

    assert run.returncode == 2
    assert run.status == "refused"
    assert run.report["checks"]["engine_pin"]["ok"] is False
    assert "1.0.168" in run.report["checks"]["engine_pin"]["detail"]


def test_refuses_non_empty_evidence_dir(fake_stack: dict) -> None:
    evidence = fake_stack["evidence_root"] / "occupied"
    evidence.mkdir(parents=True)
    (evidence / "stale.txt").write_text("old", encoding="utf-8")
    cmd = [
        sys.executable, "-m", VERIFY_MODULE,
        "--hermes", str(fake_stack["hermes"]),
        "--engine-bin", str(fake_stack["engine_bin"]),
        "--evidence-dir", str(evidence),
        "--profile-root", str(fake_stack["profile_root"]),
    ]
    proc = subprocess.Popen(cmd, cwd=str(REPO_ROOT),
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    run = VerifyRun(proc, evidence)

    assert run.returncode == 2
    assert (evidence / "stale.txt").read_text(encoding="utf-8") == "old"


def test_interrupt_cleans_only_owned_processes(fake_stack: dict) -> None:
    decoy_pidfile = fake_stack["work"] / "decoy.pid"
    decoy_home = fake_stack["work"] / "decoy-home"
    decoy_home.mkdir()
    evidence = fake_stack["evidence_root"] / f"ev{time.monotonic_ns()}"
    cmd = [
        sys.executable, "-m", VERIFY_MODULE,
        "--hermes", str(fake_stack["hermes"]),
        "--engine-bin", str(fake_stack["engine_bin"]),
        "--evidence-dir", str(evidence),
        "--profile-root", str(fake_stack["profile_root"]),
        "--command-timeout", "600",
    ]
    env = {k: v for k, v in os.environ.items() if k != "PYTEST_CURRENT_TEST"}
    env.update({"FAKE_MODE": "slow-first-command",
                "FAKE_DECOY_PIDFILE": str(decoy_pidfile),
                "FAKE_DECOY_HOME": str(decoy_home)})
    proc = subprocess.Popen(cmd, cwd=str(REPO_ROOT), env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        # Wait until the decoy (started by the owned REPL) and the transcript exist.
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            if decoy_pidfile.exists() and (evidence / "transcript" / "hermes-chat.log").exists():
                break
            assert proc.poll() is None, f"smoke exited early: {proc.returncode}"
            time.sleep(0.25)
        decoy_pid = int(decoy_pidfile.read_text(encoding="utf-8").strip())

        proc.send_signal(signal.SIGINT)
        stdout, stderr = proc.communicate(timeout=120)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.communicate(timeout=30)

    assert proc.returncode == 130, (stdout, stderr)
    report = json.loads((evidence / "verify-report.json").read_text(encoding="utf-8"))
    assert report["status"] == "interrupted"
    assert report["default_profile"]["unchanged"] is True
    assert (evidence / "transcript" / "hermes-chat.log").exists(), \
        "interruption must retain partial evidence"
    signals = report["processes"]["signals"]
    assert signals, "interruption cleanup must record what it signaled"
    for entry in signals:
        assert entry["identity"] == "owned"
    assert Path(f"/proc/{decoy_pid}/stat").exists(), \
        "a similar-looking process in a different profile must never be killed"
    try:
        os.kill(decoy_pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def test_unresponsive_owned_child_escalates_bounded(fake_stack: dict) -> None:
    run = run_verify(fake_stack, "--command-timeout", "8", "--exit-timeout", "10",
                     env={"FAKE_MODE": "stuck"})

    assert run.returncode == 1
    assert run.status == "fail"
    signals = run.report["processes"]["signals"]
    assert any(s["signal"] == "SIGKILL" for s in signals), signals
    assert run.report["checks"]["no_owned_survivors"]["ok"]
    assert run.report["processes"]["swept_clean"] is True
    for entry in signals:
        assert entry["identity"] == "owned"


def test_ambient_hermes_home_never_reaches_the_child(fake_stack: dict) -> None:
    run = run_verify(fake_stack)

    assert run.returncode == 0
    proofs = run.report["processes"]["identity_proofs"]
    assert proofs, "the smoke must record a live identity proof for the owned child"
    for proof in proofs:
        assert proof["hermes_home"] == run.report["owned_home"]["path"]
        assert proof["hermes_home"] != str(fake_stack["default_home"])
    assert "CONTEXT_MODE_MCP_SENTINEL_DIR" in run.report["env"]["scrubbed"], (
        "an ambient sentinel dir must never reach a smoke child")
    assert (fake_stack["default_home"] / "config.yaml").read_text(encoding="utf-8") == "model: untouched\n"


def test_relative_executable_args_canonicalize_once_before_any_child(fake_stack: dict, tmp_path: Path) -> None:
    """Public verify run from a NON-repo cwd with RELATIVE --hermes and
    --engine-bin: both canonicalize once against the operator's cwd before any
    child is launched, so the installer child (cwd = repo root) resolves the
    same engine binary, the stored engine_binary matches, and the exact pin
    checks hold. Asserted on the independently retrieved report body — the
    engine symlink resolves to the real CLI, as the pinned nested layout
    checks expect."""
    cwd = tmp_path / "operator-cwd"
    cwd.mkdir()
    evidence = fake_stack["evidence_root"] / f"ev{time.monotonic_ns()}"
    cmd = [
        sys.executable, "-m", VERIFY_MODULE,
        "--hermes", os.path.relpath(fake_stack["hermes"], cwd),
        "--engine-bin", os.path.relpath(fake_stack["engine_bin"], cwd),
        "--evidence-dir", str(evidence),
        "--profile-root", str(fake_stack["profile_root"]),
    ]
    merged = {k: v for k, v in os.environ.items() if k != "PYTEST_CURRENT_TEST"}
    proc = subprocess.Popen(cmd, cwd=str(cwd), env=merged,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    run = VerifyRun(proc, evidence)

    assert run.returncode == 0, run.stderr
    assert run.status == "pass"
    assert run.report["checks"]["engine_pin"]["ok"]
    assert run.report["checks"]["installer_configured_owned_profile"]["ok"]
    assert Path(run.report["engine_bin"]) == fake_stack["engine_bin"].resolve()
    assert Path(run.report["hermes"]) == fake_stack["hermes"]


@pytest.mark.parametrize("observation", ["exec_unreadable", "foreign_profile"])
def test_initial_proc_identity_observation_is_safe(fake_stack, monkeypatch, observation):
    from context_mode_hermes import verify

    original = verify._proc_environ
    first = True

    def initial_observation(pid):
        nonlocal first
        if first:
            first = False
            if observation == "exec_unreadable":
                return None
            return {"HERMES_HOME": str(fake_stack["work"] / "foreign-profile")}
        return original(pid)

    monkeypatch.setattr(verify, "_proc_environ", initial_observation)
    evidence = fake_stack["evidence_root"] / f"identity-{observation}"
    result = verify.main([
        "--hermes", str(fake_stack["hermes"]),
        "--engine-bin", str(fake_stack["engine_bin"]),
        "--evidence-dir", str(evidence),
        "--profile-root", str(fake_stack["profile_root"]),
        "--command-timeout", "8",
    ])
    report = json.loads((evidence / "verify-report.json").read_text())
    if observation == "exec_unreadable":
        assert result == 0, report
        assert report["checks"]["mcp_stored_content"]["ok"]
    else:
        assert result == 2, report
        assert report["dispatches"] == []
        assert not report["checks"]["owned_profile_isolated"]["ok"]
    assert report["checks"]["default_profile_unchanged"]["ok"]
    assert report["checks"]["no_owned_survivors"]["ok"]
