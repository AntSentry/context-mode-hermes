"""Safe, operator-run verification smoke for the context-mode Hermes bridge.

This is a *diagnostic*, not a benchmark and not a model-quality claim: it
proves, on a verification-owned temporary Hermes profile, that the pinned
interactive Hermes CLI (``hermes chat --cli`` — never one-shot ``-q``) dispatches
the real ``/ctx-*`` slash commands through the pinned context-mode MCP stack,
that indexing independent content and searching it back works, and that all of
this happens with **zero model inference**. A loopback provider on 127.0.0.1
records every raw request (credentials redacted) and rejects any
``/v1/chat/completions`` call; an inference attempt is a *failure* with retained
evidence, never a successful output.

Safety envelope (enforced here, regression-tested in
``tests/test_verify_contract.py``):

- the smoke runs only against an owned temporary profile (fresh ``mkdtemp``),
  with the ambient ``HERMES_HOME`` scrubbed from every child it spawns;
- the default profile (the ambient ``HERMES_HOME``, else ``~/.hermes``) is
  fingerprinted before and after and must stay identical;
- cleanup only ever signals *identity-verified* owned resources: tracked child
  handles, that child's process group, and sweep hits whose own
  ``/proc/<pid>/environ`` carries ``HERMES_HOME=<owned home>`` (re-verified
  against the process start time before any escalation signal). Nothing is ever
  matched or killed by process name, command-line pattern or a bare PID read
  from a file, so similar-looking gateways and workers in other profiles always
  survive;
- SIGINT/SIGTERM interrupt the smoke: owned children are cleaned up in bounded
  time, partial evidence is retained, and the run exits 130/143 — interruption
  is never reported as success;
- every raw provider request is retained under the evidence directory as it
  arrives (interrupt-safe), with ``Authorization``-style headers and the
  generated API key redacted; no environment dump is ever taken.

Prerequisites: the pinned Hermes v0.22.5 CLI (``--hermes``), the pinned
context-mode 1.0.169 engine (``--engine-bin``) and a Python that can import both
this package and Hermes's own ``hermes_cli`` for the installer subprocess.

Exit codes: ``0`` pass, ``1`` fail, ``2`` refused (preflight), ``130``/``143``
interrupted. The final stdout line is ``CM-VERIFY: <STATUS> evidence=<dir>``.

``CM_VERIFY_EXTRA_PYTHONPATH`` (os.pathsep-separated) is appended to every
child's ``PYTHONPATH``; the tests use it to inject a stub Hermes API.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import pty
import re
import select
import shutil
import signal
import struct
import subprocess
import sys
import tempfile
import termios
import threading
import time
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .engine import Engine
from .install import ENGINE_VERSION

HERMES_PIN = "0.22.5"
MODEL_NAME = "context-mode-verify/model"
MODEL_CONTEXT = 65_536
NO_INFERENCE_MESSAGE = (
    "verification loopback provider rejects inference: "
    "no model calls are permitted in the safe smoke"
)
REPORT_SCHEMA = "context-mode-hermes/verify-report/1"

#: Environment keys never inherited by any smoke child; the owned profile wins.
SCRUB_KEYS = (
    "HERMES_HOME", "CLAUDE_CONFIG_DIR", "CLAUDE_PROJECT_DIR", "CLAUDE_SESSION_ID",
    "CLAUDE_PLUGIN_ROOT", "CLAUDE_CODE_ENTRYPOINT", "CONTEXT_MODE_DIR",
    "CONTEXT_MODE_BIN", "CONTEXT_MODE_PLATFORM", "XDG_CONFIG_HOME", "OMP_PLUGIN_ROOT",
    "PI_CODING_AGENT_DIR", "PI_PROJECT_DIR",
)

REPO_ROOT = Path(__file__).resolve().parent.parent

ANSI_RE = re.compile(
    rb"\x1b\[[0-9;?]*[A-Za-z]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)|\x1b[()][A-Z0-9]|\x1b."
)
TELEMETRY_OFF_RE = re.compile(r"enabled['\"]?\s*:\s*['\"]?false")

_EXIT_CODES = {"pass": 0, "fail": 1, "refused": 2, "interrupted": 130}


class Refused(Exception):
    """Preflight refusal: nothing was launched, nothing will be cleaned up."""


class Interrupted(Exception):
    def __init__(self, signum: int) -> None:
        super().__init__(f"interrupted by signal {signum}")
        self.signum = signum


class _InterruptFlag:
    """Signal tripline checked by every wait loop; raised as Interrupted."""

    def __init__(self) -> None:
        self.signum: int | None = None

    def trip(self, signum: int) -> None:
        if self.signum is None:
            self.signum = signum

    def check(self) -> None:
        if self.signum is not None:
            raise Interrupted(self.signum)


# ─── Credential redaction ──────────────────────────────────────────────────────

_REDACTED_HEADERS = {"authorization", "x-api-key", "proxy-authorization", "cookie"}


def _redact(value: str, secrets: list[str]) -> str:
    for secret in secrets:
        if secret:
            value = value.replace(secret, "REDACTED")
    return value


# ─── Loopback provider: records raw requests, rejects inference ────────────────

class LoopbackProvider:
    """Verification-owned 127.0.0.1 provider: complete raw request logs, no inference."""

    def __init__(self, capture_dir: Path, api_key: str) -> None:
        self._capture_dir = capture_dir
        self._capture_dir.mkdir(parents=True, exist_ok=True)
        self._secrets = [api_key]
        self._lock = threading.Lock()
        self._seq = 0
        self.records: list[dict] = []
        provider = self

        class Handler(BaseHTTPRequestHandler):
            def _read_body(self) -> bytes:
                length = int(self.headers.get("Content-Length") or 0)
                return self.rfile.read(length) if length else b""

            def _respond(self, method: str, payload: bytes, status: int) -> None:
                path = self.path
                body = self._read_body()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("Connection", "close")
                self.end_headers()
                try:
                    self.wfile.write(payload)
                except (BrokenPipeError, ConnectionResetError):
                    pass
                provider._write_record(method, path, body, status, payload, dict(self.headers))

            def do_GET(self) -> None:  # noqa: N802 — http.server API
                if self.path.split("?")[0] == "/v1/models":
                    self._respond("GET", json.dumps({
                        "object": "list",
                        "data": [{
                            "id": MODEL_NAME, "object": "model",
                            "context_length": MODEL_CONTEXT,
                        }],
                    }).encode("utf-8"), 200)
                else:
                    self._respond("GET", json.dumps({"error": "not found"}).encode("utf-8"), 404)

            def do_POST(self) -> None:  # noqa: N802 — http.server API
                if self.path.split("?")[0] == "/v1/chat/completions":
                    self._respond("POST", json.dumps({
                        "error": {"message": NO_INFERENCE_MESSAGE, "type": "no_inference"},
                    }).encode("utf-8"), 503)
                else:
                    self._respond("POST", json.dumps({"error": "not found"}).encode("utf-8"), 404)

            def log_message(self, *args: object) -> None:  # silence stderr
                return

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._server.daemon_threads = True
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self._server.server_address[1]}/v1"

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()

    def _write_record(self, method: str, path: str, body: bytes, status: int,
                      response: bytes, headers: dict) -> None:
        with self._lock:
            self._seq += 1
            seq = self._seq
        redacted_headers = {
            key: ("REDACTED" if key.lower() in _REDACTED_HEADERS
                  else _redact(value, self._secrets))
            for key, value in headers.items()
        }
        record = {
            "seq": seq,
            "time": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
            "method": method,
            "path": path.split("?")[0],
            "query": path.split("?")[1] if "?" in path else "",
            "headers": redacted_headers,
            "body": _redact(body.decode("utf-8", "replace"), self._secrets),
            "response_status": status,
            "response_body": response.decode("utf-8", "replace"),
        }
        # Retained as it arrives so an interrupted run still keeps its evidence.
        (self._capture_dir / f"req-{seq:04d}.json").write_text(
            json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
        with self._lock:
            self.records.append(record)

    def summarize(self) -> dict:
        return {
            "base_url": self.base_url,
            "requests_total": len(self.records),
            "inference_requests": sum(
                1 for r in self.records
                if r["method"] == "POST" and r["path"] == "/v1/chat/completions"),
        }


# ─── Process identity: /proc-backed, never name-based ─────────────────────────

def _proc_environ(pid: int) -> dict[str, str] | None:
    try:
        raw = Path(f"/proc/{pid}/environ").read_bytes()
    except OSError:
        return None
    env = {}
    for entry in raw.split(b"\0"):
        if b"=" in entry:
            key, _, value = entry.partition(b"=")
            env[key.decode("utf-8", "replace")] = value.decode("utf-8", "replace")
    return env


def _proc_cmdline(pid: int) -> str:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return ""
    return " ".join(p for p in raw.decode("utf-8", "replace").split("\0") if p)[:400]


def _proc_starttime(pid: int) -> str | None:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    try:  # field 22 (starttime) after the parenthesized comm
        return stat[stat.rindex(")") + 2:].split()[19]
    except (ValueError, IndexError):
        return None


def owned_hermes_pids(owned_home: Path, exclude: set[int]) -> list[dict]:
    """Live processes whose own environment says they run under *owned_home*."""
    marker = str(owned_home)
    hits = []
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        pid = int(entry)
        if pid in exclude:
            continue
        env = _proc_environ(pid)
        if env is None or env.get("HERMES_HOME") != marker:
            continue
        hits.append({
            "pid": pid,
            "hermes_home": env.get("HERMES_HOME"),
            "cmdline": _proc_cmdline(pid),
            "starttime": _proc_starttime(pid),
        })
    return hits


class ProcessLedger:
    """Records every signal sent and the identity proof that justified it."""

    def __init__(self) -> None:
        self.signals: list[dict] = []
        self.identity_proofs: list[dict] = []
        self.strays: list[dict] = []

    def note_proof(self, pid: int, env: dict[str, str] | None, cmdline: str) -> None:
        self.identity_proofs.append({
            "pid": pid,
            "hermes_home": (env or {}).get("HERMES_HOME"),
            "home": (env or {}).get("HOME"),
            "cmdline": cmdline,
            "verified": bool(env) and env.get("HERMES_HOME") is not None,
        })

    def _record(self, target: int, sig: int, scope: str, proof: dict) -> None:
        try:
            if scope == "process-group":
                os.killpg(target, sig)
            else:
                os.kill(target, sig)
        except (ProcessLookupError, PermissionError) as exc:
            outcome = f"not-delivered ({type(exc).__name__})"
        else:
            outcome = "delivered"
        self.signals.append({
            "target": target, "scope": scope,
            "signal": signal.Signals(sig).name, "identity": "owned", "outcome": outcome,
            "proof": {k: proof.get(k) for k in ("hermes_home", "cmdline", "starttime")},
            "time": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
        })

    def signal_child_group(self, proc: subprocess.Popen, sig: int) -> None:
        """Signal the tracked child's whole process group while the handle is live."""
        if proc.poll() is not None:
            return  # reaped handles can never go stale
        try:
            pgid = os.getpgid(proc.pid)
        except ProcessLookupError:
            return
        if pgid != proc.pid:
            # Not a group leader (unexpected: we spawn with start_new_session).
            self._record(proc.pid, sig, "process", {
                "hermes_home": "tracked-child-handle",
                "cmdline": _proc_cmdline(proc.pid), "starttime": _proc_starttime(proc.pid)})
            return
        self._record(pgid, sig, "process-group", {
            "hermes_home": "tracked-child-handle (spawned with the owned profile)",
            "cmdline": _proc_cmdline(proc.pid), "starttime": _proc_starttime(proc.pid)})

    def signal_owned_pid(self, pid: int, sig: int, proof: dict) -> None:
        """Signal a swept PID only while its start time still matches the proof."""
        if _proc_starttime(pid) != proof.get("starttime"):
            return  # PID was reaped and reused between discovery and the signal
        self._record(pid, sig, "process", proof)


def stop_child(proc: subprocess.Popen, ledger: ProcessLedger, exit_timeout: float) -> None:
    """Bounded shutdown of an owned child: group TERM, wait, group KILL, reap."""
    if proc.poll() is None:
        ledger.signal_child_group(proc, signal.SIGTERM)
        try:
            proc.wait(timeout=exit_timeout)
        except subprocess.TimeoutExpired:
            ledger.signal_child_group(proc, signal.SIGKILL)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass


def sweep_owned(owned_home: Path, ledger: ProcessLedger,
                interrupt: _InterruptFlag) -> bool:
    """Terminate any process still carrying the owned HERMES_HOME identity."""
    me = {os.getpid()}
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        interrupt.check()
        hits = owned_hermes_pids(owned_home, exclude=me)
        if not hits:
            return True
        for hit in hits:
            ledger.strays.append(hit)
            ledger.signal_owned_pid(hit["pid"], signal.SIGTERM, hit)
        end = time.monotonic() + 5
        while time.monotonic() < end and owned_hermes_pids(owned_home, exclude=me):
            time.sleep(0.2)
    for hit in owned_hermes_pids(owned_home, exclude=me):
        ledger.signal_owned_pid(hit["pid"], signal.SIGKILL, hit)
    time.sleep(1)
    return not owned_hermes_pids(owned_home, exclude=me)


# ─── PTY-driven interactive Hermes dispatch ────────────────────────────────────

def normalize_terminal_output(raw: bytes) -> str:
    text = ANSI_RE.sub(b"", raw).decode("utf-8", "replace")
    return re.sub(r"\s+", " ", text)


class PtyChatDriver:
    """Drives the real interactive ``hermes chat --cli`` over a pseudo-terminal.

    The raw byte stream (ANSI included) is retained verbatim as evidence;
    marker matching runs on a whitespace-normalized, escape-stripped view so
    terminal wrapping and repaints cannot split a marker.
    """

    _KEEP_TAIL = 1 << 20

    def __init__(self, cmd: list[str], env: dict[str, str], cwd: Path,
                 transcript_path: Path) -> None:
        transcript_path.parent.mkdir(parents=True, exist_ok=True)
        self._transcript = open(transcript_path, "wb")
        self._buffer = b""
        self._trimmed = 0
        master, slave = pty.openpty()
        fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", 40, 200, 0, 0))
        self.proc = subprocess.Popen(
            cmd, stdin=slave, stdout=slave, stderr=slave,
            env=env, cwd=str(cwd), start_new_session=True, close_fds=True)
        os.close(slave)
        os.set_blocking(master, False)
        self._master = master

    def _pump(self, wait: float) -> None:
        ready, _, _ = select.select([self._master], [], [], wait)
        if not ready:
            return
        try:
            chunk = os.read(self._master, 65536)
        except (BlockingIOError, OSError):
            return
        if chunk:
            self._buffer += chunk
            self._transcript.write(chunk)
            self._transcript.flush()
            if len(self._buffer) > 2 * self._KEEP_TAIL:
                cut = len(self._buffer) - self._KEEP_TAIL
                self._buffer = self._buffer[cut:]
                self._trimmed += cut

    def send(self, line: str) -> None:
        os.write(self._master, line.encode("utf-8") + b"\r")

    def mark(self) -> int:
        """Transcript position (trim-adjusted) to scope a later ``wait_for``."""
        return self._trimmed + len(self._buffer)

    def _view(self, since: int) -> bytes:
        return self._buffer[max(0, since - self._trimmed):]

    def wait_for(self, marker: str, timeout: float, interrupt: _InterruptFlag,
                 since: int = 0) -> tuple[bool, float]:
        """Wait for *marker* in output emitted after ``since`` (an echoed older
        command can never satisfy a later dispatch's marker)."""
        deadline = time.monotonic() + timeout
        started = time.monotonic()
        while time.monotonic() < deadline:
            interrupt.check()
            self._pump(0.25)
            if marker in normalize_terminal_output(self._view(since)):
                return True, time.monotonic() - started
            if self.proc.poll() is not None:
                drain_until = time.monotonic() + 2  # child gone: drain, then fail fast
                while time.monotonic() < drain_until:
                    self._pump(0.2)
                return (marker in normalize_terminal_output(self._view(since)),
                        time.monotonic() - started)
        return False, time.monotonic() - started

    def close(self) -> None:
        try:
            os.close(self._master)
        except OSError:
            pass
        self._transcript.close()


# ─── Profile helpers ───────────────────────────────────────────────────────────

def fingerprint_default_profile(default_root: Path) -> dict:
    """Read-only fingerprint of the profile the operator actually uses."""
    if not default_root.exists():
        return {"root": str(default_root), "exists": False}
    entries = sorted(p.name for p in default_root.iterdir())
    config = default_root / "config.yaml"
    config_digest = hashlib.sha256(config.read_bytes()).hexdigest() if config.exists() else None
    return {"root": str(default_root), "exists": True, "entries": entries,
            "config_sha256": config_digest}


def child_environment(owned_home: Path) -> dict[str, str]:
    """Child env: ambient profile keys scrubbed, owned profile and this checkout first."""
    env = {k: v for k, v in os.environ.items() if k not in SCRUB_KEYS}
    extra = [p for p in os.environ.get("CM_VERIFY_EXTRA_PYTHONPATH", "").split(os.pathsep) if p]
    inherited = env.get("PYTHONPATH", "").split(os.pathsep) if env.get("PYTHONPATH") else []
    env.update({
        "HERMES_HOME": str(owned_home),
        "HOME": str(owned_home),
        "PYTHONPATH": os.pathsep.join([str(REPO_ROOT), *extra, *inherited]),
    })
    return env


def write_owned_config(owned_home: Path, provider: LoopbackProvider,
                       engine_bin: Path, api_key: str) -> None:
    """Pre-seed the owned profile before the installer merges its own wiring.

    JSON is valid YAML, so Hermes's own reader round-trips this file; the
    installer subprocess then rewrites it through Hermes's atomic config writer.
    The MCP server environment pins the bridge to *this* checkout (PYTHONPATH)
    and gives engine workers an owned HOME, so nothing ambient leaks in.
    """
    host_root = owned_home / "context-mode-host"
    config = {
        "model": {
            "provider": "custom",
            "base_url": provider.base_url,
            "api_key": api_key,
            "default": MODEL_NAME,
        },
        "telemetry": {"shared_metrics": {"enabled": False, "send": False}},
        "mcp_servers": {
            "context-mode": {
                "command": sys.executable,
                "args": ["-m", "context_mode_hermes.mcp_proxy", "--engine-bin", str(engine_bin)],
                "env": {
                    "HERMES_HOME": str(owned_home),
                    "CONTEXT_MODE_DIR": str(owned_home / "context-mode"),
                    "CLAUDE_CONFIG_DIR": str(host_root),
                    "XDG_CONFIG_HOME": str(host_root / "xdg-config"),
                    "CONTEXT_MODE_PLATFORM": "claude-code",
                    "CLAUDE_CODE_ENTRYPOINT": "hermes-context-mode-bridge",
                    "HOME": str(owned_home),
                    "PATH": os.environ.get("PATH", ""),
                    "PYTHONPATH": str(REPO_ROOT),
                },
                "timeout": 300,
                "connect_timeout": 60,
            },
        },
    }
    owned_home.joinpath("config.yaml").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")


# ─── Orchestration ─────────────────────────────────────────────────────────────

class Smoke:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.evidence = args.evidence_dir.expanduser().resolve()
        self.interrupt = _InterruptFlag()
        self.ledger = ProcessLedger()
        self.checks: dict[str, dict] = {}
        self.dispatches: list[dict] = []
        self.owned_home: Path | None = None
        self.default_root: Path | None = None
        self.default_before: dict | None = None
        self.default_after: dict | None = None
        self.provider: LoopbackProvider | None = None
        self.api_key = f"cm-verify-{uuid.uuid4().hex}"
        self.nonce = f"CM-VERIFY-NONCE-{uuid.uuid4().hex}"
        self.status = "fail"
        self.error: str | None = None
        self._interrupted: int | None = None

    # -- report plumbing ------------------------------------------------------

    def check(self, name: str, ok: bool, detail: str) -> bool:
        self.checks[name] = {"ok": bool(ok), "detail": detail}
        return bool(ok)

    def report(self) -> dict:
        provider_summary = (self.provider.summarize() if self.provider else {
            "base_url": None, "requests_total": 0, "inference_requests": 0})
        return {
            "schema": REPORT_SCHEMA,
            "status": self.status,
            "hermes": str(self.args.hermes) if self.args.hermes else None,
            "engine_bin": str(self.args.engine_bin) if self.args.engine_bin else None,
            "owned_home": {
                "path": str(self.owned_home) if self.owned_home else None,
                "removed": bool(self.owned_home) and not self.owned_home.exists(),
            },
            "evidence_dir": str(self.evidence),
            "checks": self.checks,
            "dispatches": self.dispatches,
            "provider": {**provider_summary,
                         "records_dir": str(self.evidence / "provider")},
            "processes": {
                "signals": self.ledger.signals,
                "identity_proofs": self.ledger.identity_proofs,
                "strays": self.ledger.strays,
                "swept_clean": self.checks.get("no_owned_survivors", {}).get("ok", False),
            },
            "default_profile": {
                "root": str(self.default_root) if self.default_root else None,
                "before": self.default_before,
                "after": self.default_after,
                "unchanged": self.checks.get("default_profile_unchanged", {}).get("ok", False),
            },
            "env": {
                "scrubbed": list(SCRUB_KEYS),
                "set": {"HERMES_HOME": "owned temporary profile",
                        "HOME": "owned temporary profile",
                        "PYTHONPATH": "this checkout first"},
            },
            "error": self.error,
            "interrupted": self._interrupted is not None,
        }

    def write_report(self) -> None:
        self.evidence.mkdir(parents=True, exist_ok=True)
        (self.evidence / "verify-report.json").write_text(
            json.dumps(self.report(), indent=2, ensure_ascii=False), encoding="utf-8")

    # -- phases ----------------------------------------------------------------

    def preflight(self) -> None:
        hermes = self.args.hermes
        if not hermes:
            raise Refused("--hermes is required: point it at the pinned Hermes "
                          f"v{HERMES_PIN} CLI")
        hermes = Path(str(hermes)).expanduser()
        if not (hermes.is_file() and os.access(hermes, os.X_OK)):
            raise Refused(f"--hermes is not an executable file: {hermes}")
        self.args.hermes = hermes
        try:
            version = subprocess.run([str(hermes), "--version"],
                                     capture_output=True, text=True, timeout=120)
        except (OSError, subprocess.SubprocessError) as exc:
            raise Refused(f"cannot execute {hermes} --version: {exc}") from None
        lines = (version.stdout or version.stderr).strip().splitlines()
        version_line = lines[0] if lines else ""
        if not self.check("hermes_pin", HERMES_PIN in version_line,
                          f"required Hermes v{HERMES_PIN}; --version said: {version_line!r}"):
            raise Refused("Hermes version is not the pinned one; refusing to launch anything")
        if not self.args.engine_bin:
            raise Refused("--engine-bin is required: point it at the pinned context-mode binary")
        engine_bin = Path(str(self.args.engine_bin)).expanduser()
        engine = Engine(binary=str(engine_bin))
        found = getattr(engine, "version", "unknown")
        if not self.check("engine_pin", engine.available and found == ENGINE_VERSION,
                          f"required context-mode {ENGINE_VERSION}; found {found} at {engine_bin}"):
            raise Refused(f"engine is not context-mode {ENGINE_VERSION} (found {found})")
        self.args.engine_bin = engine_bin

    def create_owned_profile(self) -> None:
        self.default_root = Path(os.environ.get(
            "HERMES_HOME", str(Path.home() / ".hermes"))).expanduser()
        self.default_before = fingerprint_default_profile(self.default_root)
        profile_root = self.args.profile_root.expanduser()
        profile_root.mkdir(parents=True, exist_ok=True)
        self.owned_home = Path(tempfile.mkdtemp(prefix="hermes-verify-", dir=str(profile_root)))
        os.chmod(self.owned_home, 0o700)
        (self.owned_home / ".verify-owned").write_text(json.dumps({
            "run": "context-mode-hermes safe verification smoke",
            "uuid": uuid.uuid4().hex,
            "created": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
            "parent_pid": os.getpid(),
        }, indent=2), encoding="utf-8")
        self.check("owned_profile_isolated", True,
                   f"fresh mkdtemp profile {self.owned_home}; the ambient HERMES_HOME "
                   "is scrubbed from every child")

    def run_installer(self, env: dict[str, str]) -> None:
        proc = subprocess.Popen(
            [sys.executable, "-m", "context_mode_hermes.install",
             "--home", str(self.owned_home), "--engine-bin", str(self.args.engine_bin)],
            cwd=str(REPO_ROOT), env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, start_new_session=True)
        try:
            stdout, stderr = proc.communicate(timeout=self.args.command_timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout, stderr = proc.communicate(timeout=10)
            stderr = (stderr or "") + \
                f"\ninstaller exceeded {self.args.command_timeout}s and was killed"
        (self.evidence / "installer.json").write_text(json.dumps({
            "returncode": proc.returncode, "stdout": stdout,
            "stderr": stderr[-8000:] if stderr else "",
        }, indent=2), encoding="utf-8")
        config_path = self.owned_home / "config.yaml"
        config_text = config_path.read_text(encoding="utf-8") if config_path.exists() else ""
        ok = proc.returncode == 0 and all([
            str(self.args.engine_bin) in config_text,
            "127.0.0.1" in config_text,
            str(REPO_ROOT) in config_text,
        ])
        detail = (f"installer rc={proc.returncode} against the owned profile; "
                  f"engine_binary={str(self.args.engine_bin) in config_text}, "
                  f"loopback base_url={'127.0.0.1' in config_text}, "
                  f"bridge PYTHONPATH={str(REPO_ROOT) in config_text}")
        if not ok and stderr:
            detail += f"; stderr tail: {stderr[-400:]}"
        self.check("installer_configured_owned_profile", ok, detail)
        self.check("telemetry_opted_out",
                   TELEMETRY_OFF_RE.search(config_text) is not None,
                   "owned profile config sets telemetry.shared_metrics.enabled=false")

    def dispatch_cycle(self, driver: PtyChatDriver) -> None:
        sentence = f"{self.nonce} the amber lighthouse catalogs quiet harbors before dawn"
        plan = [
            ("stats", "/ctx-stats", "v1.0.169"),
            ("doctor", "/ctx-doctor", "mcp_servers['context-mode'] configured"),
            ("index", f'/ctx-index --text "{sentence}" --source cm-verify-smoke',
             "from: cm-verify-smoke"),
            ("search", "/ctx-search amber lighthouse", self.nonce),
        ]
        for name, command, marker in plan:
            self.interrupt.check()
            position = driver.mark()
            driver.send(command)
            matched, elapsed = driver.wait_for(marker, self.args.command_timeout,
                                               self.interrupt, since=position)
            self.dispatches.append({
                "name": name, "command": command, "marker": marker,
                "matched": matched, "elapsed_s": round(elapsed, 2),
            })
            self.check(f"dispatch_{name}", matched,
                       f"interactive dispatch of {command!r} "
                       + ("echoed its marker" if matched else "timed out waiting for its marker")
                       + f" in {elapsed:.1f}s")

    def run(self) -> int:
        args = self.args
        for received in (signal.SIGINT, signal.SIGTERM):
            signal.signal(received, lambda s, frame, snum=received: self.interrupt.trip(snum))
        if self.evidence.exists() and any(self.evidence.iterdir()):
            raise Refused(f"evidence directory is not empty: {self.evidence}")
        self.evidence.mkdir(parents=True, exist_ok=True)

        try:
            self.preflight()
            self.create_owned_profile()
            env = child_environment(self.owned_home)
            workspace = self.owned_home / "workspace"
            workspace.mkdir(parents=True, exist_ok=True)

            self.provider = LoopbackProvider(self.evidence / "provider", self.api_key)
            self.provider.start()
            write_owned_config(self.owned_home, self.provider, args.engine_bin, self.api_key)
            self.run_installer(env)

            chat_env = dict(env)
            chat_env["TERM"] = "xterm-256color"
            driver = PtyChatDriver([str(args.hermes), "chat", "--cli"], chat_env,
                                   workspace, self.evidence / "transcript" / "hermes-chat.log")
            try:
                live_env = _proc_environ(driver.proc.pid)
                self.ledger.note_proof(driver.proc.pid, live_env,
                                       _proc_cmdline(driver.proc.pid))
                isolated = bool(live_env) and live_env.get("HERMES_HOME") == str(self.owned_home)
                self.check("owned_profile_isolated", isolated,
                           f"child pid {driver.proc.pid} runs with HERMES_HOME="
                           f"{(live_env or {}).get('HERMES_HOME')!r} "
                           f"(owned: {str(self.owned_home)!r}); refusal would fire before "
                           "any dispatch if this ever diverged")
                self.dispatch_cycle(driver)
                driver.send("/exit")
                try:
                    returncode = driver.proc.wait(timeout=args.exit_timeout)
                    self.check("child_exited", True,
                               f"interactive Hermes exited cleanly rc={returncode} after /exit")
                except subprocess.TimeoutExpired:
                    self.check("child_exited", False,
                               f"Hermes did not exit within {args.exit_timeout}s of /exit; "
                               "escalating bounded cleanup")
            finally:
                stop_child(driver.proc, self.ledger, args.exit_timeout)
                driver.close()
        except Refused as exc:
            self.status = "refused"
            self.error = str(exc)
        except Interrupted as exc:
            self.status = "interrupted"
            self.error = str(exc)
        except Exception as exc:  # a bug in the smoke itself is a failure, never a crash
            self.status = "fail"
            self.error = f"{type(exc).__name__}: {exc}"
            (self.evidence / "smoke-traceback.log").write_text(
                traceback.format_exc(), encoding="utf-8")
        finally:
            # Cleanup must always complete: capture the trip, then defuse it so
            # the already-received signal cannot abort cleanup midway.
            self._interrupted = self.interrupt.signum
            self.interrupt.signum = None
            self._cleanup()

        if (self.status == "fail" and self.error is None
                and self._interrupted is None and self.checks
                and all(c.get("ok") for c in self.checks.values())):
            self.status = "pass"
        self.write_report()
        code = _EXIT_CODES.get(self.status, 1)
        if self._interrupted == signal.SIGTERM:
            code = 143
        print(f"CM-VERIFY: {self.status.upper()} evidence={self.evidence}")
        if args.json:
            print(json.dumps(self.report(), indent=2, ensure_ascii=False))
        return code

    def _cleanup(self) -> None:
        """Bounded cleanup of owned resources; never touches foreign processes."""
        try:
            if self.provider is not None:
                self.provider.stop()
            swept = sweep_owned(self.owned_home, self.ledger, self.interrupt) \
                if self.owned_home else True
            self.check("no_owned_survivors", swept,
                       "no live process still carries the owned HERMES_HOME identity "
                       "(verified through /proc environ, never by name)")
            inference = self.provider.summarize()["inference_requests"] \
                if self.provider else 0
            self.check("no_inference", inference == 0,
                       f"loopback provider saw {inference} inference request(s); "
                       "the smoke must not call a model at all")
            self.default_after = fingerprint_default_profile(self.default_root) \
                if self.default_root else None
            unchanged = self.default_before == self.default_after
            self.check("default_profile_unchanged", unchanged,
                       f"default profile {self.default_root} fingerprint "
                       f"{'unchanged' if unchanged else 'CHANGED'} across the smoke")
            if not self.args.keep and self.owned_home and self.owned_home.exists():
                shutil.rmtree(self.owned_home, ignore_errors=True)
        except Exception as exc:  # cleanup must always finish and report
            self.error = self.error or f"cleanup error: {type(exc).__name__}: {exc}"
            if self.status not in ("interrupted", "refused"):
                self.status = "fail"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="context-mode-hermes-verify",
        description="Safe profile-isolated verification smoke for the context-mode "
                    "Hermes bridge (interactive CLI, loopback provider, zero inference).")
    parser.add_argument("--hermes", help="pinned Hermes v0.22.5 CLI executable")
    parser.add_argument("--engine-bin", help="pinned context-mode 1.0.169 executable")
    parser.add_argument("--evidence-dir", type=Path, default=None,
                        help="directory for the retained evidence bundle "
                             "(default: ./verify-evidence-<timestamp>-<id>)")
    parser.add_argument("--profile-root", type=Path, default=None,
                        help="parent directory for the owned temporary profile "
                             "(default: the system temporary directory)")
    parser.add_argument("--command-timeout", type=float, default=300.0,
                        help="seconds to wait for each interactive dispatch marker "
                             "(default: 300; also bounds the installer subprocess)")
    parser.add_argument("--exit-timeout", type=float, default=30.0,
                        help="seconds to wait for a clean exit before bounded "
                             "escalation (default: 30)")
    parser.add_argument("--keep", action="store_true",
                        help="keep the owned temporary profile for inspection "
                             "(evidence is always kept)")
    parser.add_argument("--json", action="store_true",
                        help="print the full JSON report to stdout as well")
    args = parser.parse_args(argv)
    if args.evidence_dir is None:
        args.evidence_dir = Path.cwd() / (
            f"verify-evidence-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}")
    if args.profile_root is None:
        args.profile_root = Path(tempfile.gettempdir())
    try:
        return Smoke(args).run()
    except Refused as exc:  # the evidence-dir refusal must not touch the occupied dir
        print(f"CM-VERIFY: REFUSED ({exc})", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
