"""Ticket #4 operator harness: exact GLM-5.3-Flash on local llama.cpp b11669 CPU.

Brings up ``unsloth/GLM-5.3-Flash-GGUD`` @ a38483c8… UD-IQ4_XS (5 shards,
156,822,111,200 bytes; every shard SHA256-verified against the HF tree API at
the pinned revision) with the pinned llama.cpp b11669 CPU build under a
``systemd-run --user`` scope capped at MemoryHigh=38G/MemoryMax=44G/
MemorySwapMax=0, loopback-only, mmap residency (no mlock), effective context
65536 with -np 1 (Hermes MINIMUM_CONTEXT_LENGTH floor is 64000). Native
Hermes then proves real tool-calling against the server through its custom
OpenAI-compatible provider. Everything writes structured evidence; teardown
signals only the owned llama-server PID (start-time + cmdline + cgroup
proof), never a process group.

Run:  python -m benchmarks.glm_local verify-model
      python -m benchmarks.glm_local prove --cycles 2
"""

from __future__ import annotations
from contextlib import suppress
import argparse
import hashlib
import json
import os
import re
import secrets
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# ── Pinned artifacts (authoritative; see docs/glm-5.3-flash-local-runtime.md) ──
GLM_REPO = "unsloth/GLM-5.3-Flash-GGUF"
GLM_REVISION = "a38483c8cd5df544f53d70fb281afe97369d5ab6"
MODEL_ALIAS = "GLM-5.3-Flash-UD-IQ4_XS"
SHARDS: tuple[tuple[str, int, str], ...] = (
    ("GLM-5.3-Flash-UD-IQ4_XS-00001-of-00005.gguf", 9_429_984,
     "c2f3b6ab98d3df844f322ea376059202d2eaeedcee1f0eff455345b86a137dbd"),
    ("GLM-5.3-Flash-UD-IQ4_XS-00002-of-00005.gguf", 49_989_334_176,
     "7d64cf0395672c4322012841abec502ea7e20518299bb5e3069003f06f9e6de9"),
    ("GLM-5.3-Flash-UD-IQ4_XS-00003-of-00005.gguf", 49_607_025_280,
     "7c2c63c9c30f8060428fdf2ac935dbf2ee9ad8f771d62b6ae47a7f7f2c1520e7"),
    ("GLM-5.3-Flash-UD-IQ4_XS-00004-of-00005.gguf", 49_486_530_144,
     "06c90f191871317c92dd9d25a353b687aed44c50f3ac6c713e9fe410bc2d26dd"),
    ("GLM-5.3-Flash-UD-IQ4_XS-00005-of-00005.gguf", 7_729_791_616,
     "66ebf9ec85e04d3f44af674ae4f694acbc89cc53db75e42f2f4d3646bf321c0d"),
)
TOTAL_BYTES = sum(size for _, size, _ in SHARDS)

ENGINE_PATH = Path("/home/a1234/.cache/hermes-context-mode/llama-b11669/"
                   "llama-b11669/llama-server")
ENGINE_SHA256 = ("3f2b53013c2eb596c8aca87e4740240a41f95bcebf72e61084d6d5ba0db761cc")
# The GitHub release tarball digest (what the fetch step verified).
ENGINE_ARCHIVE_SHA256 = ("cf07c1c112d709bd67e522017471de57ed0ec8bea031d732c5422351c07f65ac")
PERSISTENT_ROOT = Path("/home/a1234/.local/share/hermes-glm")
PERSISTENT_PROFILE = PERSISTENT_ROOT / "profile"
ENGINE_VERSION = "0.7.0-dev b11669 commit 620d2225c"
HERMES_BIN = Path("/home/a1234/.cache/hermes-context-mode/hermes-agent/.venv/bin/hermes")

DEFAULT_MODELS_DIR = Path("/home/a1234/.local/share/hermes-glm/models/"
                          "GLM-5.3-Flash-UD-IQ4_XS")
DEFAULT_EVIDENCE_DIR = Path("/home/a1234/.cache/hermes-context-mode/ticket-4-evidence")

SCOPE_UNIT = "hermes-glm-resource-probe"
CGROUP_PROPERTIES = ("MemoryHigh=38G", "MemoryMax=44G", "MemorySwapMax=0")
PORT = 18443  # deliberate: 18434 is the native supervisor's stable port
CTX = 65536  # Hermes MINIMUM_CONTEXT_LENGTH floor 64000 → 65536
THREADS = 4  # i5-6500 physical cores

# Readiness contract copied from the native supervisor: health-200 is NOT
# readiness — every claim needs a touch generation.
TOUCH_PROMPT = "Reply with exactly one word: the capital of France."
TOUCH_EXPECT = "paris"

TOOLPROOF_FILENAME = "harbor-notes.txt"


# ── Manifest verification ────────────────────────────────────────────────────

def _sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def _engine_identity() -> dict:
    identity = {"path": str(ENGINE_PATH), "sha256_expected": ENGINE_SHA256,
                "sha256_computed": None, "verified": False,
                "archive_sha256_pinned": ENGINE_ARCHIVE_SHA256,
                "version": ENGINE_VERSION}
    try:
        identity["sha256_computed"] = _sha256_file(ENGINE_PATH)
    except OSError as exc:
        identity["error"] = f"{type(exc).__name__}: {exc}"
    identity["verified"] = identity["sha256_computed"] == ENGINE_SHA256
    return identity


def verify_model(models_dir: Path, *, deep: bool = False) -> dict:
    """Check every shard's size; with ``deep`` also re-hash each shard."""
    report: dict = {"models_dir": str(models_dir), "deep": deep, "shards": [],
                    "ok": True}
    for name, want_size, want_sha in SHARDS:
        path = models_dir / name
        entry = {"name": name, "want_size": want_size}
        try:
            entry["size"] = path.stat().st_size
        except OSError as exc:
            entry.update(ok=False, error=f"{type(exc).__name__}: {exc}")
            report["ok"] = False
            report["shards"].append(entry)
            continue
        entry["size_ok"] = entry["size"] == want_size
        if deep:
            entry["sha256"] = _sha256_file(path)
            entry["sha256_ok"] = entry["sha256"] == want_sha
            entry["ok"] = entry["size_ok"] and entry["sha256_ok"]
        else:
            entry["ok"] = entry["size_ok"]
        report["ok"] = report["ok"] and entry["ok"]
        report["shards"].append(entry)
    on_disk = sum((models_dir / name).stat().st_size for name, _, _ in SHARDS
                  if (models_dir / name).exists())
    report["bytes_on_disk"] = on_disk
    report["total_ok"] = on_disk == TOTAL_BYTES
    report["ok"] = report["ok"] and report["total_ok"]
    return report


# ── Launch contract ──────────────────────────────────────────────────────────
def build_server_cmd(*, models_dir: Path, port: int, api_key: str,
                     prompts_dir: Path | None, log_path: Path,
                     unit: str = SCOPE_UNIT,
                     api_key_file: Path | None = None) -> list[str]:
    """The exact, shell-free argv that brings up the capped owned runtime.

    ``--log-prompts-dir`` (raw request capture) is opt-in: prove/debug only,
    never default production telemetry. ``api_key_file`` keeps the stable
    persistent key out of argv/logs (0600 file, native supervisor
    convention); otherwise a per-run key rides argv for the evidence log.
    """
    identity = _engine_identity()
    if not identity["verified"]:
        raise RuntimeError(f"Refusing unverified engine: {identity}")
    first_shard = models_dir / SHARDS[0][0]
    engine_argv = [
        str(ENGINE_PATH),
        "--host", "127.0.0.1",
        "--port", str(port),
        "-m", str(first_shard),
        "-a", MODEL_ALIAS,
        "-c", str(CTX),
        "-np", "1",
        "-t", str(THREADS),
        "--load-mode", "mmap",  # disk-backed residency: pageable, NOT mlock
        "--jinja",          # GGUF-embedded GLM template drives tool protocol
        "--metrics", "--slots",
        "--no-webui",
    ]
    if api_key_file is not None:
        engine_argv += ["--api-key-file", str(api_key_file)]
    else:
        engine_argv += ["--api-key", api_key]
    if prompts_dir is not None:
        engine_argv += ["--log-prompts-dir", str(prompts_dir)]
    return [
        "systemd-run", "--user", "--quiet", "--scope",
        f"--unit={unit}",
        *[prop for limit in CGROUP_PROPERTIES for prop in ("-p", limit)],
        *engine_argv,
    ]


# ── Timing log parsing (llama.cpp "timings" lines) ──────────────────────────

_TIMING_RE = re.compile(
    r"prompt eval time\s*=\s*(?P<pm>[\d.]+)\s*ms\s*/\s*(?P<pt>\d+)\s*tokens.*?"
    r"(?P<pps>[\d.]+)\s*tokens per second", re.IGNORECASE)
_EVAL_RE = re.compile(
    r"(?<!prompt )eval time\s*=\s*(?P<em>[\d.]+)\s*ms\s*/\s*(?P<et>\d+)\s*(?:runs|tokens).*?"
    r"(?P<eps>[\d.]+)\s*tokens per second", re.IGNORECASE)
# llama-server slot lines, keyed by (id, task) because "prompt processing"
# checkpoints fire repeatedly per request (progress 0.39, 0.83, 1.00 …):
#   I slot print_timing: id 0 | task 0 | prompt processing, n_tokens = 9,
#     progress = 0.39, t = 41.04 s / 0.22 tokens per second
#   I slot print_timing: id 0 | task 0 | eval, n_tokens = 12, t = 5.2 s / …
_SLOT_TIMING_RE = re.compile(
    r"print_timing:\s*id\s+(?P<id>\d+)\s*\|\s*task\s+(?P<task>-?\d+)\s*\|\s*"
    r"(?P<kind>prompt processing|eval|generation),"
    r"\s*n_tokens\s*=\s*(?P<tokens>\d+).*?t\s*=\s*(?P<seconds>[\d.]+)\s*s\s*/\s*"
    r"(?P<tps>[\d.]+)\s*tokens per second", re.IGNORECASE)


def parse_timings(log_text: str) -> list[dict]:
    """One record per request, pairing its prompt-eval with its eval line.

    Understands both the CLI format (``prompt eval time = … ms / N tokens``)
    and llama-server's per-slot lines (``prompt processing`` checkpoints,
    then ``eval``/``generation``), paired by slot id + task id.
    """
    records: list[dict] = []
    index: dict[tuple[str, str], dict] = {}
    current: dict | None = None
    for line in log_text.splitlines():
        slot = _SLOT_TIMING_RE.search(line)
        if slot:
            key = (slot["id"], slot["task"])
            record = index.get(key)
            if record is None:
                record = index[key] = {}
                records.append(record)
            kind = slot["kind"].lower()
            if kind == "prompt processing":
                # Later checkpoints supersede earlier partial ones.
                record.update(prompt_tokens=int(slot["tokens"]),
                              prompt_s=float(slot["seconds"]),
                              prompt_tps=float(slot["tps"]))
            else:
                record.update(eval_tokens=int(slot["tokens"]),
                              eval_s=float(slot["seconds"]),
                              eval_tps=float(slot["tps"]))
            continue
        if match := _TIMING_RE.search(line):
            current = {"prompt_ms": float(match["pm"]),
                       "prompt_tokens": int(match["pt"]),
                       "prompt_tps": float(match["pps"])}
            records.append(current)
        elif (match := _EVAL_RE.search(line)) and current is not None:
            current.update(eval_ms=float(match["em"]),
                           eval_tokens=int(match["et"]),
                           eval_tps=float(match["eps"]))
            current = None
    return records


# ── HTTP against the owned server ────────────────────────────────────────────

def _open(route: str, port: int, api_key: str, *, body: dict | None = None,
           timeout_s: float = 30.0):
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}{route}",
        headers={"Authorization": f"Bearer {api_key}"},
        data=json.dumps(body).encode() if body is not None else None)
    return urllib.request.urlopen(request, timeout=timeout_s)


def wait_health(port: int, api_key: str, deadline_s: float,
                *, stop_check=lambda: None) -> float:
    """Return seconds to first /health 200 (NOT readiness by itself)."""
    started = time.monotonic()
    deadline = started + deadline_s
    while time.monotonic() < deadline:
        stop_check()
        try:
            with _open("/health", port, api_key, timeout_s=3) as response:
                if response.status == 200:
                    return time.monotonic() - started
        except (urllib.error.URLError, OSError, TimeoutError):
            pass
        time.sleep(1.0)
    raise TimeoutError(f"no /health 200 after {deadline_s:.0f}s")


def list_models(port: int, api_key: str, timeout_s: float = 30.0) -> list[str]:
    with _open("/v1/models", port, api_key, timeout_s=timeout_s) as response:
        return [entry["id"] for entry in json.load(response).get("data", [])]


def generation(port: int, api_key: str, prompt: str, *, max_tokens: int,
               timeout_s: float) -> dict:
    """One chat completion; captures content + reasoning + wall time."""
    started = time.monotonic()
    body = {"model": MODEL_ALIAS, "max_tokens": max_tokens, "temperature": 0,
            "messages": [{"role": "user", "content": prompt}]}
    with _open("/v1/chat/completions", port, api_key, body=body,
               timeout_s=timeout_s) as response:
        payload = json.load(response)
    choice = (payload.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    usage = payload.get("usage") or {}
    return {"wall_s": time.monotonic() - started,
            "content": message.get("content") or "",
            "reasoning_content": message.get("reasoning_content") or "",
            "finish_reason": choice.get("finish_reason"),
            "usage": usage}


def wait_touch(port: int, api_key: str, deadline_s: float) -> dict:
    """Native readiness contract: temp-0 touch generation, reasoning scanned."""
    started = time.monotonic()
    deadline = time.monotonic() + deadline_s
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            result = generation(port, api_key, TOUCH_PROMPT, max_tokens=512,
                                timeout_s=min(deadline_s, 3600.0))
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            last_error = exc
            time.sleep(5.0)
            continue
        haystack = (result["content"] + result["reasoning_content"]).lower()
        if TOUCH_EXPECT in haystack:
            result["ready_s"] = time.monotonic() - started
            return result
        time.sleep(2.0)
    raise TimeoutError(f"touch generation never answered {TOUCH_EXPECT!r} "
                       f"within {deadline_s:.0f}s")


# ── Resource observation ─────────────────────────────────────────────────────

def proc_field(pid: int, filename: str) -> str:
    return Path(f"/proc/{pid}/{filename}").read_text(encoding="utf-8")


def sample_process(pid: int) -> dict:
    """I/O counters, current RSS (statm) and kernel peak RSS (VmHWM).

    VmHWM is the process's true high-water mark; statm is a snapshot.
    """
    io: dict[str, int] = {}
    for pair in proc_field(pid, "io").splitlines():
        key, _, value = pair.partition(":")
        if value.strip().isdigit():
            io[key.strip()] = int(value.strip())
    statm = [int(field) for field in proc_field(pid, "statm").split()]
    page = os.sysconf("SC_PAGE_SIZE")
    vmhwm = None
    for line in proc_field(pid, "status").splitlines():
        if line.startswith("VmHWM:"):
            with suppress(ValueError):
                vmhwm = int(line.split()[1]) * 1024
            break
    return {"read_bytes": io.get("read_bytes"), "write_bytes": io.get("write_bytes"),
            "rss_bytes": statm[1] * page, "virtual_bytes": statm[0] * page,
            "peak_rss_bytes": vmhwm}


def cgroup_sample(server_pid: int) -> dict:
    """Memory controller state of the scope the server actually runs in.

    The path is read from the live process's own cgroup file, so nesting
    under ``user@<uid>.service`` (or anywhere else) cannot break sampling.
    """
    sample: dict[str, str | None] = {"scope": None, "memory.current": None,
                                     "memory.peak": None,
                                     "memory.events.local": None}
    try:
        raw = proc_field(server_pid, "cgroup").strip().splitlines()[-1]
        relative = raw.split(":", 2)[-1].lstrip("/")
        base = Path("/sys/fs/cgroup") / relative
        sample["scope"] = relative
        for name in ("memory.current", "memory.peak", "memory.events.local"):
            try:
                sample[name] = (base / name).read_text(encoding="utf-8").strip()
            except OSError:
                sample[name] = None
    except (OSError, IndexError):
        pass
    return sample


def metrics_text(port: int, api_key: str) -> str:
    with _open("/metrics", port, api_key, timeout_s=15) as response:
        return response.read().decode("utf-8", "replace")


def llm_metrics(port: int, api_key: str) -> dict:
    """All non-comment metric rows verbatim (actual engine names are
    llamacpp:* — e.g. prompt_tokens_total, tokens_predicted_total,
    requests_processing). Callers retain the raw body as evidence."""
    found: dict[str, str] = {}
    for line in metrics_text(port, api_key).splitlines():
        if line.startswith("#") or not line.strip():
            continue
        parts = line.rsplit(" ", 1)
        if len(parts) == 2:
            found[parts[0]] = parts[1]
    return found


# ── Owned-process lifecycle (per-PID proof; never a process group) ──────────

@dataclass
class ServerState:
    proc: subprocess.Popen | None
    server_pid: int
    start_time: str
    cgroup: str
    port: int
    api_key: str
    log_path: Path
    prompts_dir: Path
    unit: str = SCOPE_UNIT


def _fresh_unit() -> str:
    """Immutable unique-per-launch scope name (no unit reuse between runs)."""
    return f"{SCOPE_UNIT}-{secrets.token_hex(6)}"


def _start_time(pid: int) -> str | None:
    try:
        stat = proc_field(pid, "stat")
        return stat[stat.rindex(")") + 2:].split()[19]
    except (OSError, ValueError, IndexError):
        return None


def _children_of(pid: int) -> list[int]:
    try:
        raw = Path(f"/proc/{pid}/task/{pid}/children").read_text()
    except OSError:
        return []
    return [int(token) for token in raw.split()]


def _owned_server_pid(proc: subprocess.Popen, engine: Path,
                      unit: str) -> tuple[int, str, str] | None:
    """(pid, start_time, cgroup) for OUR llama-server, or None.

    P3 identity rule: a match without a readable start_time is NOT accepted
    (never signal a PID whose captured or current identity is absent).
    """
    for pid in [proc.pid, *_children_of(proc.pid)]:
        try:
            cmdline = proc_field(pid, "cmdline").split("\0")[:1]
        except OSError:
            continue
        if cmdline and cmdline[0] == str(engine):
            try:
                cgroup = proc_field(pid, "cgroup").strip()
            except OSError:
                cgroup = ""
            start_time = _start_time(pid)
            if unit in cgroup and start_time is not None:
                return pid, start_time, cgroup
    return None


def launch_server(*, port: int, prompts_dir: Path, log_path: Path,
                  models_dir: Path) -> ServerState:
    # Immutable unique-per-launch unit: no reuse, no reset-failed races.
    unit = _fresh_unit()
    api_key = secrets.token_urlsafe(24)
    prompts_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    # Secrets never ride argv/logs: per-run 0600 key file beside the log.
    key_file = log_path.parent / f".api-key-{unit[-12:]}"
    fd = os.open(key_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(api_key)
    cmd = build_server_cmd(models_dir=models_dir, port=port, api_key="",
                           prompts_dir=prompts_dir, log_path=log_path,
                           unit=unit, api_key_file=key_file)
    log_handle = log_path.open("a", encoding="utf-8", errors="replace")
    log_handle.write(f"\n# spawn: {cmd}\n")
    log_handle.flush()
    proc = subprocess.Popen(cmd, stdout=log_handle, stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL)
    deadline = time.monotonic() + 30
    owned = None
    while time.monotonic() < deadline:
        owned = _owned_server_pid(proc, ENGINE_PATH, unit)
        if owned:
            break
        if proc.poll() is not None:
            raise RuntimeError(f"systemd-run exited rc={proc.returncode} "
                               f"(log: {log_path})")
        time.sleep(0.2)
    if not owned:
        proc.terminate()
        raise RuntimeError(f"owned llama-server never appeared (log: {log_path})")
    pid, start_time, cgroup = owned
    return ServerState(proc=proc, server_pid=pid, start_time=start_time,
                       cgroup=cgroup, port=port, api_key=api_key,
                       log_path=log_path, prompts_dir=prompts_dir, unit=unit)


def _proc_absent(pid: int) -> bool:
    try:
        Path(f"/proc/{pid}").stat()
    except FileNotFoundError:
        return True
    except OSError:
        return False  # inaccessible is not evidence of absence
    return False


def stop_server(state: ServerState, *, term_timeout_s: float = 45.0,
                scope_settle_timeout_s: float = 30.0) -> dict:
    """Stop exactly the owned llama-server, by fresh identity proof only.

    Absent /proc entry = already exited (truthful success). Present but
    unreadable identity = REFUSE to signal (never None==None). SIGKILL is
    preceded by an immediate identity re-proof. Wrapper descendants are
    enumerated with captured start-times BEFORE the wrapper is touched and
    re-proven before each signal. Bounded wait for the scope to reach a
    terminal state; no new launch; never a cgroup-wide or unit-wide kill.
    """
    events: list[dict] = []
    captured = state.start_time
    if not isinstance(captured, str) or not captured.strip():
        return {"refused": "captured start-time empty/absent", "events": events,
                "pid_gone": None, "unit": state.unit}

    def identity_current() -> str | None:
        """Reap a direct child before observing its remaining identity."""
        if state.proc is not None and state.proc.pid == state.server_pid:
            state.proc.poll()
        return _start_time(state.server_pid)

    def observed_gone() -> bool | None:
        current = identity_current()
        if _proc_absent(state.server_pid):
            return True
        return None if current is None else current != captured

    # Enumerate owned descendants with captured identity BEFORE wrapper kill.
    descendants: list[tuple[int, str]] = []
    if state.proc is not None and state.proc.pid:
        for pid in _children_of(state.proc.pid):
            child_start = _start_time(pid)
            try:
                owned_argv = (proc_field(pid, "cmdline").split("\0")[0]
                              == str(ENGINE_PATH))
                owned_scope = state.unit in proc_field(pid, "cgroup")
            except OSError:
                continue
            if owned_argv and owned_scope and child_start:
                descendants.append((pid, child_start))

    gone: bool | None
    if observed_gone() is True:
        gone = True  # already exited before us; nothing to signal
    else:
        current = identity_current()
        if current is None:
            return {"refused": "identity unreadable; no signal sent",
                    "events": events, "pid_gone": None, "unit": state.unit}
        if current != captured:
            gone = True  # PID recycled/exists under a different start
        else:
            try:
                os.kill(state.server_pid, signal.SIGTERM)
                events.append({"signal": "SIGTERM", "pid": state.server_pid})
            except (ProcessLookupError, PermissionError) as exc:
                events.append({"signal": "SIGTERM", "pid": state.server_pid,
                               "error": str(exc)})
            deadline = time.monotonic() + term_timeout_s
            gone = observed_gone()
            while gone is not True and time.monotonic() < deadline:
                time.sleep(0.05)
                gone = observed_gone()
            if not gone:
                # Re-prove identity immediately; missing data never signals.
                if not _proc_absent(state.server_pid) \
                        and identity_current() == captured:
                    try:
                        os.kill(state.server_pid, signal.SIGKILL)
                        events.append({"signal": "SIGKILL",
                                       "pid": state.server_pid,
                                       "reproven": True})
                        deadline = time.monotonic() + 15
                        while time.monotonic() < deadline:
                            if observed_gone() is True:
                                break
                            time.sleep(0.05)
                    except (ProcessLookupError, PermissionError) as exc:
                        events.append({"signal": "SIGKILL",
                                       "pid": state.server_pid,
                                       "error": str(exc)})
                gone = observed_gone()

    if (state.proc is not None and state.proc.pid != state.server_pid
            and state.proc.poll() is None):
        try:
            state.proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            state.proc.kill()
            state.proc.wait(timeout=10)
    gone = observed_gone()
    # Descendant sweep re-proves each captured identity before signaling.
    for pid, child_start in descendants:
        if _proc_absent(pid) or _start_time(pid) != child_start:
            continue
        try:
            os.kill(pid, signal.SIGKILL)
            events.append({"signal": "SIGKILL", "pid": pid,
                           "reason": "verified-descendant",
                           "reproven": True})
        except (ProcessLookupError, PermissionError):
            continue

    unit_state = "unknown"
    settle_deadline = time.monotonic() + scope_settle_timeout_s
    while time.monotonic() < settle_deadline:
        try:
            unit_state = subprocess.run(
                ["systemctl", "--user", "is-active", state.unit],
                capture_output=True, text=True, timeout=10).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            unit_state = "unknown"
            break
        if unit_state not in ("active", "activating", "reloading"):
            break
        time.sleep(1.0)
    return {"events": events, "pid_gone": gone,
            "unit_state_after_stop": unit_state, "unit": state.unit}


# ── Native Hermes tool-calling proof ─────────────────────────────────────────

def _local_aux(base_url: str, api_key: str, timeout: float,
               **extra) -> dict:
    block = {"provider": "custom", "model": MODEL_ALIAS,
             "base_url": base_url, "api_key": api_key,
             "reasoning_effort": "", "timeout": timeout}
    block.update(extra)
    return block


def hermes_proof_config(*, base_url: str, api_key: str) -> dict:
    """Local-only routing for the owned profile.

    Main model: explicit ``custom`` provider at the owned llama-server — no
    cloud fallback exists for that shape. Every ENABLED auxiliary task
    (native auto-chain would otherwise fall back to OpenRouter cloud when
    the main model errors and OPENROUTER/OPENAI keys leak in) is pinned to
    the same local endpoint with ``free_only``; the native compressor stays
    enabled with unbounded no-progress patience for slow CPU decode. No
    ``.env`` is written into the owned home.
    """
    aux = _local_aux
    return {
        # Lean native coding toolset: fewer tool-schema bytes in the
        # system prompt (disk-bound prefill on this box is ~0.2 tok/s cold).
        "coding_context": "focus",
        "model": {"provider": "custom", "base_url": base_url,
                  "api_key": api_key, "default": MODEL_ALIAS,
                  "context_length": CTX},
        "telemetry": {"shared_metrics": {"enabled": False, "send": False,
                                         "offer_version": 2}},
        # providers.<id>.stale_timeout_seconds (selected by
        # _resolve_stale_timeout before the local default) and
        # Single deadline namespace (native GREEN probe): providers.custom
        # request/stale fields only — no redundant agent/model overrides.
        "providers": {"custom": {
            "stale_timeout_seconds": 21600.0,
            "request_timeout_seconds": 21600.0}},
        "auxiliary": {
            "transient_retries": 2,
            "free_only": True,
            "vision": aux(base_url, api_key, 3600, download_timeout=120),
            "compression": aux(base_url, api_key, 21600,
                               no_progress_timeout=21600),
            "skills_hub": aux(base_url, api_key, 600),
            "approval": aux(base_url, api_key, 1800),
            "review": {"provider": "custom", "model": MODEL_ALIAS,
                       "base_url": base_url, "api_key": api_key,
                       "api_mode": "chat_completions"},
            "mcp": aux(base_url, api_key, 600),
            "title_generation": {"provider": "custom", "model": MODEL_ALIAS,
                                 "base_url": base_url, "api_key": api_key,
                                 "reasoning_effort": "", "language": "",
                                 "timeout": 21600},
            "memory_query_rewrite": aux(base_url, api_key, 900),
            "tts_audio_tags": aux(base_url, api_key, 600),
            "voice_chat": aux(base_url, api_key, 3600,
                              reasoning_effort="none"),
            "triage_specifier": aux(base_url, api_key, 1800),
            "kanban_decomposer": aux(base_url, api_key, 1800),
            "profile_describer": aux(base_url, api_key, 900),
            "goal_judge": aux(base_url, api_key, 1800),
            "curator": aux(base_url, api_key, 3600),
            "monitor": aux(base_url, api_key, 900),
            "background_review": aux(base_url, api_key, 21600),
            "moa_reference": aux(base_url, api_key, 3600),
            "moa_aggregator": aux(base_url, api_key, 3600),
        },
    }


# Ambient cloud credentials a Hermes child must NEVER inherit: the auxiliary
# auto-chain consumes OPENAI/OPENROUTER keys after main-model errors, and
# other providers' keys would enable silent cloud calls. The fork's
# SCRUB_KEYS does not cover these.
CLOUD_CREDENTIAL_KEYS = (
    "OPENAI_API_KEY", "OPENAI_API_BASE", "OPENAI_BASE_URL", "OPENROUTER_API_KEY",
    "ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL", "GOOGLE_API_KEY", "GEMINI_API_KEY",
    "GROQ_API_KEY", "MISTRAL_API_KEY", "DEEPSEEK_API_KEY", "XAI_API_KEY",
    "TOGETHER_API_KEY", "FIREWORKS_API_KEY", "CEREBRAS_API_KEY",
    "AZURE_OPENAI_API_KEY", "NOUS_API_KEY", "ZAI_API_KEY", "KIMI_API_KEY",
    "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN",
)


def local_only_environment(owned_home: Path) -> dict[str, str]:
    """Fork child_environment minus every cloud credential key."""
    from context_mode_hermes.verify import child_environment

    env = child_environment(owned_home)
    for key in CLOUD_CREDENTIAL_KEYS:
        env.pop(key, None)
    return env


def toolproof_prompt() -> str:
    return ("Use one of your file-reading tools to read the file "
            f"{TOOLPROOF_FILENAME} in the current working directory, then "
            "reply with the exact codeword from the codeword line, and "
            "nothing else.")


def _verify_child_home(pid: int, home: Path, *, wait_s: float = 5.0) -> bool:
    """Exec-aware HERMES_HOME proof: /proc/<pid>/environ must show the owned
    home (a re-exec'd child may briefly lack an environ — bounded wait,
    matching the verify-live-6 semantics; wrong-profile value fails at once)."""
    from context_mode_hermes.verify import _proc_environ

    deadline = time.monotonic() + wait_s
    while time.monotonic() < deadline:
        env = _proc_environ(pid)
        if env is not None:
            return env.get("HERMES_HOME") == str(home)
        time.sleep(0.2)
    return False


def _verify_saved_transcript(export_path: Path, nonce: str) -> dict:
    """Structural acceptance from the NATIVE ``/save json`` export.

    Requires: initial user prompt WITHOUT the nonce; an assistant
    ``tool_calls`` entry with an id; a ``role=tool`` response whose
    ``tool_call_id`` matches and whose content carries the nonce; and a
    LATER assistant message containing the nonce — the final answer is
    derived from the executed tool, never from echoed intermediate stdout.
    """
    result = {"checked": False, "user_prompt_clean": None,
              "tool_call_id": None, "tool_response_with_nonce": None,
              "assistant_answer_with_nonce": None,
              "export": str(export_path)}
    try:
        document = json.loads(export_path.read_text(encoding="utf-8",
                                                    errors="replace"))
    except (OSError, ValueError):
        return result
    messages = (document.get("messages") if isinstance(document, dict)
                else document)
    if not isinstance(messages, list):
        return result

    def records_by_role(role: str) -> list[tuple[int, dict]]:
        found = []
        for index, message in enumerate(messages):
            if isinstance(message, dict) and str(message.get(
                    "role", "")).lower() == role:
                found.append((index, message))
        return found

    def text_of(message: dict) -> str:
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            joined = " ".join(part.get("text", "") for part in content
                              if isinstance(part, dict))
            return joined
        return ""

    users = records_by_role("user")
    if not users:
        return result
    result["checked"] = True
    result["user_prompt_clean"] = nonce not in text_of(users[0][1])
    call_ids = set()
    for _, message in records_by_role("assistant"):
        for call in message.get("tool_calls") or []:
            if isinstance(call, dict) and call.get("id"):
                call_ids.add(call["id"])
    for index, message in records_by_role("tool"):
        if (message.get("tool_call_id") in call_ids
                and nonce in text_of(message)):
            result["tool_call_id"] = message.get("tool_call_id")
            result["tool_response_with_nonce"] = True
            assistants = records_by_role("assistant")
            if assistants:
                last_index, last = assistants[-1]
                # Completed FINAL answer: the last assistant record is a
                # plain text answer (no pending tool calls) carrying the
                # nonce, positioned after the matched tool response.
                if (last_index > index and nonce in text_of(last)
                        and not last.get("tool_calls")):
                    result["assistant_answer_with_nonce"] = True
            break
    return result


def hermes_toolproof(state: ServerState, evidence_dir: Path,
                     *, toolproof_timeout_s: float) -> dict:
    """Real native-Hermes conversation whose answer is derivable only via a
    tool. Exclusive owned temporary proof profile (full sweep is valid ONLY
    here). Identity: PTY child proven by /proc environ before keystrokes and
    re-proven after prompt; teardown reuses the fork's verified
    ProcessLedger + stop_child — never a raw os.kill. Acceptance is the
    structural session proof (tool message → later assistant answer), not
    intermediate stdout; engine prompt capture stays as corroborating
    evidence. Exit only after the turn completes."""
    from context_mode_hermes.verify import (_InterruptFlag, PtyChatDriver,
                                            normalize_terminal_output,
                                            stop_child, sweep_owned,
                                            ProcessLedger)

    home = evidence_dir / "hermes-home"
    workspace = evidence_dir / "hermes-workspace"
    home.mkdir(parents=True, exist_ok=True)
    workspace.mkdir(parents=True, exist_ok=True)
    nonce = secrets.token_hex(8)
    (workspace / TOOLPROOF_FILENAME).write_text(
        f"# harbor notes\ncodeword: {nonce}\n", encoding="utf-8")
    _write_private_json(home / "config.yaml", hermes_proof_config(
        base_url=f"http://127.0.0.1:{state.port}/v1",
        api_key=state.api_key))

    env = local_only_environment(home)
    env["TERM"] = "xterm-256color"
    interrupt = _InterruptFlag()
    ledger = ProcessLedger()
    ledger.owned_home_marker = str(home)  # authorize the sweep below
    since_ts = time.time()
    report: dict = {"nonce": nonce, "owned_home": str(home)}
    driver = PtyChatDriver([str(HERMES_BIN), "chat", "--cli"], env, workspace,
                           evidence_dir / "hermes-chat.log")
    from context_mode_hermes.verify import _proc_cmdline, _proc_environ
    ledger.note_proof(driver.proc.pid, _proc_environ(driver.proc.pid),
                      _proc_cmdline(driver.proc.pid))
    try:
        identity_ok = _verify_child_home(driver.proc.pid, home, wait_s=15.0)
        ready, ready_elapsed = driver.wait_for("❯", 300.0, interrupt)
        report["interactive_ready"] = {"ready": ready,
                                       "elapsed_s": round(ready_elapsed, 1)}
        turn_started = time.time()
        proof_input_sent = False
        if ready and _verify_child_home(driver.proc.pid, home, wait_s=15.0):
            report["child_identity_before_input"] = True
            # busy_policy=dispatch native command: capacity observation only.
            driver.send("/context")
            driver.wait_for("context", 120.0, interrupt)
            time.sleep(10.0)
            nonce_mark = driver.mark()  # before prompt: excludes prior echo
            if _verify_child_home(driver.proc.pid, home, wait_s=15.0):
                report["child_identity_after_prompt"] = True
                driver.send(toolproof_prompt())
                proof_input_sent = True
            else:
                report["child_identity_after_prompt"] = False
        else:
            report["child_identity_before_input"] = False
            report["child_identity_after_prompt"] = False
        if proof_input_sent:
            # Turn completion, not first nonce sighting: wait for the nonce
            # (progress), then the CLI prompt returning (generation finished) —
            # then export via the NATIVE /save command and inspect the export.
            answered, answer_elapsed = driver.wait_for(
                nonce, toolproof_timeout_s, interrupt)
            report["nonce_seen_in_tty"] = {"ok": answered,
                                           "elapsed_s": round(answer_elapsed, 1)}
            # Nonce + idle ❯ often arrive in ONE 64KB PTY chunk: recognize
            # idle in the already-buffered view FIRST (scan after the
            # nonce), only waiting when absent. ❯ renders ONLY when idle
            # (cli_tui_mixin 296–303; busy shows ☤; /save is busy-reject).
            driver._pump(0.5)
            buffered = normalize_terminal_output(driver._view(nonce_mark))
            nonce_at = buffered.find(nonce)
            prompt_back = bool(
                nonce_at >= 0 and "❯" in buffered[nonce_at + len(nonce):])
            if not prompt_back:
                prompt_back, _ = driver.wait_for("❯", 3600.0, interrupt,
                                                 since=driver.mark())
            report["prompt_returned_after_turn"] = prompt_back
            if prompt_back and driver.proc.poll() is None:
                driver.send("/save json glm-toolproof-export.json")
                driver.wait_for("glm-toolproof-export.json", 60.0, interrupt)
                export = workspace / "glm-toolproof-export.json"
                deadline = time.monotonic() + 120
                while time.monotonic() < deadline and not export.exists():
                    time.sleep(2.0)
                if export.exists():
                    report["save_export"] = str(export)
                    report["session_proof"] = _verify_saved_transcript(
                        export, nonce)
            if driver.proc.poll() is None:
                driver.send("/exit")
        try:
            report["hermes_exit"] = f"rc={driver.proc.wait(timeout=60)}"
        except subprocess.TimeoutExpired:
            report["hermes_exit"] = "timeout-after-exit"
    finally:
        stop_child(driver.proc, ledger, 30.0)  # TERM→KILL, ledger-verified
        driver.close()
        report["no_owned_survivors"] = sweep_owned(home, ledger, interrupt)
    transcript = normalize_terminal_output(
        (evidence_dir / "hermes-chat.log").read_bytes())
    report["nonce_in_transcript"] = nonce in transcript
    # Corroborating engine-side capture: the follow-up request must exist
    # with the nonce (secondary to the structural session proof).
    tool_roundtrip = False
    for path in sorted(state.prompts_dir.rglob("*")):
        if path.is_file():
            blob = path.read_bytes()
            if nonce.encode() in blob and b"tool" in blob:
                tool_roundtrip = True
    report["engine_tool_roundtrip"] = tool_roundtrip
    proof = report.get("session_proof") or {"checked": False}
    report["accepted"] = bool(
        report.get("child_identity_before_input")
        and report.get("child_identity_after_prompt")
        and proof.get("checked") and proof.get("user_prompt_clean")
        and proof.get("tool_response_with_nonce")
        and proof.get("assistant_answer_with_nonce"))
    return report


# ── Orchestration ────────────────────────────────────────────────────────────

def serve_cycle(cycle_dir: Path, models_dir: Path, *, port: int,
                health_deadline_s: float, touch_deadline_s: float,
                run_toolproof: bool, toolproof_timeout_s: float) -> dict:
    cycle: dict = {"cycle_dir": str(cycle_dir)}
    state = launch_server(port=port, prompts_dir=cycle_dir / "prompts",
                          log_path=cycle_dir / "server.log",
                          models_dir=models_dir)
    try:
        cycle["launch"] = {"server_pid": state.server_pid,
                           "start_time": state.start_time,
                           "cgroup": state.cgroup,
                           "port": state.port}
        cycle["health_s"] = wait_health(port, state.api_key, health_deadline_s)
        cycle["models"] = list_models(port, state.api_key)
        if MODEL_ALIAS not in cycle["models"]:
            raise RuntimeError(f"{MODEL_ALIAS} not served: {cycle['models']}")
        before = sample_process(state.server_pid)
        touch = wait_touch(port, state.api_key, touch_deadline_s)
        after = sample_process(state.server_pid)
        cycle["touch"] = touch
        cycle["touch_io"] = {"read_bytes_delta":
                             after["read_bytes"] - before["read_bytes"]}
        cycle["touch_cgroup"] = cgroup_sample(state.server_pid)
        paragraph = generation(
            port, state.api_key,
            "Explain in one short paragraph why the sky is blue.",
            max_tokens=512, timeout_s=touch_deadline_s)
        cycle["paragraph_generation"] = paragraph
        cycle["after_paragraph"] = {"process": sample_process(state.server_pid),
                                    "cgroup": cgroup_sample(state.server_pid),
                                    "llm_metrics": llm_metrics(port, state.api_key)}
        if run_toolproof:
            cycle["toolproof"] = hermes_toolproof(
                state, cycle_dir, toolproof_timeout_s=toolproof_timeout_s)
            cycle["after_toolproof"] = {
                "process": sample_process(state.server_pid),
                "cgroup": cgroup_sample(state.server_pid),
                "llm_metrics": llm_metrics(port, state.api_key)}
        time.sleep(1.0)
        cycle["timings"] = parse_timings(
            state.log_path.read_text(encoding="utf-8", errors="replace"))
    except Exception as exc:
        import traceback
        cycle["error"] = f"{type(exc).__name__}: {exc}"
        (cycle_dir / "cycle-error.log").write_text(
            traceback.format_exc(), encoding="utf-8")
    finally:
        cycle["stop"] = stop_server(state)
    return cycle


def cmd_verify_model(args: argparse.Namespace) -> int:
    report = verify_model(Path(args.models_dir).expanduser(), deep=args.deep)
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


def cmd_prove(args: argparse.Namespace) -> int:
    evidence = Path(args.evidence_dir).expanduser()
    evidence.mkdir(parents=True, exist_ok=True)
    summary: dict = {"engine": _engine_identity(), "cycles": []}
    if not summary["engine"]["verified"]:
        (evidence / "summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps({"state": "refused-engine", **summary}, indent=2))
        return 2
    manifest = verify_model(Path(args.models_dir).expanduser(),
                            deep=not args.skip_deep_verify)
    summary["model_manifest"] = manifest
    if not manifest["ok"]:
        (evidence / "summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary["model_manifest"], indent=2))
        return 2
    for cycle_index in range(args.cycles):
        cycle_dir = evidence / f"cycle-{cycle_index}"
        cycle_dir.mkdir(parents=True, exist_ok=True)
        try:
            cycle = serve_cycle(
                cycle_dir, Path(args.models_dir).expanduser(), port=args.port,
                health_deadline_s=args.health_timeout,
                touch_deadline_s=args.touch_timeout,
                run_toolproof=bool(
                    getattr(args, "toolproof_every_cycle", False)
                    or cycle_index == 0),
                toolproof_timeout_s=args.toolproof_timeout)
        except Exception as exc:  # record and continue: later cycles still
            # exercise restart; every failure keeps its full logs on disk.
            import traceback
            (cycle_dir / "cycle-error.log").write_text(
                traceback.format_exc(), encoding="utf-8")
            cycle = {"index": cycle_index, "error": f"{type(exc).__name__}: {exc}"}
        cycle["index"] = cycle_index
        summary["cycles"].append(cycle)
        (evidence / "summary.json").write_text(
            json.dumps(summary, indent=2, default=str))
    def cycle_ok(cycle: dict, index: int) -> bool:
        if cycle.get("error"):
            return False
        if not cycle.get("health_s") or not (cycle.get("touch") or {}).get("ready_s"):
            return False
        stop = cycle.get("stop") or {}
        if stop.get("pid_gone") is not True:
            return False
        unit_state = str(stop.get("unit_state_after_stop", "")).lower()
        if unit_state in ("active", "activating", "reloading"):
            return False
        required = bool(getattr(args, "toolproof_every_cycle", False)
                        or index == 0)
        if not required:
            return True
        proof = cycle.get("toolproof")
        if not proof or not proof.get("accepted"):
            return False
        if proof.get("no_owned_survivors") is not True:
            return False
        return True
    cycles_summary = summary["cycles"]
    ok = bool(cycles_summary) and all(
        cycle_ok(c, i) for i, c in enumerate(cycles_summary))
    print(f"GLM-LOCAL: cycles={args.cycles} ok={ok} evidence={evidence}")
    return 0 if ok else 1


# ── Persistent owned operation (single native model, no fleet manager) ──────
#
# The selected persistent profile is HERMES_HOME=/home/a1234/.local/share/
# hermes-glm/profile; the runtime state lives under the machine root's
# native runtimes dir. START seeds the profile ONCE (never rewrites operator
# or #5 plugin/MCP keys on restart), keeps the stable API key in a 0600 file
# (never argv), launches under a fresh unique scope, and writes a private
# internal endpoint record for these harnesses. This is NOT native supervisor
# state: the custom provider uses its explicitly configured endpoint. STOP
# touches only the manifest-owned PID/unit with nonempty fresh start-time —
# never a HERMES_HOME-wide sweep (other clients may share this profile).

OPS_STATE = PERSISTENT_ROOT / "runtimes" / "llamacpp"
OPS_SERVER_JSON = OPS_STATE / "server.json"
OPS_KEY = OPS_STATE / ".api_key"
OPS_MANIFEST = OPS_STATE / "glm-ops-manifest.json"
OPS_WORKSPACE = PERSISTENT_ROOT / "workspace"


def _write_private_json(path: Path, payload: dict) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2))


def _stable_api_key() -> str:
    OPS_STATE.mkdir(parents=True, exist_ok=True)
    with suppress(OSError):
        existing = OPS_KEY.read_text(encoding="utf-8-sig").strip()
        if len(existing) >= 16:
            return existing
    key = secrets.token_urlsafe(24)
    fd = os.open(OPS_KEY, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(key)
    return key


def ops_seed_profile(api_key: str, port: int) -> str:
    """Seed the persistent profile once; never overwrite later operator,
    plugin or MCP edits (native atomic merge is the owners' future path)."""
    config_path = PERSISTENT_PROFILE / "config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    if config_path.exists():
        return "existing"
    config = hermes_proof_config(
        base_url=f"http://127.0.0.1:{port}/v1", api_key=api_key)
    fd = os.open(config_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(config, indent=2))
    return "seeded"


def _ops_identity(manifest: dict) -> tuple[int, str] | None:
    """(pid, start_time) when the manifest-owned server is really alive."""
    pid = manifest.get("pid")
    captured = manifest.get("start_time")
    if not isinstance(pid, int) or not captured or not str(captured).strip():
        return None
    unit = manifest.get("unit")
    if not isinstance(unit, str) or not unit.strip():
        return None
    try:
        if proc_field(pid, "cmdline").split("\0")[0] != str(ENGINE_PATH):
            return None
        if unit not in proc_field(pid, "cgroup"):
            return None
    except OSError:
        return None
    current = _start_time(pid)
    return (pid, captured) if current == captured else None


def ops_state() -> tuple[dict | None, str]:
    """(manifest, classification): running | gone | uncertain.

    ``uncertain`` = a live PID at the recorded position whose identity we
    cannot prove — the manifest is RETAINED and stop/start fail closed on
    it, never unlinking ownership records for a possibly-live runtime."""
    try:
        manifest = json.loads(OPS_MANIFEST.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return None, "gone"
    pid = manifest.get("pid")
    if not isinstance(pid, int) or not Path(f"/proc/{pid}").exists():
        return manifest, "gone"
    return (manifest, "running") if _ops_identity(manifest) else (manifest, "uncertain")


def ops_running() -> dict | None:
    manifest, state = ops_state()
    return manifest if state == "running" else None


def ops_preflight(models_dir: Path) -> dict:
    """Honest recorded preflight: exact artifact sizes + engine binary
    digest, and ACTUAL RAM/disk/controllers now (never historical figures).
    No invented memory floor: the cgroup cap bounds the unit; recorded
    headroom documents host capacity."""
    engine = _engine_identity()
    if not engine["verified"]:
        return {"engine": engine}
    mem = {}
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        name, _, rest = line.partition(":")
        if name in ("MemTotal", "MemAvailable"):
            mem[name] = int(rest.split()[0]) * 1024
    try:
        controllers = {name: f"{name} " in Path(
            "/sys/fs/cgroup/cgroup.controllers").read_text()
            for name in ("memory", "cpu")}
    except OSError:
        controllers = {"memory": None, "cpu": None}
    return {"memory": mem,
            "disk": {"free_bytes": os.statvfs(models_dir).f_bavail
                     * os.statvfs(models_dir).f_frsize},
            "controllers": controllers,
            "engine": engine,
            "manifest": verify_model(models_dir, deep=False)}


def cmd_ops_start(args: argparse.Namespace) -> int:
    preflight = ops_preflight(DEFAULT_MODELS_DIR)
    if not preflight["engine"]["verified"]:
        print(json.dumps({"state": "refused-engine",
                          "preflight": preflight}, indent=2))
        return 2
    if not preflight["manifest"]["ok"]:
        print(json.dumps({"state": "refused-manifest",
                          "preflight": preflight}, indent=2))
        return 2
    record, state = ops_state()
    if state == "uncertain":
        print(json.dumps({"state": "refused-identity-uncertain",
                          "manifest_retained": True,
                          "pid": (record or {}).get("pid")}, default=str))
        return 2
    running = record if state == "running" else None
    if running:
        print(json.dumps({"state": "already-running", **running}, indent=2))
        return 0
    api_key = _stable_api_key()
    ops_seed_profile(api_key, PORT)
    OPS_WORKSPACE.mkdir(parents=True, exist_ok=True)
    unit = _fresh_unit()
    log_path = OPS_STATE / "llama-server.log"
    cmd = build_server_cmd(models_dir=DEFAULT_MODELS_DIR, port=PORT,
                           api_key="", prompts_dir=None, log_path=log_path,
                           unit=unit, api_key_file=OPS_KEY)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8", errors="replace") as handle:
        handle.write(f"\n# spawn: {cmd}\n")
        handle.flush()
        proc = subprocess.Popen(cmd, stdout=handle, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL)
    deadline = time.monotonic() + 30
    owned = None
    while time.monotonic() < deadline:
        owned = _owned_server_pid(proc, ENGINE_PATH, unit)
        if owned:
            break
        if proc.poll() is not None:
            print(json.dumps({"state": "failed", "rc": proc.returncode,
                              "log": str(log_path)}))
            return 1
        time.sleep(0.2)
    if not owned:
        proc.terminate()
        print(json.dumps({"state": "failed", "reason": "no-owned-server",
                          "log": str(log_path)}))
        return 1
    pid, start_time, cgroup = owned
    state = {"base_url": f"http://127.0.0.1:{PORT}/v1", "api_key": api_key,
             "pid": pid, "start_time": start_time, "executable": str(ENGINE_PATH)}
    _write_private_json(OPS_SERVER_JSON, state)
    manifest = {"pid": pid, "start_time": start_time, "unit": unit,
                "port": PORT, "cgroup": cgroup, "log": str(log_path),
                "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    _write_private_json(OPS_MANIFEST, manifest)
    try:
        health_s = wait_health(PORT, api_key, args.health_timeout)
        touch = wait_touch(PORT, api_key, args.touch_timeout)
        # HTTP liveness alone must not claim a foreign occupant: re-prove
        # the SAME owned PID/start-time/argv/unit after health+touch.
        if _ops_identity({"pid": pid, "start_time": start_time,
                          "unit": unit}) is None:
            cmd_ops_stop(argparse.Namespace())
            print(json.dumps({"state": "failed",
                              "reason": "post-health-identity-mismatch"}))
            return 1
    except Exception:
        # Failed health after publication: clean up the verified-owned
        # server by manifest identity, then fail (no orphaned scope).
        cmd_ops_stop(argparse.Namespace())
        raise
    print(json.dumps({"state": "running", "health_s": round(health_s),
                      "touch_ready_s": round(touch.get("ready_s", 0)),
                      "preflight": preflight, **manifest}, indent=2))
    return 0


def cmd_ops_stop(args: argparse.Namespace) -> int:
    manifest, state = ops_state()
    if state == "uncertain":
        print(json.dumps({"state": "refused-identity-uncertain",
                          "manifest_retained": True,
                          "pid": (manifest or {}).get("pid")}, default=str))
        return 2
    if state == "gone" or manifest is None:
        for path in (OPS_MANIFEST, OPS_SERVER_JSON):
            with suppress(OSError):
                path.unlink()
        print(json.dumps({"state": "not-running", "stale_cleared": True}))
        return 0
    holder = ServerState(proc=None,  # adopt-by-manifest: no wrapper process
                         server_pid=manifest["pid"],
                         start_time=manifest["start_time"],
                         cgroup=manifest.get("cgroup", ""),
                         port=manifest["port"], api_key="",
                         log_path=Path(manifest.get("log", "/dev/null")),
                         prompts_dir=Path("/dev/null"),
                         unit=manifest["unit"])
    report = stop_server(holder)
    if report.get("pid_gone"):
        with suppress(OSError):
            OPS_MANIFEST.unlink()
            OPS_SERVER_JSON.unlink()
    print(json.dumps(report, indent=2))
    return 0 if report.get("pid_gone") else 1


def cmd_ops_status(args: argparse.Namespace) -> int:
    manifest = ops_running()
    if manifest is None:
        print(json.dumps({"state": "stopped"}))
        return 1
    # Read-only: status never creates/replaces runtime credentials.
    try:
        key = OPS_KEY.read_text(encoding="utf-8-sig").strip()
    except OSError:
        key = ""
    state = {"state": "running", "key_present": bool(key), **manifest}
    if not key:
        state["state"] = "degraded-missing-key"
        print(json.dumps(state, indent=2, default=str))
        return 1
    try:
        state["health_5s"] = wait_health(manifest["port"], key, 5.0)
    except Exception:  # honest unhealthy: nonzero, never green
        state["state"] = "unhealthy"
        print(json.dumps(state, indent=2, default=str))
        return 1
    print(json.dumps(state, indent=2, default=str))
    return 0


def cmd_ops_chat(args: argparse.Namespace) -> int:
    manifest = ops_running()
    if manifest is None and cmd_ops_start(args) != 0:
        return 1
    manifest = ops_running() or {}
    api_key = _stable_api_key()
    ops_seed_profile(api_key, manifest.get("port", PORT))
    OPS_WORKSPACE.mkdir(parents=True, exist_ok=True)
    env = local_only_environment(PERSISTENT_PROFILE)
    env["TERM"] = env.get("TERM", "xterm-256color")
    os.chdir(OPS_WORKSPACE)
    os.execvpe(str(HERMES_BIN), [str(HERMES_BIN), "chat", "--cli"], env)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="benchmarks.glm_local")
    sub = parser.add_subparsers(dest="command", required=True)

    verify = sub.add_parser("verify-model", help="check shard sizes/digests")
    verify.add_argument("--models-dir", default=str(DEFAULT_MODELS_DIR))
    verify.add_argument("--deep", action="store_true",
                        help="re-hash every shard (slow on 146 GiB)")
    verify.set_defaults(fn=cmd_verify_model)

    prove = sub.add_parser("prove", help="serve, observe, Hermes toolproof, stop")
    prove.add_argument("--models-dir", default=str(DEFAULT_MODELS_DIR))
    prove.add_argument("--evidence-dir", default=str(DEFAULT_EVIDENCE_DIR))
    prove.add_argument("--port", type=int, default=PORT)
    prove.add_argument("--cycles", type=int, default=2)
    prove.add_argument("--health-timeout", type=float, default=1800.0)
    prove.add_argument("--touch-timeout", type=float, default=7200.0)
    prove.add_argument("--toolproof-timeout", type=float, default=21600.0)
    prove.add_argument("--toolproof-every-cycle", action="store_true",
                       help="run the Hermes toolproof in every cycle")
    prove.add_argument("--skip-deep-verify", action="store_true",
                       help="size-only manifest gate (digests verified at download)")
    prove.set_defaults(fn=cmd_prove)

    chat = sub.add_parser("start", help="launch the persistent owned runtime")
    chat.add_argument("--health-timeout", type=float, default=1800.0)
    chat.add_argument("--touch-timeout", type=float, default=7200.0)
    chat.set_defaults(fn=cmd_ops_start)
    chat = sub.add_parser("stop", help="stop the manifest-owned runtime only")
    chat.set_defaults(fn=cmd_ops_stop)
    chat = sub.add_parser("status", help="report owned runtime state")
    chat.set_defaults(fn=cmd_ops_status)
    chat = sub.add_parser("chat", help="native interactive chat on the owned profile")
    chat.add_argument("--health-timeout", type=float, default=1800.0)
    chat.add_argument("--touch-timeout", type=float, default=7200.0)
    chat.set_defaults(fn=cmd_ops_chat)
    args = parser.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
