"""Benchmark cleanup must be identity-scoped (SafetyStandardsReview P1).

Consumer-visible contract for ``benchmarks.run_bench``'s production cleanup:
it reuses the verification smoke's ``ProcessLedger`` machinery — every signal
goes to an individually verified PID whose own ``/proc`` environ carries an
explicitly owned arm profile as ``HERMES_HOME`` (start time still matching),
TERM first, bounded KILL after. Nothing is ever selected by command line or
path prefix, so an inspector, editor or foreign-profile process that merely
mentions the benchmark temp root in argv always survives. The arm-timeout
path and the final sweep are scoped the same way.

These tests are deliberately decoy-generic: the decoys are ordinary sleep
processes whose argv happens to mention the benchmark root — there is no
known-decoy list to special-case because cleanup never looks at command
lines at all.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path


from benchmarks import run_bench
from context_mode_hermes.verify import ProcessLedger, owned_hermes_pids, stop_child


def _wait_visible(home: Path, *pids: int, timeout: float = 10.0) -> None:
    """Wait until each pid shows up in /proc with the owned home in its env."""
    deadline = time.monotonic() + timeout
    outstanding = set(pids)
    while outstanding and time.monotonic() < deadline:
        found = {entry["pid"] for entry in owned_hermes_pids(home, exclude={os.getpid()})}
        outstanding -= found
        if not outstanding:
            return
        time.sleep(0.05)
    raise AssertionError(f"children never showed up under {home}: {sorted(outstanding)}")


def _spawn(home: Path, *argv_markers: str, ignore_term: bool = False) -> subprocess.Popen:
    code = ("import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
            "time.sleep(300)" if ignore_term else "import time; time.sleep(300)")
    return subprocess.Popen(
        [sys.executable, "-c", code, *argv_markers],
        env={**os.environ, "HERMES_HOME": str(home)},
        start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _spawn_decoy(root: Path) -> subprocess.Popen:
    """A foreign process whose command line merely MENTIONS the benchmark
    root (the editor/inspector shape the old cmdline sweep SIGKILLed)."""
    marker = str(root / "logs" / "inspect.log")
    return subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(300)", marker],
        env={**os.environ, "HERMES_HOME": str(root.parent / "foreign-profile")},
        start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _reap(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)


def test_sweep_kills_only_identity_verified_children_and_spares_root_mentioning_decoys(tmp_path):
    root = tmp_path / "cm-bench-root"
    owned = root / "home-plugin"
    owned.mkdir(parents=True)

    decoy = _spawn_decoy(root)
    owned_child = _spawn(owned, str(root))  # root in argv too, like engine workers
    try:
        _wait_visible(owned, owned_child.pid)
        assert decoy.poll() is None

        ledger = ProcessLedger()
        killed = run_bench.sweep_profile(owned, ledger)

        owned_child.wait(timeout=10)  # TERM-honoring owned child is gone
        assert decoy.poll() is None, "a process that merely mentions the root must survive"
        signals = [s for s in ledger.signals if s["target"] == owned_child.pid]
        assert signals and all(s["identity"] == "owned" for s in signals)
        assert {s["signal"] for s in signals} <= {"SIGTERM", "SIGKILL"}
        assert all(s["target"] != decoy.pid for s in ledger.signals)
        assert killed and all(k["hermes_home"] == str(owned) for k in killed)
    finally:
        _reap(decoy)
        _reap(owned_child)


def test_timeout_path_is_bounded_term_then_kill_per_verified_pid_and_spares_decoys(tmp_path):
    """The arm-timeout seam (verify.stop_child): an owned child that ignores
    TERM is KILLed individually after the bounded grace window — never via an
    unverified process-group SIGKILL — and the decoy survives."""
    root = tmp_path / "cm-bench-root"
    owned = root / "home-baseline"
    owned.mkdir(parents=True)

    decoy = _spawn_decoy(root)
    stubborn = _spawn(owned, str(root), ignore_term=True)
    try:
        _wait_visible(owned, stubborn.pid)
        ledger = ProcessLedger()
        ledger.owned_home_marker = str(owned)

        stop_child(stubborn, ledger, exit_timeout=3.0)

        assert stubborn.wait(timeout=10) is not None, "stubborn owned child must be reaped"
        by_signal = [s["signal"] for s in ledger.signals if s["target"] == stubborn.pid]
        assert "SIGTERM" in by_signal and "SIGKILL" in by_signal
        assert all(s["identity"] == "owned" and s["scope"] == "process"
                   for s in ledger.signals if s["target"] == stubborn.pid)
        assert decoy.poll() is None, "the decoy must never be signaled"
        assert all(s["target"] != decoy.pid for s in ledger.signals)
    finally:
        _reap(decoy)
        _reap(stubborn)


def test_final_sweep_covers_every_owned_arm_profile_only(tmp_path):
    """The main()-finally seam: every explicitly owned arm profile is swept
    (each swept PID carries that home's identity), and root-mentioning
    decoys survive even though all homes live under the same root."""
    root = tmp_path / "cm-bench-root"
    homes = {arm: root / f"home-{arm}" for arm in ("baseline", "plugin")}
    for home in homes.values():
        home.mkdir(parents=True)

    decoy = _spawn_decoy(root)
    children = {arm: _spawn(home, str(root)) for arm, home in homes.items()}
    try:
        for arm, home in homes.items():
            _wait_visible(home, children[arm].pid)

        ledger = ProcessLedger()
        killed = run_bench.sweep_profiles(homes.values(), ledger)

        for arm, child in children.items():
            child.wait(timeout=10)
        assert decoy.poll() is None, "decoy survives the final sweep"
        swept_homes = {entry["hermes_home"] for entry in killed}
        assert swept_homes == {str(home) for home in homes.values()}
        assert all(s["target"] != decoy.pid for s in ledger.signals)
    finally:
        _reap(decoy)
        for child in children.values():
            _reap(child)
