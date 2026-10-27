"""Ticket #4 contract: GLM-5.3-Flash local runtime harness.

Permanent tests cover only real behavior: llama.cpp timing-log parsing
(both CLI and server-slot formats), the local-only environment boundary,
the structural /save-export tool-proof parser (positive + two negatives),
and cmd_prove's consumer-visible exit code when a REQUIRED tool proof
fails acceptance (no silent false-success). Pinned-constant copies and
config/default dict tests are prohibited here (review); the deadline
namespace was verified through the native consumer as a throwaway probe.
"""

from __future__ import annotations

import json

import pytest

from benchmarks import glm_local


@pytest.fixture
def owned_engine_artifact(tmp_path, monkeypatch):
    """Keep exit-code tests independent of the operator's installed runtime."""
    artifact = tmp_path / "llama-server"
    artifact.write_bytes(b"controlled proof-gate artifact\n")
    monkeypatch.setattr(glm_local, "ENGINE_PATH", artifact)


def test_parse_llama_timings() -> None:
    log = """info: prompt eval time =  123456.78 ms /    456 tokens (  270.74 ms per token,     3.69 tokens per second)
info: eval time =      5123.45 ms /    32 runs   (  160.11 ms per token,     6.25 tokens per second)
garbage line without numbers
info: prompt eval time =    200.00 ms /      8 tokens (   25.00 ms per token,    40.00 tokens per second)
info: eval time =      1000.00 ms /    20 runs   (   50.00 ms per token,    20.00 tokens per second)
info: srv  llama_server: model loaded
3.54.796 I slot print_timing: id  0 | task 0 | prompt processing, n_tokens =      9, progress = 0.39, t =  41.04 s / 0.22 tokens per second
4.37.340 I slot print_timing: id  0 | task 0 | prompt processing, n_tokens =     19, progress = 0.83, t =  83.59 s / 0.23 tokens per second
5.00.000 I slot print_timing: id  0 | task 0 | prompt processing, n_tokens =     23, progress = 1.00, t =  101.00 s / 0.23 tokens per second
9.00.000 I slot print_timing: id  0 | task 0 | eval, n_tokens =      5, t =  120.00 s / 0.04 tokens per second
"""
    timings = glm_local.parse_timings(log)
    assert [t["prompt_tokens"] for t in timings] == [456, 8, 23]
    assert timings[0]["prompt_tps"] == pytest.approx(3.69, rel=0.05)
    assert timings[0]["eval_tokens"] == 32
    assert timings[0]["eval_tps"] == pytest.approx(6.25, rel=0.05)
    assert timings[1]["eval_tps"] == pytest.approx(20.00, rel=0.02)
    assert timings[2]["prompt_tokens"] == 23
    assert timings[2]["prompt_tps"] == pytest.approx(0.23, rel=0.05)
    assert timings[2]["eval_tokens"] == 5
    assert timings[2]["eval_tps"] == pytest.approx(0.04, rel=0.05)


def test_local_only_environment_scrubs_cloud_credentials(tmp_path,
                                                          monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-inherited")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-inherited")
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    env = glm_local.local_only_environment(tmp_path)
    for key in glm_local.CLOUD_CREDENTIAL_KEYS:
        assert key not in env, key
    assert env["HERMES_HOME"] == str(tmp_path)


def _write_export(tmp_path, messages):
    path = tmp_path / "export.json"
    path.write_text(json.dumps({"messages": messages}), encoding="utf-8")
    return path


def test_saved_transcript_parser_accepts_real_tool_chain(tmp_path) -> None:
    nonce = "0123456789abcdef"
    path = _write_export(tmp_path, [
        {"role": "user", "content": "read harbor-notes.txt please"},
        {"role": "assistant", "tool_calls": [
            {"id": "call_1", "function": {"name": "read_file"}}]},
        {"role": "tool", "tool_call_id": "call_1",
         "content": f"# harbor notes\ncodeword: {nonce}\n"},
        {"role": "assistant", "content": f"The codeword is {nonce}."},
    ])
    result = glm_local._verify_saved_transcript(path, nonce)
    assert result["checked"] is True
    assert result["user_prompt_clean"] is True
    assert result["tool_call_id"] == "call_1"
    assert result["tool_response_with_nonce"] is True
    assert result["assistant_answer_with_nonce"] is True


def test_saved_transcript_parser_rejects_unmatched_call_id(tmp_path) -> None:
    nonce = "0123456789abcdef"
    path = _write_export(tmp_path, [
        {"role": "user", "content": "read the file"},
        {"role": "assistant", "content": f"the codeword is {nonce}"},
        {"role": "tool", "tool_call_id": "unmatched",
         "content": f"codeword: {nonce}"},
    ])
    result = glm_local._verify_saved_transcript(path, nonce)
    assert result["tool_response_with_nonce"] is None
    assert result["assistant_answer_with_nonce"] is None


def test_saved_transcript_parser_requires_completed_final_answer(
        tmp_path) -> None:
    # Nonce present in an INTERMEDIATE assistant record that still carries
    # pending tool calls must NOT satisfy the completed-final-answer rule:
    # the LAST assistant message must be the plain answer.
    nonce = "0123456789abcdef"
    path = _write_export(tmp_path, [
        {"role": "user", "content": "read the file"},
        {"role": "assistant", "content": f"reading, codeword may be {nonce}",
         "tool_calls": [{"id": "call_1",
                         "function": {"name": "read_file"}}]},
        {"role": "tool", "tool_call_id": "call_1",
         "content": f"codeword: {nonce}"},
    ])
    result = glm_local._verify_saved_transcript(path, nonce)
    assert result["tool_response_with_nonce"] is True
    assert result["assistant_answer_with_nonce"] is None


def test_prove_exit_code_fails_when_required_toolproof_rejected(
        tmp_path, monkeypatch, owned_engine_artifact) -> None:
    """Consumer-visible regression: ``prove`` must exit nonzero when a
    REQUIRED tool proof fails acceptance — a cycle error is not the only
    failure mode. Controlled local seam around the real cmd_prove."""
    cycles = {"n": 0}

    def fake_serve_cycle(cycle_dir, models_dir, **kwargs):
        cycles["n"] += 1
        accepted = cycles["n"] != 1  # cycle 0 fails acceptance, cycle 1 passes
        cycle = {"index": cycles["n"] - 1,
                 "toolproof": {"accepted": accepted,
                               "session_proof": {"checked": accepted},
                               "no_owned_survivors": True},
                 "health_s": 1.0, "touch": {"ready_s": 1.0},
                 "stop": {"pid_gone": True,
                          "unit_state_after_stop": "inactive"},
                 "timings": [], "models": [glm_local.MODEL_ALIAS],
                 "launch": {}, "touch_io": {}, "touch_cgroup": {}}
        (cycle_dir / "server.log").write_text("", encoding="utf-8")
        return cycle

    monkeypatch.setattr(glm_local, "serve_cycle", fake_serve_cycle)
    monkeypatch.setattr(glm_local, "verify_model",
                        lambda *a, **k: {"ok": True, "shards": [],
                                         "bytes_on_disk": glm_local.TOTAL_BYTES,
                                         "total_ok": True, "deep": False,
                                         "models_dir": "x"})
    argv = ["prove", "--cycles", "2", "--evidence-dir", str(tmp_path / "e"),
            "--models-dir", str(tmp_path / "m")]
    assert glm_local.main(argv) == 1  # required tool proof rejected
    # Second run with acceptance satisfied exits zero.
    cycles["n"] = 0

    def fake_passing(cycle_dir, models_dir, **kwargs):
        cycle = fake_serve_cycle(cycle_dir, models_dir, **kwargs)
        cycle["toolproof"]["accepted"] = True
        cycle["toolproof"]["session_proof"]["checked"] = True
        return cycle

    monkeypatch.setattr(glm_local, "serve_cycle", fake_passing)
    import shutil
    shutil.rmtree(tmp_path / "e", ignore_errors=True)
    assert glm_local.main(argv) == 0


def test_prove_exit_code_fails_on_omitted_required_proof_and_failed_stop(
        tmp_path, monkeypatch, owned_engine_artifact):
    """Omitted required toolproof and a failed stop both exit nonzero —
    absence of evidence is not success."""
    import shutil

    def make_cycle(**over):
        cycle = {"index": 0, "health_s": 1.0, "touch": {"ready_s": 1.0},
                 "stop": {"pid_gone": True,
                          "unit_state_after_stop": "inactive"},
                 "timings": [], "models": [], "launch": {},
                 "touch_io": {}, "touch_cgroup": {}}
        cycle.update(over)
        return cycle

    def run_with(cycle):
        monkeypatch.setattr(glm_local, "serve_cycle", lambda *a, **k: cycle)
        monkeypatch.setattr(glm_local, "verify_model",
                            lambda *a, **k: {"ok": True, "shards": [],
                                             "bytes_on_disk": glm_local.TOTAL_BYTES,
                                             "total_ok": True, "deep": False,
                                             "models_dir": "x"})
        shutil.rmtree(tmp_path / "e2", ignore_errors=True)
        return glm_local.main(["prove", "--cycles", "1",
                               "--evidence-dir", str(tmp_path / "e2"),
                               "--models-dir", str(tmp_path / "m2")])

    assert run_with(make_cycle()) == 1  # required toolproof omitted entirely
    assert run_with(make_cycle(
        toolproof={"accepted": True, "no_owned_survivors": True},
        stop={"pid_gone": False,
              "unit_state_after_stop": "inactive"})) == 1
