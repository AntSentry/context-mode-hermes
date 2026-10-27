# GLM-5.3-Flash local runtime on the OptiPlex 5040 (ticket #4)

Operator/reproduction guide for bringing up the exact GLM-5.3-Flash
(`unsloth/GLM-5.3-Flash-GGUF` @ `a38483c8cd5df544f53d70fb281afe97369d5ab6`,
`UD-IQ4_XS`, 5 shards, **156,822,111,200 bytes** total, every shard
SHA256-verified against the HF tree API at that pinned revision) on
llama.cpp **b11669** CPU (`0.7.0-dev b11669 commit 620d2225c`,
`llama-server` SHA256 `3f2b53013c2eb596c8aca87e4740240a41f95bcebf72e61084d6d5ba0db761cc`;
release-archive SHA256 `cf07c1c112d709bd67e522017471de57ed0ec8bea031d732c5422351c07f65ac`)
under the pinned Hermes v0.22.5 checkout — then proving real tool-calling
through native Hermes against it.

Everything below is driven by `benchmarks/glm_local.py` (tested contract in
`tests/test_glm_local_contract.py`); raw evidence lands outside git in
`/home/a1234/.cache/hermes-context-mode/ticket-4-evidence/`.

## Hardware facts (from ticket #3, verified)

- i5-6500: 4 cores, AVX2, no AVX-512/AMX; no usable GPU (HD 530, 256 MiB).
- RAM `MemTotal` 63,627,908 KiB ≈ 60.68 GiB usable.
- Fast NVMe `/dev/nvme0n1p2` (~286 GiB free before download).

## Why this is not the native catalog bootstrap

Hermes' native `local_runtime` catalog admission plans against
`usable_vram_bytes`; on CPU-only that is RAM × (1 − 0.20 UMA headroom)
≈ 48.5 GiB. **Every** GLM-5.3-Flash GGUF exceeds that — smallest is
`UD-IQ1_S` at 86.7 GiB, `UD-IQ4_XS` is 146.05 GiB — so `recommended_entry`
would (correctly) refuse. The viable regime on this box is llama.cpp's
disk-backed residency: `--load-mode mmap` (explicitly **not** mlock), file
pages reclaimed under pressure, RAM holding the ~6.3 GiB
(config-derived, 65,536-token) KV state plus compute. The native facilities
we still keep are: the supervisor's readiness contract (health-200 is NOT
readiness — a temp-0 touch generation must answer), the custom
OpenAI-compatible provider consumption path, and the fork's PTY smoke
driver/owned-process ledger.

## Layout

- Model shards: `/home/a1234/.local/share/hermes-glm/models/GLM-5.3-Flash-UD-IQ4_XS/`
  (owned machine root `/home/a1234/.local/share/hermes-glm`; the default
  `~/.hermes` is never touched).
- Engine: `/home/a1234/.cache/hermes-context-mode/llama-b11669/llama-b11669/llama-server`.
- Hermes: `/home/a1234/.cache/hermes-context-mode/hermes-agent/.venv/bin/hermes`.
- Evidence: `/home/a1234/.cache/hermes-context-mode/ticket-4-evidence/`.

## Artifact acquisition

The recorded resumable download script verifies each shard's SHA256:
`/home/a1234/.cache/hermes-context-mode/ticket-4-evidence/download-ud-iq4xs.sh`.
The weights are already present on this machine. Manifest verification,
proof runs and persistent operation use the invocations below.

## Launch contract (see `build_server_cmd`)

`systemd-run --user --quiet --scope --unit=<unique-owned-scope>
-p MemoryHigh=38G -p MemoryMax=44G -p MemorySwapMax=0` wrapping:

`llama-server --host 127.0.0.1 --port 18443 --api-key-file <private-key-file>
-m <shard-1> -a GLM-5.3-Flash-UD-IQ4_XS -c 65536 -np 1 -t 4
--load-mode mmap --jinja --metrics --slots --no-webui`

Proof runs additionally use `--log-prompts-dir <evidence>/prompts`.

- Port 18443 deliberately avoids the native supervisor's stable 18434.
- `-c 65536` honors Hermes' `MINIMUM_CONTEXT_LENGTH` floor (64000); `-np 1`
  keeps one decode stream (KV ≈ 6.3 GiB config-derived).
- b11669 has no `--tool-call-parser`/`--reasoning-parser`: `--jinja` uses
  the GGUF-embedded GLM template, which carries the tool-call protocol.
- `--log-prompts-dir` retains engine-side prompt captures, corroborating the
  follow-up tool-result request; it is not a claim of complete HTTP wire capture.
- Ownership: the harness records the llama-server PID, `/proc` start time and
  unique cgroup at launch. Teardown verifies identity before signalling that
  PID (TERM→45s→KILL), never a process group.

## Resource observation

Per cycle the harness records: time to `/health` 200, `/v1/models` listing,
touch-generation readiness, per-request wall time and usage, `/proc/<pid>/io`
`read_bytes` deltas (disk paging), `statm` RSS, cgroup `memory.current` /
`memory.peak` / `memory.events.local`, raw `/metrics` and parsed metric rows, and llama.cpp's
own `prompt eval time … tokens per second` / `eval time … tokens per second`
log lines (parsed into `timings` in `summary.json`).

## Native Hermes consumption (no cloud, compressor intact)

`hermes_proof_config()` writes an owned profile (isolated under the evidence
dir, never inside `~/.hermes`) with:

- main model: explicit `custom` provider → `http://127.0.0.1:18443/v1`,
  `default: GLM-5.3-Flash-UD-IQ4_XS` (that shape has no cloud fallback);
- every enabled auxiliary task pinned to the same local endpoint with
  explicit per-task slow-CPU timeouts. Compression, title generation and
  background review use 21600s; other tasks use 600–3600s.
  `providers.custom` deadlines do NOT cover auxiliary tasks.
  Native activation of title generation/background review/compression is untouched.
  `free_only: true` is the one non-default routing flag — a safe tightening
  of the cloud lane that explicit custom pins already make unreachable; it
  is a tightening, NOT a standalone routing guarantee (an empty
  openrouter_model means the default free SKU if that lane were reached).
- the native compressor stays **enabled**, with explicit 21600s timeout and
  no-progress budget. A null no-progress value would select the native 60s default;
- `local_only_environment()` additionally scrubs ambient cloud credential
  keys (`OPENAI_API_KEY`, `OPENROUTER_API_KEY`, `ANTHROPIC_API_KEY`, …) the
  fork's `SCRUB_KEYS` does not cover; no `.env` is written into the owned
  home.

These long auxiliary deadlines accommodate CPU inference; they do not prove
that every native auxiliary task finishes successfully. Title/approval
classifiers use the same local GLM endpoint.

## Tool-calling proof

A fresh codeword is placed in `harbor-notes.txt`, absent from the user prompt.
Success requires a model-issued tool call, a matching tool-result row with the
codeword, and a later completed assistant answer in the native `/save json`
export. Terminal echo and engine prompt captures alone cannot pass the gate.
The PTY profile identity is verified before input; idle-prompt detection handles
the codeword and prompt arriving in one buffered chunk.

## Evidence manifest (ticket-4-evidence/)

- `download.log`, `download-ud-iq4xs.sh` — resumable digest-verified fetch;
- `llama-server-b11669-help.txt` — engine flag surface (loopback-only);
- `summary.json` — per-cycle launch/readiness/observations/timings/stop,
  plus the manifest verification;
- `cycle-N/server.log`, `cycle-N/prompts/`, `cycle-N/hermes-chat.log`,
  `cycle-N/hermes-home/config.yaml` (local-only routing).

## Measured results (2026-10-27, live arcs)

Attempt 1 (`cycle-0-attempt1-shared-load/`) ran while other adapter suites
shared the CPU — its 0.22 tok/s cold prefill is a **shared-load** figure.
The main arc (`summary.json`, cycles 0–1) ran isolated:

| Observable | cycle 0 | cycle 1 (restart) |
|---|---|---|
| deep manifest verify (5×SHA256) | ok | ok (gate re-run) |
| launch → `/health` 200 | 124 s | 114 s |
| touch generation → "paris" | 114 s | 105 s |
| prefill (warm cache) | 0.46–0.48 tok/s | same class |
| decode, 359-token answer | **0.27 tok/s** | 0.23 tok/s (359-tok 1565s) |
| disk reads during touch | 72.2 GB | — |
| stop | PID TERM→KILL gone, scope `inactive` | TERM→KILL gone, `inactive` |

- Earlier cycles' cgroup counters were not sampled because of a path bug.
  They remain unmeasured. The later adopted scope recorded MemoryPeak
  40,803,491,840 B; its 47,244,640,256 B MemoryMax is a cap, not a measured peak.
- Clean restart is proven by cycle 1 (full arc again after the first stop).
- The toolproof arc (`toolproof-rerun/`) drives `hermes chat --cli` against
  the live endpoint with the corrected `chat` subcommand and telemetry
  opt-out keys; its per-request engine captures live under
  `toolproof-rerun/cycle-0/prompts/` and its summary lands in
  `toolproof-rerun/summary.json`.

## Corrections & operations (2026-10-27, post-review)

- Decode: 0.27 tok/s (cycle 0) / 0.23 tok/s (cycle 1) — not a range claim.
- Both cycle stops were SIGTERM then SIGKILL (45s window) — no TERM-only stop.
- The toolproof attempts observed earlier were 900s `local_stream_stale_timeout`
  cancellations, not completions (see `native-deadline-diagnosis.json`); the
  owned profile now sets native `providers.custom.{stale,request}_timeout_seconds`
  = 21600 (single namespace, parent-native GREEN probe).
- Persistent owned operation: `python -m benchmarks.glm_local
  {start,status,chat,stop}` — seed-once profile at
  `/home/a1234/.local/share/hermes-glm/profile`, stable 0600 keyfile
  (`--api-key-file`, key never in argv), unique per-launch scope units,
  manifest-only per-PID stop. Published archive redacts loopback API keys
  (retired, loopback-only, never valid off-box); raw originals stay local;
  the toolproof codeword nonce is intentionally retained (non-secret,
  required correctness proof).
- Raw request capture (`--log-prompts-dir`) is opt-in (prove/debug only).


## Final live evidence (2026-10-27)

**Tool-calling gate ACCEPTED** (`toolflow-warm/toolflow-report.json`): on the
still-warm owned server (adopted by PID/start-time/cgroup proof, reusing the
cancelled attempts' 6153-token slot-cache prefix), native `hermes chat --cli`
with the 21600s `providers.custom` deadlines completed a real tool turn in
one heavy window (~98 min): PTY identity proven before every input, `/context`
dispatched, nonce file read via a model-issued tool call (matched
`tool_call_id` → tool response containing the nonce → LAST plain assistant
answer carrying the codeword — structural proof from the NATIVE `/save json`
export, not terminal echo). Hermes exited rc=0; the server was stopped by
SIGTERM→(identity-re-proven)SIGKILL; scope `inactive`; zero owned survivors.
Measured on that window: peak RSS (VmHWM) 40.6 GB; cgroup MemoryPeak
40,803,491,840 B (< MemoryMax 44 GiB); memory.events.local oom 0 (high
623,399 reclaim events recorded); lifetime disk reads 4.50 TB; decode at the
class measured earlier (0.27/0.23 tok/s).

**Persistent operator smoke** (`ops-smoke2.json`, `ops-chat-smoke2.log`):
public CLI `start`/`status`/`stop` all rc=0 (unique per-launch scope, 0600
keyfile, honest preflight: recorded RAM/disk/controllers + artifact sizes +
engine digest; post-health identity re-proof of the same PID/start-time/
unit). `chat` on the persistent profile rendered the idle prompt with NO
consent modal and its status bar showed the local model
(`☤ GLM-5.3-Flash-UD-IQ4_XS · ctx …`). The operator's unrelated config edit
and the stable key survived start/stop unchanged; manifests cleared; no
owned server or client survivors. The earlier smoke report recorded
`exit_sent: false` and forced cleanup, so it does not prove an `/exit` timeout.
A subsequent independent public-CLI smoke (`parent-ops-smoke/report.json`)
sent `/exit` after fresh identity verification and exited rc=0 without signals.
Its start/status/stop commands also exited rc=0; the profile and stable key
remained unchanged.

### Operator invocations (run from ANY working directory)

```bash
PY=/home/a1234/.cache/hermes-context-mode/hermes-agent/.venv/bin/python
SRC=/home/a1234/Projects/context-mode-hermes        # main integration checkout
PYTHONPATH=$SRC $PY -m benchmarks.glm_local start    # + /status /chat /stop
PYTHONPATH=$SRC $PY -m benchmarks.glm_local prove --cycles 2
PYTHONPATH=$SRC $PY -m benchmarks.glm_local verify-model [--deep]
```

`chat` execs the native interactive CLI with `HERMES_HOME=/home/a1234/.
local/share/hermes-glm/profile` and a workspace cwd of
`/home/a1234/.local/share/hermes-glm/workspace`. Raw evidence (unredacted,
local-only) lives in `/home/a1234/.cache/hermes-context-mode/ticket-4-evidence/`;
published archives are sanitized (all actual loopback API keys replaced,
including spawn-argv reprs; the codeword nonce is intentionally retained as
correctness proof).

### Portable regression checks

The proof-command exit-code regressions use an owned temporary artifact for
provenance hashing and controlled cycle outcomes. They do not require this
machine's llama-server path or weights on a CI runner. Actual GLM execution is
evidenced by the live transcripts above, not those isolated gate tests.

### Runtime review corrections and evidence provenance

Current proof and operator-start commands refuse a binary whose digest differs
from the approved engine pin; the shared launch builder enforces it too. Failed
generation cycles retain their error and verified shutdown report together.
Direct children are reaped before exit classification. If identity becomes
unreadable while a PID still exists, `pid_gone` remains unknown (`null`), not
successful cleanup. The retained controlled-HTTP diagnostic exercises failed
cycle/error retention and physical child shutdown without model inference.

The earlier archived serve cycles and adopted warm toolflow used the original
reused `hermes-glm-resource-probe` scope and a per-run key on argv. They predate
the unique-scope/key-file launcher. Retired keys are redacted in the public
bundle; the later independent operator smoke exercised the current launcher.
Do not attribute current flags retrospectively to those earlier runs.

Public evidence: `benchmarks/results/glm-local-runtime-evidence.tar.gz`.
Its detached `benchmarks/results/glm-local-runtime-evidence.review.json` records
the final archive digest and publication scan. The bundle intentionally retains
the failed review of the **earlier** archive `cdbc9cf10ea8c4963b0b44634451d31ba594c1ae536a20637cda876a98984bd0`;
that historical rejection is not the verdict on the final bundle. A provenance
correction in the published summary distinguishes its originally mislabeled
archive digest from the binary digest. Unredacted raw originals remain local
and unchanged.
