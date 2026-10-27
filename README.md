# context-mode-hermes

A Hermes-native integration for the [context-mode](https://github.com/mksglu/context-mode) engine: a lifecycle-hook bridge plus an MCP proxy that connect a Hermes Agent profile to a pinned, profile-owned context-mode runtime.

This repository is a local engineering fork of [christopher-s/context-mode-hermes](https://github.com/christopher-s/context-mode-hermes) (fork point `154ea3f`). It is **not published on PyPI** — install it from a checkout as described below.

## Honest scope

What this integration does:

- Captures routed tool results and user prompts into the engine's per-project session database (session memory).
- Re-injects a routing/restore block at session start, after resume, and after committed compactions, so captured knowledge stays reachable across context loss.
- Exposes the engine's MCP tools (`ctx_execute`, `ctx_search`, `ctx_index`, …) through an isolated per-session proxy.
- Adds operator slash commands (`/ctx-stats`, `/ctx-doctor`, `/ctx-index`, `/ctx-search`, `/ctx-upgrade`, `/ctx-purge`, `/ctx-insight`).

What it does **not** do:

- **It is not a token-saving optimization.** The controlled benchmark below measured a net *increase* in wire bytes and rough token estimates with the plugin enabled, on both workloads. Do not deploy it always-on expecting context-window savings; native Hermes already spills oversized terminal output to files on its own.
- **It has not been validated against any real model.** No LLM quality, speed, or cost claims are made (and none are possible from the controlled benchmark). In particular there is no GLM validation of any kind — no GLM weights or inference were used anywhere in this work.
- **It is not a security sandbox.** See [Security & isolation](#security--isolation).

## Compatibility

| Component | Requirement |
|---|---|
| Python | ≥ 3.11 (developed and exercised on 3.14) |
| Node.js / npm | ≥ 22.5 (validated run used Node v22.23.2) |
| Hermes Agent | **Tested: v0.22.5** — the installer checks required plugin/MCP interfaces, not the release tag; other releases are unvalidated |
| context-mode engine | **1.0.169**, pinned by the packaged npm lock |

The adapter relies on the pinned Hermes venv providing `hermes_cli` and the `mcp` Python package (it declares no runtime dependencies of its own).

## Installation

Use the pinned Hermes environment with its MCP extra. For a fresh checkout:

```bash
git clone --branch v0.22.5 https://github.com/NousResearch/hermes-agent.git /path/to/hermes-agent
uv sync --project /path/to/hermes-agent --locked --python 3.14 --no-dev --extra mcp

# Install this local adapter checkout, not the older PyPI package.
uv pip install --python /path/to/hermes-agent/.venv/bin/python -e /path/to/context-mode-hermes

# Install the locked engine and configure THIS profile only.
/path/to/hermes-agent/.venv/bin/context-mode-hermes --home /path/to/hermes-profile --install-engine

# Alternative to --install-engine: --engine-bin /absolute/path/to/context-mode
# After configuring the profile's model/provider, launch that same profile:
HERMES_HOME=/path/to/hermes-profile /path/to/hermes-agent/.venv/bin/hermes --cli
```

Restart an already-running gateway for that profile, then run `/ctx-doctor`.

What the installer does:

- `--install-engine` npm-installs `context-mode@1.0.169` into `<home>/context-mode-runtime` using the packaged `package.json`/`package-lock.json` with `--install-strategy=nested`, so transitive dependencies (including `better-sqlite3`) stay inside the profile-owned prefix. It refuses symlinked or shared/global npm prefixes and verifies the installed version is exactly 1.0.169.
- Writes `config.yaml`: enables the plugin (removing only its own previous `plugins.disabled` entry), records `plugins.entries.context-mode.settings.{engine_binary, storage_dir}`, grants `mcp_allowlist: [context-mode]`, and registers the MCP server `context-mode` (`<venv-python> -m context_mode_hermes.mcp_proxy --engine-bin …`, 300 s tool / 60 s connect timeouts) with an isolated environment.
- Backs up the existing config to `config.yaml.before-context-mode` before writing.
- Modifies no other agent's profile/configuration. Installing Python/npm dependencies also uses their ordinary environment and package caches.

**Activation is installer-gated.** The lifecycle hooks engage only when both installer-written settings (`engine_binary` and `storage_dir`) are present and the binary is executable. There is no ambient `PATH`/`CONTEXT_MODE_BIN` activation, so the plugin can never silently capture into an unrelated profile's store. If the settings are missing or the engine is unavailable, neither the hooks nor the `/ctx-*` slash commands are registered (a `[context-mode]` warning is logged) — an unconfigured profile gets no bridge and no dead command handlers.

**Model/provider:** the installer does not configure models. The profile still needs its own `model:`/provider configuration for real use; only the engine, plugin, and MCP wiring are managed here.

## How it works

**Lifecycle bridge** (`context_mode_hermes/__init__.py`) — one bridge per plugin registration, mapping Hermes hooks onto the engine's `hook claude-code` CLI (subprocess, 8 s timeout):

| Hermes hook | Bridge behavior |
|---|---|
| `pre_tool_call` | Routed tools (`terminal`, `read_file`, `search_files`, `web_extract`) → engine `pretooluse`; deny→block, ask→approve, `updatedInput`→modify. Every `ctx_*` call gets the trusted `_hermes_context` injected (LLM-supplied values are overwritten). Admin tools (`ctx_purge`/`ctx_upgrade`/`ctx_insight`) are operator-only and blocked for model-side calls. |
| `post_tool_call` | Captured tools → engine `posttooluse` (per-project session memory). |
| `transform_tool_result` | Appends deferred PreToolUse guidance to the tool result. |
| `pre_llm_call` | `sessionstart` injection once per boundary (startup / resume / compact); `userpromptsubmit` once per assistant turn (delegated turns excluded). |
| `post_llm_call` | Turn end → engine `stop`. |
| `post_compaction` | Hermes has no precompaction hook; after a committed `summarize`/`provider_native` compaction the bridge snapshots the already-captured session DB (engine `precompact`) and queues re-injection. Micro-summarize/prune housekeeping is ignored. |
| `on_session_start` / `on_session_reset` / `on_session_delete` / `on_session_finalize` | Mark-for-injection and state cleanup; `on_session_end` is deliberately not registered (it fires every turn). |

Hermes' native context compressor is left completely in charge of compaction itself — the bridge only reacts to committed compactions.

**MCP proxy** (`context_mode_hermes/mcp_proxy.py`) — Hermes launches the proxy as its `context-mode` stdio MCP server. The proxy augments the engine's tool schemas with the trusted `_hermes_context` property and routes each `tools/call` to a per-session engine worker subprocess keyed by `(session_id, project_dir)`, in a bounded LRU pool (default 4 workers). Path-like arguments are anchored to the session's live cwd before forwarding.

**cwd anchor** — the first workspace a session uses is recorded atomically under `<home>/context-mode-host/session-roots/`; later `cd`s keep the live cwd for execution but never move the conversation's captured memory to a different database.

**Isolation** — every engine subprocess runs with foreign client identifiers stripped (`CLAUDE_*`, `PI_*`, `OMP_*`, `CONTEXT_MODE_PLATFORM`, …) and a profile-local `CLAUDE_CONFIG_DIR`/`XDG_CONFIG_HOME` under `<home>/context-mode-host`, so the engine cannot pick up another agent's configuration. Redirect-shaped denies are only issued when this Hermes process itself reports the `context-mode` MCP server as available; security-policy denies never depend on MCP availability.

**Fail-open by design** — hooks only advise; every hook failure is swallowed (logged as `[context-mode] …`) so a bridge problem can never veto a tool call or kill a session. Hermes' own approval/trust controls remain authoritative.

## Storage layout

```
<home>/context-mode/                  engine storage (knowledge base, session DBs)
<home>/context-mode-runtime/          pinned npm engine (nested deps)
<home>/context-mode-host/             isolated client config + session anchors
<home>/config.yaml.before-context-mode  pre-install config backup
```

## Slash commands

Seven operator commands are registered in-session. The engine's admin MCP tools (`ctx_purge`, `ctx_upgrade`, `ctx_insight`) are operator-only: model-side calls to them are blocked with guidance to ask the user to run the matching slash command instead.

| Command | Purpose |
|---|---|
| `/ctx-stats` | Engine counters. **Approximation:** these are profile-server-wide process counters, not scoped to the current conversation. |
| `/ctx-doctor` | The upstream engine report verbatim, plus real Hermes-side validation (engine version pin, `mcp_servers` config, `mcp_allowlist` grant). The engine's `settings.json` hook-wiring check is Claude-Code-specific and annotated as such — a `FAIL` there is expected and *not applicable* under Hermes, which wires hooks through its plugin API. No check is reported green that was not actually validated. |
| `/ctx-index <path> \| --text "…"` | Index a path (or inline text) with `--source LABEL`, `--include/--exclude GLOB`, `--extensions LIST`, `--max-depth/--max-files`, `--no-gitignore`, `--follow-symlinks`. |
| `/ctx-search` | BM25 search over indexed content + session memory: `-q QUERY` (repeatable), `-s/--source`, `-l/--limit`, `-c/--content-type`, `--sort relevance\|timeline`. |
| `/ctx-upgrade [confirm]` | Staged, profile-owned update to the plugin-tested engine version: npm-install into a staging prefix with the packaged lock, validate execution plus a real SQLite/FTS index-and-retrieve round-trip, then reversible rename swap keeping the old runtime as `context-mode-runtime-backup-<ts>`. Refuses runtimes newer than the plugin-tested version; the upstream global/Claude upgrade script is never executed. Requires a gateway restart afterward. |
| `/ctx-purge` | Irreversible purge; the literal word `confirm` is required — `/ctx-purge session confirm` (current session), `/ctx-purge session <id> confirm`, or `/ctx-purge project confirm`. Without `confirm` nothing is deleted. |
| `/ctx-insight [open]` | Without arguments, explains the hosted dashboard. `open` explicitly launches context-mode.com/insight; it does not enable analytics collection. |

Use these commands in the interactive CLI (`hermes --cli`). The verified `hermes chat -q ...` path sends its argument to the model; it is not a slash-command dispatcher.

## Security & isolation

- **`ctx_execute` is not a security sandbox.** Code runs as the user on the host with full user access; the engine bounds what *enters the conversation*, never what the code can do. Treat it as output hygiene, not containment.
- All bridge hooks are advisory and fail-open (see above); blocking decisions come from the engine's routing rules, and Hermes' approvals remain the gate for anything dangerous.
- Session identity (`_hermes_context`) is bridge-owned and injected over any model-supplied value; MCP calls carry it invisibly and the proxy strips it before the engine sees the arguments.
- Admin operations (purge/upgrade/insight) are operator-only via slash commands.

## Measured results

The benchmark (`benchmarks/run_bench.py`) drives the **real pinned Hermes AIAgent** — fresh child process per run, native plugin manager, native MCP discovery, native context compressor and terminal spill — against a scripted, deterministic OpenAI-compatible provider on loopback. It reports exact wire request bytes and Hermes' own `estimate_request_tokens_rough` (a byte-derived estimate, **not real model tokens**). It self-labels as a controlled-provider measurement: no LLM quality or speed claims are possible from it.

Run the paired comparison with the pinned Hermes interpreter and engine:

```bash
<pinned-hermes-venv>/bin/python -m benchmarks.run_bench \
    --engine-bin "$CONTEXT_MODE_TEST_BIN" \
    --arms baseline,plugin --probe-untrusted-context
```

Approvals are off in **both** arms because the run is unattended; native compression, terminal caps and progressive tool disclosure stay enabled. Both profiles are independently configured and isolated.

### Run 2026-10-26 · `benchmarks/results/integrated-compact-guidance.json`

Environment: engine 1.0.169, Node v22.23.2, pinned Hermes venv (Python 3.14), advertised context 65,536, large-workload fixture 266,674 bytes (4,546 values, sum 2,279,940). Both arms answered both workloads correctly (`correct: true`); the numbers below are totals over all scripted requests.

| Workload | Arm | Requests | Wire bytes | Rough tokens |
|---|---|---:|---:|---:|
| small | baseline | 2 | 19,578 | 4,682 |
| small | plugin | 2 | 22,502 | 5,400 |
| large (266,674 bytes) | baseline | 4 | 49,887 | 11,922 |
| large (266,674 bytes) | plugin | 5 | 76,435 | 18,249 |

Net effect of enabling the plugin: **+2,924 bytes / +718 rough tokens** (small) and **+26,548 bytes / +6,327 rough tokens / +1 request** (large). The injected routing guidance and MCP tool schemas cost more than the routing saves on these workloads — the performance gate was failed, and the plugin is not recommended as an always-on token-saving measure. The baseline already spilled the oversized terminal output to files via native Hermes caps, which is where most of the theoretical saving would have come from.

## Tests

Integration tests exercise the real pinned engine and are skipped unless `CONTEXT_MODE_TEST_BIN` points at the pinned nested engine binary. Run the suite under the pinned Hermes venv:

```bash
export CONTEXT_MODE_TEST_BIN=<home>/context-mode-runtime/node_modules/.bin/context-mode
<pinned-hermes-venv>/bin/python -m pytest tests/
```

Observed locally on the pinned stack: **41 tests passed in 110.45 seconds**. Recovery tests dispatch Hermes's compaction lifecycle event and verify a captured decision is restored and searchable through the real MCP transport; they do not exercise a real model's summarization quality. Tests also cover destructive-command confirmation, preservation of sibling sessions, independent workspaces, forged metadata and worker cleanup.
The actual PTY-driven interactive CLI also exercised stats, doctor, relative-file/inline indexing, search, and a real staged upgrade from engine 1.0.168 to 1.0.169. A fresh session loaded the new engine and retrieved previously indexed content. A separate command smoke retained the raw loopback-provider log: five capability GETs and zero inference POSTs. The doctor still reports the annotated Claude-only hook-file failure; this is not hidden.

Validation incident: an early smoke-driver environment error launched against the default profile, and cleanup terminated a gateway before verifying its ownership. It may have interrupted the user's normal gateway. Writes were observed in that profile during the incident window, including `config.yaml`; attribution between these launches and gateway initialization is uncertain, and no pre-incident config copy was available for a safe rollback. The current config contains only disabled shared telemetry. Later smoke runs used explicit temporary profiles and checked process ownership. The retained runtime evidence includes the corrected incident notes; the default profile was not modified further to guess at a repair.
Evidence: [`benchmarks/results/runtime-cli-evidence.tar.gz`](benchmarks/results/runtime-cli-evidence.tar.gz) contains the CLI transcripts, installer output and incident notes. The paired request-cost measurements are in [`integrated-compact-guidance.json`](benchmarks/results/integrated-compact-guidance.json).


- `tests/test_lifecycle.py` — real plugin registration on the pinned `PluginContext`/`PluginManager`, real Hermes dispatch seams, real engine subprocesses, assertions against the actual per-project session DB.
- `tests/test_engine_contract.py`, `tests/test_install_contract.py`, `tests/test_mcp_proxy_contract.py`, `tests/test_native_hermes_contract.py`, `tests/test_commands.py`, `tests/test_verify_contract.py` — engine CLI contract, installer output/config, proxy protocol, native Hermes integration, slash-command behavior against the real engine, and the safe verification smoke (owned profile, loopback provider with zero inference, identity-verified cleanup under interruption).
- `tests/test_removal_contract.py` — `--remove` contract: configuration-only removal preserving later-edited settings/data, no engine/Node/npm needed, no-op and refusal exit statuses, offline install→remove→re-enable with retained searchable content.

## Safe verification smoke

`context-mode-hermes-verify` (also `python -m context_mode_hermes.verify`) is an operator-run **diagnostic** — not a benchmark and not a model-quality or GLM-benefit claim. On a verification-owned temporary Hermes profile it proves that the pinned **interactive** Hermes CLI (`hermes chat --cli`, never one-shot `-q`) dispatches the real `/ctx-stats`, `/ctx-doctor`, `/ctx-index` and `/ctx-search` commands through the pinned engine, that the indexed content is independently re-retrieved through the engine's public MCP stdio interface (same project) using a phrase that was never typed as a query — so no terminal echo, repaint or history replay can satisfy the proof — and that all of this happens with **zero model inference**: a loopback provider on 127.0.0.1 records every raw request (credentials redacted) and rejects `/v1/chat/completions` with HTTP 503, and any inference attempt aborts the dispatch wait promptly and is reported as a refusal with retained evidence — never as successful output.

```bash
# from this checkout, with the pinned Hermes venv python (it must be able to
# import both this checkout and hermes_cli for the installer subprocess):
<pinned-hermes-venv>/bin/python -m context_mode_hermes.verify \
    --hermes <pinned-hermes-venv>/bin/hermes \
    --engine-bin <home>/context-mode-runtime/node_modules/.bin/context-mode \
    --evidence-dir ./verify-evidence-<run>
```

Safety envelope, regression-tested in `tests/test_verify_contract.py`:

- **Owned temporary profile only.** The smoke runs in a fresh `mkdtemp` profile; the ambient `HERMES_HOME` (else `~/.hermes`) is scrubbed from every child and fingerprinted before/after — it must stay identical. The default profile is never launched against, configured, or cleaned up, and evidence/profile-root placements inside it are refused outright.
- **Identity-verified process handling.** Children are spawned in their own process group and tracked by handle; cleanup signals that group and any process whose own `/proc/<pid>/environ` carries `HERMES_HOME=<owned profile>` (a valid unchanged start time plus the owned `HERMES_HOME`; a missing start time on both sides is absence of proof, never identity). Nothing is ever matched or killed by process name, command line, or an unverified PID, so similar-looking gateways and workers in other profiles always survive.
- **Bounded, interrupt-safe.** SIGINT/SIGTERM interrupt bounded polls instead of waiting out full subprocess timeouts, clean up owned children within bounded waits (TERM, then KILL), retain the partial evidence, and exit 130/143 — interruption is never success; if any owned survivor outlives bounded cleanup, the owned profile is retained and its path recorded. Unresponsive owned workers are escalated to SIGKILL; foreign processes are never touched.
- **Evidence, not environment dumps.** The evidence directory keeps the raw PTY transcript (ANSI included), one JSON file per provider request as it arrives (bodies complete, `Authorization`-style headers and the generated API key redacted), the installer output, identity proofs for every signal sent, and `verify-report.json`.

Output convention: the final stdout line is `CM-VERIFY: PASS|FAIL|REFUSED|INTERRUPTED evidence=<dir>` with exit codes 0/1/2/130(-143); `--json` additionally prints the full report. The report's `checks` map names every criterion (exact pins, profile isolation, each dispatch, `mcp_stored_content`, `no_inference`, `child_exited`, `default_profile_unchanged`, `no_owned_survivors`, `telemetry_opted_out`); a refusal fires before any interactive command is dispatched (wrong Hermes pin — exact-match, not substring; wrong engine version; non-empty evidence directory; child identity mismatch; unsafe evidence/profile placement).

The owned profile is deleted at the end unless `--keep` is passed or an owned survivor outlives bounded cleanup; the evidence directory is always kept.

## Hosted CI

[`.github/workflows/ci.yml`](.github/workflows/ci.yml) runs on a clean GitHub-hosted runner and needs no credentials, model weights or access to the original workstation:

- **`integration`** — the pinned stack above (Hermes v0.22.5 via locked `uv` sync on Python 3.14, Node 22): editable adapter install, nested engine install through the public `context-mode-hermes --install-engine` command, then the full real-engine suite with `CONTEXT_MODE_TEST_BIN` pointed at that nested engine. The run fails if any test is *skipped*, so a missing engine prerequisite can never become a silent green check.
- **`clean-distribution-smoke`** — builds the wheel from the checkout and installs it (never editable) into the locked pinned Hermes v0.22.5 environment on Python 3.14 — the same stack the integration job exercises. From outside the checkout, the wheel's own console command installs the locked `context-mode@1.0.169` engine, and [`clean_distribution_smoke.py`](.github/scripts/clean_distribution_smoke.py) then drives the real engine over its public MCP stdio interface: `ctx_execute` must print a value it computed, `ctx_index` + `ctx_search` must round-trip stored text that an echo of the query cannot satisfy, and `context_mode_hermes` must import from the wheel install rather than the source tree. The installed engine version must equal the packaged lock.

The adapter's metadata floor is Python 3.11, but hosted CI exercises only the pinned Hermes v0.22.5 + MCP stack on Python 3.14 (both jobs); Python 3.11–3.13 are not exercised anywhere in this repository's validation.

Both jobs use only a read-only `contents: read` token; the workflow holds no secrets.

Hosted CI proves packaging and adapter integration behavior only. It runs no model, no benchmark and no hardware-specific measurement: GLM runtime, continuity and benefit evidence is the separate local reporting under [Measured results](#measured-results), and a green pipeline is not a real-model benefit claim. Hosted run links for the published integration branch and its PR are recorded by the maintainer in [issue #8](https://github.com/AntSentry/context-mode-hermes/issues/8).

## Removal & reversibility

```bash
/path/to/hermes-agent/.venv/bin/context-mode-hermes --home /path/to/hermes-profile --remove
```

`--remove` is mutually exclusive with `--install-engine`/`--engine-bin` and needs neither the engine nor Node/npm nor network access. It removes only this profile's registration — the `context-mode` plugin entry (and its `plugins.enabled`/`plugins.disabled` mentions) and the `context-mode` MCP server — rewriting the **current** `config.yaml` through Hermes' atomic writer, so settings changed after installation (model/provider, unrelated plugins, other MCP servers, other profiles) are preserved rather than rolled back to the pre-install snapshot. Before the rewrite, Hermes' native config-backup helper stores a verified current recovery copy under `<home>/backups/config/` (`config.yaml.context-mode-removal.<timestamp>`, newest five kept; if the newest copy already matches byte-for-byte it is reused); if no verified copy can be created, removal fails (exit 1) without touching the file. It never deletes the engine/data locations recorded at install — by default `<home>/context-mode` (indexed knowledge, captured sessions), `<home>/context-mode-runtime`, `<home>/context-mode-host`, plus `config.yaml.before-context-mode` — or the installed adapter package. A profile without a prior integration is a successful no-op that writes nothing; an unreadable/malformed `config.yaml` makes removal fail (exit 1) without touching the file. Restart the profile's gateway to finish removal: running gateways keep hooks, `/ctx-*` commands and MCP tools in-process until they restart; new sessions start native Hermes with none of them.

Re-enable later with the ordinary install command for the same `--home`; the retained engine is reused (no npm/network needed) and previously indexed content and captured sessions are reachable without re-indexing.

To additionally drop the retained data or the adapter — explicit, destructive, operator-owned choices the removal above deliberately never makes:

```bash
uv pip uninstall --python <pinned-hermes-venv>/bin/python context-mode-hermes
rm -rf <home>/context-mode <home>/context-mode-runtime <home>/context-mode-host
```

A failed `/ctx-upgrade` swap restores the previous runtime automatically; a successful one keeps the backup directory beside the runtime for manual rollback. Any engine-data removal goes through `/ctx-purge` before deleting the storage directory.

## License

MIT © 2026 Chris Staley. Upstream projects: [context-mode](https://github.com/mksglu/context-mode) (engine, pinned here at 1.0.169 — see [context-mode.com](https://context-mode.com)) and [Hermes Agent](https://github.com/NousResearch/hermes-agent) (v0.22.5); this repository is a fork of [christopher-s/context-mode-hermes](https://github.com/christopher-s/context-mode-hermes).
