# Removal acceptance evidence (issue #2)

Operator-runnable, end-to-end proof that `context-mode-hermes --remove` is a
reversible, profile-scoped, configuration-only return to native Hermes —
exercised through the **public console CLI and the real interactive Hermes
client**, with zero model inference, on owned temporary profiles only.

- Ticket: [issue #2](https://github.com/AntSentry/context-mode-hermes/issues/2)
- Baseline: integration `eadf496` (ticket branch `ticket/2-reversible-removal`)
- Evidence bundle: `benchmarks/results/removal-acceptance-evidence.tar.gz`
  (sha256 `9282fd4e678eaa9123f28a73e274fd26257750ad0a5d6fd74dfd5dbd5c42f04d`)
- Raw run (outside git, retained):
  `/home/a1234/.cache/hermes-context-mode/evidence/ticket-2/run-20261027-032706-70c373`
- Verdict line: `CM-TICKET2: PASS evidence=…` — **57/57 checks passed**

The published archive redacts the generated loopback key from configuration snapshots. Recorded byte-exact/hash comparisons describe the original local run; the unredacted originals remain only in the local evidence directory above.

## What was proven (issue #2 acceptance criteria)

| Criterion | Evidence |
|---|---|
| Public CLI install / remove / re-enable | `artifacts/A-install.json`, `A-remove-1.json`, `A-reenable-offline.json` (exit 0 + JSON report each) |
| Removal is configuration-only; no engine/Node/npm/network needed | `C-remove-offline.json` ran with `PATH` = empty dir (no `node`/`npm` resolvable); `D-remove-engine-missing.json` removed a profile whose configured engine binary had been **deleted** first |
| Only this profile's integration is removed; model/provider + later native edits + unrelated plugins/MCP survive | `remove_preserves_operator_config` — config parsed by Hermes' own reader, `theme: light`, `operator_note`, `history`, `example-plugin` entry and loopback provider settings identical before/after; `other_profile_untouched` — profile B's `config.yaml` byte-identical across A's removal |
| No snapshot rollback; Hermes-native recovery copy, byte-exact | `recovery_backup_byte_exact`: `backups/config/config.yaml.context-mode-removal.<ts>` equals the just-removed `config.yaml` byte-for-byte; restoring that copy alone re-registers context-mode (`recovery_restores_registration`) while keeping every operator key |
| Data/runtime retained by default | `indexed_data_retained_on_disk` — storage tree under `<home>/context-mode` byte-identical (names+sizes+sha256) across removal; `shared_engine_not_uninstalled` — engine 1.0.169 binary and shared npm package untouched; adapter importable throughout |
| Repeated removal safe; restart explained; new process is native | `A-remove-2.json` (after manual recovery, `changed: true`), `A-remove-3-noop.json` (`changed: false`, config bytes unchanged, exit 0); console prints the gateway-restart guidance; session 2 ran in a **new gateway process** after an identity-verified gateway stop and proved: native `❯` prompt, `/ctx-stats` → `Unknown command`, zero context-mode processes under the profile, clean `/exit` |
| Native Hermes remains usable; re-enabled profile retrieves retained content | `transcript/session1-installed.log` (`/ctx-stats`, `/ctx-doctor`, `/ctx-index`, `/ctx-search` live), `session2-removed.log` (native, no ctx), `session3-reenabled.log` (ctx live again; interactive search over the pre-removal index returned the sentence indexed before removal) |
| Independent retrieval via public MCP | `artifacts/mcp-probe.txt` — engine 1.0.169's public MCP stdio interface, new process, answered a **never-queried phrase** (`harbors before dawn`) with the sentence indexed before removal; no re-indexing performed |
| Safety envelope | `no_model_inference` — loopback provider rejected every `/v1/chat/completions` (0 attempts; 18 recorded requests were all non-inference); every interactive Hermes PID was `/proc`-identity-verified (`HERMES_HOME` == owned home) **before the first keystroke**; cleanup signaled only per-PID identity-verified owned processes (HERMES_HOME + start time, never process groups); `default_profile_unchanged` — `~/.hermes` fingerprint identical before/after; `no_owned_survivors` |

## Reproduction

Prerequisites: pinned Hermes v0.22.5 CLI + adapter venv
(`/home/a1234/.cache/hermes-context-mode/hermes-agent/.venv/bin`), a context-mode
1.0.169 engine executable, and a scratch directory owned by you.

1. `PYTHONPATH=<checkout> <pinned-venv>/bin/context-mode-hermes --home <fresh-owned-home> --engine-bin <engine-169>`
2. Use the profile (or edit config with Hermes' own writer) — later edits must survive.
3. `.venv/bin/context-mode-hermes --home <home> --remove` — note the reported `recovery_backup`, then restart the profile's gateway so new sessions run native Hermes.
4. Repeat `--remove` (no-op success), restore the recovery copy (registration returns), re-enable offline with the same `--engine-bin` (no npm/network).
5. The bundle contains the captured `removal_acceptance.py` at its archive root. It records the full automated lifecycle above. Its checkout, Hermes, engine and evidence-root paths are machine-specific constants near the top; update those paths before a fresh run from an extracted copy. It refuses to start if a process already carries `HERMES_HOME` under the chosen evidence root.

## Operational notes discovered while producing this evidence

These are Hermes-platform behaviors an operator must know when validating
removal (they are not adapter defects — the adapter behaved correctly in every
phase):

1. **Never park test/owned homes under a directory literally named `profiles/`.**
   Hermes v0.22.5 treats `<root>/profiles/<name>` as a *named profile* of a
   multiplexing host gateway (`hermes_cli.gateway_runtime_multiplex`); the CLI
   then parks it ("only one profile exists (nothing to multiplex)") and the
   session never gets a gateway. Use any other container name (the runner uses
   `homes/`).
2. **`/exit` ends the client, not the gateway.** The gateway daemon survives to
   serve the next session — so "new sessions start native Hermes" after
   `--remove` is only observable after that profile's gateway process ends.
   The remove console already prints this restart requirement; validation (and
   operators scripting removal) must stop the old gateway first. The runner
   does this strictly per PID (`/proc` `HERMES_HOME` identity + start time
   re-verified before each signal).
3. A gateway start can take ~15–35 s on this machine; driver code must wait
   for real content markers (the `❯` glyph also appears in the status bar).

