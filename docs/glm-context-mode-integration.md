# Hermes / GLM / context-mode on the OptiPlex 5040

> **Retired GLM deployment.** The owner requested deleting GLM and researching
> a RAM-resident Qwen replacement. GLM's server/gateway were stopped and its
> model weights deleted; the unfinished lifecycle check was cancelled, not
> passed. Hermes, context-mode and indexed data are retained in the same profile.
> The GLM launch instructions below are historical, not the replacement setup.

The selected working profile is `/home/a1234/.local/share/hermes-glm/profile`;
its workspace is `/home/a1234/.local/share/hermes-glm/workspace`. The normal
`~/.hermes` profile is separate. Do not substitute the normal profile in these
commands or stop gateways by process name.

**Owner-approved delivery priority: setup first.** Complete the reversible
working setup before the lengthy real-model continuity and paired-benefit
certification. Tickets [#6](https://github.com/AntSentry/context-mode-hermes/issues/6)
and [#7](https://github.com/AntSentry/context-mode-hermes/issues/7) remain open
and explicitly deferred; their prepared branches are retained for resumption.
This staged handover does not certify a context-mode improvement.

## Evidence status

The pinned engine and adapter are installed in the selected profile. The real
GLM conversation exported by native Hermes contains eight rows: user request,
model-issued tool search, schema discovery, the matching context-mode search
call/result, and a completed assistant answer exactly equal to the independent
codeword. The codeword was absent from the user prompt. All seven operator
surfaces were exercised, including upgrade confirmation, purge preview and
Insight browser opt-in (the browser launch was intercepted, not hosted analytics
silently enabled).

The first lifecycle controller failed: it treated the codeword appearing in a
tool result plus a rendered prompt glyph as final-turn completion, requested an
export too early, and misclassified the detached native gateway as failed
cleanup. The export completed subsequently and contains the real final answer.
The failed report/transcript remain unchanged at
`/home/a1234/.cache/hermes-context-mode/evidence/ticket-5/run-xd7p0ao8/`.
The verified idle gateway was then stopped using Hermes' profile-scoped public
command, without forced interruption or process-group signals.

**The complete real-GLM removal/native-use/re-enable acceptance is still pending.**
Its recovery controller uses the public structured single-query terminal result
for model completion, then a native conversation export; it never treats silent
CPU prefill or terminal prompt decorations as completion. Recovery evidence is
under `/home/a1234/.cache/hermes-context-mode/evidence/ticket-5/recovery3-uzv7aoen/`.
Earlier recovery failures remain retained: session purge intentionally removed
its session-owned indexed chunks, contrary to the first recovery controller's
expectation; the next attempt correctly found that session already empty.
The continuation adopts the completed confirmed purge rather than inventing
another positive pre-purge count. The retention fixture is indexed **before**
removal, never afterwards. This document does not claim the pending cycle passed.

The third recovery exposed a real operator-path bug: after purge, `/ctx-index`
stored the new chunks, but project-scoped search hid them because the operator
path bypassed the model-tool capture hook and left no session-event membership.
Successful operator indexing now records its actual result through the existing
engine `posttooluse` hook, using trusted Hermes identity. The fix does not alter
the pinned engine, relax search scope or fabricate conversation history.
A real-engine regression failed before and passed after; all 12 command checks
passed. Actual native interactive index/search on the purged working session
also passed, with three real operator-capture events, correct workspace and no
owned survivors. That zero-inference repair smoke is retained at
`/home/a1234/.cache/hermes-context-mode/evidence/ticket-5/operator-index-fix-uas2urha/`.

Native Hermes compression remains in control. Real-model continuity and paired
net-benefit measurements remain separate acceptance gates. Existing controlled
provider measurements increased request bytes with context-mode enabled; they
are neither real GLM token counts nor proof of an improvement.

## Start and chat

The exact model, engine binary, memory limit and approved disk-backed runtime
are documented in [the local runtime guide](glm-5.3-flash-local-runtime.md).
The model does not fit entirely in this machine's RAM. Slow execution is
expected; a health response alone is not generation readiness.

Run from any directory:

```bash
PY=/home/a1234/.cache/hermes-context-mode/hermes-agent/.venv/bin/python
SRC=/home/a1234/Projects/context-mode-hermes
PROFILE=/home/a1234/.local/share/hermes-glm/profile
VENV=/home/a1234/.cache/hermes-context-mode/hermes-agent/.venv/bin

PYTHONPATH=$SRC "$PY" -m benchmarks.glm_local start
PYTHONPATH=$SRC "$PY" -m benchmarks.glm_local status
PYTHONPATH=$SRC "$PY" -m benchmarks.glm_local chat
```

The chat launcher selects that profile and workspace explicitly; it does not
replace the model/provider or disable native memory, toolsets or compression.
Startup and every model round trip can take a long time on CPU/mmap.

## Enable or re-enable

The retained engine is pinned to context-mode **1.0.169**; Hermes is the pinned
**v0.22.5** environment. For offline re-enable with the already installed engine:

```bash
ENGINE="$PROFILE/context-mode-runtime/node_modules/.bin/context-mode"
PYTHONPATH=$SRC "$VENV/context-mode-hermes" --home "$PROFILE" --engine-bin "$ENGINE"
```

For a fresh engine installation, use `--install-engine` instead of
`--engine-bin`; the installer uses its packaged nested dependency lock and
checks the engine version. Do not install an unpinned global package as a
replacement for this profile's runtime.

Restart the selected profile's gateway after registration changes. In a new
chat, `/ctx-stats` reports the engine version and `/ctx-doctor` reports real
checks. Claude-only upstream doctor checks are not Hermes compatibility proof;
preserve their annotated limitation instead of claiming every check is green.

## Operator commands

- `/ctx-stats` — context-mode statistics.
- `/ctx-doctor` — engine/integration diagnostics with platform limitations.
- `/ctx-index relative-file.md --source notes` — relative to the live workspace.
- `/ctx-search independent-phrase --source notes` — retrieve indexed content.
- `/ctx-upgrade`, then `/ctx-upgrade confirm` — operator-controlled upgrade.
- `/ctx-purge session`, then `/ctx-purge session <id> confirm` — irreversible
  deletion of the explicitly selected session's captured events **and its
  indexed chunks**. Project purge is broader. Neither is an uninstall command.
- `/ctx-insight` — explanation only; `/ctx-insight open` explicitly requests the
  hosted dashboard. Do not opt in automatically.

The model can use permitted context-mode tools through native Hermes discovery.
Runtime session/project identity is supplied by Hermes, not trusted from model
arguments. Model-side administrative purge/upgrade/Insight calls are blocked;
normal Hermes human-approval policy remains authoritative.

## Remove without deleting models or indexed content

Exit the client using `/exit`. This does **not** necessarily stop its detached
gateway. Stop only the selected profile's gateway, without `--all` or forced
interruption:

```bash
HERMES_HOME="$PROFILE" "$VENV/hermes" gateway stop
PYTHONPATH=$SRC "$VENV/context-mode-hermes" --home "$PROFILE" --remove
```

If Hermes reports active work, preserve it and let it finish; do not add `--yes`
or kill a guessed PID. The stop command above targets the selected profile,
not the llama.cpp model server. Start a new client with the normal `chat`
launcher to use native Hermes/GLM after removal.

`--remove` is configuration-only. It removes this adapter's plugin/MCP
registration while preserving the current model/provider, unrelated settings,
other profiles, model weights, retained engine and indexed knowledge. It needs
neither Node nor the engine nor network access. The reported recovery copy is
of the configuration at removal time; it is not an automatic rollback to an
old installation snapshot. Ordinary offline re-enable preserves later edits.

To re-enable, run the retained-engine command above and restart only this
profile's gateway again. Search the old indexed source to recover retained
knowledge; re-indexing should not be necessary. See the separate
[57-check configuration-only removal evidence](removal-acceptance.md) for the
already exercised zero-inference lifecycle; it does not replace the pending
real-GLM lifecycle gate.

Stopping the model server is a separate optional operation:

```bash
PYTHONPATH=$SRC "$PY" -m benchmarks.glm_local stop
```

Default removal never purges data or uninstalls the engine or Python package.
