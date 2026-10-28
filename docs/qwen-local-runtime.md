# Qwen on the OptiPlex 5040

The owner requested deletion of GLM and a fresh Qwen comparison before replacement. The GLM gateway/server were stopped, its unfinished validation cancelled, and its 146.05 GiB weight set deleted. Its historical evidence is not Qwen acceptance evidence.

## Selected artifact

- Model: **Qwen3.8-27B**, dense, Apache-2.0; text-only deployment.
- Repository: [ggml-org/Qwen3.8-27B-GGUF](https://huggingface.co/ggml-org/Qwen3.8-27B-GGUF/tree/a316ea527fde32c12487be7574e2db5ac2ad2a7e).
- Revision: `a316ea527fde32c12487be7574e2db5ac2ad2a7e`.
- File: `Qwen3.8-27B-Q8_0.gguf`.
- Bytes: `28,595,763,648` (26.63 GiB).
- SHA-256: `aab65c67ef0dad127960efef9247f1832bca105faa1c7a052cc039b223cf86a1`.
- Download: Hermes' native resumable downloader, pinned HTTPS URL and hash verification before atomic publication; completed in 614.78 seconds on this machine.
- Runtime: Hermes v0.22.5, its package-manager-pinned CPU llama.cpp b11671 (`ba3c05f71`). No GPU, vision projector, draft model, cloud provider or GLM compatibility server is required.

[Selection research and benchmark caveats](qwen-selection.md). The reference AA Intelligence Index is 33.68 at xhigh effort and 27.55 at medium; neither is a measured score for this local quantized artifact. Flash-Next scores higher but its smallest published quant already exceeds physical RAM before runtime memory. The 35B-A3B alternative favors speed rather than the highest reference intelligence score.

## Profile isolation

The preserved profile remains `/home/a1234/.local/share/hermes-glm/profile`; its workspace remains `/home/a1234/.local/share/hermes-glm/workspace`. The historical directory name is **not** the active model name. Keeping it preserves existing project anchors and context-mode indexed identities. The normal `~/.hermes` profile is separate.

The GGUF lives in the native managed model store:

```text
/home/a1234/.local/share/hermes-glm/profile/models/Qwen3.8-27B-Q8_0.gguf
```

The profile uses `model.provider=llamacpp`, model ID `Qwen3.8-27B-Q8_0`, and `local_runtime.backend=cpu`, with **no fixed `model.base_url` or API key**. Hermes resolves the current managed endpoint. Only one model is admitted concurrently. All 19 explicit GLM auxiliary-task pins were migrated, including internal tasks absent from the model picker's bulk assignment. The normal profile's configuration remained unchanged.

Native admission generated a **262,144-token** runtime window with q8_0 K/V caches, flash attention enabled, and `spilled=false`. This is Hermes' automatically admitted runtime configuration, not a demonstrated 262k-token workload or a peak-memory measurement. The research's 65,536-token/f16 figures are comparison estimates, not the generated launch settings.

## Launch and stop

From the retained workspace:

```bash
cd /home/a1234/.local/share/hermes-glm/workspace
HERMES_HOME=/home/a1234/.local/share/hermes-glm/profile \
  /home/a1234/.cache/hermes-context-mode/hermes-agent/.venv/bin/hermes --cli
```

The native managed runtime starts on demand. A clean exit stops a server owned by that Hermes process; a process that adopted another Hermes process's server does not stop its owner. `/exit` closes the interactive client, not necessarily its gateway. If a gateway owns the server, stop only this selected profile with `HERMES_HOME=/home/a1234/.local/share/hermes-glm/profile /home/a1234/.cache/hermes-context-mode/hermes-agent/.venv/bin/hermes gateway stop`; an active turn requires the command's explicit interruption confirmation. No separate custom model daemon is installed by this cutover.

The selected generic GGUF carries a thinking-enabled template with an xhigh default; unlike the curated Q4 entry, this file does not automatically inherit the catalog's medium-effort sampling recipe. Xhigh is verbose and can be slow on four CPU cores. Hermes exposes `/reasoning` and the invocation-specific `--reasoning` option; a lower-effort run must not be described using the xhigh benchmark score. Actual wire behavior is part of runtime verification, not inferred from the model's name.

**Native endpoint caveat:** the model activation helper saved a loopback URL, but the managed server can select a different port when its preferred port is busy. An explicit URL overrides dynamic `llamacpp` resolution. The fixed URL was removed through Hermes' atomic config writer, and resolution to the live managed endpoint was verified. This did **not** make the full tool turn complete within its deadline. Connection-refused errors after the test-owned server shut down must not be misread as the cause of its earlier slow prefill. When selecting a model again, avoid an obsolete explicit endpoint and restart the selected gateway to discard cached provider clients.

The gateway-backed `chat -q` path rejects the otherwise advertised `--run-budget` CLI flag in this pinned release. Its error directs operators to `agent.run_budget_seconds` instead. The bounded smoke used an external deadline; it did not disable approval controls or enable cloud fallback.

## Context-mode and removal

The existing engine is retained at version 1.0.169, along with profile-owned indexes, captured sessions, lifecycle hooks, MCP registration and operator commands. Inside the selected Hermes profile, `/ctx-doctor` checks the integration; `/ctx-index`, `/ctx-search` and `/ctx-stats` operate without requesting inference from the language model. `/ctx-upgrade`, `/ctx-purge` and `/ctx-insight` remain operator-only, confirmation-gated controls.

To remove **context-mode registration only**, first exit this profile's active Hermes session/gateway, then run:

```bash
/home/a1234/.cache/hermes-context-mode/hermes-agent/.venv/bin/context-mode-hermes \
  --home /home/a1234/.local/share/hermes-glm/profile --remove
```

This retains Qwen, Hermes, the engine and indexed data. It does not restore a stale pre-install config or switch back to GLM. Restart the selected profile after removal.

To re-enable the retained engine without downloading it again:

```bash
/home/a1234/.cache/hermes-context-mode/hermes-agent/.venv/bin/context-mode-hermes \
  --home /home/a1234/.local/share/hermes-glm/profile \
  --engine-bin /home/a1234/.local/share/hermes-glm/profile/context-mode-runtime/node_modules/.bin/context-mode
```

Restart the selected profile afterward. Detailed removal semantics and existing real-engine regression coverage are in the [README](../README.md#removal--reversibility).

## Observed local checks

| Check | Observed result |
|---|---|
| Download integrity | Exact byte size and pinned SHA-256 passed |
| Native activation | Qwen selected on CPU; main route and all 19 auxiliary routes migrated; dynamic endpoint resolution verified |
| Direct model response | Correct `QWEN_READY`; 23 prompt tokens, 5 completion tokens; 89.4 seconds including cold startup |
| Short-probe server timings | 1.35 prompt tokens/s; 0.26 generated tokens/s. Only five generated tokens: not a sustained throughput benchmark |
| Memory snapshot after direct response | Model child RSS 38,740,869,120 bytes (36.08 GiB); process swap 0; system available RAM 20,488,753,152 bytes (19.08 GiB). Snapshot, not peak memory |
| Interactive operator commands | Actual selected-profile `/ctx-doctor` and `/ctx-stats` dispatched; client exited 0 |
| Context-mode removal/re-enable | Public installer commands passed; Qwen main/auxiliary routes unchanged; four indexed-data files byte-identical; engine retained; normal profile unchanged |
| Model-driven Hermes `ctx_execute` turn | **Not passed.** Both inference-capable attempts exceeded the 960-second external deadline before a completed tool/final-answer event |

The corrected-route attempt reached native prompt processing: the server logged 1,209 prompt tokens at 37% completion after 686.39 seconds (1.76 tokens/s). That implies roughly 3.3k prompt tokens; a ~31-minute initial prefill is a **linear extrapolation**, not an observed completed turn. Removing the fixed endpoint did not solve this CPU latency limit. These attempts used `--reasoning none`; xhigh performance was not measured.

The timed-out test clients and their verified selected-profile gateways were stopped. Qwen and context-mode remain installed and configured, but practical end-to-end Hermes use is **not yet validated**. No continuity/net-benefit ticket was closed on this evidence.

Local raw reports and failed attempts are preserved under `/home/a1234/.cache/hermes-context-mode/evidence/qwen-switch-2c2df9__/`; the first attempt records the rejected CLI flag, the next records the direct probe and initial tool timeout, and the final report records dynamic route resolution and its bounded timeout. Historical failures were not overwritten with passing claims.

## Evidence boundaries

A short inference and tool smoke establishes functional operation only. It cannot establish an intelligence-index score, long-context continuity, improved answer quality, lower latency, or net context savings. The paired net-benefit and continuity certification gates remain separate; do not substitute successful installation for them. The controlled adapter benchmark already recorded increased wire/context costs on its measured workloads.

Weights-plus-KV estimates are not total-memory measurements. mmap alone does not guarantee RAM residency; report observed process memory and system swap activity separately from planned budgets.
