# Qwen model selection for the 64 GiB CPU-only Hermes box

- Date of research: **2026-10-27**. Selection: **ggml-org Qwen3.8-27B Q8_0**, fidelity-first within the measured RAM budget.
- Question: after deleting GLM-5.3-Flash, which Qwen should replace it under Hermes v0.22.5 + the pinned native CPU llama.cpp engine (b11671 as of 2026-10-27, updated from b11669; standard path, CPU-only)?
- Hardware (from [glm-5.3-flash-feasibility.md](glm-5.3-flash-feasibility.md) §5, measured 2026-10-26): i5-6500, 4 cores/4 threads, AVX2, no AVX-512/AMX/GPU; **60.68 GiB usable RAM** (64 GiB installed; `MemAvailable` 49.69 GiB observed under load); NVMe 286.1 GiB free. GLM weights (146 GiB) are deleted; gateway/model server stopped.
- Comparison assumptions: a 65,536-token window and f16 KV cache, with a conservative ~48.5 GiB native planning allowance versus 60.68 GiB physical RAM. These are not fixed launch settings: native admission prices each actual GGUF and available hardware. The selected deployment instead admitted 262,144 tokens with q8_0 KV; see the [runtime guide](qwen-local-runtime.md).
- Artifact sizes are exact HF API byte sums converted to GiB (÷ 2^30), fetched 2026-10-27 at the revisions shown. This research preceded download; subsequent checksum, activation and local smoke observations belong to the runtime guide.

## 1. Shortlist

| Rank | Model (open, Apache-2.0) | AA Intelligence Index v4.3.2 | Why |
|---|---|---|---|
| 1 | **Qwen3.8-27B** (dense, hybrid GDN, VL; Aug 2026) | **33.68 Xhigh** / 27.55 medium / 26.20 low / 20.15 non-reasoning | Highest-scoring open Qwen that is RAM-resident; native Hermes catalog entry exists |
| 2 | **Qwen3.6-35B-A3B** (MoE, 3B active; Apr 2026) | 18.23 reasoning / 15.23 non-reasoning (est.) | Same score class as 122B but 3B active → substantially fewer per-token weight reads (derived, not a fixed speed multiple); native catalog entry exists |
| 3 | Qwen3.5-122B-A10B (MoE, 10B active; Feb 2026) | 17.75 non-reasoning (est.) / 15.57 reasoning | Only fits at ≤3-bit-class quants; dominated by both above |
| — | Qwen3-Next-80B-A3B Thinking / Instruct; Qwen3-Coder-Next | 11.20 / 9.64 (est.); 9.24 | Fit at Q4_K_M but score far below 27B dense; only argument is 3B-active speed, where 3.6-35B-A3B is newer and scores higher |

Rejected for this box (weights cannot be RAM-resident; disk-paging regime only):

| Model | AA | Smallest relevant quant | Verdict |
|---|---|---|---|
| Qwen3.8-Flash-Next (Aug 2026; `qwen4_exp` arch; `qwen-community-1.0` license) | 39.82 | floor **UD-IQ1_S 67.57 GiB** weights-only — full verified inventory in §4.5 | Rejected: every published quant exceeds 60.68 GiB physical RAM before KV. SSD paging would be required; its throughput on this machine is unmeasured. |
| Qwen3.8-2.4T-A95B (Aug 2026) | 39.95 | exceeds 286 GiB free disk | Unacceptable |
| Qwen3.5-397B-A17B (Feb 2026) | 21.44 / 18.42 | UD-IQ2_M 114.60 GiB | Unacceptable (paging) |

**Selected deployment**: Qwen3.8-27B **Q8_0**, using the native Hermes arbitrary-GGUF flow. Qwen3.6-35B-A3B remains the lower-index, speed-oriented alternative. No optional draft or vision weights are needed for text-only Hermes.

## 2. Freshness check (is anything newer/better available?)

- Newest open-weight Qwen text models on HF are the **Qwen3.8 generation, Aug 5–24, 2026** ([Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), Apache-2.0; 2.4T-A95B; Flash-Next under `qwen-community-1.0`). Between then and 2026-10-27 Qwen published only image models (Qwen-Image-2.1-Turbo, Oct 9) — verified via the HF org listing sorted by creation date.
- **Qwen3.7-Plus (AA 25) and Qwen3.8-Max (AA 45) are proprietary with no open weights** — AA-stated (per QwenBenchmarks); HF author=Qwen search "Qwen3.7" also returns nothing, and the Qwen3.7-Plus column in the Qwen3.8-27B card is a closed comparison model.
- No released Qwen4 dense checkpoint was found in the official Hugging Face inventory during this check. Rumors and code sightings are not installable model options.
- Predecessors (Qwen3.5-27B, Qwen3.6-27B, Qwen3.5-35B-A3B) are superseded: the Qwen3.8-27B card benchmarks it above Qwen3.6-27B on every listed benchmark (Terminal-Bench 73.0 vs 63.4; SWE-bench Pro 61.7 vs 53.5; LiveCodeBench v6 90.3 vs 83.9), and none of the predecessors appears on the current AA leaderboard.

## 3. Architecture, context and KV cost at Hermes' 65,536-token context

All four candidates share the Qwen3.5-generation hybrid layout: Gated DeltaNet (linear attention) with a full-attention layer every 4th layer, so KV exists **only on the full-attention layers**. KV cost per token (f16, K+V) = full_layers × kv_heads × head_dim × 2 × 2 B. Linear-attention state is constant per sequence (not context-scaled; `mamba_ssm_dtype: float32`). Derived from official `config.json`:

| Model | Layers (full) | KV heads × head_dim | KV @ 65,536 (f16) | Native context | Total/active params | Vision |
|---|---|---|---|---|---|---|
| Qwen3.8-27B | 64 (16) | 4 × 256 | **4.00 GiB** (64 KiB/token; matches native catalog `per_layer_f16 4096`) | 262,144 (YaRN → 1M) | 27B / 27B | yes (mmproj optional) |
| Qwen3.6-35B-A3B | 40 (10) | 2 × 256 | **1.25 GiB** | 262,144 | 35B / 3B | yes (optional) |
| Qwen3.5-122B-A10B | 48 (12) | 2 × 256 | **1.50 GiB** | 262,144 | 122B / 10B | yes (optional) |
| Qwen3-Next-80B-A3B | 48 (12) | 2 × 256 | **1.50 GiB** | 262,144 | 80B / 3B | no |
| Qwen3.8-Flash-Next (`qwen4_exp`, rejected) | 48 (12) | 2 × 256 | **1.50 GiB** (+ n-gram/indexer state) | 262,144 | 180B / 6B (AA digest) | yes (optional) |

65,536 tokens is a quarter of the 262,144 native window — no RoPE scaling needed for any candidate. q8_0 KV cache would halve these figures if ever needed; at 4 GiB max it is not needed. Flash-Next keeps the same every-4th-layer full-attention cadence but adds n-gram sparse attention and an indexer (constant-scale extra state, plus a hybrid MTP draft layer).

## 4. Quantized deployment options (exact sizes, pinned revisions)

### 4.1 Qwen3.8-27B — primary

`ggml-org/Qwen3.8-27B-GGUF` @ `a316ea527fde32c12487be7574e2db5ac2ad2a7e` (last modified 2026-10-19):

| File | Bytes | GiB |
|---|---|---|
| Q8_0 | 28,595,763,648 | **26.63** |
| Q4_K_M | 18,973,870,528 | 17.67 |
| BF16 | 53,808,282,048 | 50.13 (exceeds 48.5 GiB catalog budget) |
| mtp-Q8_0 / mtp-Q4_0 (speculative draft, optional) | 3,164,006,816 / 1,680,271,776 | 2.95 / 1.56 |
| dflash-Q8_0 / dflash-Q4_0 (draft, optional) | 2,056,414,560 / 1,094,346,080 | 1.91 / 1.02 |
| mmproj-Q8_0 (vision, optional; not needed for text-only Hermes) | 629,247,008 | 0.59 |

Selected file: `Qwen3.8-27B-Q8_0.gguf`; SHA-256 `aab65c67ef0dad127960efef9247f1832bca105faa1c7a052cc039b223cf86a1`. The pinned native downloader verifies this digest before publishing the final filename.

`unsloth/Qwen3.8-27B-GGUF` @ `4ca720788d1e01f1bff70c033e0d0028fd02e502` (2026-08-20) — full UD ladder; highlights: **UD-Q4_K_M 16,464,440,224 B = 15.34 GiB** (this is the native catalog artifact), UD-Q4_K_XL 16.35, UD-Q5_K_XL 19.45, UD-Q6_K 20.47, UD-Q6_K_XL 23.56, UD-Q8_K_L 26.12, UD-Q8_K_XL 29.29, Q8_0 27.05 GiB.

**Q8_0 vs UD-Q4_K_M — honest weighing (no invented quality deltas):**

- No published measurement quantifies this exact ggml-org Q8 versus unsloth UD-Q4 capability difference; the AA index describes the reference benchmark configuration, **not** either GGUF. Q8_0 uses roughly 8.5 bits per weight; UD-Q4_K_M is unsloth's importance-matrix 4-bit-class quantization.
- Nearest measured signal (indirect): [bartowski's Qwen3.8-27B GGUF card](https://huggingface.co/bartowski/Qwen3.8-27B-GGUF) reports KLD versus BF16 on a wiki-text calibration corpus: Q8_0 0.0009 / 98.7% token agreement; Q4_K_M 0.0139 / 95.0%; Q3_K_M 0.0564 / 89.8%; IQ2_XXS 0.2834 / 78.0%. These are not results for the selected ggml-org file, nor an AA capability score.
- Dense decode must repeatedly read most weights; Q8_0 has ~1.7× the artifact bytes of the catalog UD-Q4_K_M. This suggests a speed cost, but quantized kernels and CPU compute also matter: neither proportional slowdown nor a fixed MoE speed multiple is established on this machine. Hosted GPU/API throughput is not CPU evidence.
- RAM subtotal: Q8_0 26.63 GiB + 4.00 GiB f16 KV ≈ 30.63 GiB. This excludes recurrent state, compute buffers, Hermes and the OS. It leaves substantial planning headroom, but actual peak memory and residency must be observed during the smoke run.
- Fidelity-first choice: Q8_0 (26.63 GiB, ggml-org). Speed/headroom-first choice (and the supported catalog default): UD-Q4_K_M at `reasoning_effort=medium`. UD-Q6_K_XL (23.56 GiB) is the untested middle.
- Xhigh effort is AA's top config (33.68) but AA also flags Qwen3.8-27B as **very verbose** (200M output tokens across its index run, 25th of 27 in its class); on this CPU, verbosity multiplies wall-clock time. The catalog default `reasoning_effort=medium` (27.55) is the sensible operating point; the model card itself warns that in multi-turn agentic tasks lower effort does not always increase total task time.

### 4.2 Qwen3.6-35B-A3B — fast alternative

`unsloth/Qwen3.6-35B-A3B-GGUF` @ `a483e9e6cbd595906af30beda3187c2663a1118c` (2026-04-20): **UD-Q4_K_M 22,134,528,992 B = 20.62 GiB** (native catalog entry), MXFP4_MOE 20.22, UD-Q4_K_XL 20.83, UD-Q5_K_XL 24.77, UD-Q6_K_XL 29.66, **Q8_0 36,903,140,320 B = 34.37 GiB**. All fit RAM comfortably.

⚠ Catalog discrepancy: the native Hermes catalog records UD-Q4_K_M as 22,663,387,424 B; the current HF revision above says 22,134,528,992 B (Δ ≈ 0.49 GiB). Verify expected byte size at download time. It does not change any conclusion.

### 4.3 Qwen3.5-122B-A10B — bigger-model bucket, rejected

`unsloth/Qwen3.5-122B-A10B-GGUF` @ `51eab4d59d53f573fb9206cb3ce613f1d0aa392b` (2026-03-05): Q3_K_S 48.87 GiB (52,495,013,376 B), Q3_K_M 52.56, UD-Q3_K_XL 53.05, UD-IQ3_XXS 41.67, Q4_K_S 66.77 (exceeds RAM). The Q3_K_XL class breaches the 48.5 GiB catalog budget; even Q3_K_S (48.87) is over it by 0.3 GiB. Only the ~2–3-bit files (UD-IQ3_XXS 41.67, UD-Q2_K_XL 38.97) would admit, at 3-bit quality on an est.-scored 17.75 model — worse score than 3.6-35B-A3B's Q8_0 at 34.37 GiB, with 10B active vs 3B. Dominated; reject.

### 4.4 Qwen3-Next-80B-A3B and Qwen3-Coder-Next — fit, but dominated

Official `Qwen/Qwen3-Next-80B-A3B-Thinking-GGUF`: Q4_K_M 45.09 GiB (48,410,989,824 B), Q5_K_M 52.81, Q6_K 61.02 (exceeds RAM). `unsloth/Qwen3-Coder-Next-GGUF`: Q4_K_M 45.20, MXFP4_MOE 44.74. Q4_K_M fits the 48.5 GiB budget with ~1.5 GiB KV. But AA scores (11.20 est. / 9.64 / 9.24) are a third of Qwen3.8-27B; the only argument for them (3B active, low verbosity — Coder-Next is AA's least verbose class) is better served by Qwen3.6-35B-A3B.

### 4.5 Qwen3.8-Flash-Next — verified complete quant inventory (rejection basis)

`unsloth/Qwen3.8-Flash-Next-GGUF`, split-file sums verified 2026-10-27 via the HF tree API. Base-model license: `qwen-community-1.0` (HF card, `license: other`). Architecture `qwen4_exp` — newer than `qwen3_5_moe` (48 layers, 12 full-attention, 2 KV heads × 256, 512 experts / 10 active + shared expert, n-gram sparse attention with indexer; 180B total / 6B active per the AA/QwenBenchmarks digest).

| Quant | Sum bytes | GiB |
|---|---|---|
| UD-IQ1_S (floor) | 72,546,461,344 | **67.57** |
| UD-IQ1_M | 74,538,755,776 | 69.42 |
| UD-Q2_K_XL | 78,869,128,864 | 73.45 |
| UD-IQ3_XXS | 81,961,823,936 | 76.31 |
| UD-IQ4_XS | 93,682,584,224 | 87.25 (matches the native catalog's byte sum exactly) |
| mtp-Q8_0 draft (optional) | 4,137,429,120 | 3.85 |

Every published quant's **weights alone exceed the 60.68 GiB physical RAM** (UD-IQ1_S 67.57 GiB + ~1.50 GiB KV ≈ 69.1 GiB, excluding other runtime memory). RAM-resident deployment is therefore impossible at a currently published quantization. The earlier "≥ 103 GiB floor" came from a partial inventory; the verified floor is 67.57 GiB. Flash-Next throughput has not been measured here. AA 39.82 describes the reference model, not these low-bit files.

## 5. CPU / llama.cpp / Hermes support

- Architectures: `qwen35` (Qwen3.8-27B) and `qwen35moe` (3.6-35B-A3B, 122B) are mainline llama.cpp archs — `qwen35` support landed in Feb 2026 (llama.cpp PR #19468, per QwenBenchmarks), ggml-org (the llama.cpp maintainers) publishes official GGUFs with `convert.log`, and a stock `b11069` build has been community-verified CPU-only on the 35B-A3B. The native engine is now **b11671** (installed 2026-10-27, `--version` passed; supersedes b11669) and Hermes v0.22.5's native catalog already drives both models on the pinned standard path (`model_setup_flows_local.py` → shared `web_routers/local_models` quickstart/activate; per Main). **No llama.cpp update is required.**
- Tool calling and thinking control ship inside the GGUF chat templates (verified from the GGUF metadata served by HF for both ggml-org and unsloth files): hermes-style `<tool_call><function=…><parameter=…>` XML with `<tools>` JSON injection and `<tool_response>` returns; `enable_thinking` (default true), `preserve_thinking`, and `reasoning_effort ∈ {xhigh (default), medium, low}` on Qwen3.8 (the 3.6/3.5-generation templates have `enable_thinking` but no effort knob). On b11671, `--jinja` renders these embedded templates — there is no separate tool-call parser flag in this build ([glm-5.3-flash-local-runtime.md](glm-5.3-flash-local-runtime.md), same mechanism as the GLM setup).
- Recommended sampling from the Qwen3.8-27B card: thinking mode T=1.0, top_p=0.95, top_k=20; instruct (non-thinking) T=0.7, top_p=0.80, top_k=20, presence_penalty=1.5.
- Draft-weight options (optional, unmeasured on this box): ggml-org `dflash-Qwen3.8-27B-Q4_0` 1.02 GiB (the catalog's "DFlash 2-draft 1.14 GB" most closely matches this file; exact identity unverified) and `mtp-*` 1.56–2.95 GiB. CPU speculative decoding adds verification compute; benefit here is unknown — treat as an experiment, not a plan.
- Base vs Instruct/Thinking: Qwen3.8-27B is a single post-trained checkpoint — thinking/non-thinking is a per-request template switch, not separate weights. Qwen3.5-generation *-Base repos (e.g. Qwen3.5-35B-A3B-Base, Qwen3.5-9B-Base) are pre-training artifacts without chat/tool templates — unusable for Hermes. Qwen3-Next ships as separate Instruct and Thinking repos; Qwen3-Coder-Next is a single non-reasoning coding model. Uncensored/abliterated derivatives exist but are out of scope.
- Native integration: models are stored under the isolated `PROFILE/models`. The curated 27B entry uses UD-Q4_K_M; the selected ggml-org Q8_0 was downloaded with its pinned checksum and activated through Hermes' native arbitrary-GGUF flow, with `backend=cpu` and no spill in the generated admission plan.

## 6. RAM budget summary (65,536-token context, f16 KV)

| Candidate quant | Weights | KV | Weights+KV | vs 48.5 GiB catalog budget | vs 60.68 GiB physical |
|---|---|---|---|---|---|
| 27B Q8_0 (ggml-org) | 26.63 | 4.00 | 30.63 | pass (+17.9) | pass |
| 27B UD-Q6_K_XL | 23.56 | 4.00 | 27.56 | pass (+20.9) | pass |
| 27B UD-Q4_K_M (catalog default) | 15.34 | 4.00 | 19.34 | pass (+29.2) | pass |
| 3.6-35B-A3B Q8_0 | 34.37 | 1.25 | 35.62 | pass (+12.9) | pass |
| 3.6-35B-A3B UD-Q4_K_M (catalog) | 20.62 | 1.25 | 21.87 | pass (+26.6) | pass |
| 122B-A10B UD-Q3_K_XL | 53.05 | 1.50 | 54.55 | **fail** | marginal |
| Next-80B-A3B Q4_K_M (official) | 45.09 | 1.50 | 46.59 | pass (+1.9) | pass |
| Flash-Next UD-IQ1_S (smallest published) | 67.57 | 1.50 | 69.07 | **fail** | **fail — weights alone exceed RAM** |

The table gives weights-plus-KV subtotals, not total process memory or admission guarantees. Recurrent state, compute buffers, other processes and OS memory remain additional costs. mmap does not guarantee residency under memory pressure. Observe actual memory and swap activity before claiming a paging-free deployment; no local Qwen peak-memory measurement was available during research.

## 7. Unknowns and uncertainties (explicit)

1. AA scores describe original configurations, not GGUFs. The only measured quant-quality signal is bartowski's KLD/same-top-p corpus work (§4.1) — indirect (bartowski's files, wiki-text-style calibration; not the ggml-org/unsloth artifacts, not AA). A Q8-vs-Q4 capability delta for Qwen3.8-27B specifically remains unmeasured.
2. No comparative CPU-only throughput benchmark was available during selection. Subsequent local Qwen smoke measurements are reported separately in the runtime guide; active parameter counts and hosted GPU results are not local speed measurements.
3. The unselected native 35B-A3B catalog byte count differs from its current HF revision by 0.49 GiB; an optional DFlash catalog description also lacks exact artifact identity. Resolve these only if installing those artifacts. Neither was downloaded for this cutover.
4. The selected Q8_0 is admitted through the generic native GGUF path; it does not inherit the curated Q4 recipe's optional assets or medium-effort default.
5. Flash-Next's `qwen4_exp` architecture has unverified CPU support in b11671; no published quant fits this machine's RAM regardless.
6. MTP/DFlash speculative-decoding benefit on this 4-core CPU is workload-dependent and unmeasured; optional experiment only.

## Sources

- Official: [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) (card, config, thinking/tool docs), [Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B), [Qwen/Qwen3.5-122B-A10B](https://huggingface.co/Qwen/Qwen3.5-122B-A10B), [Qwen/Qwen3-Next-80B-A3B-Thinking](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Thinking) configs; HF tree/API for every size and SHA cited (2026-10-27).
- Artificial Analysis model pages (Intelligence Index v4.3.2, fetched 2026-10-27): [Qwen3.8 27B](https://artificialanalysis.ai/models/qwen3-8-27b) (34/Xhigh), [Qwen3.6 35B A3B](https://artificialanalysis.ai/models/qwen3-6-35b-a3b) (18), [Qwen3.5 122B A10B](https://artificialanalysis.ai/models/qwen3-5-122b-a10b) (16), [Qwen3 Coder Next](https://artificialanalysis.ai/models/qwen3-coder-next) (9); effort-variant scores and estimated-score flags from the 2026-10-26 snapshot `local://model-feasibility-ranking.md`.
- Quantization proxy: [bartowski/Qwen3.8-27B-GGUF](https://huggingface.co/bartowski/Qwen3.8-27B-GGUF). Runtime architecture support: [llama.cpp PR #19468](https://github.com/ggml-org/llama.cpp/pull/19468).
- [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) card + config (license `qwen-community-1.0`, `qwen4_exp` layout) and `unsloth/Qwen3.8-Flash-Next-GGUF` tree (all quant sums, 2026-10-27).
