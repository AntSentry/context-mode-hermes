# GLM-5.3-Flash on the 64 GiB OptiPlex 5040 — feasibility investigation

- Ticket: [#3](https://github.com/AntSentry/context-mode-hermes/issues/3) (parent spec [#9](https://github.com/AntSentry/context-mode-hermes/issues/9))
- Date of research and of all measurements: **2026-10-26**
- Method: read-only. Primary sources only (Hugging Face API + raw files, z.ai, GitHub upstream repos, locally pinned Hermes source). **No weights were downloaded, no benchmarks executed, no system state changed.** Every number below is either (a) fetched from a cited primary source, (b) read off this machine read-only, or (c) explicitly marked `[INFERENCE]` (derived, unmeasured).
- Consumer: ticket #4 (runtime bring-up) and the parent's feasibility decision. Nothing in this report substitutes a smaller model or a cloud endpoint — both are out of scope by spec.

## Verdict (summary)

| Question | Answer | Basis |
|---|---|---|
| Exact model exists as a downloadable artifact? | **Yes** — `zai-org/GLM-5.3-Flash`, public, ungated, MIT | HF API, revision `46a050bec13002ad29ee795c16e61cf4fa3c4a9c` |
| Any published artifact fits 64 GiB RAM **in memory**? | **No.** Smallest verified artifact is 86.7 GiB (unsloth `UD-IQ1_S`); vendor minimum for it is 100 GB total memory | HF tree-API size sums; Unsloth official guide |
| CPU-only local runtime exists? | **Yes, llama.cpp mainline** — arch `glm5-next`, merged 2026-09-30 (PR #27773), shipped in release `b11279` and `v0.7.0`; CPU build documented by Unsloth | GitHub API + master source inspection |
| Disk-backed (mmap) loading viable on this box? | **Unsupported-by-evidence, not disproven.** Artifact fits free NVMe space; config-derived KV estimate fits RAM at the Hermes-mandated context (§6.2, unmeasured, backend-dependent); but no vendor documents a working config at 64 GiB / 4 cores, and expected speed is unusable-unknown. **Must be measured live (ticket #4/#5)** | Budget math below |
| GPU serving stacks (vLLM/SGLang/KTransformers)? | **Prerequisites not met**: they require datacenter GPUs (+350 GB RAM and AVX-512 for KTransformers) | Official vLLM recipe, SGLang cookbook, KTransformers tutorial |
| Genuine blocker? | One: **no local throughput measurement is possible without downloading ≥86.7 GiB**, which this read-only ticket must not do. Everything else is resolved. | — |

## 1. Exact model identity and provenance (story 1, 2)

- Repository: `zai-org/GLM-5.3-Flash` — official Zhipu/z.ai organization, **public, not gated** (`"gated": false, "private": false`), 6,350,666 downloads at time of query.
- Pinned revision: **`46a050bec13002ad29ee795c16e61cf4fa3c4a9c`** (`lastModified` 2026-10-15T06:32:52Z). A second revision seen in the wild: `2d894c61fc2ea7708ae8e1395ea009fa616ea577`. All `main` figures below were read at the pinned sha on 2026-10-26.
- License: **MIT** (HF `license:mit` tag; `LICENSE` file in repo; inherited by the community quantizations surveyed, each states its own base relation).
- Canonical sources: [HF model card](https://huggingface.co/zai-org/GLM-5.3-Flash) · [z.ai blog](https://z.ai/blog/glm-5.3-flash) · [arXiv 2602.15763](https://arxiv.org/abs/2602.15763) ("GLM-5") · ModelScope mirror [`ZhipuAI/GLM-5.3-Flash`](https://www.modelscope.cn/models/ZhipuAI/GLM-5.3-Flash).
- Vendor description: first natively multimodal model of the GLM-5 series; **320B total / 18B active** MoE; "Flash" = low-cost tier of GLM-5.3; one-tenth the price of GLM-5.2 on z.ai's API.
- No substitution was performed anywhere in this report; every artifact below derives from this exact model.

## 2. Architecture (from `config.json` at the pinned revision)

`Glm5NextForConditionalGeneration`, `model_type: glm5_next`, `transformers_version: 5.16.0`.

Text stack (45 layers, hidden 4096, vocab 154,880, `max_position_embeddings` **1,048,576**):

- **Hybrid attention**: 34 linear-attention layers (KDA/GatedDeltaNet-style: `num_heads 64`, `head_dim 128`, `short_conv_kernel_size 4`, gated with `gate_lower_bound −5`) interleaved with **11 sparse-attention layers** (DeepSeek-style DSA with lightning indexer: `index_n_heads 32`, `index_head_dim 128`, `index_topk 2048`; every 4th layer: 3,7,…,43).
- **MLA-style compressed KV**: `kv_lora_rank 512`, `q_lora_rank 1536`, `qk_nope_head_dim 256`, `qk_rope_head_dim 0`, `v_head_dim 256` (NoPE — no rope component).
- **MoE**: 288 routed experts + 1 shared, **8 experts/token**, `moe_intermediate_size 2048`, sigmoid/noaux-tc routing, first 3 layers dense (`intermediate_size 12288`).
- **MTP**: 1 nextn predict layer. Mainline llama.cpp v0.7.0 release notes include GLM5-Next MTP via [#29928](https://github.com/ggml-org/llama.cpp/pull/29928); local CPU benefit is unmeasured. The initial claim that MTP was absent from mainline was incorrect.
- **mHC** (`hc_mult 4`): heterogeneous-horizon hyper-connections.
- Vision tower: 24-layer encoder, 448-px inputs — native image/video input (`image-text-to-text` pipeline tag).

## 3. Verified artifact inventory and sizes (story 3, 6)

All sizes summed from the Hugging Face tree API (`/api/models/<repo>/tree/main?recursive=true`) on 2026-10-26 — these are exact byte sums of shard files, not vendor marketing figures.
- Pinned revisions for these sums: official `zai-org/GLM-5.3-Flash` `46a050bec13002ad29ee795c16e61cf4fa3c4a9c`; community `unsloth/GLM-5.3-Flash-GGUF` `a38483c8cd5df544f53d70fb281afe97369d5ab6`. Worked example: `UD-IQ4_XS` = 5 shard files, exact sum 156,822,111,200 bytes = 146.051972 GiB (the table's 146.1 GiB rounds this). No weights were downloaded in this ticket — every size above comes from the HF tree API at the pinned revisions only.

### Official checkpoints (zai-org)

| Artifact | Precision | Size (GiB) | Size (GB) | Shards |
|---|---|---|---|---|
| `zai-org/GLM-5.3-Flash` | FP8 e4m3 (dynamic act) | **305.79** | 328.3 | 62 safetensors (~5.0 GiB each) |
| `zai-org/GLM-5.3-Flash-BF16` | BF16 | **598.5** | 642.6 | 124 |

### Quantized derivatives (same model, verified lineage)

| Artifact (quant) | Size (GiB) | Unsloth-reported size (GB) / top-1 fidelity* |
|---|---|---|
| `unsloth/GLM-5.3-Flash-GGUF` UD-IQ1_S (3 shards) | **86.7** ← smallest verified | 93.09 / 70.89% |
| `…` UD-IQ1_M | 90.9 | 97.58 / 73.06% |
| `…` UD-IQ2_XXS | 94.9 | 101.84 / 76.30% |
| `…` UD-Q2_K_XL | 101.3 | 108.72 / 78.34% |
| `…` UD-IQ3_XXS | 112.1 | 120.37 / 81.63% |
| `…` UD-Q3_K_XL | 137.4 | 147.54 / 86.25% |
| `…` UD-IQ4_XS | **146.1** | 156.82 / 88.18% |
| `…` UD-Q4_K_XL | 186.0 | 199.71 / 92.22% |
| `…` UD-Q5_K_XL | 223.8 | 240.31 / 94.35% |
| `…` UD-Q6_K_XL | 271.8 | 291.83 / 95.23% |
| `…` Q8_0 | 317.6 | — |
| `…` BF16 GGUF | 597.6 | 641.64 / (reference) |
| `vcruz305/GLM-5.3-Flash-GGUF` Q2_K (single file) | 108.7 | — |
| `nvidia/GLM-5.3-Flash-FP4-Local` (safetensors) | 165.3 | — |
| mmproj (vision projector) any build | ~1.1 | — |

\* Fidelity figures are Unsloth's own published benchmark vs BF16; independent reproduction is out of scope.

Note resolving a figure circulating in planning: "~146 GiB" matches `UD-IQ4_XS` (146.1 GiB exact). It is **not** the smallest usable artifact and **not** a prerequisite; the floor is `UD-IQ1_S` at 86.7 GiB.

**Unsloth's official minimum total memory (RAM+VRAM) table** — the only vendor-published hardware floors for llama.cpp-family execution:

| Precision | Minimum total memory |
|---|---|
| 1-bit | **100 GB** |
| 2-bit | 115 GB |
| 3-bit | 128–150 GB |
| 4-bit | 162–210 GB |
| 8-bit | 350 GB |
| BF16 | 650 GB |

Every published configuration therefore exceeds this machine's 60.68 GiB of usable RAM (§5); the 64 GiB box sits below even the 1-bit floor.

## 4. Runtime support matrix (story 4, 5) — verified vs untested

| Runtime | GLM-5.3-Flash support | Evidence (primary) | Runs on this machine? |
|---|---|---|---|
| **llama.cpp (mainline)** | **Supported**: `glm5-next` support merged in [#27773](https://github.com/ggml-org/llama.cpp/pull/27773), commit `649dcb1036fd47d01a83b56c5e35913e65c277ba`. The [v0.7.0 release](https://github.com/ggml-org/llama.cpp/releases/tag/v0.7.0), commit `620d2225cc436ab0b001e3200e590e3b286ef9ab`, also includes GLM5-Next MTP via #29928. Its linked nightly `b11669` provides an Ubuntu x64 CPU binary. | Primary release metadata and the downloaded CPU binary's SHA256/version output; see follow-up below | **Binary startup verified; model execution unmeasured.** CPU build is also documented (`-DGGML_CUDA=OFF`). No working 64 GiB/4-core model configuration has been established by this investigation. |
| llama.cpp (Unsloth fork) | Supported, with MTP ("3.3× decode") | branch `glm5next/upstream`, `unslothai/llama.cpp#61`; unsloth guide build instructions | Same hardware caveat as above |
| **vLLM** | Supported, **GPU-only**: vLLM ≥ 0.29.0, single-node TP=8 on H100/H200/B200/GB200/GB300, AMD gfx950 (MI355X, PR #53906), Ascend 950PR ×8; "weights: about 306 GiB" FP8 | [official recipe](https://recipes.vllm.ai/zai-org/GLM-5.3-Flash) | **No** — no CUDA/ROCm GPU, no 306 GiB |
| **SGLang** | Supported, **GPU/NPU-only**: v0.5.20+, H100/H200/B200/B300/GB200/GB300/MI355X/TPU7x/Ascend; docker default "four GPUs" | [official cookbook](https://docs.sglang.io/cookbook/autoregressive/GLM/GLM-5.3-Flash) | **No** |
| **KTransformers** | Supported (CPU+GPU offload, reads official FP8 weights directly): requires **≥350 GB system memory**, an NVIDIA SM89/SM120 GPU (RTX 40/50), and an **AVX-512 FP8 CPU kernel** | [official tutorial](https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/kt-kernel/GLM-5.3-Flash-Tutorial.md) | **No** — 60.68 GiB < 350 GB, no NVIDIA GPU, i5-6500 has no AVX-512 |
| Transformers | `glm5_next` architecture in transformers ≥ 5.16.0 (CPU-capable in principle) | `config.json` `transformers_version`; model card links HF `model_doc/glm5_next` | In-memory only → excluded by size |
| MLX builds (e.g. `pipenetwork/…-MLX-8bit`) | Apple Silicon only | HF repo tags | **No** |
| Ollama | Claimed "cloud-only" as of Oct 2026 by a secondary source; **not verified** | modemguides blog (secondary, unverified) | Untested by this ticket |

**Net:** the only runtime with a documented CPU-only path is llama.cpp (mainline b11279+ or the Unsloth fork). Its `llama-server` exposes the OpenAI-compatible API with tool calling that Hermes consumes (§8).

### Runtime prerequisite follow-up (2026-10-27)

The v0.7.0 release links nightly [b11669](https://github.com/ggml-org/llama.cpp/releases/tag/b11669), with the same commit `620d2225cc436ab0b001e3200e590e3b286ef9ab`. Its `llama-b11669-bin-ubuntu-x64.tar.gz` CPU asset was downloaded outside the repository: 17,851,611 bytes, SHA256 `cf07c1c112d709bd67e522017471de57ed0ec8bea031d732c5422351c07f65ac`, matching GitHub's asset digest. On this 5040, `llama-server --version` exited successfully and reported `0.7.0-dev (build 11669, commit 620d2225c)`, GNU 11.4.0, Linux x86_64. This avoids an unnecessary source build; it does **not** prove model loading, tool calling, memory fit or throughput. No model weights were downloaded for this prerequisite check.


## 5. Target machine inventory (read-only, measured 2026-10-26)

- **CPU**: Intel Core i5-6500 @ 3.20 GHz, 1 socket, **4 cores / 4 threads**, AVX2, **no AVX-512, no AMX** (`/proc/cpuinfo`, `lscpu`).
- **RAM**: `MemTotal` 63,627,908 kB = **60.68 GiB usable** (64 GiB installed); `MemAvailable` 49.69 GiB at measurement time; swap = 8 GiB file, 1.7 GiB used; THP `madvise`.
- **Fast storage**: `/` on `/dev/nvme0n1p2` — Samsung MZVLB512HAJQ-000H1 (PM981-class 512 GB NVMe, PCIe 3.0 x4 platform, non-rotational): 468.4 GiB total, **286.1 GiB free**.
- **Bulk storage**: `/mnt/raid1` (md1) 916 GiB, 695 GiB free — **spinning RAID1, currently DEGRADED `[U_]`**; md0 (1.8 T RAID0) present but unmounted. Neither is a candidate for latency-critical weight paging.
- **GPU**: Intel HD Graphics 530 integrated (`00:02.0`), 256 MiB prefetchable aperture, shares system DRAM; **no discrete GPU, no CUDA/ROCm/NPU**.

## 6. Memory and storage budget (stories 7, 9, 10, 11)

### 6.1 Weights vs RAM — in-memory is infeasible

Usable RAM 60.68 GiB (49.69 GiB free at idle). Smallest verified artifact 86.7 GiB (`UD-IQ1_S`). **Gap ≈ 26 GiB even before KV, activations, OS.** No published quantization of this model — official or community — fits in memory on this machine. This is a hard arithmetic result, not a judgment call.

### 6.2 KV-cache and runtime memory (derived from `config.json`; storage assumed F16 — FP8-storing runtimes halve the indexer column)

Per-token costs: 11 DSA layers × (MLA compressed KV 512×2 B = 1 KiB + indexer keys 32×128×2 B = 8 KiB) = **99 KiB/token with indexer**, 11 KiB/token MLA-only. Linear layers add a **constant** per-sequence state: 64 heads×128×128×2 B ≈ 2 MiB × 34 layers ≈ **68 MiB/sequence**, independent of context length.

| Context (tokens) | DSA KV, with indexer | DSA KV, MLA-only | + KDA state |
|---|---|---|---|
| 8,192 | 0.77 GiB | 0.09 GiB | +68 MiB |
| 32,768 | 3.09 GiB | 0.34 GiB | +68 MiB |
| **65,536 (Hermes floor, §8)** | **6.19 GiB** | **0.69 GiB** | +68 MiB |
| 131,072 | 12.38 GiB | 1.38 GiB | +68 MiB |
| 1,048,576 | 99.0 GiB | 11.0 GiB | +68 MiB |

Vendor cross-checks (implementation-specific pool sizing, not per-token math): vLLM recipe reports a 14.92M-token KV pool at TP4/128K on MI355X; SGLang splits GPU memory between KV and KDA state pools via `--mamba-full-memory-ratio 0.9`.

**Conclusion:** KV/runtime memory is **not proven** to be the binding constraint — at the Hermes-mandated 64K floor the config-derived estimate is KV + state ≤ 6.3 GiB, inside RAM. That figure is a hand-computed `[INFERENCE]` from `config.json` dimensions, **not a measured allocation**: actual usage is backend-dependent (llama.cpp pool sizing, quantized-vs-F16 cache storage, and mmap bookkeeping all differ from this arithmetic) and **must be confirmed by reading real allocator output during the #4/#5 live smoke**. The verified binding constraints are weights residency (§6.1) and, under mmap, weight paging (§6.3).

### 6.3 Disk-backed (mmap) loading — the only candidate path

llama.cpp default-loads GGUF via `mmap`, so the OS page cache streams weights from NVMe; resident RAM is bounded by page cache, and KV/state/compute buffers use §6.2 numbers. Feasibility accounting:

- **Disk space**: 86.7 GiB (UD-IQ1_S) fits the 286.1 GiB free on `/` with headroom for a second quant; the 112 GiB UD-IQ3_XXS (better fidelity) also fits. RAID1 HDD is unsuitable for paging (rotational + degraded).
- **Paging volume per generated token** `[INFERENCE]`: 8 routed experts × 42 MoE layers = 336 expert-FFN reads/token; each expert ≈ 3×2048×4096 params ≈ 25.2 M params ≈ ~6.9 MiB at IQ1-class bitrates → worst case **~2 GiB of random reads per token** before inter-token page-cache reuse. The governing metric is **QD1 random 4 KiB read latency, not sequential throughput** (MoE routing touches scattered pages across the whole 86.7+ GiB file; the file cannot be fully cached: 86.7 GiB > ~55 GiB cacheable).
- **Expected speed**: **unknown and likely unusable for interactive tool-calling** `[INFERENCE]`. Unsloth's 62.79 t/s decode for UD-IQ1_S is a **GPU measurement on 1× B200 at baseline tg32, no MTP** — not a CPU figure. No CPU inference speed can be inferred from it. This machine has 4 AVX2-only cores. Mainline v0.7.0 includes MTP support, but its CPU benefit and additional resource cost have not been measured here. No source documents a working 64 GiB / 4-core / AVX2 configuration.
- **Swap**: the 8 GiB swap file adds nothing for mmap'd weights (the page cache already provides paging; swapping anonymous compute buffers only adds latency).
- This ticket could not measure local throughput — that requires the ≥86.7 GiB download and is deferred to the ticket #4/#5 live smoke, which must treat "slow but functional" as the success bar and record honest t/s numbers.

## 7. Integrated graphics / accelerators (story: GPU research)

- Intel HD 530 shares system DRAM (256 MiB aperture); it cannot host any artifact (smallest 86.7 GiB) and adds no capacity beyond the 60.68 GiB already counted.
- llama.cpp GPU backends relevant here: Vulkan — glm5-next support **unverified**, with an open mmproj+MTP failure issue (#30204); SYCL/i915 — no glm5-next evidence found. Neither would change the weights-residency arithmetic.
- All officially supported GPU stacks (vLLM, SGLang, KTransformers) require H100-class / MI355X / RTX 40-50 hardware (§4). **No acceleration path is verified on this machine** — Vulkan and SYCL glm5-next support is *unverified, not disproven* — so **CPU+mmap is the only verified candidate**; even a working Vulkan/SYCL backend would not change the weights-residency arithmetic above.

## 8. Hermes v0.22.5 integration contract (input for ticket #4)

From the pinned Hermes source (`v0.22.5`, commit `bf0203a7`, `/home/a1234/.cache/hermes-context-mode/hermes-agent`):

- **Hard context floor: `MINIMUM_CONTEXT_LENGTH = 64_000`** (`agent/model_metadata.py:274`). Enforced at startup (`agent/agent_init.py:2069–2090`) — a local model below 64K is rejected, and the error message itself prescribes the llama.cpp remedy: **`-c 64000`** (also vLLM `--max-model-len`, Ollama `OLLAMA_CONTEXT_LENGTH`). An explicit below-floor escape exists only for LM Studio profiles. Consequence: run llama-server with `-c 65536`, KV cost ≈ ≤6.3 GiB by the §6.2 config-derived estimate (confirm the real allocation in #4/#5).
- **Transport**: OpenAI-compatible `chat.completions` with `tools` (e.g. `trajectory_compressor.py:467`, auxiliary relay). Provider profiles accept `base_url` + `api_key` (`providers/base.py:63–64,158`), so `llama-server`'s `/v1` endpoint slots in as a provider; per-model context length comes from the provider hook `get_model_context_length` or the profile `context_length` setting, which caps the detected value (`cli-config.yaml.example:135–145`).
- **Reasoning effort**: model exposes low/high/max via chat-template kwargs (vLLM recipe: `chat_template_kwargs.reasoning_effort`; llama.cpp: `--chat-template-kwargs`). Server-side parsers used by GPU stacks: `--tool-call-parser glm47`, `--reasoning-parser glm47/glm45`.
- **Sizing reality check**: the parent spec's observed Hermes figures — 19,578–76,435 total request bytes, largest single artifact a 266,674-byte JSON — are **byte totals, not token counts**, and no per-request history was provided, so they do not establish that sessions approach 64K *tokens*. The 64K floor binds because Hermes enforces it (`MINIMUM_CONTEXT_LENGTH`), not because observed session sizes were shown to need it.

## 9. License and redistribution (story 3)

MIT on the official checkpoint (and the GLM-5 repo). Distribution of quantized copies inside this project (e.g. for CI fixtures) is permitted by the license; this ticket neither downloaded nor redistributed anything. CI cannot exercise the real model regardless (165–306 GiB official artifacts; smallest community quant 86.7 GiB) — engine-only CI is the right scope, matching the parent's "distributable real-engine CI" decision.

## 10. Decision inputs and honest blockers

1. **In-memory execution: infeasible** on this machine for every published artifact of this exact model (arithmetic, §6.1). Not fixable by quant choice within the published set.
2. **CPU + NVMe mmap via llama.cpp b11279+/v0.7.0 (or unsloth fork): the only candidate**. Artifact fits disk; KV fits RAM at the Hermes 64K floor **by the §6.2 config-derived estimate (unmeasured, backend-dependent)**; runtime support is merged and CPU builds are documented. **No evidence any 64 GiB / 4-core machine has run it**; expected interactive latency is poor-to-unknown. The feasibility decision therefore hinges on the ticket #4/#5 live measurement (download once, measure t/s, tool-call correctness, and the real KV allocation at `-c 65536`), which this read-only ticket deliberately did not perform.
3. **GPU stacks: excluded** by prerequisites (§4, §7). **Cloud API and alternate models: excluded by spec.**
4. Genuine blocker encountered in this ticket: **none unresolved** — the only unverifiable-by-design item is local throughput, whose prerequisite (86.7 GiB download, live smoke) is owned by tickets #4/#5.
5. Secondary sources (codersera, atomic.chat, modemguides blogs) were used only as leads; every load-bearing claim above is pinned to a primary source.

## 11. Sources

Primary (all fetched 2026-10-26):

1. HF model card: https://huggingface.co/zai-org/GLM-5.3-Flash (rev `46a050bec13002ad29ee795c16e61cf4fa3c4a9c`)
2. HF raw config: https://huggingface.co/zai-org/GLM-5.3-Flash/raw/main/config.json
3. HF tree APIs (exact sizes): `/api/models/{zai-org/GLM-5.3-Flash, zai-org/GLM-5.3-Flash-BF16, unsloth/GLM-5.3-Flash-GGUF, nvidia/GLM-5.3-Flash-FP4-Local, vcruz305/GLM-5.3-Flash-GGUF}/tree/main?recursive=true`
4. z.ai blog: https://z.ai/blog/glm-5.3-flash · arXiv: https://arxiv.org/abs/2602.15763 · ModelScope: https://www.modelscope.cn/models/ZhipuAI/GLM-5.3-Flash
5. Unsloth official guide (memory floors, quant table, fork build): https://unsloth.ai/docs/models/glm-5.3-flash
6. llama.cpp: PR #27773 (merged 2026-09-30, merge commit `649dcb1036fd47d01a83b56c5e35913e65c277ba`); master `src/llama-arch.cpp` (`glm5-next`); releases `b11279`, `v0.7.0`; open #27917, #29867, #29950, #30204; closed #27752, #27754; unslothai/llama.cpp#61 (branch `glm5next/upstream`)
7. vLLM recipe: https://recipes.vllm.ai/zai-org/GLM-5.3-Flash
8. SGLang cookbook: https://docs.sglang.io/cookbook/autoregressive/GLM/GLM-5.3-Flash
9. KTransformers tutorial: https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/kt-kernel/GLM-5.3-Flash-Tutorial.md
10. Hermes v0.22.5 pinned source, commit `bf0203a7`: `agent/model_metadata.py:274`, `agent/agent_init.py:2069–2090`, `providers/base.py`, `cli-config.yaml.example`

Secondary (leads only, flagged in text): codersera.com, atomic.chat, modemguides.com GLM-5.3-Flash run-locally articles.

## 12. Reproducing this report's measurements (read-only)

```bash
# machine
lscpu; grep -E 'MemTotal|MemAvailable|SwapTotal' /proc/meminfo; free -h
df -h / /mnt/raid1; lsblk -o NAME,SIZE,ROTA,MOUNTPOINTS; cat /proc/mdstat
cat /sys/block/nvme0n1/device/model; lspci | grep -i vga; swapon --show
# artifact sizes (exact byte sums)
curl -s "https://huggingface.co/api/models/zai-org/GLM-5.3-Flash/tree/main?recursive=true" \
  | jq '[.[]|select(.type=="file")|.size]|add'   # 328366173639 ≈ 305.81 GiB
# revision pin
curl -s https://huggingface.co/api/models/zai-org/GLM-5.3-Flash | jq -r '.sha, .gated, .tags[]|select(startswith("license"))'
# llama.cpp support
curl -s https://api.github.com/repos/ggml-org/llama.cpp/pulls/27773 | jq -r '.state,.merged_at,.merge_commit_sha'
curl -s https://raw.githubusercontent.com/ggml-org/llama.cpp/v0.7.0/src/llama-arch.cpp | grep -c 'glm5-next'
```

KV-cache derivation inputs (`config.json`): 11 sparse layers × (512-dim MLA KV + 32×128 indexer) + 34 linear layers × 64·128·128 state — see §6.2 for the worked table.
