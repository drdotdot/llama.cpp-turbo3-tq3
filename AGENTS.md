---
description: 
alwaysApply: true
---

# CLAUDE.md — TurboQuant CUDA (Madreag/turbo3-cuda)

## Who You Are Working For

Erol Germain (@erolgermain, GitHub: Madreag). Manufacturing Engineer. Direct communicator. Does NOT tolerate:
- Skipping items
- Deferring work to "later" or "next session"
- Stopping to ask "want me to continue?"
- Implementing things halfway
- Moving to the next task before the current one is PROVEN with measurements
- Compromises, fallbacks, or "good enough"

When Erol says "figure it out" — that means investigate, debug, try multiple approaches, and solve the problem yourself. Do not give up. Do not suggest alternatives that avoid the hard work.

## Project Overview

CUDA port of Google's TurboQuant (ICLR 2026) KV cache compression for llama.cpp, targeting NVIDIA RTX 5090 (SM120 Blackwell). Goal: be the definitive TurboQuant CUDA implementation for Blackwell GPUs.

- **Repo**: `/home/erol/ai/turboquant/research/llama-cpp-turboquant/`
- **Branch**: `release/turbo3-cuda`
- **Dense model**: `/home/erol/ai/turboquant/models/opus-v2-Q6_K.gguf` (Qwen 3.5 27B, ~21 GB)
- **MoE model**: `/home/erol/ai/turboquant/models/Qwen3.5-35B-A3B-Q4_K_M.gguf` (~21 GB)
- **Hardware**: RTX 5090 32GB GDDR7, SM120, CUDA 12.8, WSL2 Ubuntu 24.04
- **Competitor reference**: `/home/erol/projects/llama-cpp-turboquant-cuda/` (spiritbuun's fork, RTX 3090)
- **Full context dump**: `.trash/resume.md` — READ THIS for complete project history, architecture decisions, and dead ends

## ⚠️ SETTLED ARCHITECTURE — DO NOT REVISIT

The decode architecture uses a **persistent fp16 shadow cache with incremental dequant**. This has been proven optimal across 11 sessions. The following alternatives have been thoroughly tested and FAILED — do NOT retry them:

1. **Fused SET_ROWS fp16 write** — speed regression (0.944x→0.919x). Tested 3 times (Sessions 2, 6, 7).
2. **Flat array shadow cache** — hash collisions corrupt data (PPL 25.93). Tested Session 5.
3. **Adaptive native/shadow** — native turbo3 vec is SLOWER than fp16 vec at ALL depths. Tested Session 7.
4. **L2 cache persistence hints** — net negative. Tested Session 2.
5. **Fused compressed attention** — register spill from dynamic indexing. Tested Session 1.
6. **Ring buffer direct shadow write from SET_ROWS** — cross-TU access blocked by CUDA compilation model. Previous attempts caused register spill regression (0.944x→0.919x). Tested Sessions 6, 7, 11.
7. **Ring buffer graph-level SET_ROWS bypass** — ggml graph is built with fixed structure, can't conditionally skip ops. Would require custom graph op or llama-graph.cpp modifications. Analyzed Session 11.

See `.trash/resume.md` for detailed failure analysis of each.

## Current Performance (Session 11 — Tier 2 Final)

### Dense Model (Qwen 3.5 27B Q6_K, RTX 5090)
| Type | Short | 8K | 32K | PPL 512 | PPL 2048 |
|------|-------|-----|------|---------|----------|
| q8_0 | 56.10 | 54.17 | 46.03 | 6.759 | 5.674 |
| turbo2 | ~53 | 51.80 | **48.73** | — | 5.929 (+4.5%) |
| turbo3 | 52.55 | ~52 | **~48** | **6.803 (+0.65%)** | 5.737 (+1.11%) |
| turbo4 | 50.55 | ~49 | ~46 | — | **5.715 (+0.72%)** |
| K=turbo3/V=q8_0 | 54.46 | 52.51 | 46.15 | 6.804 (+0.67%) | 5.650 (**-0.42%**) |

### MoE Model (Qwen 3.5 35B-A3B Q4_K_M)
| Type | Short | 32K | 128K |
|------|-------|------|------|
| q8_0 | 180.14 | 149.62 | 89.92 |
| turbo3 | 168.77 | 141.70 | **116.65 (1.30x)** |
| K=turbo3/V=q8_0 | 169.75 | — | 94.79 (1.05x) |

## ABSOLUTE RULES

### 1. NEVER skip an item. NEVER defer.
If the plan says to do something, do it. If it's blocked, find another way.

### 2. MEASURE before and after EVERY change.
No commit without benchmark data. PPL at ctx=512 AND ctx=2048 for every code change. Record in `benchmarks/CHANGELOG.md`.

### 3. PPL REJECT THRESHOLDS — non-negotiable.
- ctx=512: turbo3 PPL > 6.89 → **REJECT and revert immediately**
- ctx=2048: turbo3 PPL > 5.77 → **REJECT and revert immediately**

### 4. ONE commit per logical change.
Implement → rebuild → measure → commit → next item.

### 5. Fix regressions IMMEDIATELY.
If any metric gets worse, stop and fix before moving on.

### 6. Read reference implementations BEFORE reimplementing.
spiritbuun's fork at `/home/erol/projects/llama-cpp-turboquant-cuda/` is the CUDA reference.

### 7. NEVER ask "want me to continue?"
The answer is always yes. Save state to `.trash/resume.md` when context is low.

### 8. Understand the ARCHITECTURE before writing code.
Read the decode data flow in `.trash/resume.md` before modifying fattn.cu or turbo-quant.cu.

## Build & Test Commands

```bash
# Build
cd /home/erol/ai/turboquant/research/llama-cpp-turboquant
/home/erol/miniconda3/envs/tq/bin/cmake --build build -j$(nproc)

# PPL gate (run for EVERY code change):
MODEL=/home/erol/ai/turboquant/models/opus-v2-Q6_K.gguf
WIKI=wikitext-2-raw/wiki.test.raw
./build/bin/llama-perplexity -m $MODEL -f $WIKI -c 512 -ctk turbo3 -ctv turbo3 -fa on --chunks 8 -ngl 99
./build/bin/llama-perplexity -m $MODEL -f $WIKI -c 2048 -ctk turbo3 -ctv turbo3 -fa on --chunks 8 -ngl 99

# Full decode curve (NOTE: llama-bench uses -d for depth, NOT -c):
for DEPTH in 0 2048 4096 8192 16384 32768; do
  ./build/bin/llama-bench -m $MODEL -fa 1 -ctk turbo3 -ctv turbo3 -d $DEPTH -ngl 99 -t 1 -r 3 -p 0 -n 128
  ./build/bin/llama-bench -m $MODEL -fa 1 -ctk q8_0 -ctv q8_0 -d $DEPTH -ngl 99 -t 1 -r 3 -p 0 -n 128
done

# Prefill:
for DEPTH in 0 8192 32768; do
  ./build/bin/llama-bench -m $MODEL -fa 1 -ctk turbo3 -ctv turbo3 -d $DEPTH -ngl 99 -t 1 -p 512 -n 0 -r 1
done
```

## Architecture Knowledge

### TurboQuant turbo3 (3.25 bpv, 4.6x compression)
- Block: 16 bytes per 32 values (padded from 14), rotation group = 128 values = 4 blocks
- Lloyd-Max 8 centroids, FWHT rotation for Gaussianization
- Norm correction: `corrected_norm = raw_norm / ||reconstruction||`
- Centroids: `{-0.190685, -0.117832, -0.065717, -0.021460, 0.021460, 0.065717, 0.117832, 0.190685}`
- Independent K/V rotation signs (Session 10): V uses different FWHT sign arrays than K

### TurboQuant turbo2 (2.5 bpv, 6.4x compression)
- 2-bit PolarQuant only (no QJL). 4 centroids, same FWHT rotation as turbo3.
- Block: 10 bytes per 32 values. PPL +4.5% at ctx=2048.
- Best compression-per-quality ratio. Beats q8_0 by 5.9% at 32K.

### TurboQuant turbo4 (4.25 bpv)
- 3-bit PolarQuant + 1-bit QJL residual correction
- Block: 68 bytes per 128 values (norm + rnorm + 48 qs + 16 signs)
- Current: MSE-only shadow dequant (QJL data stored but unused in shadow path)
- QJL residual computed in original space (fixed Session 10, PPL 5.743→5.715)
- Known limitation: native vec_dot gives NaN with Q->ne[3]>1 (multi-seq PPL at ctx=512)

### SM120 (RTX 5090) Constraints
- 128 KB shared memory/SM, 99 KB max per block
- 48 max warps/SM (NOT 64 like SM100 datacenter)
- NO WGMMA, NO TMEM — only extended `mma.sync`
- 98 MB L2 cache, 1.79 TB/s GDDR7 bandwidth
- FlashAttention-3/4 does NOT work on SM120
- HAS native FP4 E2M1 Tensor Core support via `mma.sync.m16n8k64`
- CUDA 12.8 only (13.1 segfaults on MMQ kernels)

### Key Files
```
ggml/src/ggml-cuda/turbo-quant.cu   — CUDA dequant, WHT, SET_ROWS (turbo2/3/4), independent K/V signs
ggml/src/ggml-cuda/fattn-common.cuh — FA vec_dot for turbo3 + V dequant + sparse V (1e-4 threshold)
ggml/src/ggml-cuda/fattn-vec.cuh    — sparse V threshold, __expf, nthreads
ggml/src/ggml-cuda/fattn.cu         — FA dispatch: shadow cache, prefill MMA, turbo4 multi-seq bypass
ggml/src/ggml-cuda/CMakeLists.txt   — Template instance globs (turbo2/3/4, f16-q8_0)
ggml/src/ggml-turbo-quant.c         — CPU reference quantize/dequant (turbo2/3/4)
src/llama-kv-cache.cpp              — Layer-adaptive modes 1-8 (TURBO_LAYER_ADAPTIVE), partial offload guard
src/llama-context.cpp               — FA requirement check for turbo types
src/llama-graph.cpp                 — Graph-level WHT rotation (forward=K signs, inverse=V signs)
common/arg.cpp                      — CLI arg parsing for turbo2/turbo3/turbo4
```

## Common Mistakes To Avoid

### ❌ Trying to eliminate the shadow cache
The persistent fp16 shadow IS the optimal architecture. See "SETTLED ARCHITECTURE" above. The 5.6% short-context gap is inherent to the shadow approach and cannot be fixed without a fundamentally new FA kernel (e.g., FP4 Tensor Core, BitDecoding lop3).

### ❌ Re-dequanting the entire KV cache every token
O(context_length) work per token when only O(1) new data was added. Use incremental dequant.

### ❌ Reimplementing from scratch when reference code exists
spiritbuun's fork is cloned locally. Read it first.

### ❌ Batching multiple changes without measuring between them
One change → one measurement → one commit.

### ❌ Skipping PPL validation
Every code change needs PPL at ctx=512 AND ctx=2048. If it exceeds reject thresholds → revert.

### ❌ Using cudaEventSynchronize inside graph compute
CUDA graphs are enabled (USE_GRAPHS=1). Sync inside graph replay = crash.

### ❌ Stopping early to "save state"
Do NOT stop to write SESSION_STATE.md unless you literally cannot make another tool call.
Push context to the absolute limit. Save state only when forced.

## Commit Message Format

```
<type>: <short description>

<Detailed explanation of what changed and why>

<Benchmark results — REQUIRED>
  PPL impact: turbo3=X.XXX, q8_0=X.XXX, delta=+X.XX% (ctx=512)
  Decode: XX.XX tok/s (ratio vs q8_0)
```

Types: `feat`, `fix`, `perf`, `docs`, `test`, `refactor`

**NEVER add Co-Authored-By lines.** Credit spiritbuun/TheTom in the commit message body text if adapted from their code.

## Environment Variables

| Variable | Effect |
|----------|--------|
| `TURBO_LAYER_ADAPTIVE=N` | Per-layer KV type: 0=uniform, 1=first4+last4 K+V→q8_0 (best PPL), 6=V-only last8→q8_0, 7=K-only last8→q8_0, 8=V-only first2+last2→q8_0 |
| `GGML_TURBO_DECODE_NATIVE=1` | Disable fp16 shadow, use native turbo3 vec (slower, for debug) |

## Competitive Intelligence

5 competitor repos analyzed in `.trash/intel*.md`:
- **spiritbuun** (feature + turbo2 branches): CUDA reference on RTX 3090. We ported turbo2, 3 safety fixes, asymmetric LA modes 6-8.
- **TheTom** (feature + decode-experiments): Metal implementation. 12 decode approaches tested, 4-mag LUT won on Metal.
- **0xSero**: Python/Triton for vLLM. Key insights: fused score from packed data, ring buffer, QJL as additive score (not fused dequant), per-layer rotation seeds.
- **yzamari**: MLX Metal port of 0xSero. Confirmed: QJL residual must be in original space, score-level correction pattern.
- **RecursiveIntell**: Rust, polar encoding. Confirmed turbo4 residual wrong (3rd codebase). Configurable QJL projections (m < d).

**Feature parity with spiritbuun achieved** (turbo2/3/4, LA modes 1-8, all safety fixes).

## Full Context

**Read `.trash/resume.md`** for the complete context dump including:
- All implemented optimizations with files and impacts
- Complete decode data flow (decode + prefill paths)
- Shadow cache lifecycle details
- All 7 dead ends with detailed failure analysis
- Hardware specs, competitor analysis, commit history
- Research paper summaries and frontier opportunities
- WSL2 caveats

## Remaining Work

1. **Ring buffer** — needs `llama-graph.cpp` changes to skip SET_ROWS for recent tokens. 4% FWHT cost is the dominant overhead. All approaches within current architecture investigated and insufficient.
2. **turbo4 score-level QJL** — needs FA logit bias mechanism (doesn't exist in llama.cpp FA). QJL data is stored correctly, just unused in shadow path.
3. **lop3 TC-based FA kernel** — multi-day moonshot. Standalone prototype exists at `.trash/lop3_turbo3_dot.cu`. Needs full MMA-based FA kernel.
4. **Post to llama.cpp #20969** — draft at `.trash/DISCUSSION_POST.md`

## Remember

- turbo3 BEATS q8_0 at 32K (1.04x dense, 1.30x MoE at 128K) while using 4.6x less KV memory
- turbo2 BEATS q8_0 at 32K (+5.9%) with 6.4x compression
- Asymmetric K=turbo3/V=q8_0 has BETTER quality than q8_0 at ctx=2048 (PPL 5.650 vs 5.674)
- Short context gap (0.944x) is 4% FWHT in SET_ROWS — needs graph-level changes to fix
- Independent K/V rotation already improved PPL from 6.848→6.803 (free win)
- Every optimization must be MEASURED. Intuition is not data.
- DO NOT STOP. DO NOT DEFER. DO NOT SKIP. FINISH THE WORK.
