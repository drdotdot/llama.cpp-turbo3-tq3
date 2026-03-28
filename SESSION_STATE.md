# Session State — TurboQuant CUDA

**Updated**: 2026-03-28 Session 11
**Branch**: `release/turbo3-cuda`

## Performance (Session 11 Final)
| Type | Short | 32K | PPL ctx=512 | PPL ctx=2048 |
|------|-------|-----|-------------|-------------|
| q8_0 | ~55 | 45.48 | 6.759 | 5.674 |
| turbo2 | ~53 | **48.59** | — | 5.929 (+4.5%) |
| turbo3 | 52.76 | **47.86** | **6.803 (+0.65%)** | 5.737 (+1.11%) |
| turbo4 | ~49 | 44.85 | — | **5.715 (+0.72%)** |
| K=turbo3/V=q8_0 | 54.56 | 45.23 | 6.804 (+0.67%) | 5.650 (-0.42%) |

## Session 11 Summary

### Phase 1: Ring Buffer Analysis
Investigated 3 approaches for the 0.944x short-context gap:
- Direct shadow write from SET_ROWS: cross-TU access to shadow (fattn.cu ↔ turbo-quant.cu) blocked by CUDA compilation model. Previous attempts (Session 6) caused register spill regression.
- Graph-level SET_ROWS bypass: ggml graph is built with fixed structure, can't conditionally skip ops. Would require custom graph op or llama-graph.cpp modifications.
- Batched shadow sync: 32 kernel launches × ~3μs = ~100μs. Savings from batching are ~0.5%.

**Finding**: The 4% FWHT cost in SET_ROWS is the dominant overhead and requires graph-level changes to skip. Ring buffer within the current architecture saves at most ~1%. Documented in .trash/ASK.md.

### Phase 2: turbo4 Score-Level QJL (Partial)
- MSE-only shadow dequant: turbo4 now matches turbo3 quality (5.737 at ctx=2048)
- QJL signs + rnorm stored correctly for future score-level correction
- Full score-level QJL needs FA logit bias support (no existing mechanism in llama.cpp FA)

### Phase 4: Verification
- 4A: Asymmetric K=turbo3/V=q8_0 PP = 2649 tok/s (fast, no CPU fallback)
- 4B: MoE 128K turbo3 = 52.57 tok/s (no crash)
- 4D: Full Tier 2 benchmark completed

## Continuation Prompt
> Read SESSION_STATE.md. Branch: release/turbo3-cuda.
>
> Ring buffer needs graph-level changes to llama-graph.cpp (skip SET_ROWS for recent tokens).
> turbo4 score-level QJL needs FA logit bias mechanism.
> Both are deferred to dedicated sessions.
>
> All other features are complete and verified. Feature parity achieved.
