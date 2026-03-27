# Session State — TurboQuant CUDA

**Updated**: 2026-03-27 Session 9
**Branch**: `release/turbo3-cuda`

## Performance (unchanged from Session 8)
| Context | q8_0 | turbo3 | Ratio |
|---------|------|--------|-------|
| short | 55.05 | 51.95 | 0.944x |
| 32K | 45.96 | 47.76 | 1.039x |
| 128K MoE | 79 | 87 | 1.100x |

## Session 9 Summary

### Competitive Intelligence (Phase 1-3)
Cloned 5 repos across spiritbuun (feature, turbo2) and TheTom (feature, decode-experiments, turboquant_plus). Full analysis at `.trash/intelligence.md`.

Key discoveries:
- **spiritbuun has TURBO2_0** (2-bit, 2.5 bpv, 6.4x compression). PPL +8% uniform (too aggressive), but turbo2-K + turbo3-V at +3.9% is interesting.
- **spiritbuun has multi-seq fix** (6f8f923) — different arch from ours (bulk dequant vs shadow cache)
- **TheTom tested 12 decode approaches** — 4-mag LUT won at +38% on Metal. Not directly applicable to our CUDA shadow path.
- **No new external competitors** found in web search.

### Ported Fixes (Phase 4)
1. **Partial GPU offload crash** (spiritbuun c99c230): CPU type_traits registration + runtime error check
2. **KV cache tensor budget** (spiritbuun 6cdd9db): n_turbo_extra=8 safety margin
3. **Asymmetric LA modes 6-8** (spiritbuun 65e28eb): K/V independent promotion
   - Mode 6 tested: PPL 6.817 (+0.86%) — better than uniform turbo3

### turbo2 Port (stashed)
Type registration (ggml.h, ggml-common.h, ggml.c) stashed as WIP. Needs:
- CPU reference quantize/dequant in ggml-turbo-quant.c
- CUDA kernels (SET_ROWS, GET_ROWS, shadow dequant)
- FA vec_dot + V dequant
- Template instances
- Integration (context, graph, kv-cache, bench)

## Continuation Prompt
> Read SESSION_STATE.md and .trash/intelligence.md. Branch: release/turbo3-cuda.
>
> Session 9 completed: competitive intel + 3 ported fixes.
> turbo2 type registration stashed (`git stash pop` to continue).
>
> Next priorities:
> 1. Complete turbo2 port (stashed WIP, ~4-5 hours remaining)
> 2. Fix native turbo4 vec_dot for multi-seq (Q->ne[3]>1)
> 3. Post discussion to #20969 (draft at .trash/DISCUSSION_POST.md)
> 4. lop3 TC-based FA kernel prototype
