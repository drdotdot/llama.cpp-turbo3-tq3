# Session State — TurboQuant CUDA

**Updated**: 2026-03-27 Session 10
**Branch**: `release/turbo3-cuda`

## Performance (Final Tier 2 — Dense Model)
| Type | Short | 32K | Short ratio | 32K ratio | PPL (2048) |
|------|-------|-----|-------------|-----------|------------|
| q8_0 | 57.10 | 47.02 | baseline | baseline | 5.674 |
| turbo2 | 53.87 | **48.94** | 0.943x | **1.041x** | 5.929 (+4.5%) |
| turbo3 | 53.45 | **48.27** | 0.936x | **1.027x** | 5.736 (+1.1%) |
| turbo4 | 51.78 | — | 0.907x | — | 5.743 (+1.2%) |

## Session 10 Summary

### Phase 1: Bug Fixes
- turbo4 multi-seq NaN: architectural limitation (shadow strides vs ne[3]>1). Documented.
- Sparse V threshold: verified consistent 1e-4 both paths.
- Multi-GPU q_rot_buf: N/A (we use graph-level rotation).
- Partial offload: verified clear error message works.

### Phase 2: turbo2 Full Port — COMPLETE
23 files, 544 lines. Type registration, CPU reference, CUDA kernels, FA integration,
template instances, arg parser, context/graph/kv-cache integration.
- turbo2 PPL ctx=2048: 5.929 (+4.49%)
- turbo2 decode: 53.87 short, 48.94 at 32K (1.041x vs q8_0!)
- Mixed K=turbo2 V=turbo3: needs additional FA instances (future work)

### Phase 4: lop3 Research
Confirmed: lop3 helps bit extraction only (already 2 ops on CUDA). The real win
needs MMA tensor cores with new FA kernel. Multi-day project, deferred.

## Continuation Prompt
> Read SESSION_STATE.md. Branch: release/turbo3-cuda.
>
> Session 10 completed: turbo2 full port (23 files), all known fixes ported.
> Feature parity with spiritbuun achieved (turbo2/3/4, LA modes 1-8, all safety fixes).
>
> Next priorities:
> 1. Mixed turbo2/turbo3 FA instances (for K=turbo2 V=turbo3 mode)
> 2. Fix turbo4 multi-seq (needs shadow ne[3] support or native vec_dot fix)
> 3. lop3 TC-based FA kernel (multi-day moonshot)
> 4. Post to #20969 (draft at .trash/DISCUSSION_POST.md)
