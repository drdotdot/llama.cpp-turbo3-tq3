# Session State — TurboQuant CUDA

**Updated**: 2026-03-28 Session 10 (continued)
**Branch**: `release/turbo3-cuda`

## Performance (Updated)
| Type | Short | 32K | PPL ctx=512 | PPL ctx=2048 |
|------|-------|-----|-------------|-------------|
| q8_0 | 57.10 | 47.02 | 6.759 | 5.674 |
| turbo2 | 53.87 | 48.94 | — | 5.929 (+4.5%) |
| turbo3 | 52.07 | 48.27 | **6.803 (+0.65%)** | 5.737 (+1.11%) |
| turbo4 | 48.83 | 44.36 | — | **5.715 (+0.72%)** |

## Session 10 Achievements

### Intelligence (5 repos analyzed)
- spiritbuun (feature + turbo2): turbo2 port, 3 safety fixes
- TheTom (feature + decode-experiments): 12 decode approaches, 4-mag LUT
- 0xSero: Python/Triton, fused score, ring buffer, QJL score-level correction
- yzamari: MLX Metal port, 4 Metal shaders, template kernels
- RecursiveIntell: Rust, polar encoding, proper QJL residual

### Code Changes
1. **3 spiritbuun fixes ported**: partial offload, tensor budget, asymmetric LA 6-8
2. **turbo2 full port**: 23 files, 544 lines, 2-bit 2.5bpv, beats q8_0 at 32K
3. **Independent K/V rotation**: PPL 6.848 → 6.803 at ctx=512 (free quality win)
4. **turbo4 QJL original-space fix**: PPL 5.743 → 5.715 at ctx=2048
5. **Build guards**: CUDA 13.x MMQ segfault + FORCE_CUBLAS perf trap warnings
6. **5 intelligence reports**: intel.md, intel2.md, intel3.md, intel4.md, completedsession.md

### Phase F Resolution
Dimension-specific codebooks NOT needed — our FWHT rotation group is always 128
elements regardless of head_dim. The d=128 Lloyd-Max codebooks are correct for all
head dimensions. Computed and verified d=256 codebooks — they'd only be needed if
rotation group size changed.

## Continuation Prompt
> Read SESSION_STATE.md. Branch: release/turbo3-cuda.
>
> Key results this session:
> - turbo3 PPL: 6.803 (+0.65%) — improved via independent K/V rotation
> - turbo4 PPL: 5.715 (+0.72%) — fixed QJL original-space residual
> - turbo2: working, beats q8_0 by 4.1% at 32K
> - Feature parity with spiritbuun achieved
>
> Remaining:
> 1. Ring buffer (128-token exact fp16) — biggest impact for 0.94x gap, needs arch changes
> 2. turbo4 score-level QJL correction (not fused into dequant)
> 3. lop3 TC-based FA kernel (multi-day moonshot)
> 4. Post to #20969 (draft at .trash/DISCUSSION_POST.md)
