## 2026-03-28 Fused SET_ROWS Attempt

### Question
The fused SET_ROWS fp16 write approach has a fundamental lifecycle problem: the shadow buffer
doesn't properly track KV cache clears. When perplexity runs multiple chunks, each chunk clears
and repopulates the cache. The shadow serves stale data, producing wrong PPL.

The incremental sync approach (turbo_shadow_sync in FA) handles this correctly because it checks
ne[1] < filled on every call. The fused approach skips this check (zero overhead) but loses the
cache-clear detection.

### What I tried
1. Static unordered_map tracking valid rows per data pointer
2. Reset when ne[1] decreases (cache clear detection)
3. Both produced wrong PPL at ctx=2048 (stale shadow data)

### My recommendation
Keep the incremental sync approach. The overhead is small (unordered_map lookup + conditional
dequant of 1 row per token per layer). The short-context gap (5.6%) is inherent to the fp16 shadow
stride mismatch, not the sync overhead.

### What I'm doing while waiting
Moving to Task 2: MoE model benchmark for apples-to-apples comparison with spiritbuun.

---

## 2026-03-28 Ring Buffer Analysis

### Finding
Profiled the 0.944x gap breakdown:
- FWHT in SET_ROWS: ~4% (main cost)
- Shadow sync dequant: ~1% (32 kernel launches per token)
- Other (stride mismatch, map lookups): ~0.5%

### Why Ring Buffer Doesn't Help Much Within Current Architecture
A ring buffer saves only the ~1% shadow sync. The FWHT runs regardless because SET_ROWS
is a ggml graph op. To get to >0.98x, we'd need to skip SET_ROWS turbo quantization for
recent tokens — that requires modifying llama-graph.cpp to conditionally create SET_ROWS ops.

### Recommendation
Needs a dedicated session focused on graph-level modifications to llama.cpp.
