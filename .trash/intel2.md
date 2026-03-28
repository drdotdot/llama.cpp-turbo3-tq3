# Intelligence Report: 0xSero/turboquant

**Repo**: https://github.com/0xSero/turboquant
**Author**: seroxdesign (Sherif Cherfa)
**Language**: Python (PyTorch + Triton + vLLM)
**Last updated**: 2026-03-27
**Commits**: 2 (v0.2.0 initial + cleanup/license change)
**Total code**: ~2,900 lines across 12 Python modules

---

## What This Is

A **pure Python/Triton** TurboQuant implementation targeting **vLLM inference**. NOT a llama.cpp fork — this is a completely separate codebase that monkey-patches vLLM's attention backend to intercept KV cache writes and replace them with TurboQuant compressed storage.

**Key distinction from our work**: We modified llama.cpp's CUDA kernels at the ggml level. 0xSero operates at the Python/vLLM application layer with Triton kernels. These are complementary, non-competing approaches.

---

## Architecture

```
vLLM engine
  └── Attention layer (monkey-patched)
        ├── Prefill: normal flash attention → capture KV to CompressedKVStore
        └── Decode: hybrid attention
              ├── Compressed history (TurboQuant MSE + QJL for keys, group quant for values)
              └── Recent exact buffer (last N tokens uncompressed)
```

### Key Design Decisions

1. **Keys use TurboQuantProd** (Algorithm 2 from paper): MSE quantization + QJL residual correction. Unbiased inner product estimation.

2. **Values use standard group quantization** (NOT TurboQuant): min-max symmetric, 2-bit or 4-bit, per-group scales and zeros. This is a pragmatic choice — the paper's polar quantization is designed for inner products (Q·K), not for value accumulation where asymmetric quantization works better.

3. **Rotation uses full QR matrix** (not FWHT): `generate_rotation_matrix` does QR decomposition of a random Gaussian matrix. This is O(d^2) storage but exact (our FWHT is O(d log d) and approximate). For head_dim=128, the matrix is 64KB — negligible.

4. **Hybrid decode**: Compressed historical tokens + exact recent buffer. The recent buffer keeps the last N tokens unquantized for quality on recent context.

5. **Triton fused kernels**: 3 kernels for decode attention:
   - MSE score: compute Q·K from packed indices without dequantizing
   - QJL score: compute QJL correction from packed sign bits
   - Fused decode attention: online softmax over compressed KV

---

## Codebooks

Pre-computed Lloyd-Max codebooks for different dimensions and bit widths:

| d | bits | Centroids |
|---|------|-----------|
| 128 | 2 | {-0.1330, -0.0400, 0.0400, 0.1330} |
| 128 | 3 | {-0.1884, -0.1181, -0.0666, -0.0216, 0.0216, 0.0666, 0.1181, 0.1884} |
| 128 | 4 | 16 centroids |
| 64 | 2/3/4 | Different centroids (wider distribution for smaller d) |
| 576 | 3 | {-0.0895, ..., 0.0895} (smaller centroids for larger d) |

**Comparison with our centroids** (d=128, 3-bit):
- 0xSero: {-0.1884, -0.1181, -0.0666, -0.0216, 0.0216, 0.0666, 0.1181, 0.1884}
- Ours: {-0.190685, -0.117832, -0.065717, -0.021460, 0.021460, 0.065717, 0.117832, 0.190685}
- **Difference**: ~1% — both are Lloyd-Max optimal for Beta(d/2, d/2) on [-1,1], minor numerical precision differences.

**Interesting**: They have d=576 codebooks for models with large head dimensions (e.g., Qwen3.5's pooled attention). We only support d=64/128/256 in our FA instances.

---

## Benchmark Results (from README)

### RTX 5090 — Qwen3.5-27B-AWQ (dense)
| Metric | Baseline | TurboQuant |
|--------|----------|------------|
| Prefill tok/s (30K) | 1,804 | 1,907 (+5.7%) |
| Decode tok/s (30K) | 1.264 | 1.303 (+3.1%) |
| KV freed | — | 30 GB |
| Max tokens | 457K | 914K (2x) |

### 8x RTX 3090 — Qwen3.5-35B-A3B MoE (131K context)
| Context | Prefill | Decode | TTFT |
|---------|---------|--------|------|
| 1K | 7,127 | 129.7 | 0.14s |
| 32K | 9,761 | 116.7 | 3.28s |
| 131K | 8,238 | 98.3 | 15.9s |

**KV savings**: 30.9% overall (only 10 of 40 layers are full-attention and compressible). On pure dense transformer: 77% savings (4.4x).

### Quality
- Needle-in-haystack: 4/5 at all context lengths (model reformats one answer)
- 5-needle at near-max: 5/5 retrieved
- cos_sim: keys 1.000 (3-bit), values 0.940 (2-bit)

---

## What We Can Learn

### 1. Value Quantization Strategy
**They use standard group quantization for values, NOT TurboQuant.** This is significant:
- 2-bit values: cos_sim = 0.940 (the quality bottleneck)
- 4-bit values: cos_sim = 0.997

Our approach: both K and V go through TurboQuant (same PolarQuant + centroids). Their approach: K uses TurboQuant (rotation + centroids), V uses simple per-group min-max quantization.

**Takeaway**: For our asymmetric K=turbo3 V=q8_0 mode, we're already doing something similar — high-quality K compression, full-precision V. Their explicit per-group value quantization at 2-bit is a different design point we could explore.

### 2. Hybrid Decode (Compressed History + Exact Recent Buffer)
They keep the last N tokens uncompressed in an exact buffer, only compressing older tokens. During decode, attention merges both segments via log-sum-exp.

**Takeaway**: We could implement a "buffer window" where the most recent K tokens are kept as fp16 (no shadow overhead) and only older positions go through turbo compression. This would eliminate the incremental dequant cost for the hottest positions.

### 3. Dimension-Specific Codebooks
They compute and store separate codebooks per head dimension (d=64, 128, 256, 576). The centroids differ because the Beta distribution changes with d.

**Takeaway**: Our centroids are for d=128 specifically. For models with d=64 or d=256, the optimal centroids would be slightly different. In practice, the difference is small (~1-2% PPL) but worth noting.

### 4. Fused Score Computation (No Dequantization)
Their Triton kernels compute attention scores DIRECTLY from packed indices without materializing dequantized vectors:
```
score = norm * sum_j(q_rot[j] * centroid[idx[j]])
```
The query is pre-rotated (q @ Pi^T), then dot product is computed against centroid LUT values.

**Takeaway**: This is the "fused compressed attention" approach we tested and rejected in Session 1 (Dead End #5 — register spill from `partial_q[idx] +=`). However, their Triton implementation avoids the register spill issue because Triton has better register allocation than CUDA C++. If we ever move to Triton kernels, this approach becomes viable.

### 5. Adversarial Self-Audit
Their `audit_claims.py` honestly assesses their own claims. Findings:
- "5.1x compression" is misleading (doesn't count rotation matrices)
- "Needle-in-haystack passes" is trivial (query=key test too easy)
- "Recall@8 >= 0.40" is a low bar
- "Hybrid decode saves memory but not compute" (dequantizes everything per step)

**Takeaway**: The honest self-assessment is good practice. We should audit our own claims similarly — our "1.04x beats q8_0" claim is solid but context-dependent (only at 32K+).

### 6. QJL Implementation Details
Their QJL uses a **full random Gaussian matrix S** (not FWHT with random signs). The projection is:
```python
S = torch.randn(d, d, generator=rng)  # NOT sparse, NOT Hadamard
q_sketched = query @ S.T               # full d×d matmul
```

**Takeaway**: Our GPU turbo4 uses FWHT for the QJL rotation (same signs1/signs2 approach as the main rotation). The paper describes using a random Gaussian matrix. The FWHT approximation may explain why our turbo4 QJL correction is less effective than expected.

---

## What They DON'T Have (Our Advantages)

| Feature | Us | 0xSero |
|---------|-----|--------|
| C/CUDA native kernels | YES (ggml-level) | No (Python/Triton) |
| Persistent shadow cache | YES | No (dequantizes per step) |
| Sparse V dequant | YES (1e-4 threshold) | No |
| llama.cpp integration | YES (full FA path) | No (vLLM only) |
| Layer-adaptive modes | YES (8 modes) | Limited (initial layers configurable) |
| turbo2/turbo4 | YES | No (only TurboQuantProd ~turbo3 equivalent) |
| Blackwell optimization | YES | No (generic Python) |
| FWHT (fast rotation) | YES (O(d log d)) | No (full QR, O(d^2)) |

---

## What They Have That We DON'T

| Feature | 0xSero | Us |
|---------|--------|-----|
| vLLM integration | YES (monkey-patch) | No |
| Fused Triton kernels (no dequant) | YES (3 kernels) | No (dequant to fp16 first) |
| Separate value quantization (group quant) | YES (2/4-bit) | No (values use same turbo format) |
| Hybrid decode (exact recent buffer) | YES | No (all positions go through shadow) |
| Per-dimension codebooks | YES (d=64/128/256/576) | No (hardcoded d=128 centroids) |
| Paper validation suite | YES (9 tests for Theorems 1-3) | No |
| Adversarial audit script | YES | No |

---

## Should We Port Anything?

### YES — Worth implementing:
1. **Hybrid recent buffer**: Keep last N tokens as fp16, only compress older positions. Reduces shadow overhead for the hottest attention positions. Moderate effort (modify shadow sync logic).

2. **d-specific centroids**: Pre-compute codebooks for d=64 and d=256 (we currently use d=128 centroids for all). Small effort (compute once, store as constants).

### MAYBE — Interesting but different ecosystem:
3. **Fused score computation**: Computing scores directly from packed data without dequant. Our Session 1 attempt failed on CUDA C++ (register spill), but the concept is valid. May work better with lop3+MMA approach.

### NO — Not applicable:
4. **vLLM integration**: Different inference engine. Not relevant to llama.cpp.
5. **Full QR rotation**: We use FWHT which is faster and works well. QR is O(d^2) vs O(d log d).
6. **Python/Triton kernels**: We need CUDA C++ for ggml integration.

---

## Novel Insights

1. **Value quantization decoupled from key quantization**: They recognized that the TurboQuant algorithm (polar decomposition + centroids) is optimized for inner products (Q·K), not for value accumulation. Standard min-max group quantization works better for values. This validates our asymmetric K=turbo3 V=q8_0 approach.

2. **"Hybrid decode saves memory but not compute"**: Their honest admission that the hybrid approach dequantizes all compressed history during compute. Our persistent shadow cache actually avoids this — we dequant incrementally and cache the result. We have a genuine compute advantage.

3. **cos_sim 1.000 for 3-bit keys**: Their measured cosine similarity of 1.0000 for 3-bit TurboQuant keys (at d=256) confirms the algorithm's near-lossless key compression. The quality loss is entirely from value quantization.

4. **MoE models benefit less**: TQ only compresses full-attention layers. Hybrid architectures (Qwen3.5 with GatedDeltaNet) have incompressible state that limits overall impact. We see the same: our MoE results show 0.846x at short context vs 0.944x for dense.

---

---

## Algorithmic Ideas That Transfer to Our CUDA Implementation

### IDEA 1: Fused Score from Packed Data (No Dequantization)

Their Triton Kernel 1 computes `score = norm * sum_j(q_rot[j] * centroid[idx[j]])` directly from packed bytes. The query is pre-rotated ONCE, then the inner loop extracts indices and accumulates against a centroid LUT. No intermediate fp16 buffer.

**Maps to our CUDA**: This is exactly what a lop3+register-LUT kernel would do. Our native turbo3 `vec_dot` already does the same math but is slower than the shadow fp16 path because centroid lookup + bit extraction costs exceed fp16 half2 vectorized ops. Their approach shows it CAN be competitive by processing BLOCK_N=64-128 KV positions per thread block to amortize the LUT load. Key architectural difference: they process KV in blocks (flash-style), we process per-position (vec-style).

### IDEA 2: Separate Key and Value Quantization

Keys use TurboQuant (rotation + centroids + QJL). Values use standard per-group min-max quantization (2-bit or 4-bit with scale+zero per group of 32 elements). This preserves value magnitudes better than rotation-based quantization.

**Actionable**: We could add `GGML_TYPE_GQ2` (2-bit group quantization) as a new value type. K=turbo3 V=gq2 would give ~5x overall compression with better V quality than turbo2 (cos_sim 0.94 for group quant vs our turbo2's codebook approach).

### IDEA 3: Ring Buffer for Recent Tokens

Last N tokens (128) stay in exact bf16 — zero quantization overhead on the decode hot path. When buffer is full, oldest chunk flushes to compressed store. Attention merges both segments via log-sum-exp.

**Impact**: Eliminates the 5.6% shadow overhead at short context entirely. At 8K+, ring buffer is a small fraction of total KV so benefit diminishes. Would require splitting the FA dispatch into two segments + a merge step.

### IDEA 4: Online Softmax Over Compressed KV (Fused Kernel)

Flash-attention-style online softmax in a single pass over compressed KV. For each block of N positions: compute scores from packed data, update running max/normalizer, dequant values only for non-negligible positions, accumulate. Combined with sparse V, this skips BOTH score AND value work for negligible blocks.

**This is the moonshot**: Replace our entire two-step approach (shadow dequant + stock FA kernel) with a single fused compressed attention kernel. No shadow buffer needed. Only memory traffic is reading the packed turbo3 blocks. This is the architecture for the lop3 MMA kernel.

### IDEA 5: QJL as Separate Additive Score (Not Fused Into Dequant)

Their attention score splits into two independent terms:
```
total = MSE_score + QJL_score
MSE_score = norm * dot(q_rotated, centroids[indices])
QJL_score = qjl_scale * residual_norm * dot(q_sketched, signs)
```

The query is projected through TWO matrices: Pi (rotation, for MSE) and S (random Gaussian, for QJL). Each projection computed once per decode step.

**Why our turbo4 QJL is weak**: We fuse centroid + QJL into a single dequanted value `(C[idx] + s * qjl_scale) * norm`, computed in the ROTATED space. Their approach keeps them separate and adds at the SCORE level. The QJL correction uses the ORIGINAL query projected through S, preserving the unbiased estimator property from the paper. Our fused approach loses this property because the cross-space residual doesn't match the theory.

**To fix turbo4 properly**: Need to compute MSE and QJL scores independently, add them, then feed to softmax. Requires storing the S matrix and pre-computing `q @ S^T` per step. Fundamental architectural change.

---

---

## Deep Code Analysis — Line-by-Line Algorithmic Findings

### FINDING 1: Chunked Store with Lazy Flatten (store.py)

Their `CompressedKVStore` keeps quantized chunks in a LIST, not a single contiguous buffer:
```python
self._key_chunks: list[ProdQuantized] = []
self._flat: Optional[FlatCache] = None

def append_chunk(self, key, value):
    key_q = self.quantizer.quantize(k)
    self._key_chunks.append(key_q)
    self._flat = None  # invalidate

def get_flat_cache(self):
    if self._flat is not None:
        return self._flat  # cached
    flat_kq = _concat_prod_q(self._key_chunks)  # lazy concatenation
    self._flat = FlatCache(prod_q=flat_kq, ...)
    return self._flat
```

**Why this is smart**: During decode, new chunks are appended cheaply (list append). The concatenated flat view is only materialized when attention needs to read. If multiple decode tokens are generated between reads, they batch into one concat.

**For our CUDA shadow**: Our shadow dequants incrementally (1 row per token). Their approach defers all work until read time. We could adopt a similar lazy pattern: batch incremental dequants and only execute when the FA kernel actually needs the shadow. Currently we dequant synchronously on every SET_ROWS call. Deferring to FA dispatch would let us skip dequant if the KV cache is cleared before the next FA call (e.g., between perplexity chunks).

### FINDING 2: GQA Broadcast Without repeat_interleave (score.py)

Their attention computation handles GQA by BROADCASTING, not by repeating K/V:
```python
# q: (T, Q, D) -> (H_kv, G, T, D)  where G = gqa_ratio
q = query.view(T, H_kv, gqa_ratio, D).permute(1, 2, 0, 3)
k = kv_keys.unsqueeze(1)    # (H_kv, 1, N, D) — broadcasts over G
v = kv_values.unsqueeze(1)  # (H_kv, 1, N, D) — broadcasts over G

scores = torch.einsum("hgtd,hgnd->hgtn", q, k) * scale
```

**Why this matters**: `repeat_interleave` creates copies. Broadcasting uses no extra memory. At 128K context with 6:1 GQA, this saves 5x the KV memory during attention computation.

**For our CUDA FA**: Our FA vec kernel already uses GQA optimization (the `gqa_opt_applies` path in `get_best_fattn_kernel`). But confirming the broadcast-not-repeat pattern is the right approach.

### FINDING 3: The TurboQuantProd Residual — Full Dequant in Original Space (quantizer.py)

The most algorithmically significant code:
```python
def quantize(self, x):
    # Stage 1: MSE quantize
    mse_q = self.mse_quantizer.quantize(x)          # quantize in ROTATED space
    x_hat = self.mse_quantizer.dequantize(mse_q)    # FULLY DEQUANT BACK to original space
    #    dequantize does: centroids -> rotate_backward(y_hat, Pi) -> rescale by norms
    #    i.e., x_hat = Pi^T @ centroids[indices] * norms  (ORIGINAL SPACE)

    residual = x - x_hat                              # residual in ORIGINAL space
    projected = torch.matmul(residual.float(), self.S.T)  # project through GAUSSIAN S
    signs = sign(projected)
```

And the score computation:
```python
def attention_score(self, query, quantized_key):
    # MSE part: full dequant -> matmul
    k_mse = self.mse_quantizer.dequantize(mse_q)    # Pi^T @ centroids * norms
    scores_mse = matmul(query, k_mse.T)              # standard Q @ K^T

    # QJL part: sketch query, dot with signs
    q_sketched = matmul(query, self.S.T)             # project query through S
    scores_qjl = matmul(q_sketched, signs.T) * qjl_scale * residual_norms

    return scores_mse + scores_qjl
```

**The critical insight for our turbo4**: Their QJL operates on the RESIDUAL in ORIGINAL space (`x - Pi^T @ centroids * norms`), NOT in rotated space. The projection matrix is a full random Gaussian S, NOT the FWHT.

Our turbo4 SET_ROWS does:
```c
residual[j] = x[j] - recon[j];  // ROTATED space residual
turbo_fwht_128(residual);         // project through FWHT (not Gaussian S)
```

This is wrong in TWO ways:
1. Residual should be in original space (need inverse FWHT before computing residual)
2. QJL projection should use a separate random matrix S, not the same FWHT

And our turbo4 dequant fuses centroid + QJL:
```c
val = (C[idx] + s * qjl_scale) * norm;  // single fused value
```

Their approach keeps them separate:
```python
k_mse = rotate_backward(centroids[indices]) * norms   # MSE reconstruction
x_qjl = matmul(signs, S) * qjl_scale * residual_norms # QJL reconstruction
k = k_mse + x_qjl                                      # combined
```

### FINDING 4: Prefill Bypass — Ring Buffer Handles Short Context (capture.py)

Their `ingest_prefill` logic:
```python
def ingest_prefill(self, key, value, num_tokens):
    if num_tokens <= self.ring.capacity:
        # Short prompt: just put everything in the exact buffer
        self.ring.write(key, value, num_tokens)
    else:
        # Long prompt: compress old tokens, keep recent in buffer
        n_compress = num_tokens - self.ring.capacity
        self.store.append_chunk(key[:n_compress], value[:n_compress])
        self.ring.write(key[n_compress:], value[n_compress:], self.ring.capacity)
```

**Key insight**: For prompts shorter than `ring_capacity` (128), ZERO quantization happens. The entire KV cache stays exact. This means:
- Short conversations: zero turbo overhead, 1.0x of fp16 quality
- Only long prompts trigger quantization
- The ring buffer acts as a natural quality preserving window

**For our CUDA**: At short context (<128 tokens), our shadow still launches 32 dequant kernels per token. With this approach, we'd skip all of it. The FA dispatch would just use the ring buffer's fp16 data directly — identical to the non-turbo path.

### FINDING 5: Ring Buffer Overflow with Chunked Writes (capture.py)

Their ring buffer handles overflow elegantly:
```python
def write(self, key, value, num_tokens):
    while remaining > 0:
        space = self.capacity - self._pos
        if space <= 0:
            # Buffer full — drain to overflow
            overflow_parts.append(self._k[:self._pos].clone())
            self._pos = 0
            space = self.capacity
        n = min(remaining, space)
        self._k[self._pos:self._pos+n] = key[offset:offset+n]
        self._pos += n
```

When the ring overflows, the OLDEST tokens (which have lowest attention weight on average) are moved to compressed storage. The NEWEST tokens (highest attention weight, most likely to be "attended to") stay in exact precision.

**For our CUDA**: This is the inverse of our current approach where ALL tokens go through turbo quantization. With a ring buffer, only tokens older than `ring_capacity` are compressed. The decode hot path (token generation) just writes to the ring buffer — a simple memcpy, no FWHT, no centroid lookup, no norm computation.

### FINDING 6: Separate Per-Layer Seed for Rotation Matrix (store.py)

```python
self.quantizer = TurboQuantProd(
    dim=head_dim, bits=key_bits, device=self.device,
    seed=42 + layer_idx * 7,  # DIFFERENT seed per layer
)
```

Each layer gets a different rotation matrix Pi and QJL matrix S. This is important: if all layers share the same rotation, the quantization errors are correlated across layers, which can compound. Different rotations make errors independent.

**For our CUDA**: We use a single shared rotation matrix (`turbo_rotation` / `turbo_rotation_inv` allocated once in `llama-kv-cache.cpp`). The FWHT signs (`d_turbo_wht_signs1/2`) are also global. We should investigate whether per-layer rotations improve PPL. On CUDA, this would mean different sign arrays per layer — stored in constant memory indexed by layer_id.

### FINDING 7: The qjl_scale Constant (quantizer.py)

```python
self.qjl_scale = math.sqrt(math.pi / 2.0) / dim
```

For d=128: `qjl_scale = sqrt(pi/2) / 128 = 1.2533141 / 128 = 0.009791517`

Our turbo4 uses: `1.2533141f / 128.0f` — same constant. But theirs is applied to the SCORE level (`scores_qjl * qjl_scale * residual_norms`), while ours is applied per-element in the dequant (`(centroid + sign * qjl_scale_unit) * norm`). The mathematical result is different because of the norm interaction.

### FINDING 8: The MIN_HISTORY_FOR_TQ Threshold (score.py)

```python
MIN_HISTORY_FOR_TQ = 16
```

They only invoke the compressed attention path when there are at least 16 compressed tokens. Below that, the overhead of the TQ score kernel isn't worth it.

**For our CUDA**: We don't have a minimum — we use the shadow path for even 1 token. A similar threshold could help: below 16 KV positions, just use the native turbo vec_dot (or even skip the shadow entirely and use the ring buffer approach).

---

## Summary

0xSero's turboquant is a clean Python/Triton implementation targeting vLLM. The ALGORITHMIC insights that directly help us:

1. **Fused compressed attention** (score from packed data, no dequant) — the architecture for our lop3 moonshot kernel
2. **Separate K/V quantization strategies** — validates our asymmetric approach, suggests group-quant for values
3. **Ring buffer for recent tokens** — eliminates shadow overhead at short context (the biggest win for our 0.944x gap)
4. **Online softmax over compressed KV** — the fused kernel design we should target
5. **QJL as additive score, not fused dequant** — explains why our turbo4 QJL correction is weak and how to fix it
6. **Lazy flatten (chunked store)** — defer concatenation until read time, skip work if cache cleared
7. **Per-layer rotation seeds** — uncorrelate quantization errors across layers
8. **Min-history threshold** — skip TQ overhead for very short context
9. **Ring buffer overflow → compressed store** — newest tokens stay exact, oldest compress
10. **GQA broadcast** — avoid repeat_interleave, use unsqueeze + broadcast

The **single most impactful idea** for closing our 0.944x short-context gap: the **ring buffer**. If the last 128 tokens are kept in exact fp16 with zero turbo overhead, short-context decode has no penalty. Combined with our existing shadow cache for older positions, this could push short-context to ~0.98-0.99x while keeping the 1.04x advantage at 32K+.
