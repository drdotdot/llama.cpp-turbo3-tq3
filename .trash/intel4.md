# Intelligence Report: RecursiveIntell/turbo-quant

**Repo**: https://github.com/RecursiveIntell/turbo-quant
**Author**: RecursiveIntell
**Language**: Rust (no GPU kernels — pure CPU)
**Last updated**: 2026-03-27
**Commits**: 1 (initial commit, complete library)
**Total code**: ~1,800 lines across 6 modules + 2 integration test files
**License**: MIT

---

## What This Is

A **pure Rust library** implementing TurboQuant, PolarQuant, and QJL as standalone vector compression algorithms. Targets semantic search / vector databases AND KV cache compression. No GPU kernels, no inference engine integration. Clean reference implementation with excellent documentation and test coverage.

**Critical difference from ALL other repos**: Uses **POLAR COORDINATE ENCODING** (radius + angle quantization), NOT the Lloyd-Max codebook approach that everyone else uses. This is actually closer to the paper's original Algorithm 1 (PolarQuant) than the simplified centroid-based approach that 0xSero, yzamari, TheTom, spiritbuun, and we all use.

---

## Architecture

```
TurboQuantizer (turbo.rs)
  ├── PolarQuantizer (polar.rs)    — Angle quantization of coordinate pairs
  │     └── StoredRotation (rotation.rs)  — QR-based random orthogonal matrix
  └── QjlQuantizer (qjl.rs)       — 1-bit sign sketch of residual
KvCacheCompressor (kv.rs)          — Online KV cache using TurboQuantizer
```

---

## THE BIG ALGORITHMIC DIFFERENCE: Polar Coordinate Encoding

### What everyone else does (Lloyd-Max codebook):
```
rotate → each coordinate → find nearest centroid from fixed codebook → store index
```

### What RecursiveIntell does (Polar pairs):
```
rotate → group coordinates into PAIRS (a,b) → convert to polar (r, θ) →
   store r losslessly as f32 → quantize θ uniformly on [-π, π] → store index
```

**The pair-based approach**:
1. Take rotated coordinates (y₀, y₁), (y₂, y₃), ..., (y_{d-2}, y_{d-1})
2. For each pair: `r = sqrt(a² + b²)`, `θ = atan2(b, a)`
3. **Radii stored LOSSLESSLY** as f32 (4 bytes per pair = 4 bytes per 2 values)
4. **Angles uniformly quantized** to b bits on [-π, π]

**Storage per 2 values**: 4 bytes (radius) + ceil(b/8) bytes (angle) = ~4.5 bytes at 4 bits = 18 bpv

**Compare with our turbo3**: 16 bytes per 32 values = 4 bpv (much more compressed)

**Why polar pairs are interesting**: The radius captures the MAGNITUDE of each pair EXACTLY (lossless f32). Only the DIRECTION (angle) is quantized. This means:
- Outlier magnitudes are perfectly preserved
- Quantization error is purely directional
- Inner product estimation has known, bounded error

**Why we DON'T use this**: The compression ratio is much worse. PolarQuant at 4 bits gives ~1.28x compression (vs our 4.6x with turbo3). The radii dominate storage. The paper itself notes that PolarQuant is for inner-product estimation quality, not for maximum compression. The centroid approach (which we use) gives better compression at the cost of some reconstruction fidelity.

---

## QJL Implementation — The Cleanest Reference

Their `QjlQuantizer` is the most readable QJL implementation:

```rust
// Project vector onto random hyperplanes, keep only signs
pub fn sketch(&self, vector: &[f32]) -> QjlSketch {
    let G = self.projection_matrix();  // m×d random Gaussian
    let signs = G.iter().map(|row| {
        let dot: f32 = row.iter().zip(vector).map(|(g, x)| g * x).sum();
        if dot >= 0.0 { 1i8 } else { -1i8 }
    }).collect();
    QjlSketch { signs, ... }
}

// Unbiased inner product estimator
pub fn inner_product_estimate(&self, sketch: &QjlSketch, query: &[f32]) -> f32 {
    let G = self.projection_matrix();
    let scale = PI / (2.0 * m);
    scale * G.iter().zip(sketch.signs.iter())
        .map(|(row, &sign)| {
            let g_dot_query: f32 = row.iter().zip(query).map(|(g, q)| g * q).sum();
            sign as f32 * g_dot_query
        }).sum()
}
```

**Key mathematical detail**: The scale factor is `π / (2m)`, NOT `sqrt(π/2) / d` that 0xSero uses. These are different formulas!
- RecursiveIntell: `(π / 2m) * Σᵢ sign(gᵢ·x) * (gᵢ·y)` where m = number of projections
- 0xSero: `sqrt(π/2)/d * ||r|| * Σᵢ sign(gᵢ·r) * (gᵢ·y)` where d = dimension, r = residual

The difference: RecursiveIntell's QJL operates on the RAW vector, scaling by π/(2m). 0xSero's operates on the RESIDUAL (x - mse_reconstruction), scaling by sqrt(π/2)/d * ||residual||. Both are mathematically valid but the scaling conventions differ.

**For our turbo4**: We use `1.2533141f / 128.0f * rnorm` which is `sqrt(π/2) / d * ||residual||` — matching 0xSero's convention. The RecursiveIntell convention is simpler but doesn't account for the residual norm separately.

---

## TurboQuant Two-Stage — Clear Residual Computation

```rust
pub fn encode(&self, vector: &[f32]) -> TurboCode {
    // Stage 1: PolarQuant at (bits-1) bits
    let polar_code = self.polar.encode(vector)?;

    // Stage 2: Compute residual and sketch
    let reconstruction = self.polar.decode(&polar_code)?;  // FULL DECODE back to original space
    let residual: Vec<f32> = vector.iter()
        .zip(reconstruction.iter())
        .map(|(orig, rec)| orig - rec)
        .collect();

    let residual_sketch = self.qjl.sketch(&residual)?;
    TurboCode { polar_code, residual_sketch }
}
```

**Confirms 0xSero + yzamari**: The residual is computed in ORIGINAL space (`vector - decode(encode(vector))`). The decode step applies the INVERSE rotation. This is the third independent implementation confirming our turbo4 computes the residual in the wrong space.

**Inner product estimation**:
```rust
pub fn inner_product_estimate(&self, code: &TurboCode, query: &[f32]) -> f32 {
    let polar_estimate = self.polar.inner_product_estimate(&code.polar_code, query)?;
    let qjl_correction = self.qjl.inner_product_estimate(&code.residual_sketch, query)?;
    polar_estimate + qjl_correction  // ADDITIVE score, not fused
}
```

Again confirms: MSE score + QJL score are separate additive terms.

---

## KV Cache — Values Also Use TurboQuant

Unlike 0xSero/yzamari (which use group quantization for values), RecursiveIntell compresses BOTH keys AND values with the full TurboQuant pipeline:

```rust
pub fn compress_token(&mut self, key: &[f32], value: &[f32]) -> Result<()> {
    let compressed_key = self.key_quantizer.encode(key)?;
    let compressed_value = self.value_quantizer.encode(value)?;  // SAME algorithm
    self.tokens.push(CompressedToken { compressed_key, compressed_value });
}
```

BUT with independent seeds:
```rust
let value_quantizer = TurboQuantizer::new(
    head_dim, bits, projections,
    seed.wrapping_add(0x1234_5678_ABCD_EF00),  // different seed
);
```

**Key insight**: K and V use DIFFERENT rotation matrices. This is important — if they shared the same rotation, the quantization errors would be correlated between K and V, potentially amplifying attention errors.

**For our implementation**: We use a SINGLE rotation matrix (`turbo_rotation` / `turbo_rotation_inv`) shared across all layers and both K and V. RecursiveIntell + 0xSero both use independent rotations for K vs V. This could be a source of quality loss for us.

---

## Attend Logic — Values Decoded at Attention Time

```rust
pub fn attend(&self, query: &[f32]) -> Vec<f32> {
    let scores = self.attention_scores(query)?;

    // Numerically stable softmax
    let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = scores.iter().map(|s| (s - max_score).exp()).collect();
    let sum_exp: f32 = exps.iter().sum();
    let weights = exps.iter().map(|e| e / sum_exp).collect();

    // Weighted sum of DECODED values
    let mut output = vec![0.0f32; head_dim];
    for (token, &weight) in self.tokens.iter().zip(weights.iter()) {
        let decoded = self.value_quantizer.decode_approximate(&token.compressed_value)?;
        for (out, val) in output.iter_mut().zip(decoded.iter()) {
            *out += weight * val;
        }
    }
    output
}
```

**Important**: Values are DECODED (approximate reconstruction) one at a time, weighted by softmax. This is O(n × d) where n = sequence length. No sparse V skip — every position is decoded.

**Comment in code is honest about this**:
> "Values are decoded (approximate reconstruction) rather than stored in compressed form, since weighted sums cannot be computed in compressed space."

This is correct — you can't compute `Σ w_i * compressed(v_i)` without decompressing each v_i. Our sparse V skip is the optimization for this: skip decompression for positions where w_i < threshold.

---

## What's Algorithmically Novel (compared to other repos)

### 1. Polar Pair Encoding (unique to this repo)

Nobody else uses the actual PolarQuant algorithm from the paper. Everyone else (including us) uses Lloyd-Max codebook centroids. PolarQuant stores lossless radii + quantized angles.

**When polar is better**: For VECTOR SEARCH (not KV cache) where you need accurate nearest-neighbor ranking and can afford the storage. The lossless radii mean magnitude outliers are perfectly preserved.

**When codebook is better**: For KV cache compression where the goal is maximum memory reduction. The codebook approach packs 32 values into 16 bytes (turbo3) vs ~96 bytes for polar at 4 bits.

### 2. Separate Seeds for K vs V Quantizers

```rust
let key_quantizer = TurboQuantizer::new(head_dim, bits, projections, seed)?;
let value_quantizer = TurboQuantizer::new(head_dim, bits, projections,
    seed.wrapping_add(0x1234_5678_ABCD_EF00))?;  // independent rotation
```

This is the third repo to use independent K/V rotations (0xSero uses `seed + 1000`, yzamari inherits from 0xSero). We use a single shared rotation.

**Actionable**: Test whether per-K/V rotation improves PPL. On CUDA, this means allocating two rotation matrices per layer and using the appropriate one in the graph-level WHT rotation.

### 3. ChaCha8 RNG for Determinism

Uses `rand_chacha::ChaCha8Rng` seeded from a u64 for all random matrix generation. This guarantees cross-platform determinism — the same seed always produces the same rotation matrix regardless of hardware or OS.

**For our implementation**: Our FWHT sign arrays are hardcoded constants (`d_turbo_wht_signs1/2`). We don't regenerate them from a seed. This means we can't have per-layer or per-K/V sign arrays without hardcoding more constants. A seed-based approach would be more flexible.

### 4. Configurable QJL Projections (not always d)

Their QJL uses `m = projections` where m can be less than d:
```rust
TurboQuantizer::new(1536, 8, 384, 42)  // 384 projections for d=1536
```

Rule of thumb: `m = d/4` for search, `m = d/8` for KV cache. Fewer projections = smaller sketch but higher variance.

**For our turbo4**: We use m = d = 128 (one sign bit per coordinate). Their flexibility to use m < d reduces storage for the QJL component.

### 5. L2 Distance from Inner Product

```rust
pub fn l2_distance_estimate(&self, code: &TurboCode, query: &[f32]) -> f32 {
    let ip = self.inner_product_estimate(code, query)?;
    let query_norm_sq: f32 = query.iter().map(|x| x * x).sum();
    let code_norm_sq: f32 = code.polar_code.radii.iter().map(|r| r * r).sum();
    (query_norm_sq + code_norm_sq - 2.0 * ip).max(0.0)
}
```

Uses the identity `||x-y||² = ||x||² + ||y||² - 2<x,y>`. The code's norm is exact (from lossless radii). This enables L2 nearest neighbor search from inner-product-optimized codes.

**Not relevant for our KV cache** (attention uses dot products, not L2), but useful for vector search applications of turbo-quant.

---

## What We Can Learn

### LEARNING 1: Three independent implementations confirm our turbo4 is wrong

RecursiveIntell (Rust), 0xSero (Python/Triton), and yzamari (Python/MLX) ALL compute:
```
residual = original_vector - full_decode(quantize(vector))
```

Where `full_decode` includes the INVERSE rotation back to original space. Our turbo4 does:
```
residual = rotated_vector - centroid[index]  // wrong space, no inverse rotation
```

This is now confirmed by 3 independent codebases. The fix is clear and validated.

### LEARNING 2: K and V should use independent rotations

All three external repos use different seeds for K vs V quantizers. We share one rotation. This could be contributing to PPL loss. Easy to test — just allocate two sets of FWHT signs.

### LEARNING 3: Polar encoding is the "correct" Algorithm 1 but impractical for KV cache

The paper's actual Algorithm 1 (PolarQuant) uses polar coordinates, not codebook centroids. The centroid approach (our turbo2/3/4) is a simplification that trades some mathematical elegance for much better compression. RecursiveIntell implements the paper faithfully; everyone else implements the practical variant.

### LEARNING 4: QJL projections can be < d

Using m < d for QJL saves storage. For d=128 with m=32, the QJL sketch is 32 bits (4 bytes) instead of 128 bits (16 bytes). This reduces turbo4's block size from 68 bytes to ~54 bytes for d=128.

### LEARNING 5: The attend() pattern matches 0xSero

Decode values one at a time, weighted by softmax. No sparse V skip. Confirms sparse V is our unique advantage.

---

## What They Have vs What We Have

| Feature | RecursiveIntell | Us (Madreag) |
|---------|----------------|-----|
| Language | Rust (CPU only) | CUDA C++ (GPU) |
| Quantization approach | Polar pairs (paper Algorithm 1) | Lloyd-Max codebook (simplified) |
| Radii storage | Lossless f32 | Not applicable (norm correction instead) |
| QJL projections | Configurable (m ≤ d) | Fixed (m = d) |
| K/V rotation | Independent per quantizer | Shared single rotation |
| Compression ratio | ~1.2-1.3x (polar) | 4.6x (turbo3) |
| KV cache | Yes (CPU, per-token) | Yes (CUDA, FA-integrated) |
| Sparse V skip | No | Yes (our advantage) |
| GPU acceleration | No | Yes (CUDA kernels) |
| Vector search | Yes (L2 + IP) | No (KV cache only) |
| Serialization | serde JSON | N/A (ggml tensor format) |

---

## Summary

RecursiveIntell's turbo-quant is the most THEORETICALLY FAITHFUL implementation — it uses the actual polar coordinate encoding from the paper rather than the simplified codebook approach. However, the compression ratio (~1.2x) makes it impractical for KV cache compression where we need 4x+.

**Transferable ideas**:
1. **turbo4 residual is confirmed wrong by third codebase** — must be computed in original space
2. **Independent K/V rotations** — easy change, potential PPL improvement
3. **Configurable QJL projections** (m < d) — reduces turbo4 block size
4. **ChaCha8 deterministic RNG** — better than hardcoded sign arrays for flexibility
5. **Polar encoding for vector search** — different use case, but if we ever extend turbo-quant beyond KV cache, this is the right approach

The codebase is clean, well-tested (comprehensive integration tests for determinism, accuracy, ordering), and serves as an excellent reference for the paper's math. Best read alongside the paper for understanding the theoretical foundations.
