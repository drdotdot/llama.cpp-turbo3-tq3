# llama.cpp-turbo3-tq3

`llama.cpp-turbo3-tq3` is a `llama.cpp`-based fork for one specific runtime combination:

- **TQ3 / TQ3_1S model weights**
- **Turbo3 KV-cache compression**
- **one CUDA runtime that can use both together**

The practical target is straightforward:

- load a model like **`Qwen3.5-27B-TQ3_1S.gguf`**
- run with **`-ctk turbo3 -ctv turbo3`**
- keep the full path working in one real server process

## Headline result

This repo is about a simple claim:

> **TurboQuant on both sides: weights and KV cache.**

Not just TQ3/TQ3_1S model loading.
Not just Turbo3 cache compression.
**Both in the same runtime.**

## What that means technically

On the **weights** side, this repo includes support for **TQ3 / TQ3_1S GGUF tensor formats**.

On the **cache** side, this repo includes support for **Turbo3 compression for both K and V cache**.

For readers who want the technical meaning of those labels:

- **TQ3 / TQ3_1S weights** use Walsh-Hadamard rotation, centroid/codebook quantization, and a dual-scale block structure
- **Turbo3 KV cache** uses a PolarQuant-style low-bit KV compression path with a QJL-style residual bit

In other words, the names point to specific compression methods, not just marketing labels.

## Verified result

Verified on a live Hermes/Qwen run:

- model file type detected as **`TQ3_1S`**
- **`tq3_1s` tensors** recognized during load
- KV cache initialized as **`turbo3` for both K and V**
- **65/65 layers** offloaded to GPU in the verified setup
- long-context server run completed with Turbo3 KV active

Representative startup facts from the verified run:

- **file type:** `TQ3_1S`
- **model size:** ~12.91 GiB
- **GPU offload:** 65/65 layers
- **KV cache:** Turbo3 K + Turbo3 V
- **context:** 98,304 tokens

More detail: [`docs/VERIFICATION.md`](docs/VERIFICATION.md)

## Important caveat

This repo does **not** claim that every TQ3/TQ3_1S setup is universally better than every standard quant.

The practical value here is mainly:

- getting a **TQ3-family model runtime** working in `llama.cpp`
- combining that with **Turbo3 KV compression**
- making the combined path usable on real GPU hardware

So this is primarily an **integration and deployment** repo, not a blanket claim about every benchmark outcome.

## What is in this fork

### TQ3 / TQ3_1S weight support
This repo contains in-tree type definitions and weight-path support for TQ3-family formats, including:

- `TQ3_0`
- `TQ3_1S`

### Turbo3 KV-cache support
This repo supports Turbo3 cache compression through:

- `-ctk turbo3`
- `-ctv turbo3`

### Combined runtime path
This is the whole point of the fork:

- load a **TQ3-family GGUF**
- keep **K and V cache compressed with Turbo3**
- run both together in one usable `llama.cpp` server flow

## Quick start

### Build

```bash
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build -j$(nproc)
```

Adjust `CMAKE_CUDA_ARCHITECTURES` for your GPU.

### Run

```bash
./build/bin/llama-server \
  -m /path/to/Qwen3.5-27B-TQ3_1S.gguf \
  -ngl 99 \
  -ctk turbo3 \
  -ctv turbo3 \
  -c 98304
```

## Repo focus

This is a practical engineering fork.

Focus:

- make the **TQ3 weight path + Turbo3 KV path** work together
- keep the CUDA runtime path real
- separate verified behavior from unverified claims
- keep the repo understandable enough that someone else can reproduce it

## Read next

- [`docs/VERIFICATION.md`](docs/VERIFICATION.md) — what has been verified
- [`docs/REPO_SCOPE.md`](docs/REPO_SCOPE.md) — what this repo is and is not
- [`PLAN.md`](PLAN.md) — short project plan

## Related work

This is **our own standalone repo**, not just a mirror.

A key reference on the TQ3 side is:
- **turbo-tan** — <https://github.com/turbo-tan/llama.cpp-tq3>

If you mainly care about the TQ3 weight-format side, start there.
If you want **TQ3-family weights plus Turbo3 KV cache in one runtime**, that is what this repo is for.

## Credits

This repo builds on work from the broader `llama.cpp` and TurboQuant-adjacent ecosystem.

Special credit to:

- **turbo-tan** — TQ3 reference work: <https://github.com/turbo-tan/llama.cpp-tq3>
- the wider `llama.cpp` community for the base runtime and tooling
- prior CUDA and cache-compression work that helped shape the combined path

## License

MIT, matching upstream `llama.cpp` unless specific files say otherwise.
