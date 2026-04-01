# llama.cpp-turbo3-tq3

> **TQ3 weights + Turbo3 KV cache in one runtime.**

This fork exists for one reason:

**run a TQ3-family model and keep the KV cache compressed with `turbo3` in the same `llama.cpp` server.**

That means stuff like:
- `Qwen3.5-27B-TQ3_1S.gguf`
- `-ctk turbo3`
- `-ctv turbo3`
- CUDA actually working instead of the repo just sounding ambitious

## Why this repo is interesting

A lot of the work around this has been split up:

- one place has the **TQ3 / TQ3_1S weight-format work**
- another place has the **Turbo3 KV-cache path**
- then there are random local patches, half-merges, and branch piles

This repo is the combined path.

Not just weight-format support.
Not just KV-cache compression.
**Both, in one practical runtime.**

## What is actually verified

Verified on a live Hermes/Qwen run:

- model file type loaded as **`TQ3_1S`**
- **`tq3_1s` tensors** recognized during model load
- KV cache running as **`turbo3` for both K and V**
- full tested model offloaded to GPU in the verified run
- large prompt handling worked while Turbo3 KV stayed active

Representative startup facts from the verified run:

- **file type:** `TQ3_1S` (~4 bpw)
- **model size:** ~12.91 GiB
- **GPU offload:** 65/65 layers
- **KV cache:** Turbo3 K + Turbo3 V
- **context:** 98,304 tokens

More detail: [`docs/VERIFICATION.md`](docs/VERIFICATION.md)

## Terminology, kept honest

The short version is catchy:

> **TurboQuant for both weights and KV cache**

The more exact version is:

- **weights:** TQ3 / TQ3_1S model-weight formats
- **cache:** Turbo3 compression for K/V cache

Under the hood, those names point to actual mathematical ideas rather than just branding:

- **TQ3 / TQ3_1S weights**: built around **Walsh-Hadamard rotation** plus learned / codebook-style low-bit quantization paths (the codebase also labels TQ3 as using a **Lloyd-Max codebook** path)
- **Turbo3 KV cache**: built around **PolarQuant-style low-bit KV compression** with a **QJL (Johnson-Lindenstrauss) residual bit** in the `turbo3` path

So the headline can be loud.
The body should stay precise.
That is what this repo is trying to do.

## What is in this fork

### TQ3 / TQ3_1S weight path
This repo contains in-tree type definitions and weight-path support for TQ3-family formats, including:

- `TQ3_0`
- `TQ3_1S`

### Turbo3 KV-cache path
This repo supports Turbo3 cache compression through:

- `-ctk turbo3`
- `-ctv turbo3`

### The thing that actually matters

- load a **TQ3-family GGUF**
- keep **K and V cache compressed with Turbo3**
- run both together in one usable `llama.cpp` flow

That is the whole pitch.

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

If you understand why that command is cool, you already understand this repo.

## Repo focus

This is a practical engineering fork.
Not a paper mirror.
Not a vague “future of compression” repo.

Focus:

- make the **TQ3 weight path + Turbo3 KV path** work together
- keep the CUDA path real
- separate verified claims from hopeful ones
- make the repo understandable enough that someone else can reproduce it

## Read next

- [`docs/VERIFICATION.md`](docs/VERIFICATION.md) — what was actually verified
- [`docs/REPO_SCOPE.md`](docs/REPO_SCOPE.md) — what this repo is and is not
- [`PLAN.md`](PLAN.md) — implementation notes and references

## Related work

This is **our own standalone repo**, not just a mirror.

A key reference on the TQ3 side is:
- **turbo-tan** — <https://github.com/turbo-tan/llama.cpp-tq3>

If you mainly care about the TQ3 weight-format side, start there.
If you want **TQ3-family weights plus Turbo3 KV cache in one runtime**, that is what this repo is for.

## Credits

This repo stands on work from the broader `llama.cpp` and TurboQuant-adjacent ecosystem.

Special credit to:

- **turbo-tan** — TQ3 reference work: <https://github.com/turbo-tan/llama.cpp-tq3>
- the wider `llama.cpp` community for the base runtime and tooling
- prior CUDA and cache-compression work that made this integration path possible

## License

MIT, matching upstream `llama.cpp` unless specific files say otherwise.
