# llama.cpp-turbo3-tq3

> **TurboQuant for both weights and KV cache** — with the explanation below kept precise.

This is a standalone `llama.cpp` fork focused on one practical goal:

- **TQ3 / TQ3_1S model-weight support**
- **Turbo3 KV-cache compression**
- **one runtime that can use both together**

In other words: this repo is about running a **TQ3-family model** such as `Qwen3.5-27B-TQ3_1S.gguf` while also using **`turbo3`-compressed K/V cache** in the same CUDA-backed server process.

## What the headline means

The short headline is **“TurboQuant for both weights and KV cache.”**

The more exact version is:

- on the **weights** side, this repo contains support for **TQ3/TQ3_1S GGUF tensor formats**
- on the **cache** side, this repo contains support for **Turbo3 K/V cache compression**
- the main value is that these two paths work **together in one fork**

So yes: using “TurboQuant” in the headline is reasonable shorthand.
But in the body, it is more accurate to talk specifically about:
- **TQ3 / TQ3_1S model weights**
- **Turbo3 KV cache**

That is the terminology this README uses below.

## Why this repo exists

A lot of the interesting work in this area has been split across separate branches and experiments:

- one path focused on **TQ3-family weight formats**
- another path focused on **TurboQuant-style KV-cache compression**
- separate local branches, patches, and test harnesses for CUDA bring-up

This repo exists to make the combined path easier to understand, verify, and use.

## The core claim

This repo is **not only** about TQ3/TQ3_1S model weights.
And it is **not only** about Turbo3 KV cache.

The point is the combination:

- **TQ3/TQ3_1S model loading**
- **Turbo3 cache for both K and V**
- **same runtime / same server / same GPU execution path**

That is the real landing-page message.

## What is currently verified

Verified on the live Hermes/Qwen setup:

- **model file type loaded:** `TQ3_1S`
- **weight tensors recognized:** `tq3_1s` tensors detected during load
- **KV cache mode:** `turbo3` for both K and V
- **CUDA offload:** full tested model offloaded to GPU in the verified run
- **long context:** the server handled very large prompts while Turbo3 KV remained active

Representative startup characteristics from the verified run:

- **file type:** `TQ3_1S` (turbo two-scale, ~4 bpw)
- **model size:** ~12.91 GiB
- **GPU offload:** 65/65 layers
- **KV cache allocation:** Turbo3 K + Turbo3 V
- **context size:** 98,304 tokens

For the fuller verification notes, see [`docs/VERIFICATION.md`](docs/VERIFICATION.md).

## What you get in this fork

### 1) TQ3 / TQ3_1S weight-path support
This repo contains codepaths for TQ3-family weight formats, including in-tree type definitions and quant/dequant support for:

- `TQ3_0`
- `TQ3_1S`

### 2) Turbo3 KV-cache support
This repo supports Turbo3 cache compression through the standard cache flags:

- `-ctk turbo3`
- `-ctv turbo3`

### 3) The combined runtime path
This is the part that matters most:

- load a **TQ3-family GGUF model**
- keep **K and V cache compressed with Turbo3**
- run both together in a practical `llama.cpp` server workflow

## Quick start

### Build

```bash
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build -j$(nproc)
```

Adjust `CMAKE_CUDA_ARCHITECTURES` for your GPU.

### Run a TQ3 model with Turbo3 KV cache

```bash
./build/bin/llama-server \
  -m /path/to/Qwen3.5-27B-TQ3_1S.gguf \
  -ngl 99 \
  -ctk turbo3 \
  -ctv turbo3 \
  -c 98304
```

That command is the entire point of the repo.

## Repo focus

This is a practical engineering fork, not a paper mirror and not a general-purpose “everything TurboQuant” umbrella.

Primary focus:

- make the **TQ3 weight path + Turbo3 KV path** work together cleanly
- keep the CUDA path real and usable
- document what is verified versus what is still incomplete
- keep the repo understandable for someone trying to reproduce the setup

## Status and scope

See:

- [`docs/VERIFICATION.md`](docs/VERIFICATION.md) — what has been verified live
- [`docs/REPO_SCOPE.md`](docs/REPO_SCOPE.md) — what this repo is for
- [`PLAN.md`](PLAN.md) — implementation notes and references

## Relationship to other work

This repo is **our own standalone repo**, not a fork mirror.

Important reference work includes:

- turbo-tan’s TQ3-focused fork: <https://github.com/turbo-tan/llama.cpp-tq3>

This project stands on top of that broader ecosystem and tries to make one specific integration path usable in practice.

## Credits

This repo builds on work from the wider `llama.cpp` and TurboQuant-adjacent ecosystem.

Special credit to:

- **turbo-tan** for the TQ3-focused reference work: <https://github.com/turbo-tan/llama.cpp-tq3>
- the broader `llama.cpp` community for the base runtime and tooling
- prior CUDA / cache-compression experiments that helped shape the combined path

If you are here because of the weight-format side, start by looking at turbo-tan’s repo.
If you are here because you want **TQ3-family weights plus Turbo3 KV cache in one runtime**, that is the gap this repo is trying to fill.

## License

MIT, matching upstream `llama.cpp` unless specific files note otherwise.
