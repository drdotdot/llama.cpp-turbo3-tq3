# llama.cpp + Turbo3/TQ3 — TurboQuant for BOTH Weights and KV Cache

A standalone `llama.cpp` fork focused on one key capability:

> **TurboQuant for both model weights and KV cache in the same runtime.**

This repo combines:
- **TQ3/TQ3_1S weight loading** for TurboQuant-compressed GGUF models
- **Turbo3 KV cache compression** for long-context inference
- **CUDA execution path** for actually running both together on GPU

That means you can run a model like **Qwen3.5-27B-TQ3_1S.gguf** *and* keep the KV cache compressed with `-ctk turbo3 -ctv turbo3` in the same server process.

## Why This Fork Exists

Most TurboQuant work to date has been split across separate branches/forks:
- one path for **TurboQuant-compressed weights**
- another path for **TurboQuant-compressed KV cache**

This repo is about unifying those two pieces into one practical fork.

## The Important Point

**This is not just TurboQuant weights.**
**This is not just TurboQuant KV cache.**

**This repo is TurboQuant for BOTH weights and KV cache.**

That is the headline and the reason to use it.

## Current Verified Capability

Verified on the live Hermes/Qwen setup:
- **Model format loaded:** `TQ3_1S`
- **Weight tensors loaded:** TQ3_1S tensors recognized successfully
- **KV cache mode:** `turbo3` for both K and V
- **CUDA offload:** full GPU layer offload working on tested model
- **Long context:** server handled very large prompts while keeping TurboQuant KV active

Representative verified startup characteristics from the live run:
- file type: **TQ3_1S (turbo two-scale, ~4 bpw)**
- model size: **~12.91 GiB**
- layers offloaded: **65/65 to GPU**
- KV cache allocation: **turbo3 K + turbo3 V**
- context: **98,304 tokens**

## What You Get

### 1) TurboQuant weights
Support for loading TurboQuant-compressed GGUF weights, especially:
- `TQ3_1S`
- related TQ3-family paths being integrated in this codebase

### 2) TurboQuant KV cache
Support for TurboQuant KV cache compression through llama.cpp cache flags:
- `-ctk turbo3`
- `-ctv turbo3`

### 3) The combination
The main value of this repo is that the two features work together:
- **TurboQuant model weights**
- **TurboQuant KV cache**
- **same runtime / same server / same CUDA path**

## Quick Start

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

This is the core configuration this repo exists to enable.

## Repo Focus

This is a practical engineering fork, not a paper mirror.

Primary focus:
- make **TQ3 weights + turbo3 KV** work together cleanly
- keep the CUDA path real and usable
- document what is verified versus what is still in progress

## Status

See:
- [`docs/VERIFICATION.md`](docs/VERIFICATION.md) — what has been verified live
- [`docs/REPO_SCOPE.md`](docs/REPO_SCOPE.md) — what this repo is for
- [`PLAN.md`](PLAN.md) — local implementation goal and source references

## Relationship to Other Repos

Reference repo for the TQ3 side:
- <https://github.com/turbo-tan/llama.cpp-tq3>

This repo is **our own standalone repo**, not a fork of that repository.
It exists to carry the combined Turbo3 + TQ3 work as its own project.

## Credits

Core ideas and prior implementation work come from the broader TurboQuant community, including:
- TurboQuant / TQ research and implementations
- llama.cpp-based TurboQuant forks
- CUDA and KV cache compression work from prior branches

This repo’s specific goal is the **combined weights + KV integration path**.

## License

MIT, matching upstream llama.cpp unless specific files note otherwise.
