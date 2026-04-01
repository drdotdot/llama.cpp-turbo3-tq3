# Verification

This document records the core claim of this repo that has been verified in practice.

## Core Claim

**TurboQuant is active for both model weights and KV cache in the same runtime.**

That means:
- the model weights are loaded from a **TQ3/TQ3_1S GGUF**
- the KV cache is configured with **`-ctk turbo3 -ctv turbo3`**
- the system runs through the CUDA path in one live server process

## Verified Observations

From the verified live Hermes/Qwen run:
- model file type detected as **`TQ3_1S`**
- model size reported around **12.91 GiB**
- **497 `tq3_1s` tensors** recognized during model load
- **65/65 layers** offloaded to GPU on the tested configuration
- KV cache initialized with **turbo3 K** and **turbo3 V**
- turbo rotation matrices initialized for KV cache path
- context configured at **98,304 tokens**

## What This Verifies

It verifies the important product-level statement:

> This repo can run **TurboQuant weights + TurboQuant KV cache together**.

Not one or the other. Both.

## What This Document Does Not Claim

This document does **not** claim:
- full benchmark coverage across GPUs
- full quality benchmarking versus every other quant format
- final upstream readiness
- complete implementation of every possible TQ variant

Those need separate benchmark and compatibility work.

## Recommended Front-Page Message

If someone asks what this repo is for, the short answer is:

**A llama.cpp fork for TurboQuant on both weights and KV cache.**
