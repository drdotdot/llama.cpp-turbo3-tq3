# Repo Scope

## What this repo is

A standalone `llama.cpp`-based repo for integrating:
- **TQ3/TQ3_1S TurboQuant model weights**
- **Turbo3 KV cache compression**
- a practical **CUDA runtime path** that supports both together

## What makes it distinct

The repo is centered on one integration point:

> **TurboQuant for both weights and KV cache.**

That is the identity of the project.

## What this repo is not

This repo is not just:
- a mirror of another TQ3 fork
- a KV-cache-only TurboQuant fork
- a benchmark scrapbook with no runnable integration

## Immediate goals

1. Keep the combined TQ3 + turbo3 path understandable
2. Make the README communicate the real value immediately
3. Preserve a clean standalone project identity
4. Document verified behavior separately from aspirational claims

## Naming

Canonical repo name:
- `llama.cpp-turbo3-tq3`

Reason:
- `llama.cpp` base
- `turbo3` for KV cache path
- `tq3` for TurboQuant weight path

That name makes the combined purpose obvious.
