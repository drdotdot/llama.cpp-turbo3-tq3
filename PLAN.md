# Plan

## Goal
Keep this repo narrowly focused on one integration path:

- **TQ3 / TQ3_1S weight loading**
- **Turbo3 KV-cache compression**
- **one practical CUDA runtime that can use both together**

## Near-term priorities

1. Keep the combined path buildable and understandable
2. Preserve only project-specific docs in the repo
3. Separate verified claims from aspirational work
4. Avoid shipping local agent state, personal workflow files, or unrelated scratch notes

## Documentation rule
If a file does not help a new reader:
- understand what this repo does
- build it
- run it
- verify the core claim

then it probably should not be in the top-level project docs.
