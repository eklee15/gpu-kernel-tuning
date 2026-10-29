# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup

Requires an NVIDIA GPU with CUDA support. Install PyTorch for your CUDA version first (not in requirements.txt):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124  # adjust for your CUDA version
pip install -r requirements.txt
```

## Running examples

```bash
python ex1_fusion.py   # eager vs torch.compile (operator fusion)
python ex2_tiling.py   # bad tile vs good tile vs FP16 (Triton matmul)
```

Each script prints `key_averages()` sorted by CUDA time, asserts correctness with `torch.testing.assert_close`, prints `torch.utils.benchmark` timings, and writes Chrome traces to `./prof/<run-name>/`.

## Architecture

**`prof_utils.py`** is the shared harness. `run_profile(fn, name)` wraps any zero-argument callable with `torch.profiler` using `schedule(wait=1, warmup=2, active=3)` — the wait/warmup steps intentionally exclude compilation, cuDNN autotuning, and allocator growth from the recorded trace. It writes traces to `./prof/<name>/` for Perfetto/`chrome://tracing`.

**`ex1_fusion.py`** and **`ex2_tiling.py`** each follow the same four-step pattern: capture a profile, read the bottleneck from `key_averages()`, apply one technique, verify with `assert_close` and re-benchmark. They import only from `prof_utils` — there are no other shared modules.

## Viewing traces

Open `./prof/<run-name>/*.pt.trace.json` at https://ui.perfetto.dev or `chrome://tracing`. The `./prof/` directory is git-ignored.
