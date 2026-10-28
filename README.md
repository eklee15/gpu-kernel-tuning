# GPU Kernel Tuning with PyTorch

Companion code for the **GPU Kernel Tuning** slide deck (`slides/GPU_Kernel_Tuning_NYU.pptx`).
It contains a reusable `torch.profiler` harness and two reproducible examples that walk through
the four-step tuning workflow:

1. **Setup & capture**: profile a few warmed-up steps with `torch.profiler`
2. **Identify the bottleneck**: read `key_averages()` and the trace timeline
3. **Apply a technique**: fusion, memory layout / allocation / tiling, Tensor Cores
4. **Verify**: `torch.testing.assert_close`, re-profile, and benchmark

## Repository layout

```
.
├── prof_utils.py      # shared profiling harness: run_profile(fn, name)
├── ex1_fusion.py      # Example 1: operator / kernel fusion with torch.compile
├── ex2_layout.py      # Example 2: channels_last layout, convert-once allocation, cuDNN autotuning, FP16
├── requirements.txt
└── slides/
    └── GPU_Kernel_Tuning_NYU.pptx
```

## Requirements

- An NVIDIA GPU with a working CUDA driver
- Python 3.9+
- PyTorch 2.x with CUDA support (install the build for your CUDA version from https://pytorch.org/get-started/locally/)
- A Triton-capable setup for `torch.compile` (bundled with Linux CUDA wheels of PyTorch)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Run the examples

```bash
python ex1_fusion.py                 # eager vs torch.compile
python ex2_layout.py                 # NCHW FP32 -> NHWC FP32 -> NHWC FP16
python ex2_layout.py --anti-pattern  # also profile converting the input inside the loop
```

Each run prints a `key_averages()` table sorted by CUDA time, checks correctness with
`assert_close`, prints `torch.utils.benchmark` timings, and writes a trace to `./prof/<run-name>/`.

## View the traces

Open any `./prof/<run-name>/*.pt.trace.json` file at https://ui.perfetto.dev (or `chrome://tracing`).

## What to look for

| Example | Profiler signal | Before | After |
|---|---|---|---|
| 1 Fusion | CUDA kernels per step | one `aten` elementwise kernel per op (~9) | one `triton_poi_fused_*` kernel |
| 1 Fusion | HBM traffic | read + write the full 128 MiB tensor per op | one read, one write |
| 2 Layout | Layout transform / `aten::copy_` kernels | may appear in the NCHW run | none in channels_last runs |
| 2 Allocation | Self CUDA Mem per step | grows with per-step conversion (`--anti-pattern`) | flat |
| 2 Tiling | conv kernel names and CUDA time | default algorithm | cuDNN-autotuned algorithm / tile shape |
| 2 Precision | conv kernels in `ex2_nhwc_fp16` | FP32 / TF32 kernels | Tensor-Core (MMA) kernels |

Exact kernel names and timings depend on your GPU, driver, cuDNN and PyTorch versions.

## Notes

- `run_profile` uses `schedule(wait=1, warmup=2, active=3)` so compilation, cuDNN autotuning and
  allocator growth fall outside the recorded steps.
- On newer PyTorch releases `sort_by="device_time_total"` is equivalent to `cuda_time_total`.
- `./prof/` is git-ignored; traces can be large.
