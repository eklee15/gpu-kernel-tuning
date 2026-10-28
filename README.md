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
├── ex2_tiling.py      # Example 2: tile size impact and FP16 Tensor Core benefit (Triton matmul)
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
python ex1_fusion.py   # eager vs torch.compile
python ex2_tiling.py   # bad tile vs good tile vs FP16 (Triton matmul)
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
| 2 Tiling | `matmul_kernel` CUDA time | ~50 ms (16×16×16 tiles) | ~5 ms (128×128×64 tiles) |
| 2 Precision | `matmul_kernel` CUDA time | ~5 ms FP32 | ~2 ms FP16 (Tensor Cores) |

Exact kernel names and timings depend on your GPU, driver, cuDNN and PyTorch versions.

## Notes

- `run_profile` uses `schedule(wait=1, warmup=2, active=3)` so compilation, cuDNN autotuning and
  allocator growth fall outside the recorded steps.
- On newer PyTorch releases `sort_by="device_time_total"` is equivalent to `cuda_time_total`.
- `./prof/` is git-ignored; traces can be large.
