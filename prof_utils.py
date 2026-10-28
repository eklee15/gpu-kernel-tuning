"""Shared torch.profiler harness used by both examples.

run_profile(fn, name) profiles `fn` for a few steps, prints the top CUDA ops,
and writes a Chrome trace to ./prof/<name>/ (open it in https://ui.perfetto.dev).
"""
import torch
from torch.profiler import (ProfilerActivity, profile, record_function,
                            schedule, tensorboard_trace_handler)


def run_profile(fn, name, steps=6, row_limit=10):
    """Profile `fn` and return the profiler object.

    Schedule: skip 1 step, warm up 2 (excludes autotuning, compilation and
    allocator growth), then record 3. `steps` must be >= wait + warmup + active.
    """
    acts = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    sched = schedule(wait=1, warmup=2, active=3, repeat=1)
    with profile(activities=acts, schedule=sched,
                 record_shapes=True, profile_memory=True,
                 on_trace_ready=tensorboard_trace_handler(f"./prof/{name}")) as prof:
        for _ in range(steps):
            with record_function(name):
                fn()
            torch.cuda.synchronize()
            prof.step()
    print(f"\n===== {name} =====")
    # On newer PyTorch versions "device_time_total" is also accepted.
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=row_limit))
    return prof


def require_cuda():
    if not torch.cuda.is_available():
        raise SystemExit("These examples need a CUDA GPU (torch.cuda.is_available() is False).")
