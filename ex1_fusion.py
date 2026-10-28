"""Example 1 - Operator / kernel fusion.

Compares an eager bias + GELU (tanh approximation) chain, which launches one
kernel per op, with the same function compiled by torch.compile, which Inductor
fuses into a single Triton kernel.

Run:  python ex1_fusion.py
Then compare ./prof/ex1_eager and ./prof/ex1_fused in https://ui.perfetto.dev
"""
import torch
from torch.utils import benchmark

from prof_utils import require_cuda, run_profile

require_cuda()

x = torch.randn(8192, 4096, device="cuda")  # 128 MiB
b = torch.randn(4096, device="cuda")


def bias_gelu(x, b):  # eager: one kernel per op
    y = x + b
    return 0.5 * y * (1.0 + torch.tanh(
        0.7978845608 * (y + 0.044715 * y ** 3)))


fused = torch.compile(bias_gelu)  # Inductor -> 1 Triton kernel
fused(x, b)                       # compile warm-up, outside the profiler

# Step 1-2: capture and read the profiles
run_profile(lambda: bias_gelu(x, b), "ex1_eager")
run_profile(lambda: fused(x, b), "ex1_fused")

# Step 4: verify correctness, then wall-clock speed
torch.testing.assert_close(fused(x, b), bias_gelu(x, b), rtol=1e-4, atol=1e-4)
print("\nassert_close: OK")
for label, fn in (("eager", bias_gelu), ("fused", fused)):
    t = benchmark.Timer("fn(x, b)", globals={"fn": fn, "x": x, "b": b})
    print(f"{label}: {t.blocked_autorange()}")
