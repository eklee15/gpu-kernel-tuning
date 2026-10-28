"""Example 2 - Tiling and precision.

Three profiled runs on a large matrix multiply (M=4096, N=4096, K=4096):
  ex2_bad_tile   BLOCK_M/N/K = 16  -- tiny tiles, poor SM occupancy
  ex2_good_tile  BLOCK_M=128, BLOCK_N=128, BLOCK_K=64  -- fills warps, hides latency
  ex2_fp16       same good tile but inputs in FP16, using Tensor Cores

Run:  python ex2_tiling.py
Then compare ./prof/ex2_* in https://ui.perfetto.dev
"""
import torch
import triton
import triton.language as tl
from torch.utils import benchmark

from prof_utils import require_cuda, run_profile

require_cuda()

M, N, K = 4096, 4096, 4096


@triton.jit
def matmul_kernel(
    A, B, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(A + rm[:, None] * stride_am + (rk + k)[None, :] * stride_ak,
                    mask=(rm[:, None] < M) & ((rk + k)[None, :] < K), other=0.0)
        b = tl.load(B + (rk + k)[:, None] * stride_bk + rn[None, :] * stride_bn,
                    mask=((rk + k)[:, None] < K) & (rn[None, :] < N), other=0.0)
        acc += tl.dot(a, b)
    c = acc.to(tl.float32)
    tl.store(C + rm[:, None] * stride_cm + rn[None, :] * stride_cn,
             c, mask=(rm[:, None] < M) & (rn[None, :] < N))


def matmul(a, b, BLOCK_M=128, BLOCK_N=256, BLOCK_K=64):
    assert a.shape[1] == b.shape[0]
    m, k = a.shape
    _, n = b.shape
    c = torch.empty((m, n), device=a.device, dtype=torch.float32)
    grid = (triton.cdiv(m, BLOCK_M), triton.cdiv(n, BLOCK_N))
    matmul_kernel[grid](
        a, b, c, m, n, k,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return c


# ---- inputs ----
a_fp32 = torch.randn(M, K, device="cuda")
b_fp32 = torch.randn(K, N, device="cuda")
a_fp16 = a_fp32.half()
b_fp16 = b_fp32.half()

# ---- warm-up (compile all variants outside the profiler) ----
bad  = matmul(a_fp32, b_fp32, BLOCK_M=16,  BLOCK_N=16,  BLOCK_K=16)
good = matmul(a_fp32, b_fp32, BLOCK_M=128, BLOCK_N=128, BLOCK_K=64)
fp16 = matmul(a_fp16, b_fp16, BLOCK_M=128, BLOCK_N=128, BLOCK_K=64)

# ---- correctness ----
ref = torch.mm(a_fp32, b_fp32)
torch.testing.assert_close(good, ref, rtol=1e-2, atol=0.5)
torch.testing.assert_close(fp16.float(), ref, rtol=1e-1, atol=1.0)
print("assert_close: OK")

# ---- profile ----
run_profile(lambda: matmul(a_fp32, b_fp32, BLOCK_M=16,  BLOCK_N=16,  BLOCK_K=16),  "ex2_bad_tile")
run_profile(lambda: matmul(a_fp32, b_fp32, BLOCK_M=128, BLOCK_N=128, BLOCK_K=64),  "ex2_good_tile")
run_profile(lambda: matmul(a_fp16, b_fp16, BLOCK_M=128, BLOCK_N=128, BLOCK_K=64),  "ex2_fp16")

# ---- benchmark ----
for label, fn in (
    ("bad_tile  (16×16×16)",   lambda: matmul(a_fp32, b_fp32, BLOCK_M=16,  BLOCK_N=16,  BLOCK_K=16)),
    ("good_tile (128×128×64)", lambda: matmul(a_fp32, b_fp32, BLOCK_M=128, BLOCK_N=128, BLOCK_K=64)),
    ("fp16      (128×128×64)", lambda: matmul(a_fp16, b_fp16, BLOCK_M=128, BLOCK_N=128, BLOCK_K=64)),
):
    t = benchmark.Timer("fn()", globals={"fn": fn})
    print(f"{label}: {t.blocked_autorange()}")
