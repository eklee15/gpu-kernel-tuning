"""Example 3 - Tiling and precision: custom CUDA C++ kernel.

Same three-run benchmark as ex2_tiling.py but written in raw CUDA C instead of Triton.
The kernels are compiled as a standalone shared library (no PyTorch ATen headers needed)
and called via ctypes -- showing how CUDA kernels work at the metal.

  ex3_bad_tile   16×16 shared-memory tiles -- low SM occupancy, tiny working set per block
  ex3_good_tile  32×32 shared-memory tiles -- 4× more flops per tile, better latency hiding
  ex3_fp16       32×32 tiles, FP16 inputs  -- half the memory traffic

WHY THE NUMBERS ARE SMALLER THAN ex2_tiling.py
-----------------------------------------------
ex2 (Triton) shows  bad_tile=15ms → good_tile=1.4ms → fp16=0.66ms  (~23× total).
ex3 (this file) shows        ~29ms →        ~26ms  → fp16=~27ms  (~1.1× total).

The difference comes down to what limits throughput:

  ex2 uses tl.dot, which Triton compiles to wmma / MMA Tensor Core instructions.
  Those do 16 multiply-adds per CUDA core per clock → the kernel is COMPUTE-BOUND.
  Bigger tiles amortize the HBM loads over more MMA ops → large tiling benefit.
  FP16 doubles the MMA throughput → large precision benefit.

  ex3 accumulates with scalar FP32/FP16 multiply-add in a loop. Even with 32×32
  shared-memory tiles, the A100's 2 TB/s HBM bandwidth is enough to keep the SMs
  fed → the kernel is MEMORY-BANDWIDTH-BOUND. Tile size matters less once you are
  already bandwidth-bound, and FP16 inputs help only if the kernel is also
  compute-bound (here it is not).

To see large tiling and precision gains in CUDA C you would need to use the
wmma intrinsics (cuda/include/mma.h) or cuBLAS, just as Triton does under the hood.

Run:  python ex3_tiling_cuda.py
Then compare ./prof/ex3_* in https://ui.perfetto.dev
"""
import ctypes
import shutil
import subprocess
import tempfile
from pathlib import Path

import torch
from torch.utils import benchmark

from prof_utils import require_cuda, run_profile

require_cuda()

M, N, K = 4096, 4096, 4096

# ---- CUDA kernel source (no PyTorch / ATen headers) ----
CUDA_SRC = r"""
#include <cuda_fp16.h>
#include <stdint.h>

template <int BLOCK>
__global__ void matmul_fp32_kernel(const float* __restrict__ A,
                                    const float* __restrict__ B,
                                    float*       __restrict__ C,
                                    int M, int N, int K)
{
    __shared__ float As[BLOCK][BLOCK];
    __shared__ float Bs[BLOCK][BLOCK];
    int row = blockIdx.y * BLOCK + threadIdx.y;
    int col = blockIdx.x * BLOCK + threadIdx.x;
    float acc = 0.f;
    for (int t = 0; t < (K + BLOCK - 1) / BLOCK; ++t) {
        int ak = t * BLOCK + threadIdx.x;
        int bk = t * BLOCK + threadIdx.y;
        As[threadIdx.y][threadIdx.x] = (row < M && ak < K) ? A[row * K + ak] : 0.f;
        Bs[threadIdx.y][threadIdx.x] = (bk < K && col < N) ? B[bk * N + col] : 0.f;
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < BLOCK; ++k) acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        __syncthreads();
    }
    if (row < M && col < N) C[row * N + col] = acc;
}

template <int BLOCK>
__global__ void matmul_fp16_kernel(const __half* __restrict__ A,
                                    const __half* __restrict__ B,
                                    float*        __restrict__ C,
                                    int M, int N, int K)
{
    __shared__ __half As[BLOCK][BLOCK];
    __shared__ __half Bs[BLOCK][BLOCK];
    int row = blockIdx.y * BLOCK + threadIdx.y;
    int col = blockIdx.x * BLOCK + threadIdx.x;
    float acc = 0.f;
    for (int t = 0; t < (K + BLOCK - 1) / BLOCK; ++t) {
        int ak = t * BLOCK + threadIdx.x;
        int bk = t * BLOCK + threadIdx.y;
        As[threadIdx.y][threadIdx.x] = (row < M && ak < K) ? A[row * K + ak] : __float2half(0.f);
        Bs[threadIdx.y][threadIdx.x] = (bk < K && col < N) ? B[bk * N + col] : __float2half(0.f);
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < BLOCK; ++k)
            acc += __half2float(As[threadIdx.y][k]) * __half2float(Bs[k][threadIdx.x]);
        __syncthreads();
    }
    if (row < M && col < N) C[row * N + col] = acc;
}

extern "C" {
void bad_tile(const float* A, const float* B, float* C, int M, int N, int K) {
    matmul_fp32_kernel<16><<<dim3((N+15)/16,(M+15)/16), dim3(16,16)>>>(A, B, C, M, N, K);
}
void good_tile(const float* A, const float* B, float* C, int M, int N, int K) {
    matmul_fp32_kernel<32><<<dim3((N+31)/32,(M+31)/32), dim3(32,32)>>>(A, B, C, M, N, K);
}
void fp16_tile(const __half* A, const __half* B, float* C, int M, int N, int K) {
    matmul_fp16_kernel<32><<<dim3((N+31)/32,(M+31)/32), dim3(32,32)>>>(A, B, C, M, N, K);
}
}
"""

# ---- compile to a shared library ----
build_dir = Path(tempfile.mkdtemp(prefix="matmul_cuda_"))
cu_path  = build_dir / "matmul.cu"
so_path  = build_dir / "matmul.so"
cu_path.write_text(CUDA_SRC)

print("Compiling CUDA extension... ", end="", flush=True)
nvcc = shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
result = subprocess.run(
    [nvcc, "-O3", "--use_fast_math", "--shared", "-Xcompiler=-fPIC",
     "-gencode=arch=compute_80,code=sm_80",
     "-o", str(so_path), str(cu_path)],
    capture_output=True, text=True,
)
if result.returncode != 0:
    print("FAILED")
    print(result.stderr)
    raise SystemExit(1)
print("done.")

lib = ctypes.CDLL(str(so_path))
_void_p = ctypes.c_void_p
_int    = ctypes.c_int
for _fn in ("bad_tile", "good_tile", "fp16_tile"):
    getattr(lib, _fn).restype  = None
    getattr(lib, _fn).argtypes = [_void_p, _void_p, _void_p, _int, _int, _int]


def _call(fn_name, A, B):
    C = torch.zeros(A.shape[0], B.shape[1], device="cuda", dtype=torch.float32)
    getattr(lib, fn_name)(
        A.data_ptr(), B.data_ptr(), C.data_ptr(),
        A.shape[0], B.shape[1], A.shape[1],
    )
    torch.cuda.synchronize()
    return C


# ---- inputs ----
a_fp32 = torch.randn(M, K, device="cuda")
b_fp32 = torch.randn(K, N, device="cuda")
a_fp16 = a_fp32.half()
b_fp16 = b_fp32.half()

# ---- warm-up (outside the profiler) ----
_call("bad_tile",  a_fp32, b_fp32)
_call("good_tile", a_fp32, b_fp32)
_call("fp16_tile", a_fp16, b_fp16)

# ---- correctness ----
ref = torch.mm(a_fp32, b_fp32)
torch.testing.assert_close(_call("good_tile", a_fp32, b_fp32), ref, rtol=1e-2, atol=0.5)
torch.testing.assert_close(_call("fp16_tile", a_fp16, b_fp16), ref, rtol=1e-1, atol=1.0)
print("assert_close: OK")

# ---- profile ----
run_profile(lambda: _call("bad_tile",  a_fp32, b_fp32), "ex3_bad_tile")
run_profile(lambda: _call("good_tile", a_fp32, b_fp32), "ex3_good_tile")
run_profile(lambda: _call("fp16_tile", a_fp16, b_fp16), "ex3_fp16")

# ---- benchmark ----
for label, fn in (
    ("bad_tile  (16×16, FP32)", lambda: _call("bad_tile",  a_fp32, b_fp32)),
    ("good_tile (32×32, FP32)", lambda: _call("good_tile", a_fp32, b_fp32)),
    ("fp16_tile (32×32, FP16)", lambda: _call("fp16_tile", a_fp16, b_fp16)),
):
    t = benchmark.Timer("fn()", globals={"fn": fn})
    print(f"{label}: {t.blocked_autorange()}")
