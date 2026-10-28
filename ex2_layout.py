"""Example 2 - Memory layout, allocation and tiling.

Three profiled runs, one change each:
  ex2_nchw_fp32  baseline (default NCHW layout)
  ex2_nhwc_fp32  layout: model and input in channels_last, input converted ONCE
  ex2_nhwc_fp16  + autocast FP16 so convolutions use Tensor Cores
cudnn.benchmark autotunes the conv algorithm / tile shape for these sizes.

Run:  python ex2_layout.py            (add --anti-pattern to convert inside the step)
Then compare the three folders under ./prof in https://ui.perfetto.dev
"""
import sys

import torch
import torch.nn as nn
from torch.utils import benchmark

from prof_utils import require_cuda, run_profile

require_cuda()
ANTI_PATTERN = "--anti-pattern" in sys.argv

torch.backends.cudnn.benchmark = True  # tiling: autotune conv algorithm per shape
net = nn.Sequential(
    nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128),
    nn.ReLU(), nn.Conv2d(128, 128, 3, padding=1)).cuda().eval()
x = torch.randn(32, 64, 112, 112, device="cuda")

with torch.no_grad():
    ref = net(x)  # warm-up + reference output
    run_profile(lambda: net(x), "ex2_nchw_fp32")  # baseline
    timings = {"nchw_fp32": benchmark.Timer("m(x)", globals={"m": net, "x": x}).blocked_autorange()}

    net = net.to(memory_format=torch.channels_last)          # layout
    x_cl = x.contiguous(memory_format=torch.channels_last)   # allocation: convert once

    if ANTI_PATTERN:
        # Converting inside the step adds aten::copy_ and a new buffer every iteration.
        run_profile(lambda: net(x.contiguous(memory_format=torch.channels_last)),
                    "ex2_nhwc_fp32_convert_in_loop")
    run_profile(lambda: net(x_cl), "ex2_nhwc_fp32")

    def amp_step():
        with torch.autocast("cuda", dtype=torch.float16):
            return net(x_cl)

    run_profile(amp_step, "ex2_nhwc_fp16")  # Tensor Cores

    # Step 4: verify correctness, then wall-clock speed
    torch.testing.assert_close(amp_step().float(), ref, rtol=1e-2, atol=5e-2)
    print("\nassert_close: OK")
    timings["nhwc_fp32"] = benchmark.Timer("m(x)", globals={"m": net, "x": x_cl}).blocked_autorange()
    timings["nhwc_fp16"] = benchmark.Timer("f()", globals={"f": amp_step}).blocked_autorange()
    for label, m in timings.items():
        print(f"{label}: {m}")
