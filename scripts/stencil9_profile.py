# scripts/stencil9_profile.py
# 2D stencil (9-point / 3x3 blur) naive vs shared-memory tiled (halo=1) + benchmarking + optional NVTX
#
# Run:
#   python scripts/stencil9_profile.py
# Profile:
#   rm -f results/nsys/stencil9.nsys-rep
#   nsys profile --force-overwrite true -t cuda,osrt -o results/nsys/stencil9 python scripts/stencil9_profile.py
#   nsys stats results/nsys/stencil9.nsys-rep

import os
import time
import numpy as np
from numba import cuda, float32

# Optional NVTX ranges (nice-to-have)
try:
    import nvtx
    def range_push(msg): nvtx.push_range(msg)
    def range_pop(): nvtx.pop_range()
except Exception:
    def range_push(msg): pass
    def range_pop(): pass


# ----------------------------
# Kernels: 9-point stencil (3x3 blur)
# out[y,x] = average of 3x3 neighborhood
# ----------------------------

@cuda.jit
def stencil9_naive(inp, out, h, w):
    x, y = cuda.grid(2)  # (x,y)
    if 0 < x < w - 1 and 0 < y < h - 1:
        s = 0.0
        # 3x3 neighborhood
        s += inp[y-1, x-1]; s += inp[y-1, x]; s += inp[y-1, x+1]
        s += inp[y,   x-1]; s += inp[y,   x]; s += inp[y,   x+1]
        s += inp[y+1, x-1]; s += inp[y+1, x]; s += inp[y+1, x+1]
        out[y, x] = s * (1.0 / 9.0)


# Threads per block (keep these moderate)
BX = 32
BY = 8

# Shared tile dims with halo=1 (compile-time constants!)
SH_H = 10  # BY + 2
SH_W = 34  # BX + 2

@cuda.jit
def stencil9_shared(inp, out, h, w):
    tile = cuda.shared.array((SH_H, SH_W), dtype=float32)

    tx = cuda.threadIdx.x  # 0..BX-1
    ty = cuda.threadIdx.y  # 0..BY-1

    x = cuda.blockIdx.x * BX + tx
    y = cuda.blockIdx.y * BY + ty

    sx = tx + 1
    sy = ty + 1

    # Center load
    if x < w and y < h:
        tile[sy, sx] = inp[y, x]
    else:
        tile[sy, sx] = 0.0

    # Halo edges
    if tx == 0:
        xL = x - 1
        tile[sy, 0] = inp[y, xL] if (xL >= 0 and y < h) else 0.0

    if tx == BX - 1:
        xR = x + 1
        tile[sy, BX + 1] = inp[y, xR] if (xR < w and y < h) else 0.0

    if ty == 0:
        yT = y - 1
        tile[0, sx] = inp[yT, x] if (yT >= 0 and x < w) else 0.0

    if ty == BY - 1:
        yB = y + 1
        tile[BY + 1, sx] = inp[yB, x] if (yB < h and x < w) else 0.0

    # Halo corners (4 threads)
    if tx == 0 and ty == 0:
        xL, yT = x - 1, y - 1
        tile[0, 0] = inp[yT, xL] if (xL >= 0 and yT >= 0) else 0.0

    if tx == BX - 1 and ty == 0:
        xR, yT = x + 1, y - 1
        tile[0, BX + 1] = inp[yT, xR] if (xR < w and yT >= 0) else 0.0

    if tx == 0 and ty == BY - 1:
        xL, yB = x - 1, y + 1
        tile[BY + 1, 0] = inp[yB, xL] if (xL >= 0 and yB < h) else 0.0

    if tx == BX - 1 and ty == BY - 1:
        xR, yB = x + 1, y + 1
        tile[BY + 1, BX + 1] = inp[yB, xR] if (xR < w and yB < h) else 0.0

    cuda.syncthreads()

    # Compute (skip borders)
    if 0 < x < w - 1 and 0 < y < h - 1:
        s = 0.0
        s += tile[sy-1, sx-1]; s += tile[sy-1, sx]; s += tile[sy-1, sx+1]
        s += tile[sy,   sx-1]; s += tile[sy,   sx]; s += tile[sy,   sx+1]
        s += tile[sy+1, sx-1]; s += tile[sy+1, sx]; s += tile[sy+1, sx+1]
        out[y, x] = s * (1.0 / 9.0)


# ----------------------------
# Helpers
# ----------------------------

def ceil_div(a, b):
    return (a + b - 1) // b

def bench(kernel, inp_d, out_d, h, w, block, grid, iters=200, warmup=20, label="bench"):
    range_push("warmup")
    for _ in range(warmup):
        kernel[grid, block](inp_d, out_d, h, w)
    cuda.synchronize()
    range_pop()

    range_push(label)
    t0 = time.perf_counter()
    for _ in range(iters):
        kernel[grid, block](inp_d, out_d, h, w)
    cuda.synchronize()
    t1 = time.perf_counter()
    range_pop()

    return (t1 - t0) * 1000.0 / iters

def cpu_ref_blur3x3(inp):
    out = np.zeros_like(inp, dtype=np.float32)
    # 3x3 blur via slicing; borders remain 0
    out[1:-1, 1:-1] = (
        inp[0:-2, 0:-2] + inp[0:-2, 1:-1] + inp[0:-2, 2:] +
        inp[1:-1, 0:-2] + inp[1:-1, 1:-1] + inp[1:-1, 2:] +
        inp[2:,   0:-2] + inp[2:,   1:-1] + inp[2:,   2:]
    ) * (1.0 / 9.0)
    return out

def check(out_gpu, out_ref, atol=1e-5, rtol=1e-5):
    ok = np.allclose(out_gpu, out_ref, atol=atol, rtol=rtol)
    max_abs = float(np.max(np.abs(out_gpu - out_ref)))
    return ok, max_abs


# ----------------------------
# Main
# ----------------------------

def main():
    os.makedirs("results/nsys", exist_ok=True)

    dev = cuda.get_current_device()
    name = dev.name.decode("utf-8") if hasattr(dev.name, "decode") else dev.name

    print("Numba CUDA 9-point stencil (3x3 blur) profiling script")
    print(f"GPU: {name}")
    print(f"Block naive  : (32, 8)")
    print(f"Block shared : ({BX}, {BY})  shared tile = ({SH_H}, {SH_W}) with halo=1\n")

    sizes = [(2048, 2048), (4096, 4096)]

    for (h, w) in sizes:
        print(f"=== size = {h} x {w} ===")
        np.random.seed(0)
        inp_h = np.random.rand(h, w).astype(np.float32)

        range_push("cpu_ref")
        t0 = time.perf_counter()
        ref = cpu_ref_blur3x3(inp_h)
        t1 = time.perf_counter()
        range_pop()
        print(f"CPU ref (NumPy)      : {(t1 - t0)*1000.0:.2f} ms (single run)")

        range_push("H2D")
        inp_d = cuda.to_device(inp_h)
        out_d = cuda.device_array((h, w), dtype=np.float32)
        cuda.synchronize()
        range_pop()

        block = (32, 8)
        grid = (ceil_div(w, block[0]), ceil_div(h, block[1]))

        naive_ms = bench(
            stencil9_naive, inp_d, out_d, h, w,
            block=block, grid=grid, iters=200, warmup=20,
            label="bench_naive"
        )
        range_push("D2H_naive")
        out_naive = out_d.copy_to_host()
        cuda.synchronize()
        range_pop()
        ok, max_abs = check(out_naive, ref)
        print(f"GPU naive            : {naive_ms:.4f} ms/iter | correct={ok} | max_abs_err={max_abs:.3g}")

        block_shared = (BX, BY)
        grid_shared = (ceil_div(w, BX), ceil_div(h, BY))

        shared_ms = bench(
            stencil9_shared, inp_d, out_d, h, w,
            block=block_shared, grid=grid_shared, iters=200, warmup=20,
            label="bench_shared"
        )
        range_push("D2H_shared")
        out_shared = out_d.copy_to_host()
        cuda.synchronize()
        range_pop()
        ok, max_abs = check(out_shared, ref)
        print(f"GPU shared(tiling)   : {shared_ms:.4f} ms/iter | correct={ok} | max_abs_err={max_abs:.3g}")

        speedup = naive_ms / shared_ms if shared_ms > 0 else float("inf")
        print(f"Speedup shared vs naive: {speedup:.2f}x\n")


if __name__ == "__main__":
    main()
