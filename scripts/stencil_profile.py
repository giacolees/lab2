# scripts/stencil2d_profile.py
# 2D stencil (5-point) naive vs shared-memory tiled (halo) + benchmarking + NVTX ranges
#
# Why stencil for LAB2:
# - Shared memory reuse is obvious
# - Speedup is usually clearer than matmul in Numba
# - Great for Nsight Systems: you can see kernel time dominate and compare versions
#
# Run:
#   python scripts/stencil2d_profile.py
# Profile:
#   nsys profile -t cuda,nvtx,osrt -o results/nsys/stencil python scripts/stencil2d_profile.py
#   nsys stats results/nsys/stencil.nsys-rep

import os
import time
import numpy as np
from numba import cuda, float32

# Optional NVTX ranges (recommended)
try:
    import nvtx
    def range_push(msg): nvtx.push_range(msg)
    def range_pop(): nvtx.pop_range()
except Exception:
    def range_push(msg): pass
    def range_pop(): pass


# ----------------------------
# Kernels
# ----------------------------

@cuda.jit
def stencil5_naive(inp, out, h, w):
    """
    5-point stencil:
      out[y,x] = 0.25*(N+S+E+W) - 1.0*center
    Borders are left unchanged (or you can set to 0).
    """
    x, y = cuda.grid(2)  # note order: (x,y)
    if 0 < x < w - 1 and 0 < y < h - 1:
        c  = inp[y, x]
        n  = inp[y - 1, x]
        s  = inp[y + 1, x]
        e  = inp[y, x + 1]
        wv = inp[y, x - 1]
        out[y, x] = 0.25 * (n + s + e + wv) - 1.0 * c


# Tile sizes (threads per block). Keep moderate.
BX = 32
BY = 8
SH_H = 10   # BY + 2
SH_W = 34   # BX + 2

@cuda.jit
def stencil5_shared(inp, out, h, w):
    """
    Shared-memory tiled stencil with 1-cell halo.
    Shared tile dims: (BY+2) x (BX+2)
    Each thread loads its center; extra threads load halo edges.
    """
    # shared tile with halo
    tile = cuda.shared.array((SH_H, SH_W), dtype=float32)

    tx = cuda.threadIdx.x  # 0..BX-1
    ty = cuda.threadIdx.y  # 0..BY-1

    x = cuda.blockIdx.x * BX + tx
    y = cuda.blockIdx.y * BY + ty

    # coordinates in shared (shift by +1 for halo)
    sx = tx + 1
    sy = ty + 1

    # Load center
    if x < w and y < h:
        tile[sy, sx] = inp[y, x]
    else:
        tile[sy, sx] = 0.0

    # Load halo: left/right edges
    if tx == 0:
        xL = x - 1
        if xL >= 0 and y < h:
            tile[sy, 0] = inp[y, xL]
        else:
            tile[sy, 0] = 0.0

    if tx == BX - 1:
        xR = x + 1
        if xR < w and y < h:
            tile[sy, BX + 1] = inp[y, xR]
        else:
            tile[sy, BX + 1] = 0.0

    # Load halo: top/bottom edges
    if ty == 0:
        yT = y - 1
        if yT >= 0 and x < w:
            tile[0, sx] = inp[yT, x]
        else:
            tile[0, sx] = 0.0

    if ty == BY - 1:
        yB = y + 1
        if yB < h and x < w:
            tile[BY + 1, sx] = inp[yB, x]
        else:
            tile[BY + 1, sx] = 0.0

    # Load halo corners (only 4 threads do this)
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
        c  = tile[sy, sx]
        n  = tile[sy - 1, sx]
        s  = tile[sy + 1, sx]
        e  = tile[sy, sx + 1]
        wv = tile[sy, sx - 1]
        out[y, x] = 0.25 * (n + s + e + wv) - 1.0 * c


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

def cpu_ref(inp):
    out = np.zeros_like(inp, dtype=np.float32)
    h, w = inp.shape
    out[1:-1, 1:-1] = 0.25 * (
        inp[0:-2, 1:-1] + inp[2:, 1:-1] + inp[1:-1, 2:] + inp[1:-1, 0:-2]
    ) - 1.0 * inp[1:-1, 1:-1]
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

    print("Numba CUDA 2D Stencil profiling script")
    dev = cuda.get_current_device()
    name = dev.name.decode("utf-8") if hasattr(dev.name, "decode") else dev.name
    print(f"GPU: {name}")
    print(f"Block naive  : (32, 8)")
    print(f"Block shared : ({BX}, {BY})  with halo in shared\n")

    # Sizes: make it big enough to show GPU behavior clearly
    sizes = [(2048, 2048), (4096, 4096)]

    for (h, w) in sizes:
        print(f"=== size = {h} x {w} ===")
        np.random.seed(0)
        inp_h = np.random.rand(h, w).astype(np.float32)

        range_push("cpu_ref")
        t0 = time.perf_counter()
        ref = cpu_ref(inp_h)
        t1 = time.perf_counter()
        range_pop()
        print(f"CPU ref (NumPy)      : {(t1 - t0)*1000.0:.2f} ms (single run)")

        range_push("H2D")
        inp_d = cuda.to_device(inp_h)
        out_d = cuda.device_array((h, w), dtype=np.float32)
        cuda.synchronize()
        range_pop()

        # Naive config
        block_naive = (32, 8)  # (x,y)
        grid = (ceil_div(w, block_naive[0]), ceil_div(h, block_naive[1]))

        naive_ms = bench(
            stencil5_naive, inp_d, out_d, h, w,
            block=block_naive, grid=grid, iters=200, warmup=20,
            label="bench_naive"
        )

        range_push("D2H_naive")
        out_naive = out_d.copy_to_host()
        cuda.synchronize()
        range_pop()
        ok, max_abs = check(out_naive, ref)
        print(f"GPU naive            : {naive_ms:.4f} ms/iter | correct={ok} | max_abs_err={max_abs:.3g}")

        # Shared config (must match BX,BY)
        block_shared = (BX, BY)
        grid_shared = (ceil_div(w, BX), ceil_div(h, BY))

        shared_ms = bench(
            stencil5_shared, inp_d, out_d, h, w,
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
