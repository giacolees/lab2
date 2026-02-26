import os
import time
import numpy as np
from numba import cuda, float32

# ----------------------------
# 49-point stencil (7x7 blur)
# ----------------------------

@cuda.jit
def stencil49_naive(inp, out, h, w):
    x, y = cuda.grid(2)
    if 3 <= x < w - 3 and 3 <= y < h - 3:
        s = 0.0
        for dy in range(-3, 4):
            for dx in range(-3, 4):
                s += inp[y + dy, x + dx]
        out[y, x] = s * (1.0 / 49.0)


BX = 32
BY = 8
R = 3  # radius for 7x7
SH_H = BY + 2 * R   # 14
SH_W = BX + 2 * R   # 38

@cuda.jit
def stencil49_shared(inp, out, h, w):
    tile = cuda.shared.array((SH_H, SH_W), dtype=float32)

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y

    # top-left corner of the OUTPUT tile this block is responsible for
    ox = cuda.blockIdx.x * BX
    oy = cuda.blockIdx.y * BY

    # top-left corner (global) of the SHARED tile including halo
    gx0 = ox - R
    gy0 = oy - R

    # Cooperative load: cover the whole shared tile (including corners)
    # Stride by block dimensions so every thread loads multiple elements.
    y = ty
    while y < SH_H:
        x = tx
        while x < SH_W:
            gx = gx0 + x
            gy = gy0 + y
            if 0 <= gx < w and 0 <= gy < h:
                tile[y, x] = inp[gy, gx]
            else:
                tile[y, x] = 0.0
            x += BX
        y += BY

    cuda.syncthreads()

    # Compute output for threads that map inside the BXxBY output tile
    xg = ox + tx
    yg = oy + ty

    if R <= xg < w - R and R <= yg < h - R:
        sx = tx + R
        sy = ty + R
        s = 0.0
        for dy in range(-R, R + 1):
            for dx in range(-R, R + 1):
                s += tile[sy + dy, sx + dx]
        out[yg, xg] = s * (1.0 / 49.0)



# ----------------------------
# Helpers
# ----------------------------

def ceil_div(a, b):
    return (a + b - 1) // b

def bench(kernel, inp_d, out_d, h, w, block, grid, iters=100, warmup=10):
    for _ in range(warmup):
        kernel[grid, block](inp_d, out_d, h, w)
    cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        kernel[grid, block](inp_d, out_d, h, w)
    cuda.synchronize()
    t1 = time.perf_counter()

    return (t1 - t0) * 1000.0 / iters


def cpu_ref(inp):
    out = np.zeros_like(inp, dtype=np.float32)
    out[3:-3, 3:-3] = sum(
        inp[3+dy:inp.shape[0]-3+dy, 3+dx:inp.shape[1]-3+dx]
        for dy in range(-3, 4)
        for dx in range(-3, 4)
    ) * (1.0 / 49.0)
    return out


# ----------------------------
# Main
# ----------------------------

def main():
    os.makedirs("results/nsys", exist_ok=True)

    dev = cuda.get_current_device()
    name = dev.name.decode("utf-8") if hasattr(dev.name, "decode") else dev.name

    print("Numba CUDA 49-point stencil (7x7 blur)")
    print(f"GPU: {name}")
    print(f"Block naive  : (32, 8)")
    print(f"Block shared : ({BX}, {BY})  shared tile = ({SH_H}, {SH_W})\n")

    sizes = [(2048, 2048), (4096, 4096)]

    for (h, w) in sizes:
        print(f"=== size = {h} x {w} ===")
        np.random.seed(0)
        inp_h = np.random.rand(h, w).astype(np.float32)

        ref = cpu_ref(inp_h)

        inp_d = cuda.to_device(inp_h)
        out_d = cuda.device_array((h, w), dtype=np.float32)

        block = (32, 8)
        grid = (ceil_div(w, block[0]), ceil_div(h, block[1]))

        naive_ms = bench(stencil49_naive, inp_d, out_d, h, w, block, grid)
        out_naive = out_d.copy_to_host()
        ok1 = np.allclose(out_naive, ref, atol=1e-4)

        shared_ms = bench(stencil49_shared, inp_d, out_d, h, w, block, grid)
        out_shared = out_d.copy_to_host()
        ok2 = np.allclose(out_shared, ref, atol=1e-4)

        print(f"GPU naive  : {naive_ms:.3f} ms | correct={ok1}")
        print(f"GPU shared : {shared_ms:.3f} ms | correct={ok2}")
        print(f"Speedup shared vs naive: {naive_ms/shared_ms:.2f}x\n")


if __name__ == "__main__":
    main()
