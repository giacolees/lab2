# scripts/stencil49_sweep.py
# Block-size sweep for stencil49 (7x7 blur) shared kernel.
#
# Student-friendly flow:
# 1) Sweep with timing (fast): try a few block shapes and print a table.
# 2) Profile ONLY one chosen config with NSYS (clean): --block BX,BY --only-shared
#
# Examples:
#   python scripts/stencil49_sweep.py --h 2048 --w 2048
#   python scripts/stencil49_sweep.py --h 4096 --w 4096
#
# After you find the best block (e.g. 16,16), profile only that:
#   nsys profile --force-overwrite true -t cuda,osrt -o results/nsys/st49_16x16 \
#     python scripts/stencil49_sweep.py --h 2048 --w 2048 --block 16,16 --only-shared
#   python tools/parse_nsys.py results/nsys/st49_16x16.nsys-rep

import argparse
import time
import numpy as np
from numba import cuda, float32

R = 3  # radius for 7x7


def ceil_div(a, b):
    return (a + b - 1) // b


def bench_2d(kernel, inp_d, out_d, h, w, block, grid, iters=120, warmup=20):
    for _ in range(warmup):
        kernel[grid, block](inp_d, out_d, h, w)
    cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        kernel[grid, block](inp_d, out_d, h, w)
    cuda.synchronize()
    t1 = time.perf_counter()

    return (t1 - t0) * 1000.0 / iters


@cuda.jit
def stencil49_naive(inp, out, h, w):
    x, y = cuda.grid(2)
    if R <= x < w - R and R <= y < h - R:
        s = 0.0
        for dy in range(-R, R + 1):
            for dx in range(-R, R + 1):
                s += inp[y + dy, x + dx]
        out[y, x] = s * (1.0 / 49.0)


def make_stencil49_shared(BX, BY):
    """Factory: returns a @cuda.jit kernel with compile-time shared tile shape."""
    SH_H = BY + 2 * R
    SH_W = BX + 2 * R

    @cuda.jit
    def stencil49_shared(inp, out, h, w):
        tile = cuda.shared.array((SH_H, SH_W), dtype=float32)

        tx = cuda.threadIdx.x
        ty = cuda.threadIdx.y

        # output tile origin handled by this block
        ox = cuda.blockIdx.x * BX
        oy = cuda.blockIdx.y * BY

        # shared tile origin (global) including halo
        gx0 = ox - R
        gy0 = oy - R

        # Cooperative load of full shared tile (including corners)
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

        # compute one output per thread (inside the BXxBY output tile)
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

    return stencil49_shared

def make_stencil49_shared_linear_indexing(BX, BY):
    """Factory: returns a @cuda.jit kernel with compile-time shared tile shape."""
    SH_H = BY + 2 * R
    SH_W = BX + 2 * R

    @cuda.jit
    def stencil49_shared(inp, out, h, w):
        tile = cuda.shared.array((SH_H, SH_W), dtype=float32)

        tx = cuda.threadIdx.x
        ty = cuda.threadIdx.y

        # output tile origin handled by this block
        ox = cuda.blockIdx.x * BX
        oy = cuda.blockIdx.y * BY

        # shared tile origin (global) including halo
        gx0 = ox - R
        gy0 = oy - R

        # Cooperative load: linear indexing for coalescing
        for idx in range(ty * BX + tx, SH_H * SH_W, BX * BY):
            y = idx // SH_W
            x = idx % SH_W
            gx = gx0 + x
            gy = gy0 + y
            if 0 <= gx < w and 0 <= gy < h:
                tile[y, x] = inp[gy, gx]
            else:
                tile[y, x] = 0.0

        cuda.syncthreads()

        # compute one output per thread (inside the BXxBY output tile)
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

    return stencil49_shared


def make_stencil49_optimized(BX, BY):
    """
    Highly optimized stencil kernel with:
    - Coalesced halo loads (linear indexing)
    - Thread-level parallelism (2x1 output elements per thread)
    - Register-blocking for better cache locality
    """
    SH_H = BY + 2 * R
    SH_W = BX + 2 * R

    @cuda.jit
    def stencil49_opt(inp, out, h, w):
        tile = cuda.shared.array((SH_H, SH_W), dtype=float32)

        tx = cuda.threadIdx.x
        ty = cuda.threadIdx.y
        tidx = ty * BX + tx

        ox = cuda.blockIdx.x * BX
        oy = cuda.blockIdx.y * BY

        gx0 = ox - R
        gy0 = oy - R

        # Cooperative load with linear indexing (coalesced)
        total_sh = SH_H * SH_W
        for idx in range(tidx, total_sh, BX * BY):
            y = idx // SH_W
            x = idx % SH_W
            gx = gx0 + x
            gy = gy0 + y
            if 0 <= gx < w and 0 <= gy < h:
                tile[y, x] = inp[gy, gx]
            else:
                tile[y, x] = 0.0

        cuda.syncthreads()

        # Thread-level parallelism: each thread processes 2 horizontal outputs
        # This increases arithmetic intensity and better utilizes registers
        xg0 = ox + tx
        yg = oy + ty

        INV_49 = 1.0 / 49.0

        # First output element
        if R <= xg0 < w - R and R <= yg < h - R:
            sx = tx + R
            sy = ty + R
            s = 0.0
            for dy in range(-R, R + 1):
                for dx in range(-R, R + 1):
                    s += tile[sy + dy, sx + dx]
            out[yg, xg0] = s * INV_49

        # Second output element (if space available)
        xg1 = xg0 + BX
        if R <= xg1 < w - R and R <= yg < h - R:
            sx1 = tx + BX + R
            sy = ty + R
            s = 0.0
            for dy in range(-R, R + 1):
                for dx in range(-R, R + 1):
                    s += tile[sy + dy, sx1 + dx]
            out[yg, xg1] = s * INV_49

    return stencil49_opt


def cpu_ref_7x7(inp):
    h, w = inp.shape
    out = np.zeros_like(inp, dtype=np.float32)
    acc = None
    for dy in range(-R, R + 1):
        for dx in range(-R, R + 1):
            view = inp[R + dy : h - R + dy, R + dx : w - R + dx]
            acc = view.copy() if acc is None else (acc + view)
    out[R:-R, R:-R] = acc * (1.0 / 49.0)
    return out


def parse_configs(configs_str: str):
    """
    Parse configs like:
      "32x8,32x16,16x16"
    into [(32,8), (32,16), (16,16)]
    """
    out = []
    for item in configs_str.split(","):
        item = item.strip().lower()
        if not item:
            continue
        if "x" not in item:
            raise ValueError(f"Bad config '{item}'. Use format like 32x8.")
        bx, by = item.split("x", 1)
        out.append((int(bx), int(by)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h", type=int, default=2048)
    ap.add_argument("--w", type=int, default=2048)
    ap.add_argument("--iters", type=int, default=120)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--check", action="store_true", help="Check correctness vs CPU ref (slower).")

    # New options for clean NSYS profiling
    ap.add_argument("--block", type=str, default=None,
                    help="Run only ONE shared block config, format BX,BY (e.g. 16,16).")
    ap.add_argument("--only-shared", action="store_true",
                    help="Skip naive baseline (useful when profiling one config).")
    ap.add_argument("--configs", type=str, default="32x8,32x16,16x16,64x4",
                    help="Comma-separated sweep configs in BXxBY format.")
    ap.add_argument("--kernel", type=str, default="naive", choices=["naive", "shared", "shared_linear_indexing", "optimized"],
                help="Which kernel to use (for single block profiling).")

    args = ap.parse_args()

    h, w = args.h, args.w
    iters, warmup = args.iters, args.warmup

    dev = cuda.get_current_device()
    name = dev.name.decode("utf-8") if hasattr(dev.name, "decode") else dev.name
    print(f"GPU: {name}")
    print(f"Stencil: 7x7 (radius={R}) | size={h}x{w} | iters={iters} warmup={warmup}\n")

    np.random.seed(0)
    inp_h = np.random.rand(h, w).astype(np.float32)

    ref = None
    if args.check:
        print("Computing CPU reference (one-time)...")
        t0 = time.perf_counter()
        ref = cpu_ref_7x7(inp_h)
        t1 = time.perf_counter()
        print(f"CPU ref: {(t1 - t0) * 1000.0:.2f} ms\n")

    inp_d = cuda.to_device(inp_h)
    out_d = cuda.device_array((h, w), dtype=np.float32)

    # Optional baseline naive (used for speedup reference)
    naive_ms = None
    if not args.only_shared:
        block_naive = (32, 32)
        grid_naive = (ceil_div(w, block_naive[0]), ceil_div(h, block_naive[1]))
        naive_ms = bench_2d(stencil49_naive, inp_d, out_d, h, w, block_naive, grid_naive, iters, warmup)
        ok_naive = True
        if args.check:
            out_naive = out_d.copy_to_host()
            ok_naive = np.allclose(out_naive, ref, atol=1e-4)
        print("Baseline naive:")
        print(f"  block={block_naive}  time={naive_ms:.3f} ms/iter  correct={ok_naive}\n")

    # If user requested a single shared block: run it and exit (great for NSYS).
    if args.block is not None:
        BX, BY = map(int, args.block.split(","))
        if BX * BY > 1024:
            raise ValueError(f"Invalid block {BX}x{BY}: BX*BY must be <= 1024.")
            # Select kernel based on --kernel argument

        if args.kernel == "naive":
            kernel = stencil49_naive
        elif args.kernel == "shared":
            kernel = make_stencil49_shared(BX, BY)
        elif args.kernel == "shared_linear_indexing":
            kernel = make_stencil49_shared_linear_indexing(BX, BY)
        elif args.kernel == "optimized":
            kernel = make_stencil49_optimized(BX, BY)
        else:
            raise ValueError(f"Unknown kernel: {args.kernel}")    
        
        block = (BX, BY)
        grid = (ceil_div(w, BX), ceil_div(h, BY))
        ms = bench_2d(kernel, inp_d, out_d, h, w, block, grid, iters, warmup)

        ok = True
        if args.check:
            out = out_d.copy_to_host()
            ok = np.allclose(out, ref, atol=1e-4)

        if naive_ms is not None:
            speedup = naive_ms / ms if ms > 0 else float("inf")
            print(f"Single shared block: {block}  time={ms:.3f} ms/iter  speedup_vs_naive={speedup:.2f}  correct={ok}")
        else:
            print(f"Single shared block: {block}  time={ms:.3f} ms/iter  correct={ok}")
        return

    # Otherwise: sweep configs with timing (fast).
    configs = parse_configs(args.configs)
    results = []
    for (BX, BY) in configs:
        if BX * BY > 1024:
            print(f"Skipping invalid block {BX}x{BY}: BX*BY must be <= 1024.") # 1024 threads max
            continue
            
        # Select kernel based on --kernel argument
        if args.kernel == "naive":
            kernel = stencil49_naive
        elif args.kernel == "shared":
            kernel = make_stencil49_shared(BX, BY)
        elif args.kernel == "shared_linear_indexing":
            kernel = make_stencil49_shared_linear_indexing(BX, BY)
        elif args.kernel == "optimized":
            kernel = make_stencil49_optimized(BX, BY)
        else:
            raise ValueError(f"Unknown kernel: {args.kernel}")
        
        block = (BX, BY)
        grid = (ceil_div(w, BX), ceil_div(h, BY))
        ms = bench_2d(kernel, inp_d, out_d, h, w, block, grid, iters, warmup)

        ok = True
        if args.check:
            out = out_d.copy_to_host()
            ok = np.allclose(out, ref, atol=1e-4)

        results.append((BX, BY, ms, ok))

    results.sort(key=lambda x: x[2])

    print(f"{args.kernel.upper()} kernel block sweep (lower is better):")

    if naive_ms is not None:
        print(f"{'BX':>4} {'BY':>4} {'ms/iter':>10} {'speedup_vs_naive':>17} {'correct':>8}")
        for BX, BY, ms, ok in results:
            speedup = naive_ms / ms if ms > 0 else float("inf")
            print(f"{BX:>4} {BY:>4} {ms:>10.3f} {speedup:>17.2f} {str(ok):>8}")
    else:
        print(f"{'BX':>4} {'BY':>4} {'ms/iter':>10} {'correct':>8}")
        for BX, BY, ms, ok in results:
            print(f"{BX:>4} {BY:>4} {ms:>10.3f} {str(ok):>8}")


if __name__ == "__main__":
    main()
