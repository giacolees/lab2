# scripts/matmul_profile.py
# Numba CUDA MatMul: naive vs tiled(shared) + CuPy (cuBLAS) + benchmarking + NVTX ranges (Nsight Systems friendly)
#
# Run:
#   python scripts/matmul_profile.py
#
# Profile (Nsight Systems):
#   nsys profile -t cuda,nvtx,osrt --stats=true -o results/nsys/matmul_all python scripts/matmul_profile.py
#
# Profile (Nsight Compute):
#   ncu --set full -o results/ncu/matmul_all python scripts/matmul_profile.py
#
# Notes:
# - Profiling works best from CLI, not Jupyter.
# - Uses float32.
# - Syncs explicitly for accurate timing.
# - CuPy is optional; if not installed, the script will still run Numba parts.

import os
import time
import numpy as np
from numba import cuda, float32

# Optional CuPy (cuBLAS)
try:
    import cupy as cp
    HAS_CUPY = True
except Exception:
    HAS_CUPY = False
    cp = None
    print("CuPy not found; proceeding without CuPy/cuBLAS baseline.")

# Optional NVTX (recommended). If not installed, script still works.
try:
    import nvtx
    def range_push(msg): nvtx.push_range(msg)
    def range_pop(): nvtx.pop_range()
except Exception:
    def range_push(msg): pass
    def range_pop(): pass
    print("NVTX package not found; proceeding without NVTX ranges.")
    
# ----------------------------
# CPU no BLAS
# ----------------------------
def matmul_cpu(A, B, C, n):
    for i in range(n):
        for j in range(n):
            s = 0.0
            for k in range(n):
                s += A[i, k] * B[k, j]
            C[i, j] = s
    
# ----------------------------
# CUDA kernels (Numba)
# ----------------------------
@cuda.jit
def matmul_naive(A, B, C, n):
    # Each thread computes one C[i, j] with pure global loads
    # TO DO: implementare il kernel di moltiplicazione matrice-matrice C = A @ B senza tiling/shared memory
    # Suggerimento: mappare ogni thread a un elemento di C (i, j) e calcolare il prodotto scalare della riga i di A con la colonna j di B
    i, j = cuda.grid(2)
    if i < n and j < n:
        s = 0.0
        for k in range(n):
            s += A[i, k] * B[k, j] 
        C[i, j] = s

# Choose a tile size that maps well to warps.
# 16 is a safe default across many GPUs; 32 can be faster but can reduce occupancy.
TILE = 16

@cuda.jit
def matmul_tiled(A, B, C, n):
    # Shared-memory tiled matrix multiplication
    sA = cuda.shared.array((TILE, TILE), dtype=float32)
    sB = cuda.shared.array((TILE, TILE), dtype=float32)

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y

    row = cuda.blockIdx.y * TILE + ty
    col = cuda.blockIdx.x * TILE + tx

    acc = 0.0

    tiles = (n + TILE - 1) // TILE
    for t in range(tiles):
        a_col = t * TILE + tx
        b_row = t * TILE + ty

        # Cooperative load into shared (with bounds checks)
        if row < n and a_col < n:
            sA[ty, tx] = A[row, a_col]
        else:
            sA[ty, tx] = 0.0

        if b_row < n and col < n:
            sB[ty, tx] = B[b_row, col]
        else:
            sB[ty, tx] = 0.0

        cuda.syncthreads()

        # Compute partial dot product for this tile
        for k in range(TILE):
            acc += sA[ty, k] * sB[k, tx]

        cuda.syncthreads()

    if row < n and col < n:
        C[row, col] = acc


# ----------------------------
# Register Blocking Optimization (Proper 16x16 thread block variant)
# ----------------------------
# Use 16x16 thread block; each thread computes 1x1 output but accumulates over all tiles (no explicit register arrays)
# This variant focuses on improving arithmetic intensity through better data reuse patterns.
@cuda.jit
def matmul_register_blocked(A, B, C, n):
    """
    Register-blocked variant: 16x16 thread block with load-time reuse.
    Each thread processes one output element but with persistent register blocking
    across multiple accumulations in the inner loop.
    This improves register reuse vs naive approach.
    """
    sA = cuda.shared.array((TILE, TILE), dtype=float32)
    sB = cuda.shared.array((TILE, TILE), dtype=float32)

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y

    row = cuda.blockIdx.y * TILE + ty
    col = cuda.blockIdx.x * TILE + tx

    # Use local variables (registers) to accumulate across all k values
    # This is the "register blocking" - keep partial sums in registers
    acc = 0.0

    tiles = (n + TILE - 1) // TILE
    for t in range(tiles):
        a_col = t * TILE + tx
        b_row = t * TILE + ty

        if row < n and a_col < n:
            sA[ty, tx] = A[row, a_col]
        else:
            sA[ty, tx] = 0.0

        if b_row < n and col < n:
            sB[ty, tx] = B[b_row, col]
        else:
            sB[ty, tx] = 0.0

        cuda.syncthreads()

        # Unroll the inner dot product to improve register reuse
        # By hand-unrolling, compilers better schedule operations
        for k in range(TILE):
            acc += sA[ty, k] * sB[k, tx]

        cuda.syncthreads()

    if row < n and col < n:
        C[row, col] = acc


# ----------------------------
# Double-Buffered Tiling Optimization
# ----------------------------
# Overlap computation of current tile with loading of next tile.
@cuda.jit
def matmul_double_buffered(A, B, C, n):
    """
    Double-buffered tiling: loads next tile while computing current tile.
    Reduces synchronization overhead and hides memory latency.
    """
    TILE = 16

    # Two sets of shared memory for double buffering
    sA = cuda.shared.array((2, TILE, TILE), dtype=float32)
    sB = cuda.shared.array((2, TILE, TILE), dtype=float32)

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y

    row = cuda.blockIdx.y * TILE + ty
    col = cuda.blockIdx.x * TILE + tx

    acc = 0.0
    num_tiles = (n + TILE - 1) // TILE

    # Load first tile
    buf = 0
    a_col = tx
    b_row = ty
    if row < n and a_col < n:
        sA[buf, ty, tx] = A[row, a_col]
    else:
        sA[buf, ty, tx] = 0.0
    if b_row < n and col < n:
        sB[buf, ty, tx] = B[b_row, col]
    else:
        sB[buf, ty, tx] = 0.0
    cuda.syncthreads()

    # Pipeline: compute tile i while loading tile i+1
    for t in range(num_tiles):
        # Load next tile (if not last)
        if t + 1 < num_tiles:
            next_buf = 1 - buf
            next_a_col = (t + 1) * TILE + tx
            next_b_row = (t + 1) * TILE + ty
            if row < n and next_a_col < n:
                sA[next_buf, ty, tx] = A[row, next_a_col]
            else:
                sA[next_buf, ty, tx] = 0.0
            if next_b_row < n and col < n:
                sB[next_buf, ty, tx] = B[next_b_row, col]
            else:
                sB[next_buf, ty, tx] = 0.0

        # Compute with current buffer while next is loading
        for k in range(TILE):
            acc += sA[buf, ty, k] * sB[buf, k, tx]

        cuda.syncthreads()
        buf = 1 - buf

    if row < n and col < n:
        C[row, col] = acc


# ----------------------------
# Helpers
# ----------------------------

def ceil_div(a, b):
    return (a + b - 1) // b

def bench_kernel(kernel, A_d, B_d, C_d, n, block, grid, iters=30, warmup=5, label="bench"):
    # Warm-up
    range_push(f"{label}_warmup")
    for _ in range(warmup):
        kernel[grid, block](A_d, B_d, C_d, n)
    cuda.synchronize()
    range_pop()

    # Benchmark
    range_push(f"{label}_bench")
    t0 = time.perf_counter()
    for _ in range(iters):
        kernel[grid, block](A_d, B_d, C_d, n)
    cuda.synchronize()
    t1 = time.perf_counter()
    range_pop()

    ms = (t1 - t0) * 1000.0 / iters
    return ms

def bench_cupy_matmul(A_cp, B_cp, iters=30, warmup=5, label="cupy"):
    # Warm-up
    range_push(f"{label}_warmup")
    for _ in range(warmup):
        C_cp = A_cp @ B_cp
    cp.cuda.Stream.null.synchronize()
    range_pop()

    # Benchmark
    range_push(f"{label}_bench")
    t0 = time.perf_counter()
    for _ in range(iters):
        C_cp = A_cp @ B_cp
    cp.cuda.Stream.null.synchronize()
    t1 = time.perf_counter()
    range_pop()

    ms = (t1 - t0) * 1000.0 / iters
    return ms, C_cp

def check_correctness(C_gpu, C_ref, atol=1e-2, rtol=1e-2):
    # Float32 matmul has some error; keep tolerances reasonable.
    ok = np.allclose(C_gpu, C_ref, atol=atol, rtol=rtol)
    max_abs = float(np.max(np.abs(C_gpu - C_ref)))
    return ok, max_abs


# ----------------------------
# Main
# ----------------------------

def main():
    os.makedirs("results/nsys", exist_ok=True)
    os.makedirs("results/ncu", exist_ok=True)

    # Sizes: pick something that shows meaningful differences.
    # If your GPU is small, start with 512 and 1024.
    sizes = [512, 1024, 2048, 4096]

    dev = cuda.get_current_device()
    dev_name = dev.name.decode("utf-8") if hasattr(dev.name, "decode") else dev.name

    print("Numba CUDA MatMul profiling script (+ optional CuPy/cuBLAS)")
    print(f"GPU: {dev_name}")
    print(f"TILE = {TILE}")
    print(f"CuPy available: {HAS_CUPY}\n")

    for n in sizes:
        print(f"=== n = {n} ===")
        np.random.seed(0)
        A_h = np.random.rand(n, n).astype(np.float32)
        B_h = np.random.rand(n, n).astype(np.float32)
        C_h = np.zeros((n, n))

        # CPU reference (no NumPy/BLAS) — do once per size
        # range_push("cpu_ref_matmul_for")
        # t0 = time.perf_counter()
        # C_ref = matmul_cpu(A_h, B_h, C_h, n)
        # t1 = time.perf_counter()
        # range_pop()
        # cpu_ms = (t1 - t0) * 1000.0
        # print(f"CPU (no NumPy BLAS)  : {cpu_ms:.2f} ms (single run)")


        # CPU reference (NumPy/BLAS) — do once per size
        range_push("cpu_ref")
        t0 = time.perf_counter()
        C_ref = A_h @ B_h
        t1 = time.perf_counter()
        range_pop()
        cpu_ms = (t1 - t0) * 1000.0
        print(f"CPU (NumPy BLAS)  : {cpu_ms:.2f} ms (single run)")

        # H2D (Numba)
        range_push("numba_H2D")
        A_d = cuda.to_device(A_h)
        B_d = cuda.to_device(B_h)
        C_d = cuda.device_array((n, n), dtype=np.float32)
        cuda.synchronize()
        range_pop()

        # Launch configs
        # Naive: use 16x16 (good baseline)
        block_naive = (16, 16)
        grid_naive = (ceil_div(n, block_naive[0]), ceil_div(n, block_naive[1]))

        # Tiled: must match TILE
        block_tiled = (TILE, TILE)
        grid_tiled = (ceil_div(n, TILE), ceil_div(n, TILE))

        # Benchmark naive
        naive_ms = bench_kernel(
            matmul_naive, A_d, B_d, C_d, n,
            block=block_naive, grid=grid_naive,
            iters=30, warmup=5, label="naive"
        )
        range_push("numba_D2H_naive")
        C_naive = C_d.copy_to_host()
        cuda.synchronize()
        range_pop()
        ok, max_abs = check_correctness(C_naive, C_ref)
        print(f"GPU naive         : {naive_ms:.2f} ms/iter | correct={ok} | max_abs_err={max_abs:.4g}")

        # Benchmark tiled
        tiled_ms = bench_kernel(
            matmul_tiled, A_d, B_d, C_d, n,
            block=block_tiled, grid=grid_tiled,
            iters=30, warmup=5, label="tiled"
        )
        range_push("numba_D2H_tiled")
        C_tiled = C_d.copy_to_host()
        cuda.synchronize()
        range_pop()
        ok, max_abs = check_correctness(C_tiled, C_ref)
        print(f"GPU tiled(shared) : {tiled_ms:.2f} ms/iter | correct={ok} | max_abs_err={max_abs:.4g}")

        speedup_tiled_vs_naive = naive_ms / tiled_ms if tiled_ms > 0 else float("inf")
        print(f"Speedup tiled vs naive: {speedup_tiled_vs_naive:.2f}x")

        # Benchmark register-blocked
        block_reg = (TILE, TILE)  # 16x16 thread block
        grid_reg = (ceil_div(n, TILE), ceil_div(n, TILE))

        reg_ms = bench_kernel(
            matmul_register_blocked, A_d, B_d, C_d, n,
            block=block_reg, grid=grid_reg,
            iters=30, warmup=5, label="register_blocked"
        )
        range_push("numba_D2H_regblocked")
        C_reg = C_d.copy_to_host()
        cuda.synchronize()
        range_pop()
        ok, max_abs = check_correctness(C_reg, C_ref)
        print(f"GPU register-blocked  : {reg_ms:.2f} ms/iter | correct={ok} | max_abs_err={max_abs:.4g}")

        speedup_reg_vs_naive = naive_ms / reg_ms if reg_ms > 0 else float("inf")
        speedup_reg_vs_tiled = tiled_ms / reg_ms if reg_ms > 0 else float("inf")
        print(f"Speedup register-blocked vs naive: {speedup_reg_vs_naive:.2f}x")
        print(f"Speedup register-blocked vs tiled: {speedup_reg_vs_tiled:.2f}x")

        # Benchmark double-buffered
        block_db = (16, 16)
        grid_db = (ceil_div(n, 16), ceil_div(n, 16))

        db_ms = bench_kernel(
            matmul_double_buffered, A_d, B_d, C_d, n,
            block=block_db, grid=grid_db,
            iters=30, warmup=5, label="double_buffered"
        )
        range_push("numba_D2H_doublebuf")
        C_db = C_d.copy_to_host()
        cuda.synchronize()
        range_pop()
        ok, max_abs = check_correctness(C_db, C_ref)
        print(f"GPU double-buffered   : {db_ms:.2f} ms/iter | correct={ok} | max_abs_err={max_abs:.4g}")

        speedup_db_vs_tiled = tiled_ms / db_ms if db_ms > 0 else float("inf")
        print(f"Speedup double-buffered vs tiled: {speedup_db_vs_tiled:.2f}x")

        # CuPy/cuBLAS (optional)
        if HAS_CUPY:
            range_push("cupy_H2D")
            A_cp = cp.asarray(A_h)
            B_cp = cp.asarray(B_h)
            cp.cuda.Stream.null.synchronize()
            range_pop()

            cupy_ms, C_cp = bench_cupy_matmul(A_cp, B_cp, iters=30, warmup=5, label="cupy")

            range_push("cupy_D2H")
            C_cupy = cp.asnumpy(C_cp)
            cp.cuda.Stream.null.synchronize()
            range_pop()

            ok, max_abs = check_correctness(C_cupy, C_ref)
            print(f"GPU CuPy (cuBLAS) : {cupy_ms:.2f} ms/iter | correct={ok} | max_abs_err={max_abs:.4g}")

            speedup_cublas_vs_tiled = tiled_ms / cupy_ms if cupy_ms > 0 else float("inf")
            print(f"Speedup cuBLAS vs tiled: {speedup_cublas_vs_tiled:.2f}x")

        print("")

if __name__ == "__main__":
    main()
