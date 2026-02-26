# scripts/reduction_profile.py
# Compare reduction implementations for:
#   s = sum(x) / mean(x)
# for:
#   1) CPU reference (float64)
#   2) CuPy (CUB-backed reductions)
#   3) Numba "optimized" (block reduction in shared + GPU finalize)
#   4) Numba "unoptimized" (per-thread partials, CPU finalize)
#
# Run:
#   python scripts/reduction_profile.py
#
# Profile (Nsight Systems):
#   nsys profile -t cuda,nvtx,osrt --stats=true -o results/nsys/reduction_profile python scripts/reduction_profile.py
#
# Notes:
# - Uses float32 input, but CPU reference uses float64 accumulation.
# - Syncs explicitly for accurate timing.
# - "unoptimized" Numba is intentionally bad: writes one partial per thread and finalizes on CPU.
# - "optimized" Numba uses shared-memory reduction + second-stage GPU reduction until 1 value.
# - Operation simplifies to N (because mean = sum/N), but we still compute it explicitly to test reductions.

import os
import time
import numpy as np
import cupy as cp
from numba import cuda, float32

# Optional NVTX
try:
    import nvtx
    def range_push(msg): nvtx.push_range(msg)
    def range_pop(): nvtx.pop_range()
except Exception:
    def range_push(msg): pass
    def range_pop(): pass

def ceil_div(a, b):
    return (a + b - 1) // b

# ----------------------------
# Numba kernels
# ----------------------------

@cuda.jit
def reduce_block_shared(x, partial, n):
    """
    Optimized stage: each block reduces to one value using shared memory.
    partial has length = gridDim.x
    """
    tid = cuda.threadIdx.x
    bid = cuda.blockIdx.x
    i = bid * cuda.blockDim.x + tid
    stride = cuda.blockDim.x * cuda.gridDim.x

    acc = 0.0
    while i < n:
        acc += x[i]
        i += stride

    # shared reduction
    sm = cuda.shared.array(256, dtype=float32)  # requires blockdim=256
    sm[tid] = acc
    cuda.syncthreads()

    offset = cuda.blockDim.x // 2
    while offset > 0:
        if tid < offset:
            sm[tid] += sm[tid + offset]
        cuda.syncthreads()
        offset //= 2

    if tid == 0:
        partial[bid] = sm[0]

@cuda.jit
def reduce_per_thread_unoptimized(x, partial, n):
    """
    Intentionally unoptimized:
    - each thread accumulates a strided sum
    - writes one partial per thread (partial length = total threads)
    - finalize on CPU
    """
    # TO DO: implementare la riduzione per-thread non ottimizzata, dove ogni thread accumula una somma strided e scrive un valore parziale in `partial[gid]`. La finalizzazione (somma dei parziali e calcolo del rapporto) avverrà poi sulla CPU.
    gid = cuda.grid(1)
    stride = cuda.gridsize(1)

    acc = 0.0
    i = gid
    while i < n:
        acc += x[i]
        i += stride

    partial[gid] = acc

@cuda.jit
def compute_ratio(sum_out, ratio_out, n):
    """
    Given sum_out[0] = sum(x), compute:
      mean = sum / n
      ratio = sum / mean
    """
    if cuda.threadIdx.x == 0 and cuda.blockIdx.x == 0:
        s = sum_out[0]
        mean = s / float32(n)
        ratio_out[0] = s / mean  # = n ideally

# ----------------------------
# CPU reference
# ----------------------------

def cpu_reference(x_host: np.ndarray) -> float:
    s = np.sum(x_host, dtype=np.float64)
    mean = s / float(x_host.size)
    return float(s / mean)

# ----------------------------
# CuPy implementation
# ----------------------------

def cupy_sum_over_mean(x_cp, iters=30, warmup=5):
    # Warmup
    range_push("cupy_warmup")
    for _ in range(warmup):
        s = cp.sum(x_cp)
        m = cp.mean(x_cp)
        r = s / m
    cp.cuda.Stream.null.synchronize()
    range_pop()

    # Benchmark
    range_push("cupy_bench")
    t0 = time.perf_counter()
    for _ in range(iters):
        # TO DO: implementare la riduzione usando CuPy, sfruttando le funzioni di riduzione ottimizzate (es. cupy.sum, cupy.mean)
        s = cp.sum(x_cp)
        m = cp.mean(x_cp)
        r = s / m
    cp.cuda.Stream.null.synchronize()
    t1 = time.perf_counter()
    range_pop()

    ms = (t1 - t0) * 1000.0 / iters
    return ms, float(r.get())

# ----------------------------
# Numba optimized: multi-stage reduction entirely on GPU
# ----------------------------

def numba_optimized_sum_over_mean(x_d, n, blocks=1024, threads=256, iters=30, warmup=5):
    assert threads == 256, "reduce_block_shared uses shared array sized for 256 threads"

    # Allocate buffers with max needed size for first stage
    partial_d = cuda.device_array(blocks, dtype=np.float32)
    sum_d = cuda.device_array(1, dtype=np.float32)
    ratio_d = cuda.device_array(1, dtype=np.float32)

    def reduce_to_one(input_d, length, label):
        """
        Repeatedly reduce input_d (length elements) into a single value in sum_d[0].
        Uses reduce_block_shared for each stage.
        """
        cur_d = input_d
        cur_len = length

        # We’ll allocate temporary partial buffers as needed
        while True:
            grid = min(blocks, ceil_div(cur_len, threads))
            reduce_block_shared[grid, threads](cur_d, partial_d, cur_len)
            # After this stage, we have 'grid' partial sums in partial_d[0:grid]
            cuda.synchronize()

            if grid == 1:
                # Copy the single value into sum_d[0]
                # Reuse a tiny kernel via device-to-device copy is not direct in Numba;
                # easiest: run one more reduction stage with grid=1 into sum_d-sized array.
                # We'll just reuse partial_d as "sum" by reading partial_d[0].
                # Store into sum_d[0] via a tiny kernel-like pattern: compute_ratio expects sum_out[0].
                # We'll copy partial_d[:1] into sum_d by launching reduce_block_shared with cur_len=1.
                # Simpler: use cuda.to_device on host scalar (too slow).
                # Best simple approach: make sum_d an alias by reusing partial_d[:1] is not possible.
                # So we allocate sum_d and do one-thread kernel to write sum_d[0] = partial_d[0].
                _write_first_to_sum[1, 1](partial_d, sum_d)
                cuda.synchronize()
                break
            else:
                # Next stage reduces the partials
                cur_d = partial_d
                cur_len = grid

        return sum_d

    # tiny kernel to write sum_d[0] = partial_d[0]
    @cuda.jit
    def _write_first_to_sum(partial, sum_out):
        if cuda.blockIdx.x == 0 and cuda.threadIdx.x == 0:
            sum_out[0] = partial[0]

    # Warmup
    range_push("numba_opt_warmup")
    for _ in range(warmup):
        reduce_to_one(x_d, n, "opt")
        compute_ratio[1, 1](sum_d, ratio_d, n)
    cuda.synchronize()
    range_pop()

    # Benchmark end-to-end on GPU (no D2H until the end)
    range_push("numba_opt_bench")
    t0 = time.perf_counter()
    for _ in range(iters):
        reduce_to_one(x_d, n, "opt")
        compute_ratio[1, 1](sum_d, ratio_d, n)
    cuda.synchronize()
    t1 = time.perf_counter()
    range_pop()

    ms = (t1 - t0) * 1000.0 / iters
    r = float(ratio_d.copy_to_host()[0])
    return ms, r

# ----------------------------
# Numba unoptimized: per-thread partials + CPU finalize
# ----------------------------

def numba_unoptimized_sum_over_mean(x_d, n, blocks=1024, threads=256, iters=30, warmup=5):
    total_threads = blocks * threads
    partial_d = cuda.device_array(total_threads, dtype=np.float32)

    # Warmup
    range_push("numba_bad_warmup")
    for _ in range(warmup):
        reduce_per_thread_unoptimized[blocks, threads](x_d, partial_d, n)
    cuda.synchronize()
    range_pop()

    # Benchmark end-to-end: kernel + D2H + CPU finalize (this is intentionally "bad")
    range_push("numba_bad_bench")
    t0 = time.perf_counter()
    r = 0.0
    for _ in range(iters):
        reduce_per_thread_unoptimized[blocks, threads](x_d, partial_d, n)
        cuda.synchronize()
        partial_h = partial_d.copy_to_host()
        s = float(np.sum(partial_h, dtype=np.float64))
        mean = s / float(n)
        r = s / mean
    t1 = time.perf_counter()
    range_pop()

    ms = (t1 - t0) * 1000.0 / iters
    return ms, float(r)

# ----------------------------
# Main
# ----------------------------

def main():
    os.makedirs("results/nsys", exist_ok=True)

    N = 50_000_000
    iters = 30
    warmup = 5

    np.random.seed(0)
    x_host = np.random.rand(N).astype(np.float32)

    range_push("cpu_ref")
    ref = cpu_reference(x_host)
    range_pop()

    # H2D once
    range_push("H2D")
    x_d = cuda.to_device(x_host)
    cuda.synchronize()
    range_pop()

    # CuPy uses its own device arrays; reuse host data to keep experiment comparable
    range_push("cupy_H2D")
    x_cp = cp.asarray(x_host)
    cp.cuda.Stream.null.synchronize()
    range_pop()

    # CuPy
    cupy_ms, cupy_r = cupy_sum_over_mean(x_cp, iters=iters, warmup=warmup)

    # Numba optimized (GPU-only)
    numba_opt_ms, numba_opt_r = numba_optimized_sum_over_mean(
        x_d, N, blocks=1024, threads=256, iters=iters, warmup=warmup
    )

    # Numba unoptimized (kernel + D2H + CPU finalize inside loop)
    numba_bad_ms, numba_bad_r = numba_unoptimized_sum_over_mean(
        x_d, N, blocks=1024, threads=256, iters=iters, warmup=warmup
    )

    def rel_err(v):
        return abs(v - ref) / (abs(ref) + 1e-12)

    print("=== Reduction: r = sum(x) / mean(x) ===")
    print(f"N = {N}")
    print(f"CPU ref (float64)       : r={ref:.6f}")
    print("")
    print(f"CuPy (sum+mean)         : {cupy_ms:.3f} ms/iter | r={cupy_r:.6f} | rel_err={rel_err(cupy_r):.3e}")
    print(f"Numba optimized (GPU)   : {numba_opt_ms:.3f} ms/iter | r={numba_opt_r:.6f} | rel_err={rel_err(numba_opt_r):.3e}")
    print(f"Numba UNoptimized (bad) : {numba_bad_ms:.3f} ms/iter | r={numba_bad_r:.6f} | rel_err={rel_err(numba_bad_r):.3e}")
    print("")
    print("Expected r is approximately N (because mean = sum/N), deviations are due to float32 rounding and reduction order.")

if __name__ == "__main__":
    main()
