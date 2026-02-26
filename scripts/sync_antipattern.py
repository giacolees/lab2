import argparse
import time
import numpy as np
from numba import cuda

@cuda.jit
def saxpy(a, x, y, out, n):
    # TO DO: implementare il kernel SAXPY out = a * x + y
    # Suggerimento: mappare ogni thread a un elemento di out (i)
    i = cuda.grid(1)
    stride = cuda.gridsize(1)
    while i < n:
        out[i] = a*x[i] + y[i]
        i += stride

def bench(mode: str, iters=20000, warmup=20):
    n = 2_000_000
    a = np.float32(2.0)

    x_d = cuda.to_device(np.random.rand(n).astype(np.float32))
    y_d = cuda.to_device(np.random.rand(n).astype(np.float32))
    out_d = cuda.device_array(n, dtype=np.float32)

    threads = 256
    blocks = (n + threads - 1) // threads

    # warmup
    for _ in range(warmup):
        saxpy[blocks, threads](a, x_d, y_d, out_d, n)
    cuda.synchronize()

    t0 = time.perf_counter()
    if mode == "good":
        # GOOD: sync after the whole batch
        for _ in range(iters):
            saxpy[blocks, threads](a, x_d, y_d, out_d, n)
        cuda.synchronize()
    else:
        # BAD: sync dentro al loop => serializza e aumenta CPU wait
        for _ in range(iters):
            saxpy[blocks, threads](a, x_d, y_d, out_d, n) 
            cuda.synchronize()
    t1 = time.perf_counter()

    ms = (t1 - t0) * 1000 / iters
    out_h = out_d.copy_to_host()
    ok = np.isfinite(out_h).all()
    print(f"mode={mode} | {ms:.3f} ms/iter | correct={ok} | n={n}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["good", "bad"], required=True)
    args = ap.parse_args()
    bench(args.mode)

if __name__ == "__main__":
    main()
