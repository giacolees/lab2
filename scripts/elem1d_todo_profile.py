import numpy as np
import cupy as cp
from numba import cuda
import math
import time

@cuda.jit
def elem1d_kernel(x, y):
    # TO DO: implementare il kernel che calcola y[i] = sin(x[i]) + x[i] * x[i]
    i = cuda.grid(1)
    n = x.shape[0]
    stride = cuda.gridsize(1)
    while i < n:
        xi = x[i]
        y[i] = math.sin(xi) + (xi * xi)
        i += stride


@cuda.jit
def elem1d_kernel_inplace(x, y):
    # TO DO: implementare il kernel che calcola y[i] = sin(x[i]) + x[i] * x[i]
    i = cuda.grid(1)
    n = x.shape[0]
    stride = cuda.gridsize(1)
    while i < n:
        y[i] = math.sin(x[i]) + (x[i] * x[i])
        i += stride

# ---------- NUMBA: device-level ----------
def elem1d_numba_device(x_host: np.ndarray, blockdim: int = 256):
    assert x_host.dtype == np.float32
    n = x_host.size
    x_d = cuda.to_device(x_host)
    y_d = cuda.device_array(n, dtype=np.float32)
    threads = blockdim
    blocks = (n + threads - 1) // threads
    return x_d, y_d, blocks, threads

def elem1d_numba_launch(x_d, y_d, blocks, threads):
    # TO DO: lanciare il kernel `elem1d_kernel` con la configurazione di grid/blocks calcolata in `elem1d_numba_device`
    elem1d_kernel[blocks, threads](x_d, y_d)

# ---------- CuPy: device-level ----------
def elem1d_cupy_device(x_host: np.ndarray):
    assert x_host.dtype == np.float32
    # TO DO: allocare x_d e y_d su GPU usando CuPy, copiando x_host in x_d. Restituire x_d e y_d.
    x_d = cp.asarray(x_host)
    y_d = cp.empty_like(x_d)
    return x_d, y_d

def elem1d_cupy_launch(x_d, y_d):
    # write into y_d to avoid reallocations each iter
    # TO DO: calcorare y_d = sin(x_d) + x_d * x_d usando CuPy
    y_d[:] = cp.sin(x_d) + cp.pow(x_d, 2)

# ---- Test + timing ----
if __name__ == "__main__":
    N = 10_000_000
    x = np.random.rand(N).astype(np.float32)

    # CPU reference
    # TO DO: calcolare y_cpu usando NumPy, con la stessa formula usata nei kernel GPU. Misurare il tempo totale (incluso il calcolo) e salvarlo in t_cpu.
    t0 = time.time()
    
    y_cpu = np.sin(x) + np.pow(x, 2)
    t_cpu = time.time() - t0
    # --------------------
    # NUMBA: warmup/bench (NO D2H in warmup)
    # --------------------
    x_d, y_d, blocks, threads = elem1d_numba_device(x, blockdim=256)

    for _ in range(5):
        # TO DO: lanciare il kernel di warmup usando `elem1d_numba_launch` con x_d, y_d, blocks, threads
        elem1d_numba_launch(x_d, y_d, blocks, threads)
    cuda.synchronize()

    #TO-DO: lanciare il kernel di bench usando `elem1d_numba_launch` con x_d, y_d, blocks, threads. Misurare il tempo totale (incluso il lancio e la sincronizzazione) e salvarlo in t_numba.
    t0 = time.time()
    elem1d_numba_launch(x_d, y_d, blocks, threads)
    cuda.synchronize()
    y_numba = y_d.copy_to_host()   # single D2H
    t_numba = time.time() - t0

    # --------------------
    # CuPy: warmup/bench (NO D2H in warmup)
    # --------------------
    x_cp, y_cp = elem1d_cupy_device(x)

    for _ in range(5):
        elem1d_cupy_launch(x_cp, y_cp)
    cp.cuda.Stream.null.synchronize()

    t0 = time.time()
    elem1d_cupy_launch(x_cp, y_cp)
    cp.cuda.Stream.null.synchronize()
    y_cupy = cp.asnumpy(y_cp)      # single D2H
    t_cupy = time.time() - t0

    print("Elem1D max abs diff (numba vs cpu):", np.max(np.abs(y_numba - y_cpu)))
    print("Elem1D max abs diff (cupy  vs cpu):", np.max(np.abs(y_cupy  - y_cpu)))
    print(f"CPU : {t_cpu:.4f}s")
    print(f"Numba (incl one D2H): {t_numba:.4f}s")
    print(f"CuPy  (incl one D2H): {t_cupy:.4f}s")

    assert np.allclose(y_numba, y_cpu, atol=1e-5, rtol=1e-5)
    assert np.allclose(y_cupy,  y_cpu, atol=1e-5, rtol=1e-5)
