import numpy as np
from numba import cuda
import time
import cupy as cp


@cuda.jit
def stencil5_kernel(A, B):
    # B ha shape (H-2, W-2)
    # TO DO: implementare il kernel che calcola la stencil5
    i, j = cuda.grid(2)

    out_h = A.shape[0] - 2
    out_w = A.shape[1] - 2

    if i < out_h and j < out_w:
        ai = i + 1  # centro in A
        aj = j + 1

        B[i, j] = (
            A[ai,aj] + A[ai-1,aj] + A[ai+1,aj] + A[ai,aj-1] + A[ai,aj+1]
        )

def stencil5_gpu(A_host: np.ndarray, threads=(16, 16)) -> np.ndarray:
    assert A_host.dtype == np.float32
    A_host = np.ascontiguousarray(A_host)

    H, W = A_host.shape
    out_h, out_w = H - 2, W - 2

    A_d = cuda.to_device(A_host)
    B_d = cuda.device_array((out_h, out_w), dtype=np.float32)

    tx, ty = threads
    bx = (out_h + tx - 1) // tx
    by = (out_w + ty - 1) // ty

    stencil5_kernel[(bx, by), (tx, ty)](A_d, B_d)
    cuda.synchronize()
    return B_d.copy_to_host()

def stencil5_cpu(A: np.ndarray) -> np.ndarray:
    # TO DO: implementare la stencil5 su CPU usando NumPy
    H, W = A.shape
    B = np.empty((H - 2, W - 2), dtype=np.float32)
    # centro A[i+1, j+1]
    for i in range(H - 2):
        for j in range(W - 2):
            ai = i + 1
            aj = j + 1
            B[i, j] = (
               A[ai,aj] + A[ai-1,aj] + A[ai+1,aj] + A[ai,aj-1] + A[ai,aj+1]
            )
    return B

def stencil5_cupy(A_host: np.ndarray) -> np.ndarray:
    """
    B[i,j] = A[i+1,j+1] + A[i,  j+1] + A[i+2,j+1] + A[i+1,j] + A[i+1,j+2]
    Output shape: (H-2, W-2)  (valid)
    """
    assert A_host.dtype == np.float32
    A = cp.asarray(A_host)  # H2D
    # TO DO: implementare la stencil5 usando CuPy
    center = A[1:-1, 1:-1]
    up     = A[0:-2, 1:-1]
    down   = A[2:, 1:-1]
    left   = A[1:-1, 0:-2]
    right  = A[1:-1, 2:]

    B = center + up + down + left + right  # elementwise kernel(s) (often fused)
    cp.cuda.Stream.null.synchronize()
    return cp.asnumpy(B)  # D2H

# ---- Test + timing ----
if __name__ == "__main__":
    H, W = 4096, 4096
    A = np.random.rand(H, W).astype(np.float32)

    # Correctness check on smaller tile
    small = A[:512, :512]
    B_cpu = stencil5_cpu(small)

    B_numba = stencil5_gpu(small, threads=(16, 16))
    B_cupy  = stencil5_cupy(small)

    print("Stencil5 max abs diff (numba vs cpu):", np.max(np.abs(B_numba - B_cpu)))
    print("Stencil5 max abs diff (cupy  vs cpu):", np.max(np.abs(B_cupy  - B_cpu)))

    assert np.allclose(B_numba, B_cpu, atol=1e-5, rtol=1e-5)
    assert np.allclose(B_cupy,  B_cpu, atol=1e-5, rtol=1e-5)

    # Timing on large input
    t0 = time.time()
    B_numba_big = stencil5_gpu(A, threads=(16, 16))
    t_numba = time.time() - t0

    t0 = time.time()
    B_cupy_big = stencil5_cupy(A)
    t_cupy = time.time() - t0

    print(f"Stencil5 Numba (4096x4096, incl copies): {t_numba:.4f}s | out={B_numba_big.shape}")
    print(f"Stencil5 CuPy  (4096x4096, incl copies): {t_cupy:.4f}s | out={B_cupy_big.shape}")

