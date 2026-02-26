import numpy as np
from numba import cuda

# ----------------------------
# TODO: implementare il kernel di somma vettoriale C = A + B
# ----------------------------
@cuda.jit
def vec_add(a, b, c, n):
    # TODO:
    i = cuda.grid(1)
    if i < n:
        c[i] = a[i] + b[i]


def main():
    n = 10_000_000
    a_h = np.random.rand(n).astype(np.float32)
    b_h = np.random.rand(n).astype(np.float32)

    a_d = cuda.to_device(a_h)
    b_d = cuda.to_device(b_h)
    c_d = cuda.device_array_like(a_d)

    threads = 256
    blocks = (n + (threads - 1))//threads

    # warmup
    vec_add[blocks, threads](a_d, b_d, c_d, n)
    cuda.synchronize()

    # run
    vec_add[blocks, threads](a_d, b_d, c_d, n)
    cuda.synchronize()

    c_h = c_d.copy_to_host()
    ok = np.allclose(c_h, a_h + b_h, atol=1e-6)
    print(f"vec_add correct={ok} | n={n} | grid={blocks} block={threads}")


if __name__ == "__main__":
    main()
