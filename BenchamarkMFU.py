import torch

# --- CONFIGURATION ---
N = 8192  # Matrix size (N x N)
dtype = torch.float32  # Change to torch.float32 to test FP32
device = 'cuda'

# Peak TFLOPS (Check your specific card)
# Values for A100 (SXM): BF16=312, FP32=19.5
# Values for 4090: BF16=330, FP32=82.6
PEAK_TFLOPS = 330 if dtype == torch.bfloat16 else 19.5 

print(f"Benchmarking {dtype} on {torch.cuda.get_device_name(0)}")
print(f"Matrix size: {N}x{N} | Theoretical Peak: {PEAK_TFLOPS} TFLOPS")

# 1. Initialize Tensors
a = torch.randn(N, N, device=device, dtype=dtype)
b = torch.randn(N, N, device=device, dtype=dtype)

# 2. Warmup (Never skip this on a GPU!)
for _ in range(10):
    torch.matmul(a, b)
torch.cuda.synchronize()

# 3. Measurement
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

start_event.record()
for _ in range(50):
    torch.matmul(a, b)
end_event.record()

torch.cuda.synchronize()
elapsed_time = start_event.elapsed_time(end_event) / 1000 / 50 # Avg time per op in seconds

# 4. Calculations
total_flops = 2 * (N**3)
achieved_tflops = (total_flops / elapsed_time) / 1e12
mfu = (achieved_tflops / PEAK_TFLOPS) * 100

print(f"--- RESULTS ---")
print(f"Avg Time: {elapsed_time*1000:.2f} ms")
print(f"Achieved: {achieved_tflops:.2f} TFLOPS")
print(f"MFU:      {mfu:.2f}%")