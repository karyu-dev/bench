from time import perf_counter
import cupy as cp
import numpy as np


def bench_gpu_cupy() -> dict[str, float] | None:
    if not cp.cuda.is_available():
        print("⚠️  Aucun GPU CUDA détecté par CuPy.")
        return None

    dev = cp.cuda.Device(0)
    _, total_mem = dev.mem_info
    total_vram_gb = total_mem / (1024**3)

    print(f"--- Benchmark GPU CuPy ({total_vram_gb:.2f} GB VRAM) ---\n")

    # Événements CUDA pour chronométrage matériel ultra-précis
    start_evt = cp.cuda.Event()
    end_evt = cp.cuda.Event()

    # ----------------------------------------------------
    # 1. BANDE PASSANTE BUS PCIe (H2D & D2H)
    # ----------------------------------------------------
    print("=== 1. BANDE PASSANTE BUS PCIe (GB/s) ===")
    size_mb = 1024  # 1 Go de transfert
    host_arr = np.random.randn(size_mb * 1024 * 1024 // 4).astype(np.float32)

    # Host to Device (RAM -> VRAM)
    cp.cuda.Stream.null.synchronize()
    start_evt.record()
    gpu_arr = cp.asarray(host_arr)
    end_evt.record()
    cp.cuda.Stream.null.synchronize()
    h2d_ms = cp.cuda.get_elapsed_time(start_evt, end_evt)
    h2d_gbps = (size_mb / 1024) / (h2d_ms / 1000)

    # Device to Host (VRAM -> RAM)
    start_evt.record()
    _ = cp.asnumpy(gpu_arr)
    end_evt.record()
    cp.cuda.Stream.null.synchronize()
    d2h_ms = cp.cuda.get_elapsed_time(start_evt, end_evt)
    d2h_gbps = (size_mb / 1024) / (d2h_ms / 1000)

    print(f"  Host -> Device (RAM to VRAM) : {h2d_gbps:6.2f} GB/s")
    print(f"  Device -> Host (VRAM to RAM) : {d2h_gbps:6.2f} GB/s")

    # ----------------------------------------------------
    # 2. BANDE PASSANTE MEMOIRE VRAM INTERNE
    # ----------------------------------------------------
    print("\n=== 2. BANDE PASSANTE MEMOIRE VRAM (GB/s) ===")
    # 512 Mo par tenseur (128M float32)
    n_elems = 128 * 1024 * 1024
    a_vram = cp.full(n_elems, 2.5, dtype=cp.float32)
    b_vram = cp.full(n_elems, 1.5, dtype=cp.float32)
    q = np.float32(3.0)

    its = 50
    # Warm-up
    for _ in range(5):
        a_vram = b_vram + q * a_vram

    cp.cuda.Stream.null.synchronize()
    start_evt.record()
    for _ in range(its):
        a_vram = b_vram + q * a_vram
    end_evt.record()
    cp.cuda.Stream.null.synchronize()

    vram_ms = cp.cuda.get_elapsed_time(start_evt, end_evt)
    # Triad : 2 lectures + 1 écriture = 3 * 4 octets = 12 octets / élément
    vram_gbps = ((3 * n_elems * 4 * its) / 1e9) / (vram_ms / 1000)
    print(f"  Débit VRAM Interne (Triad)  : {vram_gbps:6.2f} GB/s")

    # ----------------------------------------------------
    # 3. PEAK COMPUTE FP32 (cuBLAS GEMM)
    # ----------------------------------------------------
    print("\n=== 3. PEAK COMPUTE MATRIX (TFLOPS) ===")
    dim = 8192  # Matrice 8192 x 8192

    mat_a = cp.random.randn(dim, dim, dtype=cp.float32)
    mat_b = cp.random.randn(dim, dim, dtype=cp.float32)

    # Warm-up (initialisation du contexte cuBLAS)
    _ = cp.dot(mat_a, mat_b)

    its_gemm = 20
    cp.cuda.Stream.null.synchronize()
    start_evt.record()
    for _ in range(its_gemm):
        _ = cp.dot(mat_a, mat_b)
    end_evt.record()
    cp.cuda.Stream.null.synchronize()

    gemm_ms = cp.cuda.get_elapsed_time(start_evt, end_evt)
    # 2 * N^3 FLOPs par GEMM
    total_flops = 2 * (dim**3) * its_gemm
    tflops = (total_flops / (gemm_ms / 1000)) / 1e12
    print(f"  Performance GEMM FP32 (cuBLAS) : {tflops:6.2f} TFLOPS\n")

    return {
        "h2d_gbps": h2d_gbps,
        "d2h_gbps": d2h_gbps,
        "vram_gbps": vram_gbps,
        "fp32_tflops": tflops,
    }


if __name__ == "__main__":
    bench_gpu_cupy()