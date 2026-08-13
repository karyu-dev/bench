import ctypes
import os
import subprocess
from pathlib import Path
from time import perf_counter
import numpy as np

# Compilation automatique
c_src_dir = Path(__file__).parent / "c_src"
so_path = c_src_dir / "libomp.so"

if not so_path.exists():
    print("⚡ Compilation de libomp.so...")
    subprocess.run(["make", "-C", str(c_src_dir)], check=True)

c_lib = ctypes.CDLL(str(so_path))
c_double_p = ctypes.POINTER(ctypes.c_double)

# Signatures C
c_lib.c_stream_triad_omp.argtypes = [
    c_double_p,
    c_double_p,
    c_double_p,
    ctypes.c_double,
    ctypes.c_size_t,
    ctypes.c_size_t,
    ctypes.c_int,
]
c_lib.c_fma_peak_omp.argtypes = [ctypes.c_size_t, ctypes.c_int]
c_lib.c_fma_peak_omp.restype = ctypes.c_double


def bench_thread_scaling():
    max_cores = os.cpu_count() or 1
    # Liste des configurations de threads à tester : ex [1, 2, 4, 8, 12, 16...]
    thread_counts = sorted(
        list(
            set(
                [1, 2, 4, 8, 12, 16, 24, 32, 64, max_cores]
                if max_cores >= 16
                else [1, 2, 4, 6, 8, 12, max_cores]
            )
        )
    )
    thread_counts = [t for t in thread_counts if t <= max_cores]

    print(
        f"--- Benchmark MULTI-THREAD SCALING (1 -> {max_cores} Threads) ---\n"
    )

    # ----------------------------------------------------
    # 1. SCALING COMPUTE : PEAK FMA FP64
    # ----------------------------------------------------
    print("=== 1. SCALING COMPUTE : Peak FMA FP64 (GFLOPS) ===")
    its_per_thread = 200_000_000

    for threads in thread_counts:
        # Warm-up
        _ = c_lib.c_fma_peak_omp(10_000_000, threads)

        t0 = perf_counter()
        _ = c_lib.c_fma_peak_omp(its_per_thread, threads)
        dt = perf_counter() - t0

        # Total FLOPs = threads * iterations_per_thread * 96 FLOPs
        total_gflops = (threads * its_per_thread * 96) / (dt * 1e9)
        speedup = total_gflops / (
            (its_per_thread * 96) / (dt * 1e9) / threads
        )  # approximatif vs 1 thread

        print(
            f"  Threads: {threads:<2} | Performance: {total_gflops:7.2f} GFLOPS"
        )

    # ----------------------------------------------------
    # 2. SCALING BANDWIDTH : STREAM TRIAD DRAM
    # ----------------------------------------------------
    print("\n=== 2. SCALING BANDE PASSANTE : STREAM Triad DRAM (GB/s) ===")
    # N grand (DRAM) : ~256 Mo total (16M float64)
    n = 16_777_216
    raw_a = np.empty(n + 24, dtype=np.float64)
    raw_b = np.full(n + 24, 2.5, dtype=np.float64)
    raw_c = np.full(n + 24, 1.5, dtype=np.float64)
    q = 3.0

    a_p = raw_a[0:n].ctypes.data_as(c_double_p)
    b_p = raw_b[8 : n + 8].ctypes.data_as(c_double_p)
    c_p = raw_c[16 : n + 16].ctypes.data_as(c_double_p)

    internal_its = 10
    it_size_gb = (3 * n * 8 * internal_its) / 1e9

    for threads in thread_counts:
        # Warm-up
        c_lib.c_stream_triad_omp(a_p, b_p, c_p, q, n, 2, threads)

        t0 = perf_counter()
        c_lib.c_stream_triad_omp(a_p, b_p, c_p, q, n, internal_its, threads)
        dt = perf_counter() - t0

        gbps = it_size_gb / dt
        print(f"  Threads: {threads:<2} | Débit RAM: {gbps:6.2f} GB/s")


if __name__ == "__main__":
    bench_thread_scaling()