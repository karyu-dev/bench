import ctypes
from pathlib import Path
from time import perf_counter

# Chargement du .so
so_path = Path(__file__).parent / "c_src" / "libfma.so"
c_lib = ctypes.CDLL(str(so_path))

# Signature : double c_fma_peak_fp64(size_t iterations)
c_lib.c_fma_peak_fp64.argtypes = [ctypes.c_size_t]
c_lib.c_fma_peak_fp64.restype = ctypes.c_double


def bench_fma_peak_single_core(
    iterations: int = 500_000_000,
) -> dict[str, float]:
    """Mesure le pic de calcul FP64 FMA théorique sur 1 cœur (GFLOPS)."""
    # Warm-up (monter la fréquence Turbo)
    _ = c_lib.c_fma_peak_fp64(10_000_000)

    # Chrono
    t0 = perf_counter()
    _ = c_lib.c_fma_peak_fp64(iterations)
    dt = perf_counter() - t0

    # 12 FMA * 8 FLOPs/FMA = 96 FLOPs par itération
    total_flops = iterations * 96
    gflops = (total_flops / dt) / 1e9

    print(f"--- Benchmark COMPUTE | Peak FMA FP64 (1 Cœur) ---")
    print(f"  Temps d'exécution : {dt*1000:.2f} ms")
    print(f"  Performance FMA   : {gflops:.2f} GFLOPS (FP64)")

    return {"gflops": gflops, "elapsed_s": dt}


if __name__ == "__main__":
    bench_fma_peak_single_core()