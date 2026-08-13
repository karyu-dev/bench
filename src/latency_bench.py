import ctypes
import subprocess
from pathlib import Path
from time import perf_counter
import numpy as np

# Auto-compilation
c_src_dir = Path(__file__).parent/ "c_src"
so_path = c_src_dir / "liblatency.so"

if not so_path.exists():
    print("⚡ Compilation de liblatency.so...")
    subprocess.run(["make", "-C", str(c_src_dir)], check=True)

c_lib = ctypes.CDLL(str(so_path))
c_uint64_p = ctypes.POINTER(ctypes.c_uint64)
c_lib.c_pointer_chasing.argtypes = [c_uint64_p, ctypes.c_size_t]
c_lib.c_pointer_chasing.restype = ctypes.c_uint64


def create_random_cycle(n: int, seed: int | None = None) -> np.ndarray:
    """Génère une permutation circulaire unique de taille N (Cycle Hamiltonien)."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    chain = np.empty(n, dtype=np.uint64)

    # Chaining : chain[perm[i]] = perm[i+1]
    chain[perm[:-1]] = perm[1:]
    chain[perm[-1]] = perm[0]

    return chain


def measure_latency_ns(
    size_kb: int, iterations: int = 20_000_000, seed: int | None = None
) -> float:
    """Calcule la latence moyenne en nanosecondes pour une taille donnée (KiB)."""
    n_elements = (size_kb * 1024) // 8
    chain = create_random_cycle(n_elements, seed)
    chain_p = chain.ctypes.data_as(c_uint64_p)

    # Warm-up
    _ = c_lib.c_pointer_chasing(chain_p, 100_000)

    # Chrono
    t0 = perf_counter()
    _ = c_lib.c_pointer_chasing(chain_p, iterations)
    elapsed = perf_counter() - t0

    # Latence en nanosecondes (ns) par saut
    latency_ns = (elapsed * 1e9) / iterations
    return latency_ns


def bench_memory_latency_sweep() -> dict[str, float]:
    """Sonde la latence de L1d jusqu'à la DRAM."""
    # Tailles cibles : L1 (16 KiB), L2 (128 KiB), L3 (8 MiB), DRAM (128 MiB)
    targets_kb = [16, 128, 512, 4096, 16384, 131072]
    results = {}

    print("--- Benchmark LATENCE MEMOIRE (Pointer Chasing) ---")
    for size_kb in targets_kb:
        # Moins d'itérations pour les très gros tableaux DRAM
        its = 30_000_000 if size_kb <= 512 else 10_000_000
        lat_ns = measure_latency_ns(size_kb, iterations=its)

        label = (
            f"{size_kb} KiB" if size_kb < 1024 else f"{size_kb // 1024} MiB"
        )
        print(f"  Empreinte : {label:<8} | Latence moyenne : {lat_ns:.2f} ns")
        results[label] = lat_ns

    return results


if __name__ == "__main__":
    bench_memory_latency_sweep()