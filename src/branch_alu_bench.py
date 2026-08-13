import ctypes
import subprocess
from pathlib import Path
from time import perf_counter
import numpy as np

# Auto-compilation
c_src_dir = Path(__file__).parent / "c_src"
so_path = c_src_dir / "libbranch_alu.so"

if not so_path.exists():
    print("⚡ Compilation de libbranch_alu.so...")
    subprocess.run(["make", "-C", str(c_src_dir)], check=True)

c_lib = ctypes.CDLL(str(so_path))
c_uint8_p = ctypes.POINTER(ctypes.c_uint8)

c_lib.c_branch_test.argtypes = [c_uint8_p, ctypes.c_size_t, ctypes.c_size_t]
c_lib.c_branch_test.restype = ctypes.c_uint64

c_lib.c_int_alu_peak.argtypes = [ctypes.c_size_t]
c_lib.c_int_alu_peak.restype = ctypes.c_uint64


def bench_branch_prediction(n: int = 100_000, internal_its: int = 2000):
    """Compare un tableau 100% prédictible vs 50% aléatoire (Branch Misprediction)."""
    rng = np.random.default_rng(42)

    # 1. Pattern Prédictible : [0, 1, 0, 1, 0, 1...]
    pred_arr = np.tile(np.array([0, 1], dtype=np.uint8), n // 2)
    # 2. Pattern Imprédictible : 50% de 0 et 1 aléatoires
    rand_arr = rng.integers(0, 2, size=n, dtype=np.uint8)

    p_p = pred_arr.ctypes.data_as(c_uint8_p)
    r_p = rand_arr.ctypes.data_as(c_uint8_p)

    # Warm-up
    _ = c_lib.c_branch_test(p_p, n, 100)

    # Test Prédictible
    t0 = perf_counter()
    _ = c_lib.c_branch_test(p_p, n, internal_its)
    dt_pred = perf_counter() - t0

    # Test Imprédictible (Random)
    t0 = perf_counter()
    _ = c_lib.c_branch_test(r_p, n, internal_its)
    dt_rand = perf_counter() - t0

    total_branches = n * internal_its
    # Une distribution aléatoire 50% provoque ~50% de mauvaises prédictions
    estimated_mispredicts = total_branches * 0.5
    extra_time_s = max(0.0, dt_rand - dt_pred)

    # Pénalité par erreur de prédiction en nanosecondes
    penalty_ns = (extra_time_s * 1e9) / estimated_mispredicts

    # Estimation des cycles CPU perdus (sur la base d'une fréquence à 4.8 GHz)
    estimated_cycles = penalty_ns * 4.8

    print("--- Benchmark BRANCH PREDICTION ---")
    print(
        f"  Boucle Prédictible (0% miss)   : {dt_pred*1000:.2f} ms ({dt_pred/total_branches*1e9:.2f} ns/branche)"
    )
    print(
        f"  Boucle Imprédictible (~50% miss): {dt_rand*1000:.2f} ms ({dt_rand/total_branches*1e9:.2f} ns/branche)"
    )
    print(f"  --> Pénalité de Misprediction  : {penalty_ns:.2f} ns (~{estimated_cycles:.1f} cycles CPU)\n")

    return {"penalty_ns": penalty_ns, "cycles": estimated_cycles}


def bench_integer_alu(iterations: int = 500_000_000):
    """Mesure le débit maximum d'opérations entières INT64 (GIOPS)."""
    # Warm-up
    _ = c_lib.c_int_alu_peak(10_000_000)

    t0 = perf_counter()
    _ = c_lib.c_int_alu_peak(iterations)
    dt = perf_counter() - t0

    # 8 instructions * 4 uint64 par YMM = 32 opérations INT64 par itération
    total_ops = iterations * 32
    giops = (total_ops / dt) / 1e9

    print("--- Benchmark INTEGER ALU PEAK (INT64 AVX2) ---")
    print(f"  Temps d'exécution : {dt*1000:.2f} ms")
    print(f"  Performance INT64 : {giops:.2f} GIOPS (Giga Operations/s)\n")

    return {"giops": giops}


if __name__ == "__main__":
    bench_branch_prediction()
    bench_integer_alu()