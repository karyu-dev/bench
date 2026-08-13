import numpy as np
from time import perf_counter
from tqdm import trange
import ctypes
from pathlib import Path
from psutil import virtual_memory

from get_sysinfo import get_exact_cache_sizes


# 1. Chargement de la bibliothèque partagée C
so_path = Path(__file__).parent / "c_src" / "libstream.so"
c_lib = ctypes.CDLL(str(so_path))

# 2. Définition des signatures C pour ctypes
c_double_p = ctypes.POINTER(ctypes.c_double)

# COPY(a, b, n, internal_its)
c_lib.c_stream_copy.argtypes = [c_double_p, c_double_p, ctypes.c_size_t, ctypes.c_size_t]
c_lib.c_stream_scale.argtypes = [c_double_p, c_double_p, ctypes.c_double, ctypes.c_size_t, ctypes.c_size_t]
c_lib.c_stream_add.argtypes = [c_double_p, c_double_p, c_double_p, ctypes.c_size_t, ctypes.c_size_t]

# TRIAD(a, b, c, q, n, internal_its)
c_lib.c_stream_triad.argtypes = [c_double_p, c_double_p, c_double_p, ctypes.c_double, ctypes.c_size_t, ctypes.c_size_t]

# Décalage entre les bases de tableaux (0 / 64 / 128 octets) :
# évite les conflits de sets de cache quand N est une puissance de 2
PAD = 8  # 64 octets = 1 ligne de cache

# Nombre d'accès mémoire par élément pour chaque kernel
KERNEL_ARGS = {"copy": 2, "scale": 2, "add": 3, "triad": 3}


def _l3_size_bytes() -> int:
    """Taille du L3 en octets (SysFS), fallback 32 MiB."""
    return get_exact_cache_sizes().get("l3", 32 * 1024 * 1024)


def _call_kernel(kernel: str, a_p, b_p, c_p, q: float, n: int, internal_its: int) -> None:
    if kernel == "copy":
        c_lib.c_stream_copy(a_p, b_p, n, internal_its)
    elif kernel == "scale":
        c_lib.c_stream_scale(a_p, b_p, q, n, internal_its)
    elif kernel == "add":
        c_lib.c_stream_add(a_p, b_p, c_p, n, internal_its)
    elif kernel == "triad":
        c_lib.c_stream_triad(a_p, b_p, c_p, q, n, internal_its)
    else:
        raise ValueError(f"kernel inconnu : {kernel}")


def _STREAM_bench(kernel: str, n: int, it: int, internal_its: int,
                  min_time: float, flush: bool, seed: int | None,
                  silent: bool) -> dict[str, float]:
    argn = KERNEL_ARGS[kernel]
    rng = np.random.default_rng(seed)
    q = float(rng.random())

    # Bases décalées de 0 / 64 / 128 octets pour éviter les conflits de sets
    a = np.empty(n + 2 * PAD, dtype=np.float64)
    b = np.empty(n + 2 * PAD, dtype=np.float64)
    b[PAD:PAD + n] = rng.random(n)

    a_p = a[:n].ctypes.data_as(c_double_p)
    b_p = b[PAD:PAD + n].ctypes.data_as(c_double_p)
    c_p = None
    if argn == 3:
        c = np.empty(n + 2 * PAD, dtype=np.float64)
        c[2 * PAD:2 * PAD + n] = rng.random(n)
        c_p = c[2 * PAD:2 * PAD + n].ctypes.data_as(c_double_p)

    # Buffer de flush : 2x le L3, écrit entre chaque échantillon pour vider le cache
    scratch = np.zeros(_l3_size_bytes() * 2 // 8, dtype=np.float64)

    # Warm-up (~0.5 s) : first-touch des pages + montée en fréquence DVFS
    t0 = perf_counter()
    while perf_counter() - t0 < 0.5:
        _call_kernel(kernel, a_p, b_p, c_p, q, n, internal_its)

    SIZE = argn * n * 8
    speed_array = np.empty(it, dtype=np.float64)

    for i in trange(it, desc=f"STREAM | {kernel.upper()}", leave=False):
        if flush:
            scratch.fill(0.0)
        start_time = perf_counter()
        passes = 0
        # Garde de durée minimale : le bruit du timer est négligeable sur la mesure
        while perf_counter() - start_time < min_time:
            _call_kernel(kernel, a_p, b_p, c_p, q, n, internal_its)
            passes += 1
        total_time = perf_counter() - start_time

        speed_array[i] = (SIZE * internal_its * passes) / (total_time * 1e9)

    metrics = {
        "peak": float(np.max(speed_array)),
        "lowest": float(np.min(speed_array)),
        "mean": float(np.mean(speed_array)),
        "median": float(np.median(speed_array)),
        "std": float(np.std(speed_array)),
    }

    if not silent:
        footprint_gib = argn * (n * 8) / (1024**3)
        print(f"--- Benchmark MEMORY | {kernel.upper()} (allocated: {footprint_gib:.2f} GiB) | {it} éch. ---")
        print(f"Peak: {metrics['peak']:.2f} GB/s | Median: {metrics['median']:.2f} GB/s | Lowest: {metrics['lowest']:.2f} GB/s | Std: +/- {metrics['std']:.2f} GB/s\n")

    return metrics


def _STREAM_copy_run(n: int, it: int, internal_its: int = 1, seed: int | None = None,
                     min_time: float = 0.2, flush: bool = True, silent: bool = False) -> dict[str, float]:
    return _STREAM_bench("copy", n, it, internal_its, min_time, flush, seed, silent)


def _STREAM_scale_run(n: int, it: int, internal_its: int = 1, seed: int | None = None,
                      min_time: float = 0.2, flush: bool = True, silent: bool = False) -> dict[str, float]:
    return _STREAM_bench("scale", n, it, internal_its, min_time, flush, seed, silent)


def _STREAM_add_run(n: int, it: int, internal_its: int = 1, seed: int | None = None,
                    min_time: float = 0.2, flush: bool = True, silent: bool = False) -> dict[str, float]:
    return _STREAM_bench("add", n, it, internal_its, min_time, flush, seed, silent)


def _STREAM_triad_run(n: int, it: int, internal_its: int = 1, seed: int | None = None,
                      min_time: float = 0.2, flush: bool = True, silent: bool = False) -> dict[str, float]:
    return _STREAM_bench("triad", n, it, internal_its, min_time, flush, seed, silent)


def STREAM_run_DRAM(n: int, it: int, internal_its: int = 1, seed: int | None = None,
                    min_time: float = 0.2, flush: bool = True) -> dict[str, dict[str, float]]:
    return {
        "copy": _STREAM_copy_run(n, it, internal_its, seed, min_time, flush),
        "scale": _STREAM_scale_run(n, it, internal_its, seed, min_time, flush),
        "add": _STREAM_add_run(n, it, internal_its, seed, min_time, flush),
        "triad": _STREAM_triad_run(n, it, internal_its, seed, min_time, flush),
    }


def STREAM_sweep(kernel: str = "copy", it: int = 5, seed: int | None = None,
                 min_time: float = 0.2, flush: bool = True) -> dict[int, dict[str, float]]:
    """Balaie N en puissances de 2 : visualise les paliers L1/L2/L3/DRAM."""
    argn = KERNEL_ARGS[kernel]
    n_min = 256

    # Plafond : footprint max = 25 % de la RAM disponible
    max_footprint = virtual_memory().available * 0.25
    n_max = n_min
    while argn * (n_max * 2) * 8 <= max_footprint:
        n_max *= 2

    results = {}
    print(f"--- SWEEP {kernel.upper()} | N en puissances de 2 ---")
    n = n_min
    while n <= n_max:
        # internal_its auto : chaque appel C dure ~0.1 s (hypothèse 20 GB/s)
        internal_its = max(1, int(0.1 * 20e9 / (argn * n * 8)))
        metrics = _STREAM_bench(kernel, n, it, internal_its, min_time, flush, seed, silent=True)
        footprint_mib = argn * (n * 8) / (1024**2)
        print(f"n={n:>9} | footprint {footprint_mib:>10.2f} MiB | median {metrics['median']:>8.2f} GB/s | peak {metrics['peak']:>8.2f} GB/s")
        results[n] = metrics
        n *= 2
    return results


def GEMM_run(n: int, it: int, seed: int | None = None, min_time: float = 0.2) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    a = rng.random((n, n), dtype=np.float64)
    b = rng.random((n, n), dtype=np.float64)

    DOT_FLOPS = 2 * n**3
    gflops_array = np.empty(it, dtype=np.float64)

    # Warm-up (~0.5 s) : chauffe le CPU et initialise le thread pool BLAS/OpenMP
    warm_size = min(n, 1000)
    aw, bw = a[:warm_size, :warm_size], b[:warm_size, :warm_size]
    t0 = perf_counter()
    while perf_counter() - t0 < 0.5:
        _ = aw @ bw

    for i in trange(it):
        start_time = perf_counter()
        passes = 0
        while perf_counter() - start_time < min_time:
            _ = a @ b  # Affectation explicite
            passes += 1
        total_time = perf_counter() - start_time

        gflops_array[i] = DOT_FLOPS * passes / (total_time * 1e9)

    metrics = {
        "peak": float(np.max(gflops_array)),
        "mean": float(np.mean(gflops_array)),
        "median": float(np.median(gflops_array)),
        "std": float(np.std(gflops_array)),
    }

    # Formatage de l'unité (GFLOPS vs TFLOPS)
    unit = "TFLOPS" if metrics["peak"] >= 1000 else "GFLOPS"
    scale = 1000.0 if unit == "TFLOPS" else 1.0

    print(f"--- Benchmark GEMM ({n}x{n} | {it} itérations) ---")
    print(f"  Peak    : {metrics['peak'] / scale:.2f} {unit}")
    print(f"  Moyenne : {metrics['mean'] / scale:.2f} {unit}")
    print(f"  Médiane : {metrics['median'] / scale:.2f} {unit}")
    print(f"  Std Dev : +/- {metrics['std'] / scale:.2f} {unit}")

    return metrics
