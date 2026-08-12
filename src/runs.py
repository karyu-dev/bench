import numpy as np
from time import perf_counter
from tqdm import trange
import numexpr as ne

def GEMM_run(n: int, it: int, seed: int | None = None) -> dict[str, float]:

    rng = np.random.default_rng(seed)
    a = rng.random((n, n), dtype=np.float64)
    b = rng.random((n, n), dtype=np.float64)

    DOT_FLOPS = 2 * n**3 
    gflops_array = np.empty(it, dtype=np.float64)

    # Warm-up pour chauffer le CPU et initialiser le thread pool BLAS/OpenMP
    warm_size = min(n, 1000)
    _ = a[:warm_size, :warm_size] @ b[:warm_size, :warm_size]

    for i in trange(it):
        start_time = perf_counter()
        _ = a @ b  # Affectation explicite
        end_time = perf_counter()

        total_time = end_time - start_time
        gflops_array[i] = DOT_FLOPS / (total_time * 1e9)

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
    print(f"  Std Dev : +/- {metrics['std']:.2f} GFlops")

    return metrics

def STREAM_copy_run(n: int, it: int, seed: int = None) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    a = np.empty(n, dtype=np.float64)
    b = rng.random(n, dtype=np.float64)

    speed_array = np.empty(it, dtype=np.float64)

    # Warm-up / First-touch
    np.copyto(a, b)

    SIZE = 2 * n * 8  # 2 accès (1L + 1É) * 8 octets

    for i in trange(it, desc="STREAM | COPY", leave=False):
        start_time = perf_counter()
        np.copyto(a, b)
        total_time = perf_counter() - start_time

        speed_array[i] = SIZE / (total_time * 1e9)

    mem_per_array_gb = (n * 8) / (1024**3)
    metrics = {
        "peak": float(np.max(speed_array)),
        "lowest": float(np.min(speed_array)),
        "mean": float(np.mean(speed_array)),
        "median": float(np.median(speed_array)),
        "std": float(np.std(speed_array)),
    }

    print(
        f"--- Benchmark MEMORY | COPY (allocated: {2*mem_per_array_gb:.2f} GiB) | {it} itér. ---"
    )
    print(f"Peak: {metrics['peak']:.2f} GB/s | Median: {metrics['median']:.2f} GB/s | Std: +/- {metrics['std']:.2f} GB/s\n")

    return metrics


def STREAM_scale_run(n: int, it: int, seed: int = None) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    a = np.empty(n, dtype=np.float64)
    b = rng.random(n, dtype=np.float64)
    q = float(rng.random())

    speed_array = np.empty(it, dtype=np.float64)

    np.multiply(b, q, out=a)

    SIZE = 2 * n * 8

    for i in trange(it, desc="STREAM | SCALE", leave=False):
        start_time = perf_counter()
        np.multiply(b, q, out=a)
        total_time = perf_counter() - start_time

        speed_array[i] = SIZE / (total_time * 1e9)

    mem_per_array_gb = (n * 8) / (1024**3)
    metrics = {
        "peak": float(np.max(speed_array)),
        "lowest": float(np.min(speed_array)),
        "mean": float(np.mean(speed_array)),
        "median": float(np.median(speed_array)),
        "std": float(np.std(speed_array)),
    }

    print(
        f"--- Benchmark MEMORY | SCALE (allocatede: {2*mem_per_array_gb:.2f} GiB) | {it} itér. ---"
    )
    print(f"Peak: {metrics['peak']:.2f} GB/s | Median: {metrics['median']:.2f} GB/s | Std: +/- {metrics['std']:.2f} GB/s\n")

    return metrics


def STREAM_add_run(n: int, it: int, seed: int = None) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    a = np.empty(n, dtype=np.float64)
    b = rng.random(n, dtype=np.float64)
    c = rng.random(n, dtype=np.float64)

    speed_array = np.empty(it, dtype=np.float64)

    np.add(b, c, out=a)

    SIZE = 3 * n * 8  # 3 accès (2L + 1É) * 8 octets

    for i in trange(it, desc="STREAM | ADD", leave=False):
        start_time = perf_counter()
        np.add(b, c, out=a)
        total_time = perf_counter() - start_time

        speed_array[i] = SIZE / (total_time * 1e9)

    mem_per_array_gb = (n * 8) / (1024**3)
    metrics = {
        "peak": float(np.max(speed_array)),
        "lowest": float(np.min(speed_array)),
        "mean": float(np.mean(speed_array)),
        "median": float(np.median(speed_array)),
        "std": float(np.std(speed_array)),
    }

    print(
        f"--- Benchmark MEMORY | ADD (allocated: {3*mem_per_array_gb:.2f} GiB) | {it} itér. ---"
    )
    print(f"Peak: {metrics['peak']:.2f} GB/s | Median: {metrics['median']:.2f} GB/s | Std: +/- {metrics['std']:.2f} GB/s\n")

    return metrics


def STREAM_triad_run(n: int, it: int, seed: int = None) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    a = np.empty(n, dtype=np.float64)
    b = rng.random(n, dtype=np.float64)
    c = rng.random(n, dtype=np.float64)
    q = float(rng.random())

    speed_array = np.empty(it, dtype=np.float64)

    np.multiply(c, q, out=a)
    np.add(b, a, out=a)

    SIZE = 3 * n * 8

    for i in trange(it, desc="STREAM | TRIAD", leave=False):
        start_time = perf_counter()
        np.multiply(c, q, out=a)
        np.add(b, a, out=a)
        total_time = perf_counter() - start_time

        speed_array[i] = SIZE / (total_time * 1e9)

    mem_per_array_gb = (n * 8) / (1024**3)
    metrics = {
        "peak": float(np.max(speed_array)),
        "lowest": float(np.min(speed_array)),
        "mean": float(np.mean(speed_array)),
        "median": float(np.median(speed_array)),
        "std": float(np.std(speed_array)),
    }

    print(
        f"--- Benchmark MEMORY | TRIAD (allocated: {3*mem_per_array_gb:.2f} GiB) | {it} itér. ---"
    )
    print(f"Peak: {metrics['peak']:.2f} GB/s | Median: {metrics['median']:.2f} GB/s | Std: +/- {metrics['std']:.2f} GB/s\n")

    return metrics


def STREAM_run(n: int, it: int, seed: int = None) -> dict[str, dict[str, float]]:
    return {
        "copy": STREAM_copy_run(n, it, seed),
        "scale": STREAM_scale_run(n, it, seed),
        "add": STREAM_add_run(n, it, seed),
        "triad": STREAM_triad_run(n, it, seed),
    }