import os
from time import perf_counter
import numpy as np


def bench_memory_allocation(size_gb: float = 2.0, with_rng: bool = False) -> dict[str, float]:
    """Mesure la vitesse d'allocation mémoire virtuelle vs réelle (First-Touch) vs NumPy Random."""
    n_elements = int((size_gb * 1024**3) // 8)  # Nombre d'éléments float64
    bytes_allocated = n_elements * 8
    actual_gib = bytes_allocated / (1024**3)

    rng_tag = "" if with_rng else " (RNG désactivé)"
    print(f"--- Benchmark ALLOCATION MEMOIRE ({actual_gib:.2f} GiB){rng_tag} ---")

    # 1. Allocation Virtuelle Pure (malloc / mmap paresseux)
    t0 = perf_counter()
    arr = np.empty(n_elements, dtype=np.float64)
    dt_virt = perf_counter() - t0
    virt_speed_gbps = actual_gib / max(dt_virt, 1e-9)
    print(
        f"  1. Allocation Virtuelle (np.empty) : {dt_virt*1000:.3f} ms ({virt_speed_gbps:.1f} GiB/s)"
    )

    # 2. First-Touch / Kernel Page Faulting (Écriture physique dans la RAM)
    t0 = perf_counter()
    arr.fill(1.0)  # Déclenche les Page Faults Linux et l'allocation physique
    dt_touch = perf_counter() - t0
    touch_speed_gbps = actual_gib / dt_touch
    print(
        f"  2. First-Touch (Page Faults OS)   : {dt_touch*1000:.1f} ms ({touch_speed_gbps:.2f} GiB/s)"
    )

    # 3. Génération Random NumPy (Génération de données) — CPU-bound, optionnel
    metrics = {
        "virtual_alloc_ms": dt_virt * 1000,
        "first_touch_gbps": touch_speed_gbps,
    }
    if with_rng:
        t0 = perf_counter()
        _ = np.random.default_rng().random(n_elements, dtype=np.float64)
        dt_rng = perf_counter() - t0
        rng_speed_gbps = actual_gib / dt_rng
        print(
            f"  3. Génération Random (rng.random) : {dt_rng*1000:.1f} ms ({rng_speed_gbps:.2f} GiB/s)"
        )
        metrics["rng_generation_gbps"] = rng_speed_gbps

    return metrics


if __name__ == "__main__":
    bench_memory_allocation(size_gb=4.0)